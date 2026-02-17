from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from .constants import SUPPORTED_IMAGE_SIZES


PromptType = Union[str, List[str]]


class BitDanceDiffusionPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->projector->diffusion_head->autoencoder"

    def __init__(
        self,
        tokenizer,
        text_encoder,
        autoencoder,
        diffusion_head,
        projector,
        supported_image_sizes: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            autoencoder=autoencoder,
            diffusion_head=diffusion_head,
            projector=projector,
        )

        image_sizes = supported_image_sizes or SUPPORTED_IMAGE_SIZES
        self.register_to_config(supported_image_sizes=[list(size) for size in image_sizes])

        self.hidden_size = self.text_encoder.config.hidden_size
        self.vae_patch_size = self.autoencoder.patch_size
        self.parallel_num = int(self.diffusion_head.config.parallel_num)
        self.ps = int(self.parallel_num**0.5)
        if self.ps * self.ps != self.parallel_num:
            raise ValueError(
                f"parallel_num must be a perfect square (got {self.parallel_num})."
            )

        self._build_pos_embed()

    @property
    def supported_image_sizes(self) -> List[List[int]]:
        return [list(size) for size in self.config.supported_image_sizes]

    def _execution_device_fallback(self) -> torch.device:
        if getattr(self, "_execution_device", None) is not None:
            return self._execution_device
        return next(self.text_encoder.parameters()).device

    def _build_pos_embed(self) -> None:
        max_resolution = max(max(size) for size in self.supported_image_sizes)
        max_len = max_resolution // self.vae_patch_size
        pos_embed_1d = self._get_1d_sincos_pos_embed(self.hidden_size // 2, max_len)
        self.pos_embed_1d = pos_embed_1d

    @staticmethod
    def _get_1d_sincos_pos_embed(dim: int, max_len: int, pe_interpolation: float = 1.0) -> torch.Tensor:
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        pos = torch.arange(max_len, dtype=torch.float32) / pe_interpolation
        out = torch.einsum("m,d->md", pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)

    def _get_2d_embed(self, h: int, w: int, ps: int = 1) -> torch.Tensor:
        emb_v = self.pos_embed_1d[:h]
        emb_h = self.pos_embed_1d[:w]
        grid_v = emb_v.view(h, 1, self.hidden_size // 2).repeat(1, w, 1)
        grid_h = emb_h.view(1, w, self.hidden_size // 2).repeat(h, 1, 1)
        pos_embed = torch.cat([grid_h, grid_v], dim=-1)
        return rearrange(pos_embed, "(h p1) (w p2) c -> (h w p1 p2) c", p1=ps, p2=ps)

    def _encode_prompt_to_embeds(
        self,
        prompt: str,
        image_size: Tuple[int, int],
        num_images_per_prompt: int,
        guidance_scale: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        device = self._execution_device_fallback()
        model = self.text_encoder.model
        tokenizer = self.tokenizer

        cond_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        uncond_prompt = "<|im_start|>assistant\n"

        cond_ids = torch.tensor(tokenizer.encode(cond_prompt), device=device, dtype=torch.long)
        cond_emb = model.embed_tokens(cond_ids)
        uncond_emb = None
        if guidance_scale > 1.0:
            uncond_ids = torch.tensor(tokenizer.encode(uncond_prompt), device=device, dtype=torch.long)
            uncond_emb = model.embed_tokens(uncond_ids)

        image_h, image_w = image_size
        img_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        res_h_token_id = tokenizer.convert_tokens_to_ids(f"<|res_{image_h // self.vae_patch_size}|>")
        res_w_token_id = tokenizer.convert_tokens_to_ids(f"<|res_{image_w // self.vae_patch_size}|>")
        img_start_emb = model.embed_tokens(torch.tensor([img_start_id, res_h_token_id, res_w_token_id], device=device))

        for i in range(1, self.parallel_num):
            query_token_id = tokenizer.convert_tokens_to_ids(f"<|query_{i}|>")
            query_token = torch.tensor([query_token_id], device=device, dtype=torch.long)
            query_embed = model.embed_tokens(query_token)
            img_start_emb = torch.cat([img_start_emb, query_embed], dim=0)

        input_embeds_cond = torch.cat([cond_emb, img_start_emb], dim=0).unsqueeze(0).repeat(num_images_per_prompt, 1, 1)
        input_embeds_uncond = None
        if guidance_scale > 1.0 and uncond_emb is not None:
            input_embeds_uncond = torch.cat([uncond_emb, img_start_emb], dim=0).unsqueeze(0).repeat(num_images_per_prompt, 1, 1)
        return input_embeds_cond, input_embeds_uncond, img_start_emb

    def _decode_tokens_to_image(self, image_latents: torch.Tensor, image_size: Tuple[int, int], ps: int = 1) -> torch.Tensor:
        h, w = image_size
        image_latents = rearrange(image_latents, "b (h w p1 p2) c -> b c (h p1) (w p2)", h=h // ps, w=w // ps, p1=ps, p2=ps)
        return self.autoencoder.decode(image_latents)

    @torch.no_grad()
    def _generate_single_prompt(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        generator: Optional[torch.Generator],
        show_progress_bar: bool,
    ) -> torch.Tensor:
        image_size = (height, width)
        if list(image_size) not in self.supported_image_sizes:
            raise ValueError(
                f"image_size {list(image_size)} is not supported. "
                f"Please choose from {self.supported_image_sizes}"
            )

        h, w = height // self.vae_patch_size, width // self.vae_patch_size
        max_length = h * w
        step_width = self.parallel_num
        if max_length % step_width != 0:
            raise ValueError(
                f"max_length ({max_length}) must be divisible by parallel_num ({step_width})."
            )
        num_steps = max_length // step_width

        device = self._execution_device_fallback()
        model = self.text_encoder.model
        dtype = next(self.text_encoder.parameters()).dtype

        input_embeds_cond, input_embeds_uncond, _ = self._encode_prompt_to_embeds(
            prompt=prompt,
            image_size=image_size,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
        )
        pos_embed_for_diff = self._get_2d_embed(h, w, ps=self.ps).unsqueeze(0).to(device=device, dtype=dtype)

        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        with autocast_ctx:
            outputs_c = model(inputs_embeds=input_embeds_cond[:, :-step_width, :], use_cache=True)
            pkv_c = outputs_c.past_key_values

            bi_attn_mask = torch.ones(
                (input_embeds_cond.shape[0], 1, step_width, step_width + pkv_c[0][0].shape[2]),
                dtype=torch.bool,
                device=device,
            )
            outputs_c = model(
                inputs_embeds=input_embeds_cond[:, -step_width:, :],
                past_key_values=pkv_c,
                use_cache=True,
                attention_mask=bi_attn_mask,
            )
            pkv_c = outputs_c.past_key_values
            hidden_c = outputs_c.last_hidden_state[:, -step_width:]

            hidden_u = None
            pkv_u = None
            if guidance_scale > 1.0 and input_embeds_uncond is not None:
                outputs_u = model(inputs_embeds=input_embeds_uncond[:, :-step_width, :], use_cache=True)
                pkv_u = outputs_u.past_key_values
                outputs_u = model(
                    inputs_embeds=input_embeds_uncond[:, -step_width:, :],
                    past_key_values=pkv_u,
                    use_cache=True,
                    attention_mask=bi_attn_mask,
                )
                pkv_u = outputs_u.past_key_values
                hidden_u = outputs_u.last_hidden_state[:, -step_width:]

            out_tokens = []
            step_iter = range(num_steps)
            if show_progress_bar:
                step_iter = tqdm(step_iter, total=num_steps, desc="Decoding steps")

            for step in step_iter:
                if guidance_scale > 1.0 and hidden_u is not None:
                    h_fused = torch.cat([hidden_c, hidden_u], dim=0)
                else:
                    h_fused = hidden_c

                pos_slice = pos_embed_for_diff[:, step * step_width : (step + 1) * step_width, :]
                h_fused = h_fused + pos_slice
                pred_latents = self.diffusion_head.sample(
                    h_fused,
                    num_sampling_steps=num_inference_steps,
                    cfg=guidance_scale,
                    generator=generator,
                )
                curr_tokens = torch.sign(pred_latents)
                curr_embeds = self.projector(curr_tokens)
                out_tokens.append(curr_tokens[:num_images_per_prompt])

                model_input = curr_embeds + pos_slice
                bi_attn_mask = torch.ones(
                    (model_input.shape[0], 1, model_input.shape[1], model_input.shape[1] + pkv_c[0][0].shape[2]),
                    dtype=torch.bool,
                    device=device,
                )
                outputs_c = model(
                    inputs_embeds=model_input[:num_images_per_prompt],
                    past_key_values=pkv_c,
                    use_cache=True,
                    attention_mask=bi_attn_mask[:num_images_per_prompt],
                )
                pkv_c = outputs_c.past_key_values
                hidden_c = outputs_c.last_hidden_state[:, -step_width:]

                if guidance_scale > 1.0 and hidden_u is not None and pkv_u is not None:
                    outputs_u = model(
                        inputs_embeds=model_input[num_images_per_prompt:],
                        past_key_values=pkv_u,
                        use_cache=True,
                        attention_mask=bi_attn_mask[num_images_per_prompt:],
                    )
                    pkv_u = outputs_u.past_key_values
                    hidden_u = outputs_u.last_hidden_state[:, -step_width:]

        full_output = torch.cat(out_tokens, dim=1)
        return self._decode_tokens_to_image(full_output, image_size=(h, w), ps=self.ps)

    @torch.no_grad()
    def __call__(
        self,
        prompt: PromptType,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        show_progress_bar: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if len(prompts) == 0:
            raise ValueError("prompt must be a non-empty string or list of strings.")

        if isinstance(generator, list) and len(generator) != len(prompts):
            raise ValueError("When passing a list of generators, its length must equal len(prompt).")

        image_tensors = []
        for i, prompt_text in enumerate(prompts):
            prompt_generator = generator[i] if isinstance(generator, list) else generator
            images = self._generate_single_prompt(
                prompt=prompt_text,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=prompt_generator,
                show_progress_bar=show_progress_bar,
            )
            image_tensors.append(images)

        images_pt = torch.cat(image_tensors, dim=0)
        images_pt_01 = torch.clamp((images_pt + 1.0) / 2.0, 0.0, 1.0)

        if output_type == "pt":
            output_images = images_pt_01
        elif output_type == "np":
            output_images = images_pt_01.permute(0, 2, 3, 1).float().cpu().numpy()
        elif output_type == "pil":
            images_uint8 = (
                torch.clamp(127.5 * images_pt + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )
            output_images = [Image.fromarray(image) for image in images_uint8]
        else:
            raise ValueError(f"Unsupported output_type={output_type}. Expected 'pil', 'np', or 'pt'.")

        if not return_dict:
            return (output_images,)
        return ImagePipelineOutput(images=output_images)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_sampling_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        generator = None
        if seed is not None:
            device = self._execution_device_fallback()
            generator_device = "cuda" if device.type == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
        output = self(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_sampling_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            output_type="pil",
            return_dict=True,
            show_progress_bar=True,
        )
        return output.images
