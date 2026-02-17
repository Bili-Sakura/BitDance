from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

try:
    from flash_attn import flash_attn_func  # type: ignore

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None  # type: ignore
    _FLASH_ATTN_AVAILABLE = False


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    half = dim // 2
    t = time_factor * t.float()
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def time_shift_func(t: torch.Tensor, flow_shift: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    return (1.0 / flow_shift) / ((1.0 / flow_shift) + (1.0 / t - 1.0) ** sigma)


def get_score_from_velocity(velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    alpha_t, d_alpha_t = t, 1
    sigma_t, d_sigma_t = 1 - t, -1
    mean = x
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * velocity - mean) / var
    return score


def get_velocity_from_cfg(velocity: torch.Tensor, cfg: float, cfg_mult: int) -> torch.Tensor:
    if cfg_mult == 2:
        cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
        velocity = uncond_v + cfg * (cond_v - uncond_v)
    return velocity


def _randn_like(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def euler_step(x: torch.Tensor, v: torch.Tensor, dt: float, cfg: float, cfg_mult: int) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        v = get_velocity_from_cfg(v, cfg, cfg_mult)
        x = x + v * dt
    return x


def euler_maruyama_step(
    x: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    cfg: float,
    cfg_mult: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        v = get_velocity_from_cfg(v, cfg, cfg_mult)
        score = get_score_from_velocity(v, x, t)
        drift = v + (1 - t) * score
        noise_scale = (2.0 * (1.0 - t) * dt) ** 0.5
        x = x + drift * dt + noise_scale * _randn_like(x, generator=generator)
    return x


def euler_maruyama(
    input_dim: int,
    forward_fn,
    c: torch.Tensor,
    cfg: float = 1.0,
    num_sampling_steps: int = 20,
    last_step_size: float = 0.05,
    time_shift: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    cfg_mult = 1
    if cfg > 1.0:
        cfg_mult += 1

    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device, dtype=c.dtype, generator=generator)

    t_all = torch.linspace(0, 1 - last_step_size, num_sampling_steps + 1, device=c.device, dtype=torch.float32)
    t_all = time_shift_func(t_all, time_shift)
    dt = t_all[1:] - t_all[:-1]

    t = torch.tensor(0.0, device=c.device, dtype=torch.float32)
    t_batch = torch.zeros(c.shape[0], device=c.device)
    for i in range(num_sampling_steps):
        t_batch[:] = t
        combined = torch.cat([x] * cfg_mult, dim=0)
        output = forward_fn(combined, t_batch, c)
        if output.dim() == 2:
            v = (output - combined) / (1 - t_batch.view(-1, 1)).clamp_min(0.05)
        elif output.dim() == 3:
            v = (output - combined) / (1 - t_batch.view(-1, 1, 1)).clamp_min(0.05)
        else:
            raise ValueError(f"Unsupported output rank from diffusion head: {output.dim()}")

        x = euler_maruyama_step(x, v, t, float(dt[i]), cfg, cfg_mult, generator=generator)
        t += dt[i]

    combined = torch.cat([x] * cfg_mult, dim=0)
    t_batch[:] = 1 - last_step_size
    output = forward_fn(combined, t_batch, c)
    if output.dim() == 2:
        v = (output - combined) / (1 - t_batch.view(-1, 1)).clamp_min(0.05)
    elif output.dim() == 3:
        v = (output - combined) / (1 - t_batch.view(-1, 1, 1)).clamp_min(0.05)
    else:
        raise ValueError(f"Unsupported output rank from diffusion head: {output.dim()}")

    x = euler_step(x, v, last_step_size, cfg, cfg_mult)
    return torch.cat([x] * cfg_mult, dim=0)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.ada_ln_modulation = nn.Linear(channels, channels * 2, bias=True)
        self.linear = nn.Linear(channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale, shift = self.ada_ln_modulation(y).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale) + shift
        return self.linear(x)


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int) -> None:
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_head ({n_head}).")

        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_kv_dim = (self.n_head * 3) * self.head_dim
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wqkv(x).chunk(3, dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        if _FLASH_ATTN_AVAILABLE and xq.is_cuda:
            output = flash_attn_func(xq, xk, xv, causal=False)
        else:
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=0.0, is_causal=False)
            output = output.transpose(1, 2).contiguous()

        output = output.view(bsz, seqlen, self.dim)
        return self.wo(output)


class TransBlock(nn.Module):
    def __init__(self, channels: int, use_swiglu: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        self.attn = Attention(channels, n_head=channels // 128)

        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        hidden_dim = int(channels * 1.5)
        self.use_swiglu = use_swiglu
        if not use_swiglu:
            self.mlp = nn.Sequential(
                nn.Linear(channels, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, channels),
            )
        else:
            self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
            self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        gate1: torch.Tensor,
        scale2: torch.Tensor,
        shift2: torch.Tensor,
        gate2: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x) * (1 + scale1) + shift1
        h = self.attn(h)
        x = x + h * gate1

        h = self.norm2(x) * (1 + scale2) + shift2
        if not self.use_swiglu:
            h = self.mlp(h)
        else:
            h1, h2 = self.w1(h).chunk(2, dim=-1)
            h = self.w2(F.silu(h1) * h2)
        return x + h * gate2


class TransEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        z_channels: int,
        num_res_blocks: int,
        num_ada_ln_blocks: int = 2,
        grad_checkpointing: bool = False,
        parallel_num: int = 4,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.parallel_num = parallel_num

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([TransBlock(model_channels, use_swiglu) for _ in range(num_res_blocks)])
        self.ada_ln_blocks = nn.ModuleList(
            [nn.Linear(model_channels, model_channels * 6, bias=True) for _ in range(num_ada_ln_blocks)]
        )
        self.ada_ln_switch_freq = max(1, num_res_blocks // num_ada_ln_blocks)
        if (num_res_blocks % self.ada_ln_switch_freq) != 0:
            raise ValueError("num_res_blocks must be divisible by num_ada_ln_blocks")

        self.final_layer = FinalLayer(model_channels, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)

        nn.init.constant_(self.final_layer.ada_ln_modulation.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t).unsqueeze(1)
        c = self.cond_embed(c)
        y = F.silu(t + c)

        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada_ln_blocks[0](y).chunk(6, dim=-1)
        for i, block in enumerate(self.res_blocks):
            if i > 0 and i % self.ada_ln_switch_freq == 0:
                ada_ln_block = self.ada_ln_blocks[i // self.ada_ln_switch_freq]
                scale1, shift1, gate1, scale2, shift2, gate2 = ada_ln_block(y).chunk(6, dim=-1)
            x = block(x, scale1, shift1, gate1, scale2, shift2, gate2)

        output = self.final_layer(x, y)
        return 2 * torch.sigmoid(output) - 1


class BitDanceDiffusionHead(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        ch_target: int,
        ch_cond: int,
        ch_latent: int,
        depth_latent: int,
        depth_adanln: int,
        grad_checkpointing: bool = False,
        time_shift: float = 1.0,
        time_schedule: str = "logit_normal",
        P_mean: float = 0.0,
        P_std: float = 1.0,
        parallel_num: int = 4,
        diff_batch_mul: int = 1,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.ch_target = ch_target
        self.time_shift = time_shift
        self.time_schedule = time_schedule
        self.P_mean = P_mean
        self.P_std = P_std
        self.diff_batch_mul = diff_batch_mul

        self.net = TransEncoder(
            in_channels=ch_target,
            model_channels=ch_latent,
            z_channels=ch_cond,
            num_res_blocks=depth_latent,
            num_ada_ln_blocks=depth_adanln,
            grad_checkpointing=grad_checkpointing,
            parallel_num=parallel_num,
            use_swiglu=use_swiglu,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                if self.time_schedule == "logit_normal":
                    t = (torch.randn((x.shape[0]), device=x.device) * self.P_std + self.P_mean).sigmoid()
                    if self.time_shift != 1.0:
                        t = time_shift_func(t, self.time_shift)
                elif self.time_schedule == "uniform":
                    t = torch.rand((x.shape[0]), device=x.device)
                    if self.time_shift != 1.0:
                        t = time_shift_func(t, self.time_shift)
                else:
                    raise NotImplementedError(f"Unknown time_schedule={self.time_schedule}")

                e = torch.randn_like(x)
                ti = t.view(-1, 1, 1)
                z = (1.0 - ti) * e + ti * x
                v = (x - z) / (1 - ti).clamp_min(0.05)

        if self.diff_batch_mul > 1:
            chunks = self.diff_batch_mul
            x_pred_list = []
            z_chunks = torch.chunk(z, chunks, dim=0)
            t_chunks = torch.chunk(t, chunks, dim=0)
            cond_chunks = torch.chunk(cond, chunks, dim=0)
            for z_i, t_i, cond_i in zip(z_chunks, t_chunks, cond_chunks):
                x_pred_list.append(self.net(z_i, t_i, cond_i))
            x_pred = torch.cat(x_pred_list, dim=0)
        else:
            x_pred = self.net(z, t, cond)

        v_pred = (x_pred - z) / (1 - ti).clamp_min(0.05)
        with torch.autocast(device_type="cuda", enabled=False):
            v_pred = v_pred.float()
            loss = torch.mean((v - v_pred) ** 2, dim=2)
        return loss

    def sample(
        self,
        z: torch.Tensor,
        cfg: float,
        num_sampling_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return euler_maruyama(
            self.ch_target,
            self.net.forward,
            z,
            cfg,
            num_sampling_steps=num_sampling_steps,
            time_shift=self.time_shift,
            generator=generator,
        )

    def initialize_weights(self) -> None:
        self.net.initialize_weights()
