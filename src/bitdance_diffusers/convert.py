from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, Qwen3ForCausalLM

from .modeling_autoencoder import BitDanceAutoencoder
from .modeling_diffusion_head import BitDanceDiffusionHead
from .modeling_projector import BitDanceProjector
from .pipeline_bitdance import BitDanceDiffusionPipeline


def _resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype '{dtype}'. Choose from {sorted(mapping)}.")
    return mapping[dtype]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy_runtime_source(output_path: Path) -> None:
    package_root = Path(__file__).resolve().parent
    target_pkg = output_path / "bitdance_diffusers"
    shutil.copytree(package_root, target_pkg, dirs_exist_ok=True)

    loader_script = output_path / "load_pipeline.py"
    loader_script.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "",
                "from diffusers import DiffusionPipeline",
                "",
                "model_dir = Path(__file__).resolve().parent",
                "sys.path.insert(0, str(model_dir))",
                'pipe = DiffusionPipeline.from_pretrained(model_dir, trust_remote_code=True).to("cuda")',
                'images = pipe(prompt="A scenic mountain lake at sunrise.").images',
                'images[0].save("sample.png")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def convert_bitdance_to_diffusers(
    source_model_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
    device: str = "cpu",
    copy_runtime_source: bool = True,
) -> Path:
    source = Path(source_model_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_dtype(torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(source)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()

    ae_config = _load_json(source / "ae_config.json")
    ddconfig = ae_config.get("ddconfig", ae_config)
    gan_decoder = bool(ae_config.get("gan_decoder", False))
    autoencoder = BitDanceAutoencoder(ddconfig=ddconfig, gan_decoder=gan_decoder).eval()
    autoencoder.load_state_dict(load_safetensors(source / "ae.safetensors"), strict=True, assign=True)

    vision_head_config = _load_json(source / "vision_head_config.json")
    diffusion_head = BitDanceDiffusionHead(**vision_head_config).eval()
    diffusion_head.load_state_dict(load_safetensors(source / "vision_head.safetensors"), strict=True, assign=True)

    projector = BitDanceProjector(
        in_dim=int(ddconfig["z_channels"]),
        out_dim=int(text_encoder.config.hidden_size),
        hidden_act="gelu_pytorch_tanh",
    ).eval()
    projector.load_state_dict(load_safetensors(source / "projector.safetensors"), strict=True, assign=True)

    if device:
        text_encoder.to(device=device)
        autoencoder.to(device=device)
        diffusion_head.to(device=device)
        projector.to(device=device)

    pipeline = BitDanceDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        autoencoder=autoencoder,
        diffusion_head=diffusion_head,
        projector=projector,
    )
    pipeline.save_pretrained(output, safe_serialization=True)

    if copy_runtime_source:
        _copy_runtime_source(output)

    return output


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BitDance checkpoints to Diffusers format.")
    parser.add_argument("--source_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--copy_runtime_source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy self-contained runtime source into output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    converted = convert_bitdance_to_diffusers(
        source_model_path=args.source_model_path,
        output_path=args.output_path,
        torch_dtype=args.torch_dtype,
        device=args.device,
        copy_runtime_source=args.copy_runtime_source,
    )
    print(f"Saved converted Diffusers pipeline to: {converted}")


if __name__ == "__main__":
    main()
