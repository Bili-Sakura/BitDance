import torch
from diffusers import DiffusionPipeline
from pathlib import Path
import sys


# 1) One-time conversion:
# python scripts/convert_bitdance_to_diffusers.py \
#   --source_model_path models/BitDance-14B-64x \
#   --output_path models/BitDance-14B-64x-diffusers \
#   --torch_dtype bfloat16

model_path = Path("models/BitDance-14B-64x-diffusers")
device = "cuda"

# Use runtime code bundled by the conversion step.
sys.path.insert(0, str(model_path))

pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)

prompt = (
    "A close-up portrait in a cinematic photography style, capturing a girl-next-door "
    "look on a sunny daytime urban street. She wears a khaki sweater, with long, flowing "
    "hair gently draped over her shoulders. Her head is turned slightly, revealing soft "
    "facial features illuminated by realistic, delicate sunlight coming from the left. "
    "The sunlight subtly highlights individual strands of her hair. The image has a Canon "
    "film-like color tone, evoking a warm nostalgic atmosphere."
)

generator = torch.Generator(device=device).manual_seed(42)
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=50,  # try 25 for faster generation with slight quality tradeoff
    guidance_scale=7.5,
    num_images_per_prompt=1,
    generator=generator,
).images[0]
image.save("example.png")