from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        use_conv_shortcut: bool = False,
        use_agn: bool = False,
    ) -> None:
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn:
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)

        if in_filters != out_filters:
            if use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)

        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        in_channels: int,
        num_res_blocks: int,
        z_channels: int,
        ch_mult: Sequence[int] = (1, 2, 2, 4),
        resolution: Optional[int] = None,
        double_z: bool = False,
    ) -> None:
        super().__init__()
        del out_ch, double_z
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1, bias=False)
        self.down = nn.ModuleList()

        in_ch_mult = (1,) + tuple(ch_mult)
        block_out = ch * ch_mult[0]
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=3, stride=2, padding=1)
            self.down.append(down)

        self.mid_block = nn.ModuleList([ResBlock(block_out, block_out) for _ in range(self.num_res_blocks)])
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)

        for block in self.mid_block:
            x = block(x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        return x


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.dim() < 3:
        raise ValueError("Expected a channels-first (*CHW) tensor of at least 3 dims.")
    c, h, w = x.shape[-3:]
    s = block_size**2
    if c % s != 0:
        raise ValueError(f"Expected C divisible by {s}, but got C={c}.")

    outer_dims = x.shape[:-3]
    x = x.view(-1, block_size, block_size, c // s, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)
    return x


class Upsampler(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depth_to_space(self.conv1(x), block_size=2)


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel: int, in_filters: int, num_groups: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x: torch.Tensor, quantizer: torch.Tensor) -> torch.Tensor:
        bsz, channels, _, _ = x.shape

        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps
        scale = scale.sqrt()
        scale = self.gamma(scale).view(bsz, channels, 1, 1)

        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(bsz, channels, 1, 1)

        x = self.gn(x)
        return scale * x + bias


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        in_channels: int,
        num_res_blocks: int,
        z_channels: int,
        ch_mult: Sequence[int] = (1, 2, 2, 4),
        resolution: Optional[int] = None,
        double_z: bool = False,
    ) -> None:
        super().__init__()
        del in_channels, resolution, double_z
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_blocks - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, padding=1, bias=True)
        self.mid_block = nn.ModuleList([ResBlock(block_in, block_in) for _ in range(self.num_res_blocks)])

        self.up = nn.ModuleList()
        self.adaptive = nn.ModuleList()
        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        style = z.clone()
        z = self.conv_in(z)

        for block in self.mid_block:
            z = block(z)

        for i_level in reversed(range(self.num_blocks)):
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            if i_level > 0:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)
        return z


class GANDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        in_channels: int,
        num_res_blocks: int,
        z_channels: int,
        ch_mult: Sequence[int] = (1, 2, 2, 4),
        resolution: Optional[int] = None,
        double_z: bool = False,
    ) -> None:
        super().__init__()
        del in_channels, resolution, double_z
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_blocks - 1]
        self.conv_in = nn.Conv2d(z_channels * 2, block_in, kernel_size=3, padding=1, bias=True)
        self.mid_block = nn.ModuleList([ResBlock(block_in, block_in) for _ in range(self.num_res_blocks)])

        self.up = nn.ModuleList()
        self.adaptive = nn.ModuleList()
        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        style = z.clone()
        noise = torch.randn_like(z, device=z.device)
        z = torch.cat([z, noise], dim=1)
        z = self.conv_in(z)

        for block in self.mid_block:
            z = block(z)

        for i_level in reversed(range(self.num_blocks)):
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            if i_level > 0:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)
        return z


class BitDanceAutoencoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, ddconfig: Dict[str, Any], gan_decoder: bool = False) -> None:
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = GANDecoder(**ddconfig) if gan_decoder else Decoder(**ddconfig)

    @property
    def z_channels(self) -> int:
        return int(self.config.ddconfig["z_channels"])

    @property
    def patch_size(self) -> int:
        ch_mult = self.config.ddconfig["ch_mult"]
        return 2 ** (len(ch_mult) - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        codebook_value = torch.tensor([1.0], device=h.device, dtype=h.dtype)
        quant_h = torch.where(h > 0, codebook_value, -codebook_value)
        return quant_h

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        return self.decoder(quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quant = self.encode(x)
        return self.decode(quant)
