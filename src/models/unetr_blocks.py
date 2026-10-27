"""
UNETR Decoder Blocks, EV2 Residual Refinement, and Output Heads.
Implements progressive volumetric upsampling and skip fusion for SwinMamba.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EV2Block(nn.Module):
    """
    Enhanced V2 Residual Block from Rahim et al. (2026).
    Applies channel squeeze-and-excitation attention + Conv-Norm-ReLU residual refinement:
    Y = phi(A(R(X))) + A(R(X))
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Residual shortcut if channel dimensions differ
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Squeeze-and-Excitation Channel Attention A(.)
        self.se_pool = nn.AdaptiveAvgPool3d(1)
        self.se_fc1 = nn.Linear(out_channels, max(out_channels // 4, 8))
        self.se_fc2 = nn.Linear(max(out_channels // 4, 8), out_channels)

        # Refinement pathway phi(.)
        self.refine_conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.refine_norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + res)

        # Channel Attention
        b, c, _, _, _ = out.shape
        w = self.se_pool(out).view(b, c)
        w = F.relu(self.se_fc1(w), inplace=True)
        w = torch.sigmoid(self.se_fc2(w)).view(b, c, 1, 1, 1)
        attended = out * w

        # Refinement branch phi
        refine = self.act(self.refine_norm(self.refine_conv(attended)))
        return refine + attended


class UnetrBasicBlock3D(nn.Module):
    """Basic 3D convolutional block with residual connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + res)


class UnetrUpBlock3D(nn.Module):
    """
    Progressive upsampling block that fuses upsampled feature maps
    with high-resolution skip connections from the encoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: int = 2,
    ):
        super().__init__()
        self.transp_conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
        )
        # Fuse concatenated channels (out_channels + skip_channels)
        self.conv_block = EV2Block(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.transp_conv(x)

        # Pad out if rounding mismatch occurs
        diff_d = skip.size(2) - out.size(2)
        diff_h = skip.size(3) - out.size(3)
        diff_w = skip.size(4) - out.size(4)
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            out = F.pad(
                out,
                (0, diff_w, 0, diff_h, 0, diff_d)
            )

        fused = torch.cat([out, skip], dim=1)
        return self.conv_block(fused)


class UnetOutBlock3D(nn.Module):
    """Final 1x1x1 convolutional output block producing logits."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
