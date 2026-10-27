"""
3D Swin Transformer Blocks & 3D Window Attention for Stages 0 and 1.
Implements local window-based self-attention (W-MSA) and shifted window self-attention (SW-MSA).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence


def window_partition_3d(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    """
    Args:
        x: tensor of shape [B, D, H, W, C]
        window_size: (Wd, Wh, Ww)
    Returns:
        windows: [B * num_windows, Wd, Wh, Ww, C]
    """
    b, d, h, w, c = x.shape
    wd, wh, ww = window_size
    x = x.view(
        b,
        d // wd, wd,
        h // wh, wh,
        w // ww, ww,
        c
    )
    # Permute to [B, D//wd, H//wh, W//ww, wd, wh, ww, C]
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd, wh, ww, c)
    return windows


def window_reverse_3d(windows: torch.Tensor, window_size: Sequence[int], dims: Sequence[int]) -> torch.Tensor:
    """
    Args:
        windows: [B * num_windows, Wd, Wh, Ww, C]
        window_size: (Wd, Wh, Ww)
        dims: (D, H, W)
    Returns:
        x: [B, D, H, W, C]
    """
    wd, wh, ww = window_size
    d, h, w = dims
    b = int(windows.shape[0] / (d * h * w / (wd * wh * ww)))
    x = windows.view(
        b,
        d // wd,
        h // wh,
        w // ww,
        wd,
        wh,
        ww,
        -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x


def drop_path_3d(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_3d(x, self.drop_prob, self.training)


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-Head Self-Attention with relative position bias."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        wd, wh, ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Coordinate grid for relative position index
        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # [3, Wd, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [3, Wd*Wh*Ww]

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 3]

        relative_coords[:, :, 0] += wd - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= (2 * ww - 1)

        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, head_dim]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, num_heads, N, N]

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            n, n, -1
        )  # [N, N, num_heads]
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            # Mask shape: [num_windows, N, N]
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SwinTransformerBlock3D(nn.Module):
    """Single 3D Swin Transformer Block with W-MSA / SW-MSA and MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int] = (7, 7, 7),
        shift_size: Sequence[int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, D, H, W, C]
        """
        b, d, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        # Pad feature map if not divisible by window size
        wd, wh, ww = self.window_size
        pad_d = (wd - d % wd) % wd
        pad_h = (wh - h % wh) % wh
        pad_w = (ww - w % ww) % ww

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        dp, hp, wp = x.shape[1], x.shape[2], x.shape[3]

        # Cyclic shift
        sd, sh, sw = self.shift_size
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition_3d(shifted_x, self.window_size)  # [B*nW, Wd, Wh, Ww, C]
        x_windows = x_windows.view(-1, wd * wh * ww, c)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=mask_matrix)

        # Reverse window partition
        attn_windows = attn_windows.view(-1, wd, wh, ww, c)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, (dp, hp, wp))

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(sd, sh, sw), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :d, :h, :w, :].contiguous()

        # FFN & Residual
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    """3D Patch Merging layer for downsampling between stages."""
    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, H, W, C]
        Returns: [B, D/2, H/2, W/2, 2*C]
        """
        b, d, h, w, c = x.shape
        pad_d = d % 2
        pad_h = h % 2
        pad_w = w % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # [B, D/2, H/2, W/2, 8*C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, D/2, H/2, W/2, 2*C]
        return x
