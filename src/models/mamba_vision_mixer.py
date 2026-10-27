"""
Tri-Directional 3D Mamba Mixer, GSC (Gated Spatial Convolution), and MambaLayer.
Implements native 3D state-space modeling along Axial, Sagittal, and Coronal planes
from 'Bridging transformers and state-space models' (Rahim et al., 2026).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .selective_scan import selective_scan_fn


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last or channels_first."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class MambaVisionMixer3D(nn.Module):
    """
    3D Tri-Directional Mamba Mixer.
    Processes volumetric feature maps across three orthogonal anatomical planes:
    Axial (z-first), Sagittal (x-first), and Coronal (y-first).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        num_directions: int = 3,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.num_directions = num_directions

        # In-projection to 3 directional streams
        self.in_proj = nn.Linear(self.d_model, self.d_inner * self.num_directions, bias=bias, **factory_kwargs)

        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(self.num_directions)
        ])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
            for _ in range(self.num_directions)
        ])

        # Initialize dt projections
        dt_init_std = (self.dt_rank ** -0.5) * dt_scale
        for dt_proj in self.dt_proj:
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

            dt = torch.exp(
                torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        # Independent SSM parameters per anatomical direction
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n", d=self.d_inner // 2
            ).contiguous()))
            for _ in range(self.num_directions)
        ])
        for a_log in self.A_log:
            a_log._no_weight_decay = True

        self.D = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner // 2, device=device))
            for _ in range(self.num_directions)
        ])
        for d in self.D:
            d._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner * self.num_directions, self.d_model, bias=bias, **factory_kwargs)

        # 1D depthwise convolutions for local context along each scan
        self.conv1d_x = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.d_inner // 2,
                out_channels=self.d_inner // 2,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner // 2,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            ) for _ in range(self.num_directions)
        ])
        self.conv1d_z = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.d_inner // 2,
                out_channels=self.d_inner // 2,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner // 2,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            ) for _ in range(self.num_directions)
        ])

    def forward(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int] | None = None) -> torch.Tensor:
        """
        hidden_states: [B, L, D] where L = D * H * W
        spatial_shape: (D, H, W)
        """
        b, l, d = hidden_states.shape
        xz = self.in_proj(hidden_states)  # [B, L, d_inner * num_directions]
        xz = rearrange(xz, "b l (n d) -> b n l d", n=self.num_directions)

        direction_outputs = []
        for i in range(self.num_directions):
            xz_dir = xz[:, i]  # [B, L, d_inner]
            xz_dir = rearrange(xz_dir, "b l d -> b d l")
            x, z = xz_dir.chunk(2, dim=1)  # Each: [B, d_inner // 2, L]

            # 1. Apply direction-specific 3D scanning layout
            if spatial_shape is not None:
                x, z = self._apply_3d_scanning(x, z, spatial_shape, direction=i)

            # 2. Local 1D conv + SiLU
            x = F.silu(self.conv1d_x[i](x))
            z = F.silu(self.conv1d_z[i](z))
            actual_seqlen = x.shape[2]

            # 3. Discretization projections
            A = -torch.exp(self.A_log[i].float())
            x_dbl = self.x_proj[i](rearrange(x, "b d l -> (b l) d"))
            dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = rearrange(self.dt_proj[i](dt), "(b l) d -> b d l", l=actual_seqlen)
            B_param = rearrange(B_param, "(b l) dstate -> b dstate l", l=actual_seqlen).contiguous()
            C_param = rearrange(C_param, "(b l) dstate -> b dstate l", l=actual_seqlen).contiguous()

            # 4. Run Selective Scan
            y = selective_scan_fn(
                x,
                dt,
                A,
                B_param,
                C_param,
                self.D[i].float(),
                z=None,
                delta_bias=self.dt_proj[i].bias.float(),
                delta_softplus=True,
                return_last_state=False
            )

            # 5. Combine with z and restore spatial alignment
            y = torch.cat([y, z], dim=1)  # [B, d_inner, L]
            if spatial_shape is not None:
                y = self._reverse_3d_scanning(y, spatial_shape, direction=i)

            direction_outputs.append(y)

        # 6. Channel concatenation across directions + linear projection
        y_combined = torch.cat(direction_outputs, dim=1)  # [B, d_inner * 3, L]
        y_combined = rearrange(y_combined, "b d l -> b l d")
        out = self.out_proj(y_combined)  # [B, L, d_model]
        return out

    def _apply_3d_scanning(self, x: torch.Tensor, z: torch.Tensor, spatial_shape: tuple[int, int, int], direction: int):
        d_size, h_size, w_size = spatial_shape
        x = rearrange(x, "b c (d h w) -> b c d h w", d=d_size, h=h_size, w=w_size)
        z = rearrange(z, "b c (d h w) -> b c d h w", d=d_size, h=h_size, w=w_size)

        if direction == 0:  # Axial scanning (depth-first: D -> H -> W)
            x = rearrange(x, "b c d h w -> b c (d h w)")
            z = rearrange(z, "b c d h w -> b c (d h w)")
        elif direction == 1:  # Sagittal scanning (width-first: W -> D -> H)
            x = rearrange(x, "b c d h w -> b c (w d h)")
            z = rearrange(z, "b c d h w -> b c (w d h)")
        elif direction == 2:  # Coronal scanning (height-first: H -> W -> D)
            x = rearrange(x, "b c d h w -> b c (h w d)")
            z = rearrange(z, "b c d h w -> b c (h w d)")
        return x, z

    def _reverse_3d_scanning(self, y: torch.Tensor, spatial_shape: tuple[int, int, int], direction: int):
        d_size, h_size, w_size = spatial_shape
        if direction == 0:  # Axial
            y = rearrange(y, "b c (d h w) -> b c d h w", d=d_size, h=h_size, w=w_size)
        elif direction == 1:  # Sagittal
            y = rearrange(y, "b c (w d h) -> b c d h w", d=d_size, h=h_size, w=w_size)
        elif direction == 2:  # Coronal
            y = rearrange(y, "b c (h w d) -> b c d h w", d=d_size, h=h_size, w=w_size)
        y = rearrange(y, "b c d h w -> b c (d h w)")
        return y


class MambaLayer(nn.Module):
    """Mamba wrapper layer with 3D tokenization and residual connection."""
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3, expand: int = 1, layer_scale: float | None = None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = MambaVisionMixer3D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_directions=3
        )
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.use_layer_scale = True
        else:
            self.use_layer_scale = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        x_skip = x
        img_dims = x.shape[2:]
        spatial_shape = img_dims if len(img_dims) == 3 else None
        n_tokens = x.shape[2:].numel()

        x_flat = x.reshape(b, c, n_tokens).transpose(-1, -2)  # [B, L, C]
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm, spatial_shape=spatial_shape)
        out = x_mamba.transpose(-1, -2).reshape(b, c, *img_dims)

        if self.use_layer_scale:
            out = self.gamma.view(1, -1, 1, 1, 1) * out
        return out + x_skip


class GSC(nn.Module):
    """
    Global Spatial Context (GSC) Residual Convolutional Refinement Block.
    Stabilizes optimization and injects 3D spatial priors prior to Mamba state-space scans.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonlinear = nn.ReLU(inplace=True)

        self.proj2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonlinear2 = nn.ReLU(inplace=True)

        self.proj3 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonlinear3 = nn.ReLU(inplace=True)

        self.proj4 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonlinear4 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_residual = x

        x1 = self.nonlinear(self.norm(self.proj(x)))
        x1 = self.nonlinear2(self.norm2(self.proj2(x1)))

        x2 = self.nonlinear3(self.norm3(self.proj3(x)))

        x_sum = x1 + x2
        out = self.nonlinear4(self.norm4(self.proj4(x_sum)))
        return out + x_residual


class MlpChannel(nn.Module):
    """Channel projection MLP."""
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
