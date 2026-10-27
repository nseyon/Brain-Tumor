"""
Full SwinMamba Dual Architecture:
- Backbone: Stage 0-1 3D Swin Attention, Stage 2-3 Tri-Directional 3D Mamba
- Decoder: UNETR with hierarchical skip connections and EV2 residual refinement
- Dual Task Execution:
  1. Pre-training Mode: 4-Channel Volumetric Reconstruction for Intra-Modality MIM
  2. Fine-tuning Mode: Joint Volumetric Segmentation (WT, TC, ET) + Glioma Grade Classification (High-Level vs Low-Level)
"""

from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_blocks import SwinTransformerBlock3D
from .mamba_vision_mixer import MambaLayer, GSC, MlpChannel
from .unetr_blocks import UnetrBasicBlock3D, UnetrUpBlock3D, UnetOutBlock3D, EV2Block


class SwinMambaEncoder(nn.Module):
    """
    Hybrid 3D Encoder:
    - Stages 0, 1: 3D Swin Transformer (Windowed Local Self-Attention)
    - Stages 2, 3: GSC + Tri-Directional Mamba (Axial, Sagittal, Coronal State-Space Modeling)
    """
    def __init__(
        self,
        in_chans: int = 4,
        depths: Sequence[int] = (2, 2, 2, 2),
        dims: Sequence[int] = (48, 96, 192, 384),
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        window_size: Sequence[int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        num_heads: Sequence[int] = (3, 6, 12, 24),
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.depths = depths

        # Downsample layers: Stem + 3 progressive downsamplers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(dims[0], eps=1e-4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(p=drop_rate / 2.0),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample = nn.Sequential(
                nn.BatchNorm3d(dims[i], eps=1e-4),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.Dropout3d(p=drop_rate / 2.0),
            )
            self.downsample_layers.append(downsample)

        # Drop path rate schedules
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.swin_stages = nn.ModuleList()
        self.conv_stages = nn.ModuleList()
        self.mamba_stages = nn.ModuleList()

        cur = 0
        for i in range(4):
            use_swin = (i == 0 or i == 1)
            if use_swin:
                # Stage 0 & 1: 3D Swin Transformer blocks
                swin_stage = nn.ModuleList([
                    SwinTransformerBlock3D(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=(0, 0, 0) if j % 2 == 0 else tuple(w // 2 for w in window_size),
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                    ) for j in range(depths[i])
                ])
                self.swin_stages.append(swin_stage)
                self.conv_stages.append(None)
                self.mamba_stages.append(None)
            else:
                # Stage 2 & 3: GSC + Tri-Directional Mamba
                self.swin_stages.append(None)
                self.conv_stages.append(GSC(dims[i]))
                mamba_stage = nn.Sequential(*[
                    MambaLayer(
                        dim=dims[i],
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        layer_scale=layer_scale_init_value,
                    ) for _ in range(depths[i])
                ])
                self.mamba_stages.append(mamba_stage)
            cur += depths[i]

        self.mlps = nn.ModuleList([
            MlpChannel(dims[i], 2 * dims[i]) for i in range(4)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm3d(dims[i], eps=1e-4) for i in range(4)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)

            if self.swin_stages[i] is not None:
                # Swin blocks: convert [B, C, D, H, W] -> [B, D, H, W, C]
                x_swin = x.permute(0, 2, 3, 4, 1)
                for swin_block in self.swin_stages[i]:
                    x_swin = swin_block(x_swin)
                x = x_swin.permute(0, 4, 1, 2, 3).contiguous()
            else:
                # Mamba blocks: apply GSC then Tri-Directional Mamba scans
                if self.conv_stages[i] is not None:
                    x = self.conv_stages[i](x)
                x = self.mamba_stages[i](x)

            x_norm = self.norms[i](x)
            x_out = self.mlps[i](x_norm)
            outs.append(x_out)
        return outs


class ClassificationHead(nn.Module):
    """
    Classification head for High-Level vs Low-Level glioma (HGG vs LGG).
    Extracts multi-scale features from bottleneck using dual Global Pooling (GAP + GMP)
    followed by a multi-layer perceptron with dropout to prevent overfitting.
    """
    def __init__(self, in_features: int = 768, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)
        fused_in = in_features * 2

        self.mlp = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        v_avg = self.gap(x).view(b, -1)
        v_max = self.gmp(x).view(b, -1)
        v_cat = torch.cat([v_avg, v_max], dim=1)
        return self.mlp(v_cat)


class SwinMambaDual(nn.Module):
    """
    Unified SwinMamba 3D Architecture for Brain Tumor Analysis.
    Supports:
      - mode='pretrain': Intra-Modality Masked Image Modeling (IS-MIM)
      - mode='finetune': Simultaneous 3D Segmentation (WT, TC, ET) and Grade Classification (HGG vs LGG)
    """
    def __init__(
        self,
        in_chans: int = 4,
        seg_out_chans: int = 3,
        num_classes: int = 2,
        feat_size: Sequence[int] = (48, 96, 192, 384),
        hidden_size: int = 768,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] = (7, 7, 7),
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 1,
        mode: str = "finetune",
    ):
        super().__init__()
        self.in_chans = in_chans
        self.seg_out_chans = seg_out_chans
        self.num_classes = num_classes
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.mode = mode

        # 1. Hybrid 3D Encoder
        self.encoder = SwinMambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            window_size=window_size,
            num_heads=num_heads,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 2. Skip Connection Encoders
        self.enc_stem = UnetrBasicBlock3D(in_channels=in_chans, out_channels=feat_size[0])
        self.enc_stage0 = UnetrBasicBlock3D(in_channels=feat_size[0], out_channels=feat_size[0])
        self.enc_stage1 = UnetrBasicBlock3D(in_channels=feat_size[1], out_channels=feat_size[1])
        self.enc_stage2 = UnetrBasicBlock3D(in_channels=feat_size[2], out_channels=feat_size[2])
        self.enc_bottleneck = UnetrBasicBlock3D(in_channels=feat_size[3], out_channels=hidden_size)

        # 3. Progressive UNETR Decoder
        self.decoder4 = UnetrUpBlock3D(in_channels=hidden_size, out_channels=feat_size[2], upsample_kernel_size=2)
        self.decoder3 = UnetrUpBlock3D(in_channels=feat_size[2], out_channels=feat_size[1], upsample_kernel_size=2)
        self.decoder2 = UnetrUpBlock3D(in_channels=feat_size[1], out_channels=feat_size[0], upsample_kernel_size=2)
        self.decoder1 = UnetrUpBlock3D(in_channels=feat_size[0], out_channels=feat_size[0], upsample_kernel_size=2)
        self.decoder_refine = EV2Block(in_channels=feat_size[0], out_channels=feat_size[0])

        # 4. Task Heads
        # (a) Reconstruction Head for SSL Pre-training (IS-MIM lightweight 3x3x3 conv)
        self.recon_head = nn.Conv3d(feat_size[0], in_chans, kernel_size=3, stride=1, padding=1)

        # (b) Segmentation Head for Fine-tuning (WT, TC, ET)
        self.seg_head = UnetOutBlock3D(in_channels=feat_size[0], out_channels=seg_out_chans)

        # (c) Classification Head for Fine-tuning (High-Level vs Low-Level)
        self.clf_head = ClassificationHead(in_features=hidden_size, num_classes=num_classes, dropout=0.4)

    def set_mode(self, mode: str):
        assert mode in ["pretrain", "finetune"], f"Mode must be 'pretrain' or 'finetune', got {mode}"
        self.mode = mode

    def forward(self, x_in: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        # Encode features
        outs = self.encoder(x_in)

        skip_stem = self.enc_stem(x_in)           # [B, 48, D, H, W]
        skip_0 = self.enc_stage0(outs[0])         # [B, 48, D/2, H/2, W/2]
        skip_1 = self.enc_stage1(outs[1])         # [B, 96, D/4, H/4, W/4]
        skip_2 = self.enc_stage2(outs[2])         # [B, 192, D/8, H/8, W/8]
        bottleneck = self.enc_bottleneck(outs[3]) # [B, 768, D/16, H/16, W/16]

        # Progressive Decoding
        d4 = self.decoder4(bottleneck, skip_2)  # [B, 192, D/8, H/8, W/8]
        d3 = self.decoder3(d4, skip_1)          # [B, 96, D/4, H/4, W/4]
        d2 = self.decoder2(d3, skip_0)          # [B, 48, D/2, H/2, W/2]
        d1 = self.decoder1(d2, skip_stem)       # [B, 48, D, H, W]
        dec_out = self.decoder_refine(d1)       # [B, 48, D, H, W]

        if self.mode == "pretrain":
            # Return reconstructed 4-modality volume
            recon = self.recon_head(dec_out)
            return recon
        else:
            # Mode 'finetune': Dual output for Segmentation and Classification
            seg_logits = self.seg_head(dec_out)
            clf_logits = self.clf_head(bottleneck)
            return {"seg": seg_logits, "clf": clf_logits}
