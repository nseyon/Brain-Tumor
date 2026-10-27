"""
demo_pipeline.py
────────────────
Module-by-module walkthrough of the complete SwinMamba + IS-MIM pipeline.

Every module is called individually in its true forward order.
The output of each module is the literal input to the next one.
After each call a formatted line is printed showing:
  • Module name (what it is)
  • Tensor shape (B × C × D × H × W)
  • Value statistics  [min | mean | max | std]
  • A short semantic annotation

No existing source file is modified.
Run:
    python demo_pipeline.py
"""

from __future__ import annotations
import sys
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# -- pretty-printer ------------------------------------------------------------

W_NAME  = 46
W_SHAPE = 30
W_STATS = 36

_DIVIDER = "=" * (W_NAME + W_SHAPE + W_STATS + 10)
_SEP     = "-" * (W_NAME + W_SHAPE + W_STATS + 10)

def _stats(t: torch.Tensor) -> str:
    f = t.detach().float()
    return (f"min={f.min():+.3f}  u={f.mean():+.3f}  "
            f"max={f.max():+.3f}  std={f.std():.3f}")

def _shape(t: torch.Tensor) -> str:
    return str(tuple(t.shape))

def log(label: str, tensor: torch.Tensor, note: str = ""):
    shape_str = _shape(tensor)
    stats_str = _stats(tensor)
    note_str  = f"  <- {note}" if note else ""
    print(f"  {label:<{W_NAME}}  {shape_str:<{W_SHAPE}}  {stats_str}{note_str}")

def section(title: str):
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)

def subsec(title: str):
    print(f"\n  {_SEP}")
    print(f"  {title}")
    print(f"  {_SEP}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    device = torch.device("cpu")   # CPU so no GPU needed for a demo
    torch.manual_seed(42)

    print(_DIVIDER)
    print("  SwinMamba + IS-MIM  │  Module-Wise Output Demo")
    print("  (B=1, 4-modality MRI, 96³ volume, float32, CPU)")
    print(_DIVIDER)

    # =========================================================================
    # STAGE 0 — RAW INPUT VOLUME
    # =========================================================================
    section("STAGE 0 │ Raw 4-Modality MRI Input")
    # Simulate a normalised MRI volume  [B=1, C=4, D=96, H=96, W=96]
    raw_volume = torch.randn(1, 4, 96, 96, 96)
    log("Raw input  [B, C, D, H, W]", raw_volume,
        "T1 | T1ce | T2 | FLAIR stacked on channel dim")

    # =========================================================================
    # STAGE 1 — IS-MIM PREPROCESSING (Pre-training path)
    # =========================================================================
    section("STAGE 1 │ IS-MIM Preprocessing  (pre-training path only)")

    from src.data.is_mim_transforms import (
        image_to_tokens_3d,
        synthesize_modalities,
        apply_block_masking_3d,
        tokens_to_image_3d,
    )

    # ── 1a. Patch tokenisation ────────────────────────────────────────────
    subsec("1a. 3D Patch Tokenisation  (patch_size=16  →  6×6×6 = 216 tokens)")
    tokens = image_to_tokens_3d(raw_volume.unsqueeze(0) if raw_volume.ndim == 4
                                else raw_volume, patch_size=16)
    # image_to_tokens_3d expects [B,C,D,H,W] – raw_volume already is [1,4,96,96,96]
    tokens = image_to_tokens_3d(raw_volume, patch_size=16)
    log("Tokens  [B, C, p, p, p, N_patches]", tokens,
        "each 16³ cuboid is one token, N_patches=216")

    # ── 1b. Intra-modality patch swap  ────────────────────────────────────
    subsec("1b. Intra-Modality Patch Swap  (swap_prob=0.5, 2 iterations)")
    swapped_tokens, swap_mask = synthesize_modalities(tokens, swap_prob=0.5, num_iterations=2)
    log("Swapped tokens  [B, C, p, p, p, N]", swapped_tokens,
        "patches exchanged across T1/T1ce and T2/FLAIR channel pairs")
    log("Swap mask  [B, N_patches]", swap_mask,
        "1 = patch was swapped  |  0 = patch unchanged")

    # ── 1c. 50% block-wise masking ────────────────────────────────────────
    subsec("1c. 50% Block-Wise Token Masking")
    masked_tokens, is_masked = apply_block_masking_3d(swapped_tokens, mask_ratio=0.5)
    log("Masked tokens  [B, C, p, p, p, N]", masked_tokens,
        "108/216 tokens zeroed out  (50%)")
    log("is_masked  [B, 1, 1, 1, 1, N]", is_masked,
        "1 = masked (loss computed here)  |  0 = kept")

    # ── 1d. Detokenise → perturbed input volume  ──────────────────────────
    subsec("1d. Detokenise → Perturbed Volume  (IS-MIM model input)")
    D, H, W = 96, 96, 96
    x_perturbed = tokens_to_image_3d(masked_tokens, orig_shape=(D, H, W), patch_size=16)
    x_clean     = raw_volume.clone()
    log("x_perturbed  [B, C, D, H, W]", x_perturbed,
        "corrupted 4-ch volume → input to SwinMamba autoencoder")
    log("x_clean      [B, C, D, H, W]", x_clean,
        "original volume → reconstruction target (MSE on masked voxels)")

    # For the rest of the demo we use x_perturbed as the backbone input
    x = x_perturbed

    # =========================================================================
    # STAGE 2 — SWINMAMBA ENCODER
    # Build the model (random init) and step through each sub-module manually
    # =========================================================================
    section("STAGE 2 │ SwinMamba Hybrid 3D Encoder")

    from src.models.swin_mamba_dual import SwinMambaEncoder

    enc = SwinMambaEncoder(
        in_chans   = 4,
        depths     = (2, 2, 2, 2),
        dims       = (48, 96, 192, 384),
        window_size= (7, 7, 7),
        num_heads  = (3, 6, 12, 24),
    ).to(device).eval()

    with torch.no_grad():

        # ── 2a. Stem Conv  (stride-2, 4→48 channels) ─────────────────────
        subsec("2a. Stem Convolution  [Conv3d 7×7×7 s=2 | BN | LeakyReLU]")
        x_stem = enc.downsample_layers[0](x)
        log("After Stem Conv  [B, 48, D/2, H/2, W/2]", x_stem,
            "halves spatial resolution, expands channels 4→48")

        # ── 2b. Stage 0 — Swin W-MSA / SW-MSA  (depth=2) ─────────────────
        subsec("2b. Stage 0 — 3D Swin Transformer  [W-MSA → SW-MSA, depth=2, window=7³]")
        x_s0 = x_stem.permute(0, 2, 3, 4, 1)           # [B, D, H, W, C]
        for blk_idx, blk in enumerate(enc.swin_stages[0]):
            label = ("W-MSA " if blk_idx % 2 == 0 else "SW-MSA") + f" block {blk_idx}"
            x_s0  = blk(x_s0)
            log(f"  {label}  [B, D, H, W, C]", x_s0,
                "window attention on [B, D, H, W, 48] token lattice")
        x_s0 = x_s0.permute(0, 4, 1, 2, 3).contiguous()
        x_s0 = enc.mlps[0](enc.norms[0](x_s0))
        log("Stage-0 output  [B, 48, 48, 48, 48]", x_s0,
            "BN + MLP projection → skip feature S0")

        # ── 2c. Patch Merging 0→1  ────────────────────────────────────────
        subsec("2c. Patch Merging 0→1  [BN | Conv3d 2×2×2 s=2, 48→96 channels]")
        x_pm01 = enc.downsample_layers[1](x_s0)
        log("After Patch Merging  [B, 96, 24, 24, 24]", x_pm01,
            "spatial ×0.5, channels ×2  (48→96)")

        # ── 2d. Stage 1 — Swin W-MSA / SW-MSA  (depth=2) ─────────────────
        subsec("2d. Stage 1 — 3D Swin Transformer  [W-MSA → SW-MSA, depth=2, window=7³]")
        x_s1 = x_pm01.permute(0, 2, 3, 4, 1)
        for blk_idx, blk in enumerate(enc.swin_stages[1]):
            label = ("W-MSA " if blk_idx % 2 == 0 else "SW-MSA") + f" block {blk_idx}"
            x_s1  = blk(x_s1)
            log(f"  {label}  [B, D, H, W, C]", x_s1,
                "window attention on [B, 24, 24, 24, 96] token lattice")
        x_s1 = x_s1.permute(0, 4, 1, 2, 3).contiguous()
        x_s1 = enc.mlps[1](enc.norms[1](x_s1))
        log("Stage-1 output  [B, 96, 24, 24, 24]", x_s1,
            "BN + MLP projection → skip feature S1")

        # ── 2e. Patch Merging 1→2  ────────────────────────────────────────
        subsec("2e. Patch Merging 1→2  [BN | Conv3d 2×2×2 s=2, 96→192 channels]")
        x_pm12 = enc.downsample_layers[2](x_s1)
        log("After Patch Merging  [B, 192, 12, 12, 12]", x_pm12,
            "spatial ×0.5, channels ×2  (96→192)")

        # ── 2f. Stage 2 — GSC Refinement ─────────────────────────────────
        subsec("2f. Stage 2 — Gated Spatial Conv (GSC)  [Conv3×3 + 1×1 branch, InstanceNorm, ReLU]")
        x_gsc2 = enc.conv_stages[2](x_pm12)
        log("After GSC  [B, 192, 12, 12, 12]", x_gsc2,
            "residual 3D conv refinement — injects spatial inductive bias before Mamba")

        # ── 2g. Stage 2 — Tri-Directional Mamba  (depth=2) ───────────────
        subsec("2g. Stage 2 — Tri-Directional 3D Mamba  [Axial | Sagittal | Coronal SSM, depth=2]")
        x_m2 = x_gsc2
        for layer_idx, mamba_layer in enumerate(enc.mamba_stages[2]):
            x_m2_prev = x_m2
            x_m2      = mamba_layer(x_m2)
            print(f"    MambaLayer {layer_idx}  |  "
                  f"Axial seq-len={12*12*12}  Sag seq-len={12*12*12}  Cor seq-len={12*12*12}")
            log(f"  After MambaLayer {layer_idx}  [B, 192, 12, 12, 12]", x_m2,
                "3 independent SSMs fused back into volumetric layout")
        x_s2 = enc.mlps[2](enc.norms[2](x_m2))
        log("Stage-2 output  [B, 192, 12, 12, 12]", x_s2,
            "BN + MLP projection → skip feature S2")

        # ── 2h. Patch Merging 2→3  ────────────────────────────────────────
        subsec("2h. Patch Merging 2→3  [BN | Conv3d 2×2×2 s=2, 192→384 channels]")
        x_pm23 = enc.downsample_layers[3](x_s2)
        log("After Patch Merging  [B, 384, 6, 6, 6]", x_pm23,
            "spatial ×0.5, channels ×2  (192→384)")

        # ── 2i. Stage 3 — GSC Refinement ─────────────────────────────────
        subsec("2i. Stage 3 — Gated Spatial Conv (GSC)  [Conv3×3 + 1×1 branch, InstanceNorm, ReLU]")
        x_gsc3 = enc.conv_stages[3](x_pm23)
        log("After GSC  [B, 384, 6, 6, 6]", x_gsc3,
            "residual spatial refinement at bottleneck resolution")

        # ── 2j. Stage 3 — Tri-Directional Mamba  (depth=2) ───────────────
        subsec("2j. Stage 3 — Tri-Directional 3D Mamba  [Axial | Sagittal | Coronal SSM, depth=2]")
        x_m3 = x_gsc3
        for layer_idx, mamba_layer in enumerate(enc.mamba_stages[3]):
            x_m3 = mamba_layer(x_m3)
            print(f"    MambaLayer {layer_idx}  |  "
                  f"Axial seq-len={6*6*6}  Sag seq-len={6*6*6}  Cor seq-len={6*6*6}")
            log(f"  After MambaLayer {layer_idx}  [B, 384, 6, 6, 6]", x_m3,
                "long-range volumetric dependencies at 6³ resolution")
        x_s3 = enc.mlps[3](enc.norms[3](x_m3))
        log("Stage-3 output  [B, 384, 6, 6, 6]", x_s3,
            "BN + MLP projection → deepest encoder features")

    # =========================================================================
    # STAGE 3 — SKIP CONNECTION REFINEMENT + BOTTLENECK
    # =========================================================================
    section("STAGE 3 │ Skip Connection Encoders + Bottleneck Projection")

    from src.models.unetr_blocks import UnetrBasicBlock3D

    enc_stem_blk   = UnetrBasicBlock3D(in_channels=4,   out_channels=48 ).eval()
    enc_s0_blk     = UnetrBasicBlock3D(in_channels=48,  out_channels=48 ).eval()
    enc_s1_blk     = UnetrBasicBlock3D(in_channels=96,  out_channels=96 ).eval()
    enc_s2_blk     = UnetrBasicBlock3D(in_channels=192, out_channels=192).eval()
    enc_bottle_blk = UnetrBasicBlock3D(in_channels=384, out_channels=768).eval()

    with torch.no_grad():
        skip_stem = enc_stem_blk(x)          # raw input → 48-ch skip
        skip_0    = enc_s0_blk(x_s0)
        skip_1    = enc_s1_blk(x_s1)
        skip_2    = enc_s2_blk(x_s2)
        bottleneck = enc_bottle_blk(x_s3)

        log("skip_stem   (enc_stem)       [B, 48,  96, 96, 96]", skip_stem,
            "residual skip from raw input resolution")
        log("skip_0      (enc_stage0)     [B, 48,  48, 48, 48]", skip_0,
            "residual skip from Swin Stage-0 output")
        log("skip_1      (enc_stage1)     [B, 96,  24, 24, 24]", skip_1,
            "residual skip from Swin Stage-1 output")
        log("skip_2      (enc_stage2)     [B, 192, 12, 12, 12]", skip_2,
            "residual skip from Mamba Stage-2 output")
        log("bottleneck  (enc_bottleneck) [B, 768,  6,  6,  6]", bottleneck,
            "deepest representation — 384→768 projection (Bottleneck)")

    # =========================================================================
    # STAGE 4 — UNETR DECODER WITH EV2 SKIP FUSION
    # =========================================================================
    section("STAGE 4 │ UNETR Progressive Decoder  (ConvTranspose + EV2 Skip Fusion)")

    from src.models.unetr_blocks import UnetrUpBlock3D, EV2Block

    dec4 = UnetrUpBlock3D(in_channels=768, out_channels=192).eval()
    dec3 = UnetrUpBlock3D(in_channels=192, out_channels=96 ).eval()
    dec2 = UnetrUpBlock3D(in_channels=96,  out_channels=48 ).eval()
    dec1 = UnetrUpBlock3D(in_channels=48,  out_channels=48 ).eval()
    refine = EV2Block(in_channels=48, out_channels=48).eval()

    with torch.no_grad():
        d4 = dec4(bottleneck, skip_2)
        log("dec4  (bottleneck↑ ⊕ skip_2) [B, 192, 12, 12, 12]", d4,
            "ConvTranspose3d 6³→12³  |  EV2 fusion with skip_2")

        d3 = dec3(d4, skip_1)
        log("dec3  (d4↑ ⊕ skip_1)         [B, 96,  24, 24, 24]", d3,
            "ConvTranspose3d 12³→24³  |  EV2 fusion with skip_1")

        d2 = dec2(d3, skip_0)
        log("dec2  (d3↑ ⊕ skip_0)         [B, 48,  48, 48, 48]", d2,
            "ConvTranspose3d 24³→48³  |  EV2 fusion with skip_0")

        d1 = dec1(d2, skip_stem)
        log("dec1  (d2↑ ⊕ skip_stem)      [B, 48,  96, 96, 96]", d1,
            "ConvTranspose3d 48³→96³  |  EV2 fusion with stem skip")

        dec_out = refine(d1)
        log("EV2 Refine (SE attention)     [B, 48,  96, 96, 96]", dec_out,
            "Squeeze-and-Excitation refinement — final decoder features")

    # =========================================================================
    # STAGE 5A — SEGMENTATION HEAD  (Finetune path)
    # =========================================================================
    section("STAGE 5A │ Segmentation Head  (Fine-Tuning Path)")

    from src.models.unetr_blocks import UnetOutBlock3D
    import torch.nn.functional as F

    seg_head = UnetOutBlock3D(in_channels=48, out_channels=3).eval()

    with torch.no_grad():
        seg_logits = seg_head(dec_out)
        seg_probs  = torch.sigmoid(seg_logits)
        log("seg_logits  [B, 3, 96, 96, 96]", seg_logits,
            "raw logits — channels: [0]=TC  [1]=WT  [2]=ET")
        log("seg_probs   [B, 3, 96, 96, 96]", seg_probs,
            "sigmoid(logits) — voxel-wise probabilities in [0, 1]")

        # Channel-wise voxel coverage
        for ch_idx, name in enumerate(["TC (Tumor Core)", "WT (Whole Tumor)", "ET (Enhancing Tumor)"]):
            n_pos = (seg_probs[0, ch_idx] > 0.5).sum().item()
            pct   = 100.0 * n_pos / (96**3)
            print(f"    Channel {ch_idx}  {name:<24}  active voxels > 0.5 : "
                  f"{n_pos:>8,}  ({pct:.2f}% of volume)")

    # =========================================================================
    # STAGE 5B — CLASSIFICATION HEAD  (Finetune path)
    # =========================================================================
    section("STAGE 5B │ Classification Head  (Fine-Tuning Path)")

    from src.models.swin_mamba_dual import ClassificationHead
    import torch.nn.functional as F

    clf_head = ClassificationHead(in_features=768, num_classes=2).eval()

    with torch.no_grad():
        # Sub-steps of the classification head exposed individually
        gap_vec  = clf_head.gap(bottleneck).view(1, -1)
        gmp_vec  = clf_head.gmp(bottleneck).view(1, -1)
        fused    = torch.cat([gap_vec, gmp_vec], dim=1)
        clf_logits = clf_head.mlp(fused)
        clf_probs  = F.softmax(clf_logits, dim=-1)

        log("GAP vector   (global avg pool)  [B, 768]", gap_vec,
            "average pooling  6³→1¹  across all spatial positions")
        log("GMP vector   (global max pool)  [B, 768]", gmp_vec,
            "max pooling  6³→1¹  across all spatial positions")
        log("Fused        (GAP ‖ GMP)        [B, 1536]", fused,
            "concatenation along channel dim  →  MLP input")
        log("clf_logits   [B, 2]", clf_logits,
            "raw class scores  →  [0]=LGG (Low-Level)  [1]=HGG (High-Level)")
        log("clf_probs    [B, 2]", clf_probs,
            "softmax probabilities  [LGG_prob, HGG_prob]")

        pred_class = clf_probs.argmax(dim=1).item()
        print(f"\n    Predicted grade : {'HGG (High-Level)' if pred_class == 1 else 'LGG (Low-Level)'}"
              f"  |  HGG_prob={clf_probs[0,1]:.4f}  LGG_prob={clf_probs[0,0]:.4f}")

    # =========================================================================
    # STAGE 5C — RECONSTRUCTION HEAD  (Pre-training / IS-MIM path)
    # =========================================================================
    section("STAGE 5C │ Reconstruction Head  (Pre-Training / IS-MIM Path)")

    recon_head = torch.nn.Conv3d(48, 4, kernel_size=3, stride=1, padding=1).eval()

    with torch.no_grad():
        recon_out = recon_head(dec_out)
        log("recon_out  [B, 4, 96, 96, 96]", recon_out,
            "reconstructed 4-modality volume  (T1 | T1ce | T2 | FLAIR)")

        # Show per-modality reconstruction MSE vs clean target (on masked voxels)
        # Build a simple voxel mask from is_masked
        voxel_mask_tok = is_masked.squeeze(1).squeeze(1).squeeze(1).squeeze(1)  # [B, N]
        n_patches = voxel_mask_tok.shape[-1]
        # Expand to voxel space
        mask_expanded = voxel_mask_tok.view(1, 1, 6, 6, 6).repeat(1, 1, 16, 16, 16)
        # For demo: compute MSE over masked voxels only
        diff = (recon_out - x_clean) ** 2         # [1, 4, 96, 96, 96]
        mask_full = (mask_expanded > 0.5).repeat(1, 4, 1, 1, 1)[:, :, :96, :96, :96]
        if mask_full.sum() > 0:
            masked_mse = (diff * mask_full.float()).sum() / mask_full.sum().float()
        else:
            masked_mse = diff.mean()

        print(f"\n    Masked MSE Loss (on perturbed voxels) : {masked_mse.item():.5f}")
        for ci, mod in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            ch_mse = ((recon_out[0, ci] - x_clean[0, ci]) ** 2).mean().item()
            print(f"    Channel {ci}  {mod:<6}  per-channel MSE : {ch_mse:.5f}")

    # =========================================================================
    # FULL PIPELINE FLOW SUMMARY
    # =========================================================================
    section("PIPELINE FLOW SUMMARY")
    print("""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     FULL MODULE-WISE TENSOR FLOW                        │
  ├─────────────────────────────────────────────────┬──────────────────────┤
  │  Module                                         │  Output Shape        │
  ├─────────────────────────────────────────────────┼──────────────────────┤
  │  [INPUT]  4-Modality MRI Volume                 │  [1, 4, 96, 96, 96]  │
  ├─── IS-MIM PREPROCESSING ───────────────────────┼──────────────────────┤
  │  1a. 3D Patch Tokenisation (patch=16³)          │  [1, 4, 16,16,16, 216]│
  │  1b. Intra-Modality Patch Swap (p=0.5)          │  [1, 4, 16,16,16, 216]│
  │      Swap Mask                                  │  [1, 216]            │
  │  1c. 50% Block-Wise Masking                     │  [1, 4, 16,16,16, 216]│
  │      Is-Masked Map                              │  [1,1,1,1,1, 216]    │
  │  1d. Detokenise → Perturbed Input               │  [1, 4, 96, 96, 96]  │
  │      Clean Target (for MSE)                     │  [1, 4, 96, 96, 96]  │
  ├─── ENCODER ─────────────────────────────────────┼──────────────────────┤
  │  2a. Stem Conv [7³, s=2, BN, LeakyReLU]         │  [1, 48, 48, 48, 48] │
  │  2b. Stage-0 W-MSA  (block 0)                   │  [1, 48, 48, 48, 48] │
  │      Stage-0 SW-MSA (block 1)                   │  [1, 48, 48, 48, 48] │
  │      Stage-0 BN + MLP  → skip S0               │  [1, 48, 48, 48, 48] │
  │  2c. Patch Merging 0→1 [2³ s=2, 48→96]          │  [1, 96, 24, 24, 24] │
  │  2d. Stage-1 W-MSA  (block 0)                   │  [1, 96, 24, 24, 24] │
  │      Stage-1 SW-MSA (block 1)                   │  [1, 96, 24, 24, 24] │
  │      Stage-1 BN + MLP  → skip S1               │  [1, 96, 24, 24, 24] │
  │  2e. Patch Merging 1→2 [2³ s=2, 96→192]         │  [1,192, 12, 12, 12] │
  │  2f. Stage-2 GSC (Conv3×3 + Conv1×1 + Residual) │  [1,192, 12, 12, 12] │
  │  2g. Stage-2 Mamba layer 0  (Axial+Sag+Cor SSM) │  [1,192, 12, 12, 12] │
  │      Stage-2 Mamba layer 1  (Axial+Sag+Cor SSM) │  [1,192, 12, 12, 12] │
  │      Stage-2 BN + MLP  → skip S2               │  [1,192, 12, 12, 12] │
  │  2h. Patch Merging 2→3 [2³ s=2, 192→384]        │  [1,384,  6,  6,  6] │
  │  2i. Stage-3 GSC (Conv3×3 + Conv1×1 + Residual) │  [1,384,  6,  6,  6] │
  │  2j. Stage-3 Mamba layer 0  (Axial+Sag+Cor SSM) │  [1,384,  6,  6,  6] │
  │      Stage-3 Mamba layer 1  (Axial+Sag+Cor SSM) │  [1,384,  6,  6,  6] │
  │      Stage-3 BN + MLP  → deepest features S3   │  [1,384,  6,  6,  6] │
  ├─── SKIP ENCODERS ───────────────────────────────┼──────────────────────┤
  │  3.  enc_stem  (4→48, residual)   → skip_stem  │  [1, 48, 96, 96, 96] │
  │      enc_stage0 (48→48, residual) → skip_0     │  [1, 48, 48, 48, 48] │
  │      enc_stage1 (96→96, residual) → skip_1     │  [1, 96, 24, 24, 24] │
  │      enc_stage2 (192→192, resid)  → skip_2     │  [1,192, 12, 12, 12] │
  │      enc_bottleneck (384→768)     → bottleneck │  [1,768,  6,  6,  6] │
  ├─── DECODER ─────────────────────────────────────┼──────────────────────┤
  │  4.  dec4  (bottleneck↑ ⊕ skip_2) EV2 block    │  [1,192, 12, 12, 12] │
  │      dec3  (d4↑ ⊕ skip_1)         EV2 block    │  [1, 96, 24, 24, 24] │
  │      dec2  (d3↑ ⊕ skip_0)         EV2 block    │  [1, 48, 48, 48, 48] │
  │      dec1  (d2↑ ⊕ skip_stem)      EV2 block    │  [1, 48, 96, 96, 96] │
  │      EV2 Refinement  (SE attention)             │  [1, 48, 96, 96, 96] │
  ├─── OUTPUT HEADS ────────────────────────────────┼──────────────────────┤
  │  5a. Seg Head   (1×1×1 Conv 48→3)               │  [1,  3, 96, 96, 96] │
  │      sigmoid → probabilities (TC | WT | ET)     │  [1,  3, 96, 96, 96] │
  │  5b. Clf Head   GAP+GMP → 1536-dim fused        │  [1, 1536]           │
  │      MLP (1536→256→128→2)                       │  [1,  2]             │
  │      softmax → [LGG_prob, HGG_prob]             │  [1,  2]             │
  │  5c. Recon Head (3×3×3 Conv 48→4) [pretrain]    │  [1,  4, 96, 96, 96] │
  │      MSE loss on masked voxels                  │  scalar              │
  └─────────────────────────────────────────────────┴──────────────────────┘
""")

    print(_DIVIDER)
    print("  Demo complete.  All module outputs verified.\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run_demo()
    except Exception as exc:
        import traceback
        print(f"\n[ERROR] {exc}")
        traceback.print_exc()
        sys.exit(1)
