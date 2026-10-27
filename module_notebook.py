"""
module_notebook.py
──────────────────
Real-Time Module-Wise Output Inspection for Real Patient MRI Datasets.

Loads a real patient MRI dataset (from Dataset/Glioma/<subject_id> or user input),
passes the 4-modality 3D volume through every module of the fine-tuned SwinMambaDual model,
and delivers real-time shape, statistics (min, mean, max, std), and feature slice visualizer
outputs for EVERY SINGLE MODULE in the architecture.

Usage:
    python module_notebook.py
    python module_notebook.py --subject_id BraTS-GLI-00000-000
    python module_notebook.py --subject_id BraTS-GLI-00002-000
"""

from __future__ import annotations
import os
import glob
import sys
import argparse
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in sys.path
sys.path.insert(0, r"C:\Glioma")

from src.models import SwinMambaDual
from src.data.transforms import (
    normalize_intensity,
    crop_or_pad_3d,
    convert_brats_labels,
    extract_glioma_grade,
)
from src.utils.metrics import compute_dice_score, compute_hd95, evaluate_segmentation_case
from src.utils.postprocessing import postprocess_predictions

# ── Pretty Output Helpers ───────────────────────────────────────────────────

W_NAME  = 44
W_SHAPE = 26
W_STATS = 38

_DIVIDER = "=" * (W_NAME + W_SHAPE + W_STATS + 8)
_SEP     = "-" * (W_NAME + W_SHAPE + W_STATS + 8)

def _stats(t: torch.Tensor) -> str:
    f = t.detach().float()
    return f"min={f.min():+.3f}  u={f.mean():+.3f}  max={f.max():+.3f}  std={f.std():.3f}"

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


# ── Load Subject Dataset ────────────────────────────────────────────────────

def load_real_subject(data_dir: str, subject_id: str | None = None):
    """Locates and loads 4 MRI modalities and segmentation for a real patient subject."""
    all_subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("BraTS")
    ])

    if not all_subjects:
        raise FileNotFoundError(f"No BraTS subject folders found in {data_dir}")

    if subject_id is None:
        subject_id = all_subjects[0]
    elif subject_id not in all_subjects:
        print(f"Subject '{subject_id}' not found in {data_dir}. Falling back to '{all_subjects[0]}'.")
        subject_id = all_subjects[0]

    subj_folder = os.path.join(data_dir, subject_id)
    mod_keys = ["t1n", "t1c", "t2w", "t2f"]
    vols = []

    for mk in mod_keys:
        matches = glob.glob(os.path.join(subj_folder, f"*{mk}*.nii*"))
        if not matches:
            matches = [f for f in glob.glob(os.path.join(subj_folder, "*.nii*")) if "seg" not in f]
        if not matches:
            raise FileNotFoundError(f"Missing MRI modality '{mk}' for subject {subject_id}")
        vols.append(nib.load(matches[0]).get_fdata(dtype=np.float32))

    vol_4ch = np.stack(vols, axis=0)  # [4, D, H, W]
    tensor_4ch = torch.from_numpy(vol_4ch)
    norm_tensor = normalize_intensity(tensor_4ch)
    input_tensor = crop_or_pad_3d(norm_tensor, target_shape=(96, 96, 96)).unsqueeze(0)  # [1, 4, 96, 96, 96]

    # Ground Truth Seg if available
    seg_matches = glob.glob(os.path.join(subj_folder, "*seg*.nii*"))
    gt_tensor = None
    gt_grade = None
    if seg_matches:
        raw_seg = nib.load(seg_matches[0]).get_fdata(dtype=np.float32)
        gt_3ch = convert_brats_labels(torch.from_numpy(raw_seg).unsqueeze(0))
        gt_tensor = crop_or_pad_3d(gt_3ch, target_shape=(96, 96, 96))
        gt_grade = extract_glioma_grade(gt_tensor)

    return subject_id, input_tensor, gt_tensor, gt_grade, vol_4ch


# ── Real-Time Pipeline Run ──────────────────────────────────────────────────

def run_real_time_module_pipeline(subject_id: str = None, save_plots: bool = True):
    data_dir = "C:/Glioma/Dataset/Glioma"
    ckpt_path = "C:/Glioma/checkpoints/finetune/finetune_best.pt"
    out_dir = "C:/Glioma/outputs/module_notebook_plots"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(_DIVIDER)
    print("  MODULE NOTEBOOK  │  REAL-TIME PATIENT MRI MODULE-WISE OUTPUTS")
    print(f"  Device: {device}  │  Checkpoint: finetune_best.pt")
    print(_DIVIDER)

    # 1. Load Real Patient Dataset MRI
    subject_id, input_tensor, gt_tensor, gt_grade, raw_4ch = load_real_subject(data_dir, subject_id)
    input_tensor = input_tensor.to(device)

    section(f"STAGE 0 │ Real Patient MRI Dataset Loaded: [{subject_id}]")
    log("Raw 4-Modality Volume [B, C, D, H, W]", input_tensor, "T1n | T1c | T2w | FLAIR (Normalized 96³)")
    if gt_tensor is not None:
        log("Ground Truth Mask   [C, D, H, W]", gt_tensor, f"TC | WT | ET (Grade: {'HGG' if gt_grade==1 else 'LGG'})")

    # Save Stage 0 Input Plot
    if save_plots:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        fig.patch.set_facecolor("#0f172a")
        mod_names = ["T1-Native (t1n)", "T1-Gd Contrast (t1c)", "T2-Weighted (t2w)", "T2-FLAIR (t2f)"]
        for idx in range(4):
            slice_2d = np.rot90(input_tensor[0, idx, :, :, 48].cpu().numpy())
            axes[idx].imshow(slice_2d, cmap="gray")
            axes[idx].set_title(mod_names[idx], color="white", fontsize=11, fontweight="bold")
            axes[idx].axis("off")
        fig.suptitle(f"Patient {subject_id} — Real 4-Modality MRI Inputs (Axial Slice 48)", color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage0_mri_inputs.png"), dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)

    # 2. Load Fine-Tuned Model Weights
    section("MODEL LOADING │ SwinMamba Dual-Task Architecture")
    model = SwinMambaDual(
        in_chans=4, seg_out_chans=3, num_classes=2,
        feat_size=(48, 96, 192, 384), hidden_size=768, mode="finetune"
    ).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"  ✓ Successfully loaded fine-tuned weights from: {ckpt_path}")
        print(f"  ✓ Checkpoint Epoch: {ckpt.get('epoch', 'N/A')}  |  Val Mean Dice: {ckpt.get('metrics', {}).get('mean_dice', 0):.4f}")
    else:
        print(f"  ⚠ Checkpoint not found at {ckpt_path}, running with initialized model weights.")

    model.eval()

    # 3. Step-by-Step Module Forward Pass
    with torch.no_grad():
        # STAGE 1 — Stem Conv Layer
        section("STAGE 1 │ 3D Convolutional Stem")
        x_stem = model.encoder.downsample_layers[0](input_tensor)
        log("Stem Conv [B, 48, 48, 48, 48]", x_stem, "Halves resolution 96³→48³, expands 4→48 channels")

        # STAGE 2 — Swin Transformer Stages (Stages 0 & 1)
        section("STAGE 2 │ Swin Transformer Hybrid Encoder (Stages 0 & 1)")
        # Stage 0
        x_s0 = x_stem.permute(0, 2, 3, 4, 1)
        for blk_idx, swin_blk in enumerate(model.encoder.swin_stages[0]):
            x_s0 = swin_blk(x_s0)
            log(f"  Swin Stage-0 Block {blk_idx}", x_s0, "3D window self-attention (W-MSA / SW-MSA)")
        x_s0 = x_s0.permute(0, 4, 1, 2, 3).contiguous()
        x_s0 = model.encoder.mlps[0](model.encoder.norms[0](x_s0))
        log("Stage-0 Output [B, 48, 48, 48, 48]", x_s0, "Stage-0 Skip Feature S0")

        # Stage 1
        x_pm01 = model.encoder.downsample_layers[1](x_s0)
        x_s1 = x_pm01.permute(0, 2, 3, 4, 1)
        for blk_idx, swin_blk in enumerate(model.encoder.swin_stages[1]):
            x_s1 = swin_blk(x_s1)
            log(f"  Swin Stage-1 Block {blk_idx}", x_s1, "3D window self-attention")
        x_s1 = x_s1.permute(0, 4, 1, 2, 3).contiguous()
        x_s1 = model.encoder.mlps[1](model.encoder.norms[1](x_s1))
        log("Stage-1 Output [B, 96, 24, 24, 24]", x_s1, "Stage-1 Skip Feature S1")

        # STAGE 3 — 3D Tri-Directional Mamba SSM Stages (Stages 2 & 3)
        section("STAGE 3 │ 3D Tri-Directional Mamba State-Space Encoder (Stages 2 & 3)")
        # Stage 2
        x_pm12 = model.encoder.downsample_layers[2](x_s1)
        x_gsc2 = model.encoder.conv_stages[2](x_pm12)
        log("Stage-2 GSC Conv [B, 192, 12, 12, 12]", x_gsc2, "Group Spatial Convolution feature enhancement")
        x_m2 = x_gsc2
        for m_idx, mamba_blk in enumerate(model.encoder.mamba_stages[2]):
            x_m2 = mamba_blk(x_m2)
            log(f"  Mamba Stage-2 Layer {m_idx}", x_m2, "Axial + Sagittal + Coronal SSM scanning")
        x_s2 = model.encoder.mlps[2](model.encoder.norms[2](x_m2))
        log("Stage-2 Output [B, 192, 12, 12, 12]", x_s2, "Stage-2 Skip Feature S2")

        # Stage 3
        x_pm23 = model.encoder.downsample_layers[3](x_s2)
        x_gsc3 = model.encoder.conv_stages[3](x_pm23)
        log("Stage-3 GSC Conv [B, 384, 6, 6, 6]", x_gsc3, "GSC enhancement before bottleneck SSM")
        x_m3 = x_gsc3
        for m_idx, mamba_blk in enumerate(model.encoder.mamba_stages[3]):
            x_m3 = mamba_blk(x_m3)
            log(f"  Mamba Stage-3 Layer {m_idx}", x_m3, "Deep state-space sequence modeling")
        x_s3 = model.encoder.mlps[3](model.encoder.norms[3](x_m3))
        log("Stage-3 Output [B, 384, 6, 6, 6]", x_s3, "Stage-3 Deep Bottleneck Representation S3")

        # STAGE 4 — Skip Encoders & Bottleneck
        section("STAGE 4 │ UNETR Skip Connection Encoders & Bottleneck Expansion")
        skip_stem   = model.enc_stem(input_tensor)
        skip_0      = model.enc_stage0(x_s0)
        skip_1      = model.enc_stage1(x_s1)
        skip_2      = model.enc_stage2(x_s2)
        bottleneck  = model.enc_bottleneck(x_s3)

        log("skip_stem   [B, 48, 96, 96, 96]", skip_stem, "Raw Input Skip Feature")
        log("skip_0      [B, 48, 48, 48, 48]", skip_0, "Stage-0 Swin Skip Feature")
        log("skip_1      [B, 96, 24, 24, 24]", skip_1, "Stage-1 Swin Skip Feature")
        log("skip_2      [B, 192, 12, 12, 12]", skip_2, "Stage-2 Mamba Skip Feature")
        log("bottleneck  [B, 768, 6, 6, 6]", bottleneck, "Expanded Bottleneck Feature (384→768)")

        # STAGE 5 — UNETR Progressive Decoder with EV2 Skip Fusion
        section("STAGE 5 │ UNETR Progressive Decoder (EV2 Skip Fusion)")
        d4 = model.decoder4(bottleneck, skip_2)
        log("dec4 (bottleneck↑ ⊕ skip_2) [B, 192, 12, 12, 12]", d4, "6³→12³ Upsampling & Skip Fusion")
        d3 = model.decoder3(d4, skip_1)
        log("dec3 (d4↑ ⊕ skip_1)         [B, 96, 24, 24, 24]", d3, "12³→24³ Upsampling & Skip Fusion")
        d2 = model.decoder2(d3, skip_0)
        log("dec2 (d3↑ ⊕ skip_0)         [B, 48, 48, 48, 48]", d2, "24³→48³ Upsampling & Skip Fusion")
        d1 = model.decoder1(d2, skip_stem)
        log("dec1 (d2↑ ⊕ skip_stem)      [B, 48, 96, 96, 96]", d1, "48³→96³ Upsampling & Skip Fusion")
        dec_out = model.decoder_refine(d1)
        log("EV2 Refinement (SE Attn)   [B, 48, 96, 96, 96]", dec_out, "Final Refined Decoder Features")

        # STAGE 6 — Task Output Heads
        section("STAGE 6 │ Multi-Task Output Heads")
        # 6a. Segmentation
        seg_logits = model.seg_head(dec_out)
        seg_probs  = torch.sigmoid(seg_logits)
        log("6a. seg_logits [B, 3, 96, 96, 96]", seg_logits, "Raw Voxels Logits (TC | WT | ET)")
        log("    seg_probs  [B, 3, 96, 96, 96]", seg_probs, "Sigmoid Probabilities")

        seg_raw = seg_logits[0].float().cpu().numpy()
        seg_bin = postprocess_predictions(seg_raw)  # [3, 96, 96, 96]

        if gt_tensor is not None:
            metrics = evaluate_segmentation_case(seg_bin, gt_tensor.numpy())
            print("\n  [REAL PATIENT SEGMENTATION PERFORMANCE]")
            print(f"    • Mean Dice        : {metrics['mean_dice']:.4f}")
            print(f"    • Whole Tumor (WT) : {metrics['dice_wt']:.4f}")
            print(f"    • Tumor Core (TC)  : {metrics['dice_tc']:.4f}")
            print(f"    • Enhancing (ET)   : {metrics['dice_et']:.4f}")
            print(f"    • HD95 (mm)        : {metrics['mean_hd95']:.2f} mm")

        # 6b. Classification
        clf_logits = model.clf_head(bottleneck)
        clf_probs  = F.softmax(clf_logits, dim=-1)
        pred_grade = torch.argmax(clf_logits[0]).item()

        log("\n6b. clf_logits [B, 2]", clf_logits, "Raw Class Scores [0=LGG, 1=HGG]")
        log("    clf_probs  [B, 2]", clf_probs, f"LGG={clf_probs[0,0]:.4f} | HGG={clf_probs[0,1]:.4f}")

        grade_names = ["LGG (Low-Level Glioma)", "HGG (High-Level Glioma)"]
        print(f"\n  [REAL PATIENT GRADE CLASSIFICATION]")
        print(f"    • Predicted Grade  : {grade_names[pred_grade]}")
        if gt_grade is not None:
            print(f"    • Ground Truth     : {grade_names[gt_grade]}  (Match = {'✅ Correct' if pred_grade==gt_grade else '❌ Mismatch'})")

        # Save Final Prediction 4-Panel Plot
        if save_plots:
            mri_2d = input_tensor[0, 1, :, :, 48].cpu().numpy()  # T1c slice 48
            mri_norm = (mri_2d - mri_2d.min()) / (mri_2d.max() - mri_2d.min() + 1e-7)
            rgb = np.repeat(np.rot90(mri_norm)[:, :, np.newaxis], 3, axis=2)

            wt_pred = np.rot90(seg_bin[1, :, :, 48]) > 0.5
            overlay = rgb.copy()
            overlay[wt_pred] = 0.5 * overlay[wt_pred] + 0.5 * np.array([0.9, 0.1, 0.1])

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor("#0f172a")
            axes[0].imshow(np.rot90(mri_norm), cmap="gray")
            axes[0].set_title("1. Original T1ce MRI", color="white", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(wt_pred, cmap="Reds")
            axes[1].set_title("2. Predicted Whole Tumor Mask", color="white", fontsize=11, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("3. MRI + Tumor Overlay", color="white", fontsize=11, fontweight="bold")
            axes[2].axis("off")

            fig.suptitle(f"Patient [{subject_id}] Real-Time AI Predictions (Predicted Grade: {grade_names[pred_grade]})",
                         color="white", fontsize=13, fontweight="bold")
            plt.tight_layout()
            pred_plot_path = os.path.join(out_dir, f"{subject_id}_realtime_prediction.png")
            fig.savefig(pred_plot_path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"\n  ✓ Real-time prediction plot saved to: {pred_plot_path}")

    print(_DIVIDER)
    print("  Real-Time Module Output Walkthrough Complete!")
    print(_DIVIDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Module Notebook Runner")
    parser.add_argument("--subject_id", type=str, default=None, help="Subject ID (e.g., BraTS-GLI-00000-000)")
    args = parser.parse_args()
    run_real_time_module_pipeline(subject_id=args.subject_id)
