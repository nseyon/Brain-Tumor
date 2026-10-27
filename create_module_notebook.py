"""
create_module_notebook.py
─────────────────────────
Builds 'module_notebook.ipynb' — the interactive real-time Jupyter notebook
for step-by-step module output verification on real patient MRI datasets.
"""

import json

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    })

# ── Title & Setup ───────────────────────────────────────────────────────────
add_md("""# 🧠 SwinMamba + IS-MIM: Real-Time Module-Wise Output Notebook
### Step-by-Step Layer Activation Verification on Real Patient MRI Datasets

This notebook loads a **real patient MRI scan** from `Dataset/Glioma/<subject_id>`, passes the 4-modality 3D volume (`T1n`, `T1c`, `T2w`, `FLAIR`) through the fine-tuned **SwinMamba Dual-Task Neural Network** (`checkpoints/finetune/finetune_best.pt`), and delivers **real-time tensor shapes, value statistics (min, mean, max, std), and feature activations** for every module layer in the architecture.

---
### Directory & Environment Setup""")

add_code("""import os
import sys
import glob
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure workspace is in path
sys.path.insert(0, "C:/Glioma")

from src.models import SwinMambaDual
from src.data.transforms import (
    normalize_intensity,
    crop_or_pad_3d,
    convert_brats_labels,
    extract_glioma_grade,
)
from src.utils.metrics import evaluate_segmentation_case
from src.utils.postprocessing import postprocess_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ PyTorch Execution Device: {device}")
""")

# ── Stage 0 ─────────────────────────────────────────────────────────────────
add_md("""---
## Stage 0 │ Real Patient MRI Dataset Loader
Specify any patient subject ID from `Dataset/Glioma` (e.g., `BraTS-GLI-00000-000`, `BraTS-GLI-00002-000`). If omitted, automatically selects the first available patient.""")

add_code("""DATA_DIR = "C:/Glioma/Dataset/Glioma"
SUBJECT_ID = "BraTS-GLI-00000-000"  # Change to any patient ID in Dataset/Glioma

all_subjects = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("BraTS")])
if SUBJECT_ID not in all_subjects:
    SUBJECT_ID = all_subjects[0]

subj_folder = os.path.join(DATA_DIR, SUBJECT_ID)
mod_keys = ["t1n", "t1c", "t2w", "t2f"]
vols = []

for mk in mod_keys:
    matches = glob.glob(os.path.join(subj_folder, f"*{mk}*.nii*"))
    if not matches:
        matches = [f for f in glob.glob(os.path.join(subj_folder, "*.nii*")) if "seg" not in f]
    vols.append(nib.load(matches[0]).get_fdata(dtype=np.float32))

vol_4ch = np.stack(vols, axis=0)
tensor_4ch = torch.from_numpy(vol_4ch)
norm_tensor = normalize_intensity(tensor_4ch)
input_tensor = crop_or_pad_3d(norm_tensor, target_shape=(96, 96, 96)).unsqueeze(0).to(device)

# Load Ground Truth Seg & Grade if available
seg_matches = glob.glob(os.path.join(subj_folder, "*seg*.nii*"))
gt_tensor = None
gt_grade = None
if seg_matches:
    raw_seg = nib.load(seg_matches[0]).get_fdata(dtype=np.float32)
    gt_3ch = convert_brats_labels(torch.from_numpy(raw_seg).unsqueeze(0))
    gt_tensor = crop_or_pad_3d(gt_3ch, target_shape=(96, 96, 96))
    gt_grade = extract_glioma_grade(gt_tensor)

print(f"✓ Loaded Patient: {SUBJECT_ID}")
print(f"  Input Tensor Shape : {tuple(input_tensor.shape)}  [B=1, C=4, D=96, H=96, W=96]")
print(f"  Voxel Value Range  : min={input_tensor.min():+.3f} | mean={input_tensor.mean():+.3f} | max={input_tensor.max():+.3f} | std={input_tensor.std():.3f}")
if gt_grade is not None:
    print(f"  Ground Truth Grade : {'HGG (High-Level Glioma)' if gt_grade==1 else 'LGG (Low-Level Glioma)'}")

# Visualise 4 Modalities
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.patch.set_facecolor("#0f172a")
mod_names = ["T1-Native (t1n)", "T1-Gd Contrast (t1c)", "T2-Weighted (t2w)", "T2-FLAIR (t2f)"]
for idx in range(4):
    slice_2d = np.rot90(input_tensor[0, idx, :, :, 48].cpu().numpy())
    axes[idx].imshow(slice_2d, cmap="gray")
    axes[idx].set_title(mod_names[idx], color="white", fontsize=11, fontweight="bold")
    axes[idx].axis("off")
fig.suptitle(f"Patient {SUBJECT_ID} — Real 4-Modality MRI Inputs (Axial Slice 48)", color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
""")

# ── Load Model ──────────────────────────────────────────────────────────────
add_md("""---
## Load Fine-Tuned SwinMamba Weights
Loads trained weights from `checkpoints/finetune/finetune_best.pt`.""")

add_code("""CKPT_PATH = "C:/Glioma/checkpoints/finetune/finetune_best.pt"

model = SwinMambaDual(
    in_chans=4, seg_out_chans=3, num_classes=2,
    feat_size=(48, 96, 192, 384), hidden_size=768, mode="finetune"
).to(device)

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"✓ Fine-Tuned Weights Loaded from: {CKPT_PATH}")
    print(f"✓ Model Best Epoch: {ckpt.get('epoch', 'N/A')}  |  Val Mean Dice: {ckpt.get('metrics', {}).get('mean_dice', 0):.4f}")
else:
    print("⚠ Checkpoint not found. Running with initialized weights.")

model.eval()

def log(name: str, t: torch.Tensor, note: str = ""):
    f = t.detach().float()
    print(f"{name:<42}  {str(tuple(t.shape)):<24}  min={f.min():+.3f}  u={f.mean():+.3f}  max={f.max():+.3f}  std={f.std():.3f}  <- {note}")
""")

# ── Stage 1 ─────────────────────────────────────────────────────────────────
add_md("""---
## Stage 1 │ 3D Convolutional Stem
Initial stem projection layer halving spatial resolution ($96^3 \to 48^3$) and expanding channel capacity ($4 \to 48$).""")

add_code("""with torch.no_grad():
    x_stem = model.encoder.downsample_layers[0](input_tensor)
    log("Stem Conv Output", x_stem, "Halves resolution 96³→48³, expands 4→48 channels")

# Feature activations visualization
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
fig.patch.set_facecolor("#0f172a")
for i in range(4):
    axes[i].imshow(np.rot90(x_stem[0, i, :, :, 24].cpu().numpy()), cmap="viridis")
    axes[i].set_title(f"Stem Feature Ch #{i}", color="white", fontsize=10)
    axes[i].axis("off")
plt.tight_layout()
plt.show()
""")

# ── Stage 2 ─────────────────────────────────────────────────────────────────
add_md("""---
## Stage 2 │ Swin Transformer Stages (Stages 0 & 1)
Applies local windowed 3D self-attention (W-MSA and SW-MSA).""")

add_code("""with torch.no_grad():
    # Stage 0
    x_s0 = x_stem.permute(0, 2, 3, 4, 1)
    for blk_idx, swin_blk in enumerate(model.encoder.swin_stages[0]):
        x_s0 = swin_blk(x_s0)
        log(f"  Swin Stage-0 Block {blk_idx}", x_s0, "3D window self-attention")
    x_s0 = x_s0.permute(0, 4, 1, 2, 3).contiguous()
    x_s0 = model.encoder.mlps[0](model.encoder.norms[0](x_s0))
    log("Stage-0 Output", x_s0, "Stage-0 Skip Feature S0")

    # Stage 1
    x_pm01 = model.encoder.downsample_layers[1](x_s0)
    x_s1 = x_pm01.permute(0, 2, 3, 4, 1)
    for blk_idx, swin_blk in enumerate(model.encoder.swin_stages[1]):
        x_s1 = swin_blk(x_s1)
        log(f"  Swin Stage-1 Block {blk_idx}", x_s1, "3D window self-attention")
    x_s1 = x_s1.permute(0, 4, 1, 2, 3).contiguous()
    x_s1 = model.encoder.mlps[1](model.encoder.norms[1](x_s1))
    log("Stage-1 Output", x_s1, "Stage-1 Skip Feature S1")
""")

# ── Stage 3 ─────────────────────────────────────────────────────────────────
add_md("""---
## Stage 3 │ 3D Tri-Directional Mamba State-Space Encoder (Stages 2 & 3)
Group Spatial Convolution (GSC) + 3D tri-directional Mamba state-space scanning along Axial, Sagittal, and Coronal directions.""")

add_code("""with torch.no_grad():
    # Stage 2
    x_pm12 = model.encoder.downsample_layers[2](x_s1)
    x_gsc2 = model.encoder.conv_stages[2](x_pm12)
    log("Stage-2 GSC Conv", x_gsc2, "Group Spatial Convolution enhancement")
    x_m2 = x_gsc2
    for m_idx, mamba_blk in enumerate(model.encoder.mamba_stages[2]):
        x_m2 = mamba_blk(x_m2)
        log(f"  Mamba Stage-2 Layer {m_idx}", x_m2, "Axial + Sagittal + Coronal SSM scan")
    x_s2 = model.encoder.mlps[2](model.encoder.norms[2](x_m2))
    log("Stage-2 Output", x_s2, "Stage-2 Skip Feature S2")

    # Stage 3
    x_pm23 = model.encoder.downsample_layers[3](x_s2)
    x_gsc3 = model.encoder.conv_stages[3](x_pm23)
    log("Stage-3 GSC Conv", x_gsc3, "GSC enhancement before bottleneck SSM")
    x_m3 = x_gsc3
    for m_idx, mamba_blk in enumerate(model.encoder.mamba_stages[3]):
        x_m3 = mamba_blk(x_m3)
        log(f"  Mamba Stage-3 Layer {m_idx}", x_m3, "Deep state-space sequence modeling")
    x_s3 = model.encoder.mlps[3](model.encoder.norms[3](x_m3))
    log("Stage-3 Output", x_s3, "Stage-3 Deep Bottleneck Representation S3")
""")

# ── Stage 4 ─────────────────────────────────────────────────────────────────
add_md("""---
## Stage 4 │ UNETR Skip Connection Encoders & Bottleneck Expansion""")

add_code("""with torch.no_grad():
    skip_stem  = model.enc_stem(input_tensor)
    skip_0     = model.enc_stage0(x_s0)
    skip_1     = model.enc_stage1(x_s1)
    skip_2     = model.enc_stage2(x_s2)
    bottleneck = model.enc_bottleneck(x_s3)

    log("skip_stem",  skip_stem,  "Raw Input Skip Feature")
    log("skip_0",     skip_0,     "Stage-0 Swin Skip Feature")
    log("skip_1",     skip_1,     "Stage-1 Swin Skip Feature")
    log("skip_2",     skip_2,     "Stage-2 Mamba Skip Feature")
    log("bottleneck", bottleneck, "Expanded Bottleneck Feature (384→768)")
""")

# ── Stage 5 ─────────────────────────────────────────────────────────────────
add_md("""---
## Stage 5 │ UNETR Progressive Decoder (EV2 Skip Fusion)""")

add_code("""with torch.no_grad():
    d4 = model.decoder4(bottleneck, skip_2)
    log("dec4 (bottleneck↑ ⊕ skip_2)", d4, "6³→12³ Upsampling & Skip Fusion")
    d3 = model.decoder3(d4, skip_1)
    log("dec3 (d4↑ ⊕ skip_1)",        d3, "12³→24³ Upsampling & Skip Fusion")
    d2 = model.decoder2(d3, skip_0)
    log("dec2 (d3↑ ⊕ skip_0)",        d2, "24³→48³ Upsampling & Skip Fusion")
    d1 = model.decoder1(d2, skip_stem)
    log("dec1 (d2↑ ⊕ skip_stem)",     d1, "48³→96³ Upsampling & Skip Fusion")
    dec_out = model.decoder_refine(d1)
    log("EV2 Refinement (SE Attn)",   dec_out, "Final Refined Decoder Features")
""")

# ── Stage 6 ─────────────────────────────────────────────────────────────────
add_md("""---
## Stage 6 │ Multi-Task Task Output Heads & Real Patient Predictions""")

add_code("""with torch.no_grad():
    # 6a. Segmentation Head
    seg_logits = model.seg_head(dec_out)
    seg_probs  = torch.sigmoid(seg_logits)
    log("6a. seg_logits", seg_logits, "Raw Voxels Logits (TC | WT | ET)")
    log("    seg_probs",  seg_probs,  "Sigmoid Probabilities")

    seg_raw = seg_logits[0].float().cpu().numpy()
    seg_bin = postprocess_predictions(seg_raw)

    if gt_tensor is not None:
        metrics = evaluate_segmentation_case(seg_bin, gt_tensor.numpy())
        print("\\n  [REAL PATIENT SEGMENTATION PERFORMANCE]")
        print(f"    • Mean Dice        : {metrics['mean_dice']:.4f}")
        print(f"    • Whole Tumor (WT) : {metrics['dice_wt']:.4f}")
        print(f"    • Tumor Core (TC)  : {metrics['dice_tc']:.4f}")
        print(f"    • Enhancing (ET)   : {metrics['dice_et']:.4f}")
        print(f"    • HD95 (mm)        : {metrics['mean_hd95']:.2f} mm")

    # 6b. Classification Head
    clf_logits = model.clf_head(bottleneck)
    clf_probs  = F.softmax(clf_logits, dim=-1)
    pred_grade = torch.argmax(clf_logits[0]).item()

    log("\\n6b. clf_logits", clf_logits, "Raw Class Scores [0=LGG, 1=HGG]")
    log("    clf_probs",  clf_probs,  f"LGG={clf_probs[0,0]:.4f} | HGG={clf_probs[0,1]:.4f}")

    grade_names = ["LGG (Low-Level Glioma)", "HGG (High-Level Glioma)"]
    print(f"\\n  [REAL PATIENT GRADE CLASSIFICATION]")
    print(f"    • Predicted Grade  : {grade_names[pred_grade]}")
    if gt_grade is not None:
        print(f"    • Ground Truth     : {grade_names[gt_grade]}  ({'✅ Correct Match' if pred_grade==gt_grade else '❌ Mismatch'})")

# 4-Panel Visualization
mri_2d = input_tensor[0, 1, :, :, 48].cpu().numpy()
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

fig.suptitle(f"Patient [{SUBJECT_ID}] Real-Time AI Predictions (Grade: {grade_names[pred_grade]})", color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(r"C:\Glioma\module_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print("Created module_notebook.ipynb successfully!")
