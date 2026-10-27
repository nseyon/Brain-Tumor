"""
Brain AI — Glioma Segmentation & Classification System
======================================================
Interactive Streamlit Web Dashboard:
- Mode 1: 📤 Upload Raw MRI Scan for Live Inference (Segmentation + Grade Classification)
- Mode 2: 📂 Benchmark on BraTS 2023 Held-Out Test Set (Ground Truth vs Prediction)

Model: SwinMamba Dual-Task Deep Learning (Swin Transformer + Mamba SSM + UNETR Decoder)
Dataset: BraTS 2023 Adult Glioma (1,251 cases, 251 Test Cases)
"""

from __future__ import annotations
import sys
import os
import io
import gzip
import json
import glob
from io import BytesIO

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Ensure workspace is in sys.path
sys.path.insert(0, r"C:\Glioma")

from src.models import SwinMambaDual
from src.data.transforms import (
    normalize_intensity,
    crop_or_pad_3d,
    convert_brats_labels,
    extract_glioma_grade,
)
from src.utils import evaluate_segmentation_case, postprocess_predictions


# ── Constants & Paths ────────────────────────────────────────────────────────
DATA_DIR        = r"C:\Glioma\Dataset\Glioma"
PRED_DIR        = r"C:\Glioma\outputs\predictions"
CHECKPOINT_PATH = r"C:\Glioma\checkpoints\finetune\finetune_best.pt"
MANIFEST_PATH   = r"C:\Glioma\checkpoints\split\split_manifest.json"

MODALITY_LABELS = {
    "t1c": "T1-Gd (Contrast Enhanced)",
    "t2f": "T2-FLAIR",
    "t1n": "T1-Native",
    "t2w": "T2-Weighted",
}


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIG & CUSTOM STYLING
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Brain AI — Glioma Segmentation & Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 2.2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header h3 {
        margin: 6px 0 16px 0;
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.88;
    }
    .pill {
        display: inline-block;
        padding: 5px 14px;
        margin: 3px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #f8fafc;
        backdrop-filter: blur(8px);
    }
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        min-height: 85px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .result-hgg {
        border-left: 6px solid #ef4444 !important;
    }
    .result-lgg {
        border-left: 6px solid #10b981 !important;
    }
    .result-neutral {
        border-left: 6px solid #0284c7 !important;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HEADER & MODEL INFO
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🧠 Brain AI — Glioma Segmentation & Classification System</h1>
    <h3>SwinMamba Dual-Task Architecture • BraTS 2023 Adult Glioma • IS-MIM Pre-trained</h3>
    <div>
        <span class="pill">3D Multi-Parametric MRI</span>
        <span class="pill">Swin Transformer</span>
        <span class="pill">Mamba Vision SSM</span>
        <span class="pill">UNETR 3D Decoder</span>
        <span class="pill">IS-MIM Pre-training</span>
        <span class="pill">HGG / LGG Grading</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADER (AUTO-LOADS LATEST BEST CHECKPOINT)
# ════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_best_checkpoint_path() -> tuple[str, float]:
    """Locates the best fine-tune checkpoint path and its last modified timestamp."""
    best_path = r"C:\Glioma\checkpoints\finetune\finetune_best.pt"
    latest_path = r"C:\Glioma\checkpoints\finetune\finetune_latest.pt"

    target_path = best_path if os.path.exists(best_path) else latest_path
    if not os.path.exists(target_path):
        return "", 0.0
    return target_path, os.path.getmtime(target_path)


@st.cache_resource
def _load_model_cached(ckpt_path: str, mtime: float):
    """Loads and caches the model, auto-reloading whenever a new best checkpoint is saved on disk."""
    model = SwinMambaDual(
        in_chans=4,
        seg_out_chans=3,
        num_classes=2,
        feat_size=(48, 96, 192, 384),
        hidden_size=768,
        mode="finetune",
    ).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    metrics = checkpoint.get("metrics", {})
    meta = {
        "epoch": checkpoint.get("epoch", "Unknown"),
        "metrics": metrics,
        "mean_dice": metrics.get("mean_dice", 0.0),
        "clf_acc": metrics.get("clf_acc", 0.0),
        "saved_at": checkpoint.get("saved_at", ""),
        "path": ckpt_path,
        "filename": os.path.basename(ckpt_path),
    }
    return model, meta


def load_glioma_model():
    """Public helper that automatically fetches the latest best model weights."""
    ckpt_path, mtime = get_best_checkpoint_path()
    if not ckpt_path:
        st.error("No fine-tune checkpoint found in checkpoints/finetune/")
        return None, {}
    return _load_model_cached(ckpt_path, mtime)


with st.expander("ℹ️ Model Architecture & Technical Information", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card">
        <b>Architecture</b><br>
        SwinMamba Dual-Task Backbone (Hierarchical Swin Transformer + 4-way Bidirectional Mamba SSM) with UNETR Decoder.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
        <b>Input Channels (4-Modality 3D)</b><br>
        T1-Native, T1-Gd Contrast, T2-Weighted, T2-FLAIR standardized to 96 × 96 × 96 voxel grid.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        _, meta = load_glioma_model()
        ep_lbl = f"Epoch {meta.get('epoch', '?')} (Best)" if meta else "finetune_best.pt"
        dice_lbl = f" · Mean Dice: {meta.get('mean_dice', 0)*100:.2f}%" if meta and meta.get("mean_dice") else ""
        st.markdown(f"""
        <div class="info-card">
        <b>Inference Model & Weights</b><br>
        Hardware: <code>{gpu_name}</code><br>
        Weights: <code>{meta.get('filename', 'finetune_best.pt')}</code> ({ep_lbl}{dice_lbl})
        </div>
        """, unsafe_allow_html=True)


@st.cache_data
def get_test_subjects():
    """Safely extracts test subject IDs from split manifest."""
    if not os.path.exists(MANIFEST_PATH):
        return []
    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
        if "test_ids" in manifest and isinstance(manifest["test_ids"], list):
            return sorted(manifest["test_ids"])
        elif "subjects" in manifest and isinstance(manifest["subjects"], dict):
            return sorted(list(manifest["subjects"].keys()))
        elif "subjects" in manifest and isinstance(manifest["subjects"], list):
            return sorted([s["subject_id"] for s in manifest["subjects"] if s.get("split") == "test"])
    except Exception as e:
        st.warning(f"Could not load manifest test IDs: {e}")
    # Fallback to scanning predictions folder
    pred_files = glob.glob(os.path.join(PRED_DIR, "*_seg_pred.nii.gz"))
    return sorted([os.path.basename(p).replace("_seg_pred.nii.gz", "") for p in pred_files])


# ════════════════════════════════════════════════════════════════════════════
# IN-MEMORY NIFTI SERIALIZATION (NO WINDOWS FILE LOCKS)
# ════════════════════════════════════════════════════════════════════════════

def read_nifti_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """Reads 3D/4D NIfTI directly from memory without writing temp files to disk."""
    if filename.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(file_bytes)) as gz:
            decompressed = gz.read()
    else:
        decompressed = file_bytes
    file_map = nib.Nifti1Image.make_file_map()
    file_map["image"].fileobj = io.BytesIO(decompressed)
    img = nib.Nifti1Image.from_file_map(file_map)
    return img.get_fdata(dtype=np.float32)


def nifti_to_gzipped_bytes(arr: np.ndarray) -> bytes:
    """Converts numpy array into compressed .nii.gz bytes purely in memory."""
    img = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
    bio = io.BytesIO()
    fh = nib.FileHolder(fileobj=bio)
    img.to_file_map({"image": fh, "header": fh})
    raw_nii = bio.getvalue()
    buf_gz = io.BytesIO()
    with gzip.GzipFile(fileobj=buf_gz, mode="wb") as gz:
        gz.write(raw_nii)
    return buf_gz.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def extract_2d_slice(vol_3d: np.ndarray, slice_idx: int, plane: str = "Axial") -> np.ndarray:
    """Extracts a 2D slice along chosen anatomical plane."""
    if plane == "Axial":
        return np.rot90(vol_3d[:, :, slice_idx])
    elif plane == "Coronal":
        return np.rot90(vol_3d[:, slice_idx, :])
    else:
        return np.rot90(vol_3d[slice_idx, :, :])


def build_composite_color_mask(mask_slice_3c: np.ndarray, highlight: str = "All Combined") -> np.ndarray:
    """
    Constructs an RGB segmentation mask from a 3-channel slice:
      Channel 0: TC (Tumor Core)
      Channel 1: WT (Whole Tumor)
      Channel 2: ET (Enhancing Tumor)
    """
    h, w = mask_slice_3c.shape[1], mask_slice_3c.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    tc = mask_slice_3c[0] > 0.5
    wt = mask_slice_3c[1] > 0.5
    et = mask_slice_3c[2] > 0.5

    # Layer: WT (Red background) -> TC (Green core) -> ET (Yellow active enhancing)
    if highlight in ["All Combined", "Whole Tumor"]:
        rgb[wt] = [0.90, 0.20, 0.20]  # Red for Whole Tumor
    if highlight in ["All Combined", "Tumor Core"]:
        rgb[tc] = [0.15, 0.85, 0.25]  # Green for Tumor Core
    if highlight in ["All Combined", "Enhancing Tumor"]:
        rgb[et] = [1.00, 0.85, 0.05]  # Bright Yellow for Enhancing Tumor

    return rgb


def create_slice_overlay(mri_2d: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Alpha-blends grayscale MRI with colored segmentation mask."""
    norm_mri = (mri_2d - mri_2d.min()) / (mri_2d.max() - mri_2d.min() + 1e-7)
    base = np.repeat(norm_mri[:, :, np.newaxis], 3, axis=2)

    roi = mask_rgb.sum(axis=-1) > 0
    overlay = base.copy()
    overlay[roi] = (1.0 - alpha) * base[roi] + alpha * mask_rgb[roi]
    return np.clip(overlay, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE TABS
# ════════════════════════════════════════════════════════════════════════════

GRAPH_DIR = r"C:\Glioma\outputs\graphs"

tab_upload, tab_benchmark, tab_graphs = st.tabs([
    "📤 Upload Raw MRI Scan (Live Model Inference)",
    "📂 Benchmark Test Cases (Held-Out Test Set)",
    "📈 Performance Graphs & Learning Curves",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1: UPLOAD RAW MRI & LIVE INFERENCE
# ────────────────────────────────────────────────────────────────────────────

with tab_upload:
    st.markdown('<div class="section-title">1. Upload Patient MRI Scan</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a 3D Brain MRI volume in **`.nii`** or **`.nii.gz`** format. "
        "You can upload a single contrast-enhanced scan (T1ce/FLAIR) or multiple modalities."
    )

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "Select MRI NIfTI File(s)",
            type=["nii", "gz"],
            accept_multiple_files=True,
            help="Upload raw brain MRI files (e.g., patient-t1ce.nii.gz, patient-flair.nii.gz)",
        )

    with col_up2:
        primary_modality = st.selectbox(
            "Primary Uploaded Scan Type",
            ["t1c", "t2f", "t1n", "t2w"],
            format_func=lambda x: MODALITY_LABELS[x],
            help="Identify which modality is being provided if uploading a single file."
        )
        run_inference_btn = st.button("🧠 Run AI Inference", type="primary", width="stretch")

    if run_inference_btn and uploaded_files:
        model, meta = load_glioma_model()
        if model is None:
            st.error("Failed to load model checkpoint.")
            st.stop()

        with st.spinner("Processing MRI volume and running SwinMamba 3D inference..."):
            try:
                # 1. Parse uploaded NIfTI directly in memory
                first_file = uploaded_files[0]
                raw_data = read_nifti_bytes(first_file.getvalue(), first_file.name)

                # 2. Assemble 4-channel tensor [1, 4, D, H, W]
                # If single 3D volume [D, H, W], mirror to 4 channels
                if raw_data.ndim == 3:
                    vol_4ch = np.stack([raw_data] * 4, axis=0)  # [4, D, H, W]
                elif raw_data.ndim == 4:
                    if raw_data.shape[-1] == 4:
                        vol_4ch = np.transpose(raw_data, (3, 0, 1, 2))
                    else:
                        vol_4ch = raw_data[:4]
                else:
                    st.error(f"Unexpected image dimensions: {raw_data.shape}")
                    st.stop()

                # Normalize and crop/pad to [1, 4, 96, 96, 96]
                tensor_4ch = torch.from_numpy(vol_4ch)  # [4, D, H, W]
                norm_tensor = normalize_intensity(tensor_4ch)
                cropped_tensor = crop_or_pad_3d(norm_tensor, target_shape=(96, 96, 96))
                input_batch = cropped_tensor.unsqueeze(0).to(DEVICE)  # [1, 4, 96, 96, 96]

                # 3. Model forward pass
                with torch.no_grad():
                    outputs = model(input_batch)

                seg_logits = outputs["seg"][0]  # [3, 96, 96, 96]
                clf_logits = outputs["clf"][0]  # [2]

                # 4. Post-processing
                pred_binary = postprocess_predictions(seg_logits)  # [3, 96, 96, 96]
                clf_probs = torch.softmax(clf_logits, dim=0).cpu().numpy()

                hgg_prob = float(clf_probs[1])
                lgg_prob = float(clf_probs[0])
                pred_class = "High-Grade Glioma (HGG)" if hgg_prob >= 0.50 else "Low-Grade Glioma (LGG)"

                # Save results to session state for interactive inspection
                st.session_state["uploaded_mri"] = cropped_tensor[1].cpu().numpy()  # [96, 96, 96]
                st.session_state["uploaded_pred_mask"] = pred_binary  # [3, 96, 96, 96]
                st.session_state["uploaded_hgg_prob"] = hgg_prob
                st.session_state["uploaded_lgg_prob"] = lgg_prob
                st.session_state["uploaded_class"] = pred_class
                st.session_state["uploaded_filename"] = first_file.name

            except Exception as e:
                st.error(f"Inference failed with error: {e}")
                st.exception(e)
                st.stop()

    # If results are stored in session state, display them
    if "uploaded_pred_mask" in st.session_state:
        st.markdown('<div class="section-title">2. AI Diagnosis & Classification Results</div>', unsafe_allow_html=True)

        pred_mask = st.session_state["uploaded_pred_mask"]
        mri_vol = st.session_state["uploaded_mri"]
        hgg_prob = st.session_state["uploaded_hgg_prob"]
        lgg_prob = st.session_state["uploaded_lgg_prob"]
        pred_class = st.session_state["uploaded_class"]

        is_hgg = (hgg_prob >= 0.5)
        card_class = "result-hgg" if is_hgg else "result-lgg"

        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        with c_res1:
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <div class="metric-label">Predicted Glioma Grade</div>
                <div class="metric-value" style="font-size: 1.4rem;">{pred_class}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_res2:
            st.markdown(f"""
            <div class="metric-card result-hgg">
                <div class="metric-label">HGG Probability</div>
                <div class="metric-value">{hgg_prob * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c_res3:
            st.markdown(f"""
            <div class="metric-card result-lgg">
                <div class="metric-label">LGG Probability</div>
                <div class="metric-value">{lgg_prob * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c_res4:
            wt_voxels = int((pred_mask[1] > 0.5).sum())
            st.markdown(f"""
            <div class="metric-card result-neutral">
                <div class="metric-label">Estimated Whole Tumor Volume</div>
                <div class="metric-value">{wt_voxels:,} <span style="font-size: 0.9rem; font-weight: 500;">voxels</span></div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Interactive Visualization Viewer
        st.markdown('<div class="section-title">3. Interactive 3D Segmentation Slice Viewer</div>', unsafe_allow_html=True)

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            plane_choice = st.selectbox("Anatomical Plane", ["Axial", "Coronal", "Sagittal"], key="up_plane")
        with col_ctrl2:
            highlight_choice = st.selectbox(
                "Tumor Compartment",
                ["All Combined", "Whole Tumor", "Tumor Core", "Enhancing Tumor"],
                key="up_highlight"
            )
        with col_ctrl3:
            # Slices are 96 in all dimensions
            axis_size = 96
            # Find slice with maximum tumor automatically
            wt_sums = pred_mask[1].sum(axis=(1, 2) if plane_choice == "Sagittal" else ((0, 2) if plane_choice == "Coronal" else (0, 1)))
            def_slice = int(np.argmax(wt_sums)) + 1 if np.max(wt_sums) > 0 else 48
            slice_num = st.slider("Slice Index", 1, axis_size, def_slice, key="up_slice")
            slice_idx = slice_num - 1

        # Extract 2D slice
        mri_2d = extract_2d_slice(mri_vol, slice_idx, plane_choice)
        mask_2d = np.stack([extract_2d_slice(pred_mask[c], slice_idx, plane_choice) for c in range(3)], axis=0)
        color_mask = build_composite_color_mask(mask_2d, highlight_choice)
        overlay_2d = create_slice_overlay(mri_2d, color_mask, alpha=0.55)

        # Matplotlib 3-panel display
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        fig.patch.set_facecolor("#0f172a")
        plt.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.10, wspace=0.06)

        fig.suptitle(
            f"Brain MRI Segmentation • {st.session_state['uploaded_filename']}\n"
            f"{plane_choice} View • Slice {slice_num} / {axis_size} • Predicted: {pred_class}",
            fontsize=13, fontweight="bold", color="white", y=0.95
        )

        axes[0].imshow(mri_2d, cmap="gray")
        axes[0].set_title("1. Original Brain MRI", color="white", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        axes[1].imshow(color_mask)
        axes[1].set_title(f"2. Predicted Mask ({highlight_choice})", color="white", fontsize=11, fontweight="bold", pad=8)
        axes[1].axis("off")

        axes[2].imshow(overlay_2d)
        axes[2].set_title("3. MRI + Tumor Overlay", color="white", fontsize=11, fontweight="bold", pad=8)
        axes[2].axis("off")

        fig.text(0.5, 0.02, "■ Red = Whole Tumor (WT)   ■ Green = Tumor Core (TC)   ■ Yellow = Enhancing Tumor (ET)",
                 ha="center", fontsize=9, color="#94a3b8")

        st.pyplot(fig)
        plt.close(fig)

        # Download buttons
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
            st.download_button(
                "⬇️ Download Current Slice (PNG)",
                data=buf.getvalue(),
                file_name=f"tumor_segmentation_slice_{slice_num}.png",
                mime="image/png",
                width="stretch",
            )
        with d_col2:
            # Export 3D mask as NIfTI directly from memory (no disk locks)
            nii_bytes = nifti_to_gzipped_bytes(np.transpose(pred_mask, (1, 2, 3, 0)))
            st.download_button(
                "⬇️ Download Full 3D Segmentation Mask (NIfTI .nii.gz)",
                data=nii_bytes,
                file_name="predicted_tumor_mask_3d.nii.gz",
                mime="application/gzip",
                width="stretch",
            )


# ────────────────────────────────────────────────────────────────────────────
# TAB 2: BENCHMARK ON HELD-OUT TEST CASES (251 TEST CASES)
# ────────────────────────────────────────────────────────────────────────────

with tab_benchmark:
    st.markdown('<div class="section-title">BraTS 2023 Held-Out Test Set (251 Cases)</div>', unsafe_allow_html=True)
    st.markdown(
        "Evaluate pre-computed predictions on the held-out test split. "
        "Allows direct side-by-side comparison between **Ground Truth Annotation** and **Model Prediction**."
    )

    test_ids = get_test_subjects()

    if not test_ids:
        st.warning("No test subjects found in split manifest or predictions folder.")
    else:
        col_bench1, col_bench2 = st.columns([3, 1])
        with col_bench1:
            selected_test_id = st.selectbox("Select Test Patient", test_ids)
        with col_bench2:
            st.metric("Test Cohort Size", f"{len(test_ids)} Cases")

        # Load GT and Prediction
        subj_dir = os.path.join(DATA_DIR, selected_test_id)
        pred_path = os.path.join(PRED_DIR, f"{selected_test_id}_seg_pred.nii.gz")

        if not os.path.exists(pred_path):
            st.error(f"Prediction file not found for {selected_test_id}")
        else:
            # Find MRI files
            t1c_matches = glob.glob(os.path.join(subj_dir, "*t1c*.nii*"))
            seg_matches = glob.glob(os.path.join(subj_dir, "*seg*.nii*"))

            if not t1c_matches or not seg_matches:
                st.error(f"Missing raw MRI or Seg files in {subj_dir}")
            else:
                raw_mri = nib.load(t1c_matches[0]).get_fdata().astype(np.float32)
                raw_seg = nib.load(seg_matches[0]).get_fdata().astype(np.float32)
                pred_vol = nib.load(pred_path).get_fdata().astype(np.float32)  # [96, 96, 96, 3]

                # Preprocess MRI & GT to 96x96x96
                norm_mri = normalize_intensity(torch.from_numpy(raw_mri).unsqueeze(0))
                cropped_mri = crop_or_pad_3d(norm_mri, target_shape=(96, 96, 96))[0].numpy()

                gt_3ch = convert_brats_labels(torch.from_numpy(raw_seg).unsqueeze(0))
                cropped_gt = crop_or_pad_3d(gt_3ch, target_shape=(96, 96, 96)).numpy()  # [3, 96, 96, 96]
                pred_3ch = np.transpose(pred_vol, (3, 0, 1, 2))  # [3, 96, 96, 96]

                # Compute Metrics
                metrics = evaluate_segmentation_case(pred_3ch, cropped_gt)

                # Display Metric Cards
                st.markdown('<div class="section-title">Subject Segmentation Performance</div>', unsafe_allow_html=True)
                bm1, bm2, bm3, bm4, bm5 = st.columns(5)
                with bm1:
                    st.markdown(f"""
                    <div class="metric-card result-neutral">
                        <div class="metric-label">Mean Dice</div>
                        <div class="metric-value">{metrics['mean_dice']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bm2:
                    st.markdown(f"""
                    <div class="metric-card result-hgg">
                        <div class="metric-label">Dice — Whole Tumor</div>
                        <div class="metric-value">{metrics['dice_wt']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bm3:
                    st.markdown(f"""
                    <div class="metric-card result-lgg">
                        <div class="metric-label">Dice — Tumor Core</div>
                        <div class="metric-value">{metrics['dice_tc']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bm4:
                    st.markdown(f"""
                    <div class="metric-card result-neutral">
                        <div class="metric-label">Dice — Enhancing</div>
                        <div class="metric-value">{metrics['dice_et']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bm5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">HD95 (mm)</div>
                        <div class="metric-value">{metrics['mean_hd95']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 4-Panel Comparison Visualization
                st.markdown('<div class="section-title">4-Panel Ground Truth vs Model Prediction</div>', unsafe_allow_html=True)

                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    b_plane = st.selectbox("Viewing Plane", ["Axial", "Coronal", "Sagittal"], key="b_plane")
                with bc2:
                    b_highlight = st.selectbox("Tumor Region", ["All Combined", "Whole Tumor", "Tumor Core", "Enhancing Tumor"], key="b_hl")
                with bc3:
                    wt_sums = pred_3ch[1].sum(axis=(1, 2) if b_plane == "Sagittal" else ((0, 2) if b_plane == "Coronal" else (0, 1)))
                    def_b_slice = int(np.argmax(wt_sums)) + 1 if np.max(wt_sums) > 0 else 48
                    b_slice_num = st.slider("Slice Index", 1, 96, def_b_slice, key="b_slice")
                    b_slice_idx = b_slice_num - 1

                b_mri_2d = extract_2d_slice(cropped_mri, b_slice_idx, b_plane)
                b_gt_2d = np.stack([extract_2d_slice(cropped_gt[c], b_slice_idx, b_plane) for c in range(3)], axis=0)
                b_pred_2d = np.stack([extract_2d_slice(pred_3ch[c], b_slice_idx, b_plane) for c in range(3)], axis=0)

                gt_color = build_composite_color_mask(b_gt_2d, b_highlight)
                pred_color = build_composite_color_mask(b_pred_2d, b_highlight)
                b_overlay = create_slice_overlay(b_mri_2d, pred_color, alpha=0.55)

                fig_b, axes_b = plt.subplots(1, 4, figsize=(19, 5.2))
                fig_b.patch.set_facecolor("#0f172a")
                plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.10, wspace=0.05)

                fig_b.suptitle(
                    f"BraTS 2023 Held-Out Test Case • {selected_test_id}\n"
                    f"{b_plane} View • Slice {b_slice_num} / 96 • Mean Dice = {metrics['mean_dice']:.4f}",
                    fontsize=13, fontweight="bold", color="white", y=0.95
                )

                axes_b[0].imshow(b_mri_2d, cmap="gray")
                axes_b[0].set_title("1. Original MRI (T1ce)", color="white", fontsize=11, fontweight="bold", pad=8)
                axes_b[0].axis("off")

                axes_b[1].imshow(gt_color)
                axes_b[1].set_title(f"2. Ground Truth Mask", color="white", fontsize=11, fontweight="bold", pad=8)
                axes_b[1].axis("off")

                axes_b[2].imshow(pred_color)
                axes_b[2].set_title(f"3. Predicted Mask", color="white", fontsize=11, fontweight="bold", pad=8)
                axes_b[2].axis("off")

                axes_b[3].imshow(b_overlay)
                axes_b[3].set_title("4. MRI + Predicted Overlay", color="white", fontsize=11, fontweight="bold", pad=8)
                axes_b[3].axis("off")

                fig_b.text(0.5, 0.02, "■ Red = Whole Tumor (WT)   ■ Green = Tumor Core (TC)   ■ Yellow = Enhancing Tumor (ET)",
                           ha="center", fontsize=9, color="#94a3b8")

                st.pyplot(fig_b)
                plt.close(fig_b)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL PERFORMANCE GRAPHS & LEARNING CURVES
# ────────────────────────────────────────────────────────────────────────────

with tab_graphs:
    # ── Inject styles specific to graphs tab ──────────────────────────────────
    st.markdown("""
    <style>
    /* ── Graph Tab Hero Banner ── */
    .graph-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 36px 40px 32px 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    .graph-hero h2 {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0 0 10px 0;
    }
    .graph-hero p {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
    }
    .graph-hero .badge-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 18px;
        flex-wrap: wrap;
    }
    .graph-hero .badge {
        background: #1e3a5f;
        border: 1px solid #2563eb;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #93c5fd;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ── Stage header pills ── */
    .stage-pill {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
        margin-top: 8px;
    }
    .stage-pill-1  { background: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; }
    .stage-pill-2  { background: #1a2e1e; color: #86efac; border: 1px solid #22c55e; }
    .stage-pill-3  { background: #2d1a2e; color: #d8b4fe; border: 1px solid #a855f7; }

    /* ── Graph viewer card ── */
    .graph-viewer-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 28px 28px 24px 28px;
        margin-bottom: 10px;
    }
    .graph-desc-text {
        font-size: 0.92rem;
        color: #94a3b8;
        line-height: 1.65;
        margin: 0 0 20px 0;
        padding: 14px 18px;
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
    }

    /* ── Gallery card ── */
    .gallery-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px 20px 14px 20px;
        margin-bottom: 24px;
        transition: border-color 0.2s;
    }
    .gallery-card:hover { border-color: #334155; }
    .gallery-card-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin: 0 0 8px 0;
    }
    .gallery-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 14px 0;
        line-height: 1.4;
    }

    /* ── Section divider ── */
    .graph-section-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #334155, transparent);
        margin: 36px 0 30px 0;
    }
    .graph-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #cbd5e1;
        letter-spacing: -0.2px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Stats row ── */
    .stats-strip {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 28px;
    }
    .stat-tile {
        flex: 1;
        min-width: 140px;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
    }
    .stat-tile-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1;
        margin-bottom: 4px;
    }
    .stat-tile-lbl {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .stat-val-blue  { color: #60a5fa; }
    .stat-val-green { color: #4ade80; }
    .stat-val-amber { color: #fbbf24; }
    .stat-val-purple{ color: #c084fc; }
    .stat-val-rose  { color: #fb7185; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="graph-hero">
        <h2>📊 Performance Analytics Dashboard</h2>
        <p>Comprehensive evaluation of the SwinMamba Dual-Task model across<br>
           pre-training, fine-tuning, and the locked 20% held-out BraTS 2023 test cohort.</p>
        <div class="badge-row">
            <span class="badge">🗂 10 Graphs</span>
            <span class="badge">📦 251 Test Cases Evaluated</span>
            <span class="badge">🖨 300 DPI Export</span>
            <span class="badge">🧠 SwinMamba + IS-MIM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metric Tiles ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="stats-strip">
        <div class="stat-tile">
            <div class="stat-tile-val stat-val-green">86.03%</div>
            <div class="stat-tile-lbl">Mean Dice (Val)</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-val stat-val-blue">88.88%</div>
            <div class="stat-tile-lbl">Dice – Whole Tumor</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-val stat-val-amber">84.63%</div>
            <div class="stat-tile-lbl">Dice – Tumor Core</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-val stat-val-rose">84.58%</div>
            <div class="stat-tile-lbl">Dice – Enhancing</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-val stat-val-purple">11.67 mm</div>
            <div class="stat-tile-lbl">HD95 (Boundary Error)</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-val stat-val-green">98.40%</div>
            <div class="stat-tile-lbl">Grade Accuracy (HGG/LGG)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Graph Catalog Definition ──────────────────────────────────────────────
    graph_catalog = [
        {
            "id": "10_master_performance_dashboard.png",
            "stage": 0,
            "title": "Master Dashboard",
            "emoji": "🏆",
            "label": "All Stages",
            "full_title": "Master Performance Dashboard — 6-Panel Publication Summary",
            "desc": "All-in-one publication-quality dashboard unifying pre-training loss, fine-tuning convergence, segmentation Dice curves, HD95 boundary error, confusion matrix, and per-patient test-set distributions in a single figure.",
        },
        {
            "id": "01_pretrain_loss_and_lr.png",
            "stage": 1,
            "title": "IS-MIM Pre-training Loss & LR",
            "emoji": "📉",
            "label": "Stage 1 — Pre-Training",
            "full_title": "IS-MIM Self-Supervised Pre-training: Masked MSE Loss & Cosine LR Schedule",
            "desc": "Dual-axis chart tracking Masked Auto-Encoding (MSE) loss convergence from 0.5028 → 0.2473 over 10 epochs alongside the Cosine Annealing learning-rate schedule decaying from 5×10⁻⁴ to 1×10⁻⁶.",
        },
        {
            "id": "02_finetune_loss_curve.png",
            "stage": 2,
            "title": "Multi-Task Training Loss",
            "emoji": "📉",
            "label": "Stage 2 — Fine-Tuning",
            "full_title": "Downstream Multi-Task Training Loss Convergence",
            "desc": "Joint multi-task objective (Dice + Focal loss for 3D segmentation, Cross-Entropy for grade classification) converging from 0.3804 → 0.2113 over 3 fine-tuning epochs.",
        },
        {
            "id": "03_validation_dice_scores.png",
            "stage": 2,
            "title": "Segmentation Dice Evolution",
            "emoji": "🎯",
            "label": "Stage 2 — Fine-Tuning",
            "full_title": "Validation Dice Score Learning Curves (WT · TC · ET · Mean)",
            "desc": "Per-region and Mean Dice validation progression. Whole Tumor reaches 88.88%, Tumor Core 84.63%, Enhancing Tumor 84.58%, and Mean Dice 86.03% at the best epoch.",
        },
        {
            "id": "04_hausdorff_distance_hd95.png",
            "stage": 2,
            "title": "HD95 Boundary Error",
            "emoji": "📏",
            "label": "Stage 2 — Fine-Tuning",
            "full_title": "Surface Distance Error (HD95) Convergence — mm",
            "desc": "95th-percentile Hausdorff Distance drops from 15.98 mm (Epoch 1) to 11.67 mm (Epoch 3), demonstrating continuous boundary-level refinement across epochs.",
        },
        {
            "id": "05_classification_learning_curve.png",
            "stage": 2,
            "title": "Grade Classification Curve",
            "emoji": "🧠",
            "label": "Stage 2 — Fine-Tuning",
            "full_title": "Glioma Grade Classification Learning Curve (HGG vs LGG)",
            "desc": "Accuracy (98.40%) and F1-Score (99.19%) convergence for binary High- vs Low-Level Glioma classification, extracted from the dual GAP+GMP bottleneck representation.",
        },
        {
            "id": "06_confusion_matrix.png",
            "stage": 3,
            "title": "Grade Confusion Matrix",
            "emoji": "🔲",
            "label": "Stage 3 — Test Evaluation",
            "full_title": "Glioma Grade Classification Confusion Matrix (HGG vs LGG)",
            "desc": "Normalized confusion matrix showing true/predicted class distributions for High-Level (HGG) and Low-Level (LGG) glioma classification on the 251-patient held-out test cohort.",
        },
        {
            "id": "07_test_dice_distributions.png",
            "stage": 3,
            "title": "Test Cohort Dice Distributions",
            "emoji": "📦",
            "label": "Stage 3 — Test Evaluation",
            "full_title": "Test Cohort Segmentation Dice Distributions — Box & Violin Plots",
            "desc": "Per-patient Dice score distributions across all 251 held-out test subjects. Violin + box plot overlay reveals the spread, median, interquartile range, and outliers for WT, TC, ET, and Mean.",
        },
        {
            "id": "08_test_hd95_distribution.png",
            "stage": 3,
            "title": "Test HD95 Distribution",
            "emoji": "📊",
            "label": "Stage 3 — Test Evaluation",
            "full_title": "Test Cohort HD95 Boundary Distance Distribution",
            "desc": "Histogram of 95th-percentile Hausdorff Distance across test cases with Mean, Median, and 90th-percentile markers. Low overall error demonstrates robust boundary delineation.",
        },
        {
            "id": "09_multitask_radar_chart.png",
            "stage": 3,
            "title": "Multi-Task Radar Benchmark",
            "emoji": "🕸️",
            "label": "Stage 3 — Test Evaluation",
            "full_title": "Holistic Multi-Task Performance Radar Benchmark Chart",
            "desc": "Spider chart benchmarking SwinMamba across 7 clinical dimensions (WT·TC·ET Dice, Mean Dice, HD95 Quality, Grade Accuracy, Grade F1) versus clinical target baselines.",
        },
    ]

    STAGE_COLORS = {0: ("All Stages", "stage-pill-3"), 1: ("Stage 1 — Pre-Training", "stage-pill-1"), 2: ("Stage 2 — Fine-Tuning", "stage-pill-2"), 3: ("Stage 3 — Held-Out Test", "stage-pill-3")}

    # ── Featured Viewer ──────────────────────────────────────────────────────
    st.markdown('<div class="graph-section-title">🔍 &nbsp; Select & Inspect</div>', unsafe_allow_html=True)

    sel_col, _ = st.columns([2, 1])
    with sel_col:
        selected_graph_idx = st.selectbox(
            "Choose a graph to view",
            range(len(graph_catalog)),
            format_func=lambda i: f"{graph_catalog[i]['emoji']}  {graph_catalog[i]['full_title']}",
            label_visibility="collapsed",
        )

    active = graph_catalog[selected_graph_idx]
    active_path = os.path.join(GRAPH_DIR, active["id"])
    stage_lbl, stage_cls = STAGE_COLORS[active["stage"]]

    st.markdown(f'<span class="stage-pill {stage_cls}">{stage_lbl}</span>', unsafe_allow_html=True)

    if os.path.exists(active_path):
        st.markdown(f'<p class="graph-desc-text">{active["desc"]}</p>', unsafe_allow_html=True)

        st.markdown('<div class="graph-viewer-card">', unsafe_allow_html=True)
        st.image(active_path, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
        with dl_col1:
            with open(active_path, "rb") as f:
                img_bytes = f.read()
            st.download_button(
                label="⬇️  Download PNG (300 DPI)",
                data=img_bytes,
                file_name=active["id"],
                mime="image/png",
                width="stretch",
            )
    else:
        st.warning(f"Graph not found. Please run:  `python generate_all_graphs.py`")

    # ── Gallery Section ───────────────────────────────────────────────────────
    st.markdown('<div class="graph-section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-section-title">🖼 &nbsp; Complete Graph Gallery</div>', unsafe_allow_html=True)

    # Group by stage
    stages = [
        ("Stage 1 — IS-MIM Self-Supervised Pre-Training", "stage-pill-1", 1),
        ("Stage 2 — Downstream Multi-Task Fine-Tuning", "stage-pill-2", 2),
        ("Stage 3 — Held-Out Test Set Evaluation", "stage-pill-3", 3),
    ]

    # Master dashboard on its own row first
    master = graph_catalog[0]
    master_path = os.path.join(GRAPH_DIR, master["id"])
    if os.path.exists(master_path):
        st.markdown(f'<span class="stage-pill stage-pill-3">All Stages — Publication Summary</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="gallery-card-title">{master["emoji"]} {master["full_title"]}</div>', unsafe_allow_html=True)
        st.image(master_path, width="stretch")
        st.markdown("&nbsp;", unsafe_allow_html=True)

    for stage_name, pill_cls, stage_id in stages:
        stage_graphs = [g for g in graph_catalog if g["stage"] == stage_id]
        if not stage_graphs:
            continue

        st.markdown('<div class="graph-section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<span class="stage-pill {pill_cls}">{stage_name}</span>', unsafe_allow_html=True)
        st.markdown("&nbsp;", unsafe_allow_html=True)

        # 2-column grid per stage
        for row_start in range(0, len(stage_graphs), 2):
            pair = stage_graphs[row_start:row_start + 2]
            cols = st.columns(len(pair), gap="large")
            for col, g in zip(cols, pair):
                with col:
                    p = os.path.join(GRAPH_DIR, g["id"])
                    st.markdown(f"""
                    <div class="gallery-card">
                        <div class="gallery-card-label">{g["label"]}</div>
                        <div class="gallery-card-title">{g["emoji"]} {g["title"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(p):
                        st.image(p, width="stretch")
                    else:
                        st.warning("Missing — run generate_all_graphs.py")
            st.markdown("&nbsp;", unsafe_allow_html=True)

# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    🧠 <b>Brain AI — Advanced Glioma Segmentation & Classification System</b><br>
    Built with Swin Transformer + Mamba SSM (Dual-Task) • BraTS 2023 Adult Glioma Dataset<br>
    <i>Research Prototype: Model outputs are for experimental evaluation and not intended as clinical diagnostic devices.</i>
</div>
""", unsafe_allow_html=True)
