"""
Interactive 3D Slice Visualizer for Glioma Segmentation Predictions.

Displays:
  [Original MRI]  |  [Predicted Tumor Mask]  |  [MRI + Tumor Mask Overlay]
Matching the interactive 3-panel medical viewer layout.

Usage:
  # Open interactive GUI window:
  python visualize_predictions.py
  python visualize_predictions.py --subject_id BraTS-GLI-00002-000
  python visualize_predictions.py --modality t2f --mask_type wt --plane axial

  # Export static image directly to disk:
  python visualize_predictions.py --save_png outputs/figure1.png
"""

from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import nibabel as nib
import torch

from src.data.transforms import normalize_intensity, crop_or_pad_3d


def load_subject_data(data_dir: str, pred_dir: str, subject_id: str, modality: str = "t1c"):
    """Loads matching MRI modality and predicted 3D segmentation mask."""
    # 1. Locate predicted mask
    pred_path = os.path.join(pred_dir, f"{subject_id}_seg_pred.nii.gz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predicted mask not found at: {pred_path}")

    # shape: [96, 96, 96, 3] (TC, WT, ET)
    pred_mask = nib.load(pred_path).get_fdata().astype(np.float32)

    # 2. Locate raw MRI volume
    subj_folder = os.path.join(data_dir, subject_id)
    mod_matches = glob.glob(os.path.join(subj_folder, f"*{modality}*.nii*"))
    if not mod_matches:
        mod_matches = [f for f in glob.glob(os.path.join(subj_folder, "*.nii*")) if "seg" not in f]

    if not mod_matches:
        raise FileNotFoundError(f"No MRI volume found for modality '{modality}' in {subj_folder}")

    mri_raw = nib.load(mod_matches[0]).get_fdata().astype(np.float32)

    # Convert to tensor and apply standard normalization + crop/pad
    mri_tensor = torch.from_numpy(mri_raw).unsqueeze(0)  # [1, D, H, W]
    mri_norm = normalize_intensity(mri_tensor)
    mri_cropped = crop_or_pad_3d(mri_norm, target_shape=pred_mask.shape[:3])
    mri_vol = mri_cropped[0].numpy()  # [96, 96, 96]

    return mri_vol, pred_mask


def get_2d_slice(vol_3d: np.ndarray, slice_idx: int, plane: str = "axial"):
    """Extracts a 2D slice based on view plane (axial, coronal, sagittal)."""
    if plane == "axial":
        # Slicing along z-axis (axis 2)
        return np.rot90(vol_3d[:, :, slice_idx])
    elif plane == "coronal":
        # Slicing along y-axis (axis 1)
        return np.rot90(vol_3d[:, slice_idx, :])
    else:
        # Slicing along x-axis (axis 0, sagittal)
        return np.rot90(vol_3d[slice_idx, :, :])


def create_overlay(mri_2d: np.ndarray, mask_2d: np.ndarray, color=(0.85, 0.15, 0.15), alpha=0.50):
    """Alpha-blends grayscale MRI with a colored tumor mask."""
    norm_mri = (mri_2d - mri_2d.min()) / (mri_2d.max() - mri_2d.min() + 1e-7)
    rgb = np.repeat(norm_mri[:, :, np.newaxis], 3, axis=2)

    roi = mask_2d > 0.5
    if np.any(roi):
        rgb[roi] = (1.0 - alpha) * rgb[roi] + alpha * np.array(color)
    return np.clip(rgb, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Interactive Glioma 3D Slice Viewer")
    parser.add_argument("--data_dir", type=str, default=r"C:\Glioma\Dataset\Glioma")
    parser.add_argument("--pred_dir", type=str, default=r"C:\Glioma\outputs\predictions")
    parser.add_argument("--subject_id", type=str, default=None,
                        help="Subject ID (e.g. BraTS-GLI-00002-000). Defaults to first available.")
    parser.add_argument("--modality", type=str, default="t1c",
                        choices=["t1c", "t2f", "t1n", "t2w"],
                        help="MRI modality: t1c (T1ce), t2f (FLAIR), t2w (T2), t1n (T1)")
    parser.add_argument("--mask_type", type=str, default="wt",
                        choices=["wt", "tc", "et"],
                        help="Mask: wt (Whole Tumor), tc (Tumor Core), et (Enhancing Tumor)")
    parser.add_argument("--plane", type=str, default="axial",
                        choices=["axial", "coronal", "sagittal"],
                        help="Anatomical viewing plane")
    parser.add_argument("--slice", type=int, default=None,
                        help="Initial slice number (1-indexed). Defaults to max tumor slice.")
    parser.add_argument("--save_png", type=str, default=None,
                        help="Path to save static snapshot PNG (skips GUI if specified)")
    args = parser.parse_args()

    # Find subject
    if not args.subject_id:
        pred_files = glob.glob(os.path.join(args.pred_dir, "*_seg_pred.nii.gz"))
        if not pred_files:
            print(f"Error: No prediction files found in {args.pred_dir}")
            return
        args.subject_id = os.path.basename(pred_files[0]).replace("_seg_pred.nii.gz", "")

    print(f"Loading {args.subject_id} | Modality: {args.modality.upper()} | Plane: {args.plane.capitalize()}...")
    mri_vol, pred_mask = load_subject_data(args.data_dir, args.pred_dir, args.subject_id, args.modality)

    mask_channel_map = {"tc": 0, "wt": 1, "et": 2}
    mask_names = {"wt": "Whole Tumor", "tc": "Tumor Core", "et": "Enhancing Tumor"}
    selected_mask = pred_mask[:, :, :, mask_channel_map[args.mask_type]]

    color_map = {
        "wt": (0.85, 0.15, 0.15),   # Red
        "tc": (0.15, 0.85, 0.20),   # Green
        "et": (1.00, 0.80, 0.00),   # Yellow
    }
    overlay_color = color_map[args.mask_type]

    # Plane axis mapping: axial=2, coronal=1, sagittal=0
    axis_map = {"sagittal": 0, "coronal": 1, "axial": 2}
    plane_axis = axis_map[args.plane]
    total_slices = mri_vol.shape[plane_axis]

    # Find best slice (slice with maximum tumor area)
    if args.slice is None:
        slice_sums = selected_mask.sum(axis=tuple(i for i in range(3) if i != plane_axis))
        best_slice = int(np.argmax(slice_sums)) if np.max(slice_sums) > 0 else total_slices // 2
    else:
        best_slice = max(0, min(total_slices - 1, args.slice - 1))

    # Matplotlib setup
    import matplotlib
    if args.save_png:
        matplotlib.use("Agg")  # Non-interactive backend for file export
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    fig.patch.set_facecolor('#0d0e12')
    plt.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.18, wspace=0.06)

    mod_labels = {"t1c": "T1-Gd (Contrast)", "t2f": "T2-FLAIR", "t2w": "T2-Weighted", "t1n": "T1-Native"}
    mod_str = mod_labels.get(args.modality, args.modality.upper())

    # Supertitle
    title_text = fig.suptitle(
        f"Glioma Segmentation\n{args.subject_id} | Slice {best_slice + 1} / {total_slices}",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.94
    )

    mri_2d = get_2d_slice(mri_vol, best_slice, args.plane)
    mask_2d = get_2d_slice(selected_mask, best_slice, args.plane)
    overlay_2d = create_overlay(mri_2d, mask_2d, color=overlay_color)

    # 1. Original MRI
    im1 = axes[0].imshow(mri_2d, cmap="gray")
    axes[0].set_title(f"Original MRI ({mod_str})\nSlice {best_slice + 1}", color="white", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")

    # 2. Binary Tumor Mask
    im2 = axes[1].imshow(mask_2d, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Binary {mask_names[args.mask_type]} Mask\nSlice {best_slice + 1}", color="white", fontsize=12, fontweight="bold", pad=10)
    axes[1].axis("off")

    # 3. MRI + Mask Overlay
    im3 = axes[2].imshow(overlay_2d)
    axes[2].set_title(f"MRI + Tumor Mask Overlay\nSlice {best_slice + 1}", color="white", fontsize=12, fontweight="bold", pad=10)
    axes[2].axis("off")

    # If saving static image
    if args.save_png:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_png)), exist_ok=True)
        plt.savefig(args.save_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"Saved figure to: {args.save_png}")
        plt.close(fig)
        return

    # Interactive slider
    ax_slider = plt.axes([0.15, 0.06, 0.70, 0.035], facecolor="#22252e")
    slice_slider = Slider(
        ax=ax_slider,
        label="Slice  ",
        valmin=1,
        valmax=total_slices,
        valinit=best_slice + 1,
        valstep=1,
        color="#dc2626" if args.mask_type == "wt" else "#2563eb"
    )
    slice_slider.label.set_color("white")
    slice_slider.valtext.set_color("white")

    def update(val):
        curr = int(slice_slider.val) - 1
        m2 = get_2d_slice(mri_vol, curr, args.plane)
        mk2 = get_2d_slice(selected_mask, curr, args.plane)
        ov2 = create_overlay(m2, mk2, color=overlay_color)

        im1.set_data(m2)
        im2.set_data(mk2)
        im3.set_data(ov2)

        title_text.set_text(f"Glioma Segmentation\n{args.subject_id} | Slice {curr + 1} / {total_slices}")
        axes[0].set_title(f"Original MRI ({mod_str})\nSlice {curr + 1}", color="white", fontsize=12, fontweight="bold", pad=10)
        axes[1].set_title(f"Binary {mask_names[args.mask_type]} Mask\nSlice {curr + 1}", color="white", fontsize=12, fontweight="bold", pad=10)
        axes[2].set_title(f"MRI + Tumor Mask Overlay\nSlice {curr + 1}", color="white", fontsize=12, fontweight="bold", pad=10)
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)

    # Keyboard & Scroll wheel support
    def on_key(event):
        curr = int(slice_slider.val)
        if event.key in ["right", "up"] and curr < total_slices:
            slice_slider.set_val(curr + 1)
        elif event.key in ["left", "down"] and curr > 1:
            slice_slider.set_val(curr - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def on_scroll(event):
        curr = int(slice_slider.val)
        if event.button == "up" and curr < total_slices:
            slice_slider.set_val(curr + 1)
        elif event.button == "down" and curr > 1:
            slice_slider.set_val(curr - 1)

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    print("\n[Interactive Viewer Active]")
    print("  * Drag the bottom slider to change slices")
    print("  * Use Left/Right Arrow keys or Mouse Scroll wheel to navigate slices")
    print("  * Close window when done.")
    plt.show()


if __name__ == "__main__":
    main()
