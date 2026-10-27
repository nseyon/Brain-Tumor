"""
Post-processing operations for 3D Brain Tumor Segmentation.
Implements:
1. Sigmoid probability thresholding (default: 0.5)
2. Small cluster removal based on anatomical volume thresholds:
   - Whole Tumor (WT) < 250 voxels
   - Tumor Core (TC) < 150 voxels
   - Enhancing Tumor (ET) < 100 voxels
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import label as nd_label


def postprocess_predictions(
    seg_logits: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    wt_threshold: int = 250,
    tc_threshold: int = 150,
    et_threshold: int = 100,
) -> np.ndarray:
    """
    seg_logits: [C=3, D, H, W] (TC, WT, ET) or [B, 3, D, H, W]
    Returns binary mask array of identical spatial shape.
    """
    if isinstance(seg_logits, torch.Tensor):
        probs = torch.sigmoid(seg_logits).detach().cpu().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-seg_logits))

    is_batch = (probs.ndim == 5)
    if not is_batch:
        probs = np.expand_dims(probs, 0)

    b, c, d, h, w = probs.shape
    binary_masks = (probs > threshold).astype(np.uint8)

    thresholds = [tc_threshold, wt_threshold, et_threshold]

    for bi in range(b):
        for ci in range(min(c, 3)):
            channel_mask = binary_masks[bi, ci]
            if channel_mask.sum() == 0:
                continue

            # Connected component analysis in 3D
            labeled, num_features = nd_label(channel_mask)
            min_size = thresholds[ci]

            for feat_idx in range(1, num_features + 1):
                cluster_size = (labeled == feat_idx).sum()
                if cluster_size < min_size:
                    binary_masks[bi, ci][labeled == feat_idx] = 0

    if not is_batch:
        binary_masks = binary_masks[0]

    return binary_masks
