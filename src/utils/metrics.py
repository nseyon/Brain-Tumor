"""
Evaluation metrics for 3D Brain Tumor Segmentation and Classification.
Implements:
- Volumetric Dice Coefficient (TC, WT, ET, Mean Dice)
- 95th Percentile Hausdorff Distance (HD95)
- Classification Accuracy, Precision, Recall, and F1 Score
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def compute_dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Computes binary Dice similarity coefficient."""
    p = (pred > 0.5).astype(bool)
    t = (target > 0.5).astype(bool)
    intersection = np.logical_and(p, t).sum()
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0  # Perfect agreement if both empty
    return float((2.0 * intersection + smooth) / (total + smooth))


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes 95th percentile Hausdorff Distance between two binary 3D masks.
    """
    p = (pred > 0.5).astype(bool)
    t = (target > 0.5).astype(bool)

    # Edge cases
    if p.sum() == 0 and t.sum() == 0:
        return 0.0
    if p.sum() == 0 or t.sum() == 0:
        return 377.0  # Standard BraTS maximum penalty

    # Distance transforms
    dt_pred = distance_transform_edt(~p)
    dt_target = distance_transform_edt(~t)

    # Distances from boundary of target to pred, and pred to target
    dists_t_to_p = dt_pred[t]
    dists_p_to_t = dt_target[p]

    if dists_t_to_p.size == 0 or dists_p_to_t.size == 0:
        return 377.0

    hd95_1 = np.percentile(dists_t_to_p, 95)
    hd95_2 = np.percentile(dists_p_to_t, 95)
    return float(max(hd95_1, hd95_2))


def evaluate_segmentation_case(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """
    Evaluates 3 subregions (TC, WT, ET):
    pred_mask: [3, D, H, W]
    gt_mask: [3, D, H, W]
    """
    metrics = {}
    names = ["tc", "wt", "et"]

    dice_scores = []
    hd95_scores = []

    for idx, name in enumerate(names):
        d = compute_dice_score(pred_mask[idx], gt_mask[idx])
        hd = compute_hd95(pred_mask[idx], gt_mask[idx])
        metrics[f"dice_{name}"] = d
        metrics[f"hd95_{name}"] = hd
        dice_scores.append(d)
        hd95_scores.append(hd)

    metrics["mean_dice"] = float(np.mean(dice_scores))
    metrics["mean_hd95"] = float(np.mean(hd95_scores))
    return metrics


def compute_classification_metrics(preds: list[int], targets: list[int]) -> dict[str, float]:
    """Computes binary classification metrics for High-Level vs Low-Level glioma."""
    p = np.array(preds)
    t = np.array(targets)

    acc = float((p == t).mean()) if len(t) > 0 else 0.0

    tp = int(np.logical_and(p == 1, t == 1).sum())
    fp = int(np.logical_and(p == 1, t == 0).sum())
    fn = int(np.logical_and(p == 0, t == 1).sum())
    tn = int(np.logical_and(p == 0, t == 0).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
