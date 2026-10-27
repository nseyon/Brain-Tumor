"""
Multi-Task Loss for Joint Brain Tumor Segmentation and Classification.
Combines:
  1. Weighted Dice + Focal Loss across tumor subregions (WT, TC, ET)
  2. CrossEntropy / BCE Loss for Glioma Grade Classification (HGG vs LGG)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-channel Soft Dice Loss with subregion weighting."""
    def __init__(self, smooth: float = 1e-5, class_weights: list[float] | None = None):
        super().__init__()
        self.smooth = smooth
        # Default weights from Rahim et al. (Table 4): TC=1.3, WT=1.0, ET=1.5
        self.class_weights = class_weights or [1.3, 1.0, 1.5]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num_channels = logits.shape[1]
        loss = 0.0

        for c in range(num_channels):
            p = probs[:, c].contiguous().view(-1)
            t = targets[:, c].contiguous().view(-1)

            intersection = (p * t).sum()
            denominator = (p * p).sum() + (t * t).sum()
            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            weight = self.class_weights[c] if c < len(self.class_weights) else 1.0
            loss += weight * (1.0 - dice)

        return loss / sum(self.class_weights[:num_channels])


class SigmoidFocalLoss(nn.Module):
    """Focal loss for multi-label volumetric segmentation."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * probs + (1 - targets) * (1 - probs)
        loss = bce * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss

        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined Multi-Task Loss:
    L_total = (L_Dice + L_Focal) + lambda_clf * L_Classification
    """
    def __init__(
        self,
        class_weights: list[float] | None = None,
        lambda_clf: float = 0.5,
        clf_smoothing: float = 0.05,
    ):
        super().__init__()
        self.dice_loss = SoftDiceLoss(class_weights=class_weights)
        self.focal_loss = SigmoidFocalLoss()
        self.clf_loss = nn.CrossEntropyLoss(label_smoothing=clf_smoothing)
        self.lambda_clf = lambda_clf

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        seg_logits = outputs["seg"]
        seg_targets = targets["seg"]
        clf_logits = outputs["clf"]
        clf_targets = targets["label"]

        # Segmentation losses
        l_dice = self.dice_loss(seg_logits, seg_targets)
        l_focal = self.focal_loss(seg_logits, seg_targets)
        l_seg = l_dice + l_focal

        # Classification loss
        l_clf = self.clf_loss(clf_logits, clf_targets)

        # Combined total
        l_total = l_seg + self.lambda_clf * l_clf

        return {
            "loss": l_total,
            "loss_seg": l_seg,
            "loss_dice": l_dice,
            "loss_focal": l_focal,
            "loss_clf": l_clf,
        }
