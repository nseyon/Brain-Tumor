"""
Masked Image Modeling (MIM) Reconstruction Loss.
Computes Mean Squared Error (MSE) strictly on masked and modality-swapped voxels,
as defined in Qi et al. (BSPC 2024).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """
    Computes MSE loss over perturbed voxels indicated by voxel_mask.
    pred: [B, C, D, H, W]
    target: [B, C, D, H, W]
    voxel_mask: [B, 1, D, H, W] boolean/float tensor (1 where loss is evaluated)
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        voxel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if voxel_mask is None:
            return F.mse_loss(pred, target)

        # Broadcast mask across channels if mask is [B, 1, D, H, W]
        if voxel_mask.shape[1] == 1 and pred.shape[1] > 1:
            voxel_mask = voxel_mask.repeat(1, pred.shape[1], 1, 1, 1)

        mask_bool = voxel_mask.bool()
        num_masked = mask_bool.sum().float()

        if num_masked == 0:
            return F.mse_loss(pred, target)

        diff = (pred - target) ** 2
        masked_loss = (diff * mask_bool.float()).sum() / num_masked
        return masked_loss
