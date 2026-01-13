"""
Focal Loss implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for dense prediction.

    Reference: Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=2, beta=4, reduction='mean'):
        """
        Args:
            alpha: Focusing parameter for positive examples
            beta: Focusing parameter for negative examples
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target):
        """Compute focal loss.

        Args:
            pred: [B, 1, H, W] Predicted heatmap (after sigmoid)
            target: [B, 1, H, W] Target Gaussian heatmap

        Returns:
            loss: Scalar loss value
        """
        # Ensure pred is in valid range
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # Positive loss (where target > 0.99)
        pos_mask = target.ge(0.99).float()
        neg_mask = target.lt(0.99).float()

        # Positive examples
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask

        # Negative examples
        neg_loss = -((1 - target) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask

        # Combine
        num_pos = pos_mask.sum()
        if num_pos == 0:
            loss = neg_loss.sum()
        else:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


def focal_loss(pred, target, alpha=2, beta=4):
    """Functional interface for focal loss."""
    return FocalLoss(alpha, beta)(pred, target)
