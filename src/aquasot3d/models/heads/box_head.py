"""
2D Bounding Box Prediction Heads

This module implements OSTrack-style CENTER prediction heads for visual object tracking.
The CENTER head uses spatial convolutions to preserve location information and predict
target center heatmaps, sizes, and sub-pixel offsets.

Architecture:
- Score branch: Predicts center probability heatmap [B, 1, H, W]
- Size branch: Predicts normalized (w, h) at each location [B, 2, H, W]
- Offset branch: Predicts sub-pixel offset refinement [B, 2, H, W]

Reference: OSTrack (ECCV 2022) - Joint Feature Learning and Relation Modeling

Author: AquaSOT-3D Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

from aquasot3d.core import BaseHead, PREDICTION_HEAD_REGISTRY


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: Boxes in (cx, cy, w, h) format, shape (..., 4)

    Returns:
        Boxes in (x1, y1, x2, y2) format, shape (..., 4)
    """
    cx, cy, w, h = boxes.unbind(-1)
    w = w.clamp(min=1e-4)
    h = h.clamp(min=1e-4)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        boxes: Boxes in (x1, y1, x2, y2) format, shape (..., 4)

    Returns:
        Boxes in (cx, cy, w, h) format, shape (..., 4)
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_xywh_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x, y, w, h) to (cx, cy, w, h) format.

    Args:
        boxes: Boxes in (x, y, w, h) format (top-left corner + size), shape (..., 4)

    Returns:
        Boxes in (cx, cy, w, h) format (center point + size), shape (..., 4)
    """
    x, y, w, h = boxes.unbind(-1)
    cx = x + w / 2
    cy = y + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: First set of boxes in xyxy format, shape (N, 4)
        boxes2: Second set of boxes in xyxy format, shape (N, 4)

    Returns:
        IoU values, shape (N,), clamped to [0, 1]
    """
    x1_max = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1_max = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2_min = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2_min = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2_min - x1_max).clamp(min=0)
    inter_h = (y2_min - y1_max).clamp(min=0)
    inter_area = inter_w * inter_h

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6)

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area.clamp(min=1e-7)
    iou = torch.clamp(iou, min=0.0, max=1.0)
    iou = torch.where(torch.isfinite(iou), iou, torch.zeros_like(iou))

    return iou


def compute_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.

    GIoU paper: https://arxiv.org/abs/1902.09630

    Args:
        boxes1: First set of boxes in xyxy format, shape (N, 4)
        boxes2: Second set of boxes in xyxy format, shape (N, 4)

    Returns:
        GIoU values, shape (N,), in range [-1, 1]
    """
    iou = compute_iou(boxes1, boxes2)

    x1_min = torch.min(boxes1[:, 0], boxes2[:, 0])
    y1_min = torch.min(boxes1[:, 1], boxes2[:, 1])
    x2_max = torch.max(boxes1[:, 2], boxes2[:, 2])
    y2_max = torch.max(boxes1[:, 3], boxes2[:, 3])

    enclosing_w = (x2_max - x1_min).clamp(min=1e-6)
    enclosing_h = (y2_max - y1_min).clamp(min=1e-6)
    enclosing_area = enclosing_w * enclosing_h

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6)

    area1 = w1 * h1
    area2 = w2 * h2

    inter_w = (torch.min(boxes1[:, 2], boxes2[:, 2]) - torch.max(boxes1[:, 0], boxes2[:, 0])).clamp(min=0)
    inter_h = (torch.min(boxes1[:, 3], boxes2[:, 3]) - torch.max(boxes1[:, 1], boxes2[:, 1])).clamp(min=0)
    inter_area = inter_w * inter_h
    union_area = area1 + area2 - inter_area

    giou = iou - (enclosing_area - union_area) / enclosing_area.clamp(min=1e-7)
    giou = torch.clamp(giou, min=-1.0, max=1.0)
    giou = torch.where(torch.isfinite(giou), giou, torch.full_like(giou, -1.0))

    return giou


def compute_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Complete IoU (CIoU) between two sets of boxes.

    CIoU paper: https://arxiv.org/abs/2005.03572

    Args:
        boxes1: First set of boxes in xyxy format, shape (N, 4)
        boxes2: Second set of boxes in xyxy format, shape (N, 4)

    Returns:
        CIoU values, shape (N,), in range [-1, 1]
    """
    iou = compute_iou(boxes1, boxes2)

    boxes1_cxcywh = box_xyxy_to_cxcywh(boxes1)
    boxes2_cxcywh = box_xyxy_to_cxcywh(boxes2)

    cx1, cy1, w1, h1 = boxes1_cxcywh.unbind(-1)
    cx2, cy2, w2, h2 = boxes2_cxcywh.unbind(-1)
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    x1_min = torch.min(boxes1[:, 0], boxes2[:, 0])
    y1_min = torch.min(boxes1[:, 1], boxes2[:, 1])
    x2_max = torch.max(boxes1[:, 2], boxes2[:, 2])
    y2_max = torch.max(boxes1[:, 3], boxes2[:, 3])

    diagonal_w = (x2_max - x1_min).clamp(min=1e-6)
    diagonal_h = (y2_max - y1_min).clamp(min=1e-6)
    diagonal_sq = diagonal_w ** 2 + diagonal_h ** 2

    w1_safe = w1.clamp(min=1e-6)
    h1_safe = h1.clamp(min=1e-6)
    w2_safe = w2.clamp(min=1e-6)
    h2_safe = h2.clamp(min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2_safe / h2_safe) - torch.atan(w1_safe / h1_safe), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_dist_sq / (diagonal_sq + 1e-7) + alpha * v)
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)
    ciou = torch.where(torch.isfinite(ciou), ciou, torch.full_like(ciou, -1.0))

    return ciou


class GIoULoss(nn.Module):
    """Generalized IoU Loss: Loss = 1 - GIoU"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = compute_giou(pred_xyxy, target_xyxy)
        giou = torch.clamp(giou, min=-1.0, max=1.0)
        loss = 1 - giou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CIoULoss(nn.Module):
    """Complete IoU Loss: Loss = 1 - CIoU"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        ciou = compute_ciou(pred_xyxy, target_xyxy)
        ciou = torch.clamp(ciou, min=-1.0, max=1.0)
        loss = 1 - ciou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss for joint classification and quality estimation.
    QFL paper: https://arxiv.org/abs/2006.04388
    """

    def __init__(self, beta: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_labels: torch.Tensor,
        target_quality: torch.Tensor
    ) -> torch.Tensor:
        if torch.any(~torch.isfinite(pred_scores)):
            pred_scores = torch.where(
                torch.isfinite(pred_scores),
                pred_scores,
                torch.zeros_like(pred_scores)
            )

        if torch.any(~torch.isfinite(target_quality)):
            target_quality = torch.clamp(target_quality, 0.0, 1.0)
            target_quality = torch.where(
                torch.isfinite(target_quality),
                target_quality,
                torch.zeros_like(target_quality)
            )

        pred_scores_prob = torch.sigmoid(pred_scores)
        scale_factor = pred_scores_prob - target_labels * target_quality
        focal_weight = torch.abs(scale_factor) ** self.beta

        target = torch.clamp((target_labels * target_quality).float(), 0.0, 1.0)
        bce_loss = F.binary_cross_entropy_with_logits(pred_scores, target, reduction='none')
        loss = focal_weight * bce_loss

        if torch.any(~torch.isfinite(loss)):
            loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


@PREDICTION_HEAD_REGISTRY.register("CenterHead")
class CenterHead(BaseHead):
    """
    OSTrack-style CENTER prediction head with spatial convolutions.

    Architecture:
    - Score branch: Predicts center probability heatmap [B, 1, H, W]
    - Size branch: Predicts normalized (w, h) at each location [B, 2, H, W]
    - Offset branch: Predicts sub-pixel offset refinement [B, 2, H, W]

    The final bbox is computed by:
    1. Find location with highest score (soft-argmax)
    2. Extract size prediction at that location
    3. Apply offset refinement for sub-pixel accuracy

    Reference: OSTrack (ECCV 2022)

    Args:
        in_channels: Input feature dimension (e.g., 768 for ViT-Base)
        feat_sz: Spatial size of input features (e.g., 16 for 256/16)
        hidden_dim: Hidden dimension for conv layers (default: 256)
        use_offset: Whether to use offset branch for sub-pixel refinement
        loss_weight_score: Weight for score map loss (default: 1.0)
        loss_weight_size: Weight for size regression loss (default: 5.0)
        loss_weight_offset: Weight for offset regression loss (default: 1.0)
    """

    def __init__(
        self,
        in_channels: int = 768,
        feat_sz: int = 16,
        hidden_dim: int = 256,
        use_offset: bool = True,
        loss_weight_score: float = 1.0,
        loss_weight_size: float = 5.0,
        loss_weight_offset: float = 1.0,
        loss_weight_objectness: float = 1.0,
        # CRITICAL FIX (Audit 2025120902 Issue 4): Soft-argmax option
        use_soft_argmax: bool = True,  # Use soft-argmax instead of hard argmax
        soft_argmax_temperature: float = 10.0,  # Temperature for soft-argmax
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.feat_sz = feat_sz
        self.hidden_dim = hidden_dim
        self.use_offset = use_offset
        self.loss_weight_score = loss_weight_score
        self.loss_weight_size = loss_weight_size
        self.loss_weight_offset = loss_weight_offset
        self.loss_weight_objectness = loss_weight_objectness
        # CRITICAL FIX (Audit 2025120902 Issue 4): Soft-argmax parameters
        self.use_soft_argmax = use_soft_argmax
        self.soft_argmax_temperature = soft_argmax_temperature

        # Score branch: predicts center probability heatmap
        self.score_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
        )

        # Size branch: predicts normalized (w, h) at each location
        self.size_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 2, kernel_size=1),
        )

        # Offset branch: predicts sub-pixel offset refinement
        if use_offset:
            self.offset_branch = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 2, kernel_size=1),
            )
        else:
            self.offset_branch = None

        # CRITICAL FIX (Audit 2025120301 Finding 3, 9):
        # Independent objectness branch - decoupled from quality/score
        # Predicts binary target presence using has_object as supervision
        # This enables search expansion strategies to rely on true objectness
        self.objectness_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
        # Validity branch - combines padding ratio + objectness for fusion routing
        # Supervised by: label=1 when (padding_ratio<0.3 AND has_object=1)
        #                label=0 when (padding_ratio>0.5 OR has_object=0)
        # This enables single-forward validity inference for offline/online eval
        self.validity_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Loss weight for validity head
        self.loss_weight_validity = kwargs.get('loss_weight_validity', 1.0)

        self._init_weights()

        self.logger.info(
            f"Initialized CenterHead: in_channels={in_channels}, "
            f"feat_sz={feat_sz}, hidden_dim={hidden_dim}, use_offset={use_offset}"
        )

    def _init_weights(self):
        """Initialize weights following OSTrack conventions."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if hasattr(self.score_branch[-1], 'bias') and self.score_branch[-1].bias is not None:
            nn.init.constant_(self.score_branch[-1].bias, -2.0)

        if hasattr(self.size_branch[-1], 'bias') and self.size_branch[-1].bias is not None:
            nn.init.constant_(self.size_branch[-1].bias, -1.0)

    def _gather_at_index(
        self,
        feature_map: torch.Tensor,
        idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather features at specified flat indices (OSTrack style).

        This is differentiable and allows gradients to flow through the gathered values.

        Args:
            feature_map: Feature map [B, C, H, W]
            idx: Flat indices [B] (integers from argmax)

        Returns:
            Features at locations [B, C]
        """
        B, C, H, W = feature_map.shape

        # Flatten spatial dimensions: [B, C, H*W]
        feature_flat = feature_map.view(B, C, -1)

        # Expand idx for gathering: [B, C, 1]
        idx_expanded = idx.view(B, 1, 1).expand(B, C, 1).long()

        # Gather values at index: [B, C, 1] -> [B, C]
        gathered = feature_flat.gather(dim=2, index=idx_expanded).squeeze(-1)

        return gathered

    def forward(
        self,
        features: torch.Tensor,
        target_boxes: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        validity_mask: Optional[torch.Tensor] = None,
        has_object: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CENTER head.

        CRITICAL FIX (Audit 2025120902 Issue 16): Added attention_mask parameter
        for padding-aware confidence. Large padding regions can produce spuriously
        high confidence; this mask allows discounting confidence proportionally.

        Args:
            features: Input features [B, C, H, W] or [B, N, C]
            target_boxes: Target boxes in cxcywh format [B, 4], optional
            target_labels: Not used, for API compatibility
            validity_mask: Validity mask [B], optional
            has_object: Has-object flags [B], optional
            attention_mask: Spatial attention mask [B, H, W] where 1=valid, 0=padded
                           Used to adjust confidence based on padding ratio

        Returns:
            Dictionary with predictions and losses
        """
        if features.dim() == 3:
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            if H * W != N:
                raise ValueError(f"Cannot reshape {N} tokens to square grid")
            features = features.permute(0, 2, 1).reshape(B, C, H, W)

        B, C, H, W = features.shape
        feat_sz = H

        score_map_logits = self.score_branch(features)
        size_map = torch.sigmoid(self.size_branch(features))

        if self.offset_branch is not None:
            offset_map = self.offset_branch(features)
        else:
            offset_map = None

        # CRITICAL FIX (Audit 2025120902 Issue 16 - REVISED): Apply mask to score_map_logits
        # Padded regions should not contribute to score/argmax computation
        # Set logits to -inf for padded positions so they get 0 probability after softmax
        if attention_mask is not None:
            # Convert spatial mask to feature map resolution if needed
            if attention_mask.dim() == 3:  # [B, H_img, W_img]
                # Pool to feature map resolution
                mask_resized = F.adaptive_avg_pool2d(
                    attention_mask.unsqueeze(1).float(),
                    (H, W)
                )  # [B, 1, H, W]
                valid_mask_2d = (mask_resized > 0.5).float()  # Binary threshold
            elif attention_mask.dim() == 2:  # [B, N] token mask
                # Reshape to 2D
                valid_mask_2d = attention_mask.float().view(B, 1, H, W)
            else:
                valid_mask_2d = attention_mask.float().view(B, 1, H, W)

            # Mask score logits: set padded positions to -inf
            # This ensures they get 0 probability in softmax
            score_map_logits_masked = score_map_logits.clone()
            score_map_logits_masked = score_map_logits_masked.masked_fill(
                valid_mask_2d < 0.5, float('-inf')
            )
        else:
            score_map_logits_masked = score_map_logits
            valid_mask_2d = None

        score_map = torch.clamp(torch.sigmoid(score_map_logits), min=1e-4, max=1 - 1e-4)

        # CRITICAL FIX (Audit 2025120902 Issue 4): Soft-argmax option
        # Hard argmax on flat score map biases toward (0,0) when map is flat
        # Soft-argmax provides continuous, differentiable center estimation
        score_flat = score_map.view(B, -1)  # [B, H*W]

        if self.use_soft_argmax:
            # Soft-argmax: weighted average of coordinates by softmax weights
            # Temperature controls sharpness: higher = sharper (closer to hard argmax)
            # CRITICAL: Use MASKED logits for softmax so padded positions get 0 weight
            score_logits_flat = score_map_logits_masked.view(B, -1)  # [B, H*W]
            soft_weights = F.softmax(score_logits_flat * self.soft_argmax_temperature, dim=-1)  # [B, H*W]

            # Handle case where all positions are masked (all -inf -> NaN after softmax)
            if torch.isnan(soft_weights).any():
                # Fallback to uniform weights over valid positions
                if valid_mask_2d is not None:
                    valid_flat = valid_mask_2d.view(B, -1)
                    soft_weights = valid_flat / (valid_flat.sum(dim=-1, keepdim=True) + 1e-8)
                else:
                    soft_weights = torch.ones_like(soft_weights) / soft_weights.shape[-1]

            # Create coordinate grids
            device = features.device
            grid_y = torch.arange(H, device=device, dtype=torch.float32)  # [H]
            grid_x = torch.arange(W, device=device, dtype=torch.float32)  # [W]
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')  # [H, W]
            grid_y = grid_y.reshape(-1)  # [H*W]
            grid_x = grid_x.reshape(-1)  # [H*W]

            # Soft-argmax: weighted sum of coordinates
            idx_x = (soft_weights * grid_x.unsqueeze(0)).sum(dim=-1)  # [B]
            idx_y = (soft_weights * grid_y.unsqueeze(0)).sum(dim=-1)  # [B]

            # For size/offset, use soft-weighted average instead of gather
            # This makes the entire pipeline differentiable
            size_map_flat = size_map.view(B, 2, -1)  # [B, 2, H*W]
            pred_w = (soft_weights.unsqueeze(1) * size_map_flat[:, 0:1, :]).sum(dim=-1).squeeze(1)  # [B]
            pred_h = (soft_weights.unsqueeze(1) * size_map_flat[:, 1:2, :]).sum(dim=-1).squeeze(1)  # [B]

            if offset_map is not None:
                offset_map_flat = offset_map.view(B, 2, -1)  # [B, 2, H*W]
                offset_x = (soft_weights.unsqueeze(1) * offset_map_flat[:, 0:1, :]).sum(dim=-1).squeeze(1)  # [B]
                offset_y = (soft_weights.unsqueeze(1) * offset_map_flat[:, 1:2, :]).sum(dim=-1).squeeze(1)  # [B]
            else:
                offset_x = torch.zeros_like(idx_x)
                offset_y = torch.zeros_like(idx_y)

            # For quality score, use max over VALID positions only
            if valid_mask_2d is not None:
                score_flat_masked = score_flat.clone()
                score_flat_masked = score_flat_masked.masked_fill(
                    valid_mask_2d.view(B, -1) < 0.5, 0.0
                )
                max_score = score_flat_masked.max(dim=-1)[0]  # [B]
            else:
                max_score = score_flat.max(dim=-1)[0]  # [B]

            # Also compute flat_idx for compatibility (use hard argmax on masked scores)
            if valid_mask_2d is not None:
                score_for_argmax = score_flat.clone()
                score_for_argmax = score_for_argmax.masked_fill(
                    valid_mask_2d.view(B, -1) < 0.5, -float('inf')
                )
                _, flat_idx = torch.max(score_for_argmax, dim=1)
            else:
                _, flat_idx = torch.max(score_flat, dim=1)
        else:
            # Original hard argmax (OSTrack style) - also apply mask
            if valid_mask_2d is not None:
                score_for_argmax = score_flat.clone()
                score_for_argmax = score_for_argmax.masked_fill(
                    valid_mask_2d.view(B, -1) < 0.5, -float('inf')
                )
                max_score, flat_idx = torch.max(score_for_argmax, dim=1)
                # Fix max_score for masked batches
                max_score = torch.clamp(max_score, min=0.0)
            else:
                max_score, flat_idx = torch.max(score_flat, dim=1)

            # Convert to 2D coordinates
            idx_y = (flat_idx // W).float()
            idx_x = (flat_idx % W).float()

            # Gather size at peak location: [B, 2]
            size_at_loc = self._gather_at_index(size_map, flat_idx)
            pred_w = size_at_loc[:, 0]
            pred_h = size_at_loc[:, 1]

            # Gather offset at peak location (if available)
            if offset_map is not None:
                offset_at_loc = self._gather_at_index(offset_map, flat_idx)
                offset_x = offset_at_loc[:, 0]
                offset_y = offset_at_loc[:, 1]
            else:
                offset_x = torch.zeros_like(idx_x)
                offset_y = torch.zeros_like(idx_y)

        # Final center = (integer_idx + learned_offset) / feat_sz
        # Gradient flows through offset addition (OSTrack style) or soft-argmax
        pred_cx = (idx_x + offset_x) / feat_sz
        pred_cy = (idx_y + offset_y) / feat_sz

        pred_cx = torch.clamp(pred_cx, 0.0, 1.0)
        pred_cy = torch.clamp(pred_cy, 0.0, 1.0)
        pred_w = torch.clamp(pred_w, 1e-4, 1.0)
        pred_h = torch.clamp(pred_h, 1e-4, 1.0)

        pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)

        # Quality score: max of score map (localization quality, not objectness)
        pred_quality = score_map.view(B, -1).max(dim=-1)[0]

        # CRITICAL FIX (Audit 2025120301 Finding 3, 9):
        # Independent objectness prediction - decoupled from quality
        # Objectness = binary target presence, supervised by has_object
        # Quality = localization confidence, supervised by IoU with GT
        objectness_logits = self.objectness_branch(features)  # [B, 1]
        objectness_logits = objectness_logits.squeeze(-1)  # [B]
        pred_objectness = torch.sigmoid(objectness_logits)  # [B], range [0, 1]

        # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
        # Validity prediction - combines padding info + objectness for fusion routing
        # This replaces pre-forward padding-only routing with learned, single-forward validity
        validity_logits = self.validity_branch(features)  # [B, 1]
        validity_logits = validity_logits.squeeze(-1)  # [B]
        pred_validity = torch.sigmoid(validity_logits)  # [B], range [0, 1]

        # Compute valid_ratio for fallback rule and loss weighting
        valid_ratio = torch.ones(B, device=features.device)
        if attention_mask is not None:
            # attention_mask: [B, H, W] where 1=valid, 0=padded
            # Compute valid ratio per sample
            if attention_mask.dim() == 3:
                valid_ratio = attention_mask.float().mean(dim=(1, 2))  # [B]
            elif attention_mask.dim() == 2:
                valid_ratio = attention_mask.float().mean(dim=1)  # [B]
            else:
                valid_ratio = attention_mask.float()

        # CRITICAL FIX (Audit 2025120902): Fallback rule for extreme padding
        # When padding_ratio >= 0.7 (valid_ratio <= 0.3), force invalid regardless of head output
        # This prevents overconfident predictions on mostly-padded crops
        fallback_invalid = valid_ratio <= 0.3  # [B], True when mostly padded
        pred_validity_final = torch.where(
            fallback_invalid,
            torch.zeros_like(pred_validity),
            pred_validity
        )

        # CRITICAL FIX (Audit 2025120902 Issue 16): Padding-aware confidence adjustment
        # When a crop has high padding ratio, the objectness prediction is unreliable
        # because most features come from padding (zeros or replicated values).
        # Discount confidence proportionally to the padding ratio.
        padding_discount = valid_ratio.clamp(min=0.1, max=1.0)  # Avoid total zeroing

        # CRITICAL FIX: Confidence is now objectness (target presence) * padding discount
        # Quality remains as localization quality (score map max)
        # This decoupling allows search expansion to rely on true objectness
        pred_confidence = pred_objectness * padding_discount

        outputs = {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_quality,
            'score_map': score_map,
            'size_map': size_map,
            'boxes': pred_boxes,
            'quality': pred_quality,           # Localization quality (score map max)
            'objectness': pred_objectness,     # Target presence (independent branch)
            'objectness_logits': objectness_logits,  # Raw logits for loss computation
            'pred_confidence': pred_confidence,  # Confidence = objectness (decoupled)
            'confidence': pred_confidence,       # Alias for compatibility
            # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
            # Validity outputs for fusion routing (replaces pre-forward padding-only routing)
            'validity_logits': validity_logits,     # Raw logits for loss computation
            'validity': pred_validity,              # Learned validity (before fallback)
            'validity_final': pred_validity_final,  # Final validity (with fallback rule)
            'valid_ratio': valid_ratio,             # Padding ratio for diagnostics
        }

        if offset_map is not None:
            outputs['offset_map'] = offset_map

        if target_boxes is not None:
            losses = self._compute_losses(
                score_map_logits, size_map, offset_map,
                objectness_logits, validity_logits, valid_ratio,
                pred_boxes, target_boxes,
                validity_mask, has_object, feat_sz,
                attention_mask=valid_mask_2d  # CRITICAL FIX: Pass mask to loss computation
            )
            outputs.update(losses)
            outputs['total_loss'] = losses.get('total_loss', torch.tensor(0.0, device=features.device))

        return outputs

    def _compute_losses(
        self,
        score_map_logits: torch.Tensor,
        size_map: torch.Tensor,
        offset_map: Optional[torch.Tensor],
        objectness_logits: torch.Tensor,
        validity_logits: torch.Tensor,
        valid_ratio: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        validity_mask: Optional[torch.Tensor],
        has_object: Optional[torch.Tensor],
        feat_sz: int,
        attention_mask: Optional[torch.Tensor] = None  # CRITICAL FIX: Add attention_mask
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CENTER head losses with proper has_object and attention_mask handling.

        CRITICAL FIX (Audit 2025120301 Finding 3, 9):
        - score_map target is all-zeros (background) when has_object=0
        - size_loss and iou_loss are masked by has_object
        - Independent objectness loss with has_object as supervision
        - This enables proper training on occlusion/no-object frames

        CRITICAL FIX (Audit 2025120902): Added attention_mask support
        - Padded regions should have target score = 0 (background)
        - Loss at padded positions is down-weighted

        CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
        - validity_logits: Learned validity prediction for fusion routing
        - valid_ratio: padding ratio for supervision label generation
        - Supervision: label=1 when (padding_ratio<0.3 AND has_object=1)
                       label=0 when (padding_ratio>0.5 OR has_object=0)
        """
        B = target_boxes.shape[0]
        device = target_boxes.device

        target_cx = target_boxes[:, 0]
        target_cy = target_boxes[:, 1]
        target_w = target_boxes[:, 2]
        target_h = target_boxes[:, 3]

        target_idx_x = target_cx * feat_sz
        target_idx_y = target_cy * feat_sz

        y_coords = torch.arange(feat_sz, device=device, dtype=torch.float32)
        x_coords = torch.arange(feat_sz, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        sigma = torch.clamp(target_w * feat_sz / 4, min=1.0)

        # Generate Gaussian target heatmap
        target_score_map = torch.exp(
            -((xx.unsqueeze(0) - target_idx_x.view(B, 1, 1)) ** 2 +
              (yy.unsqueeze(0) - target_idx_y.view(B, 1, 1)) ** 2) /
            (2 * sigma.view(B, 1, 1) ** 2)
        )
        target_score_map = target_score_map.unsqueeze(1)  # [B, 1, H, W]

        # CRITICAL FIX (Audit 2025120301 Finding 3):
        # When has_object=0, the target heatmap should be all background (zeros)
        # This teaches the model to predict low scores when target is absent
        if has_object is not None:
            has_object_mask = has_object.float().view(B, 1, 1, 1)  # [B, 1, 1, 1]
            target_score_map = target_score_map * has_object_mask  # Zero out when no object

        # CRITICAL FIX (Audit 2025120902): Zero target at padded positions
        # Model should learn to predict low scores for padding regions
        if attention_mask is not None:
            # attention_mask: [B, 1, H, W], 1=valid, 0=padded
            target_score_map = target_score_map * attention_mask

        score_map_prob = torch.sigmoid(score_map_logits)

        pos_weight = target_score_map
        neg_weight = 1 - target_score_map

        pos_loss = -pos_weight * torch.log(score_map_prob.clamp(min=1e-6))
        neg_loss = -neg_weight * ((1 - score_map_prob) ** 2) * torch.log((1 - score_map_prob).clamp(min=1e-6))

        # CRITICAL FIX (Audit 2025120902): Weight loss by attention_mask
        # Padded regions contribute less to the loss
        if attention_mask is not None:
            # Compute loss with mask weighting
            loss_weight = attention_mask + 0.1 * (1 - attention_mask)  # Small weight for padding
            weighted_loss = (pos_loss + neg_loss) * loss_weight
            score_loss = weighted_loss.sum() / (loss_weight.sum() + 1e-8)
        else:
            score_loss = (pos_loss + neg_loss).mean()

        target_size = torch.stack([target_w, target_h], dim=-1)
        pred_size = pred_boxes[:, 2:4]

        size_loss = F.l1_loss(pred_size, target_size, reduction='none').mean(dim=-1)

        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = compute_giou(pred_xyxy, target_xyxy)
        iou_loss = 1.0 - giou

        # CRITICAL FIX (Audit 2025120301 Finding 3):
        # Combine validity_mask with has_object for size/iou losses
        # Only samples with valid annotations AND object present should contribute
        effective_mask = None
        if validity_mask is not None and has_object is not None:
            effective_mask = validity_mask.float() * has_object.float()
        elif validity_mask is not None:
            effective_mask = validity_mask.float()
        elif has_object is not None:
            effective_mask = has_object.float()

        if effective_mask is not None:
            num_valid = effective_mask.sum().clamp(min=1.0)
            size_loss = (size_loss * effective_mask).sum() / num_valid
            iou_loss = (iou_loss * effective_mask).sum() / num_valid
        else:
            size_loss = size_loss.mean()
            iou_loss = iou_loss.mean()

        # CRITICAL FIX (Audit 2025120301 Finding 3, 9):
        # Independent objectness loss - BCE with has_object as target
        # This decouples confidence (objectness) from quality (localization)
        #
        # CRITICAL FIX (Audit 2025120909): Mask objectness loss by attention_mask
        # Samples with high padding ratio have unreliable objectness predictions
        # because most features come from padding. Weight loss by valid ratio.
        if has_object is not None:
            target_objectness = has_object.float()  # [B]

            # Compute per-sample loss (no reduction yet)
            per_sample_objectness_loss = F.binary_cross_entropy_with_logits(
                objectness_logits, target_objectness, reduction='none'
            )  # [B]

            # Apply padding-based weighting
            if attention_mask is not None:
                # attention_mask: [B, 1, H, W] or [B, H, W]
                # Compute valid ratio per sample (fraction of non-padded pixels)
                if attention_mask.dim() == 4:
                    valid_ratio = attention_mask.float().mean(dim=[1, 2, 3])  # [B]
                else:
                    valid_ratio = attention_mask.float().mean(dim=[1, 2])  # [B]

                # Weight loss: heavily-padded samples (low valid_ratio) contribute less
                # Use valid_ratio as weight, but ensure minimum weight of 0.1 to avoid zeroing
                sample_weights = valid_ratio.clamp(min=0.1, max=1.0)  # [B]

                # Weighted mean loss
                objectness_loss = (per_sample_objectness_loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
            else:
                # No attention mask: simple mean
                objectness_loss = per_sample_objectness_loss.mean()
        else:
            # No has_object supervision: skip objectness loss
            objectness_loss = torch.tensor(0.0, device=device)

        # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
        # Validity loss - learns to predict valid modality for fusion routing
        # Supervision labels:
        #   label=1 when (padding_ratio<0.3 AND has_object=1) -> "valid, reliable"
        #   label=0 when (padding_ratio>0.5 OR has_object=0)  -> "invalid, unreliable"
        #   Intermediate padding (0.3-0.5) is ignored (no supervision) for margin
        #
        # valid_ratio = 1 - padding_ratio, so:
        #   valid_ratio > 0.7 means padding_ratio < 0.3 -> valid if has_object
        #   valid_ratio < 0.5 means padding_ratio > 0.5 -> invalid
        if has_object is not None:
            has_obj = has_object.float()  # [B]

            # Compute validity target labels
            # valid_ratio > 0.7 AND has_object=1 -> label=1
            # valid_ratio < 0.5 OR has_object=0 -> label=0
            # Otherwise (margin region) -> ignore
            target_validity = torch.zeros_like(validity_logits)
            validity_weight = torch.zeros_like(validity_logits)

            # Positive labels: low padding AND object present
            positive_mask = (valid_ratio > 0.7) & (has_obj > 0.5)
            target_validity[positive_mask] = 1.0
            validity_weight[positive_mask] = 1.0

            # Negative labels: high padding OR no object
            negative_mask = (valid_ratio < 0.5) | (has_obj < 0.5)
            target_validity[negative_mask] = 0.0
            validity_weight[negative_mask] = 1.0

            # Margin region (0.5 <= valid_ratio <= 0.7 with has_object=1): no supervision
            # validity_weight remains 0 for these samples

            # Focal BCE loss for validity (handles class imbalance)
            # Focal weight: (1-p)^gamma for positive, p^gamma for negative
            gamma = 2.0
            p = torch.sigmoid(validity_logits)
            focal_weight = torch.where(
                target_validity > 0.5,
                (1 - p) ** gamma,  # Positive: focus on hard positives
                p ** gamma         # Negative: focus on hard negatives
            )

            per_sample_validity_loss = F.binary_cross_entropy_with_logits(
                validity_logits, target_validity, reduction='none'
            )  # [B]
            weighted_validity_loss = per_sample_validity_loss * focal_weight * validity_weight

            # Normalize by number of supervised samples
            num_supervised = validity_weight.sum().clamp(min=1.0)
            validity_loss = weighted_validity_loss.sum() / num_supervised
        else:
            # No has_object supervision: skip validity loss
            validity_loss = torch.tensor(0.0, device=device)

        total_loss = (
            score_loss * self.loss_weight_score +
            size_loss * self.loss_weight_size +
            iou_loss * 2.0 +
            objectness_loss * self.loss_weight_objectness +
            validity_loss * self.loss_weight_validity
        )

        return {
            'score_loss': score_loss,
            'size_loss': size_loss,
            'iou_loss': iou_loss,
            'objectness_loss': objectness_loss,
            'validity_loss': validity_loss,  # CRITICAL FIX: Add validity loss
            'total_loss': total_loss,
            'cls_loss': score_loss,
            'l1_loss': size_loss,
        }


@PREDICTION_HEAD_REGISTRY.register("RGBCenterHead")
class RGBCenterHead(CenterHead):
    """RGB-specific CENTER prediction head."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("Initialized RGB CenterHead")


@PREDICTION_HEAD_REGISTRY.register("SonarCenterHead")
class SonarCenterHead(CenterHead):
    """Sonar-specific CENTER prediction head."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("Initialized Sonar CenterHead")


def build_box_head(
    head_type: str = 'rgb',
    in_channels: int = 768,
    **kwargs
) -> CenterHead:
    """
    Factory function to build box prediction head.

    Args:
        head_type: Modality type ('rgb', 'sonar', or 'base')
        in_channels: Input feature dimension
        **kwargs: Additional arguments for CenterHead

    Returns:
        CenterHead instance
    """
    if head_type == 'rgb':
        return RGBCenterHead(in_channels=in_channels, **kwargs)
    elif head_type == 'sonar':
        return SonarCenterHead(in_channels=in_channels, **kwargs)
    else:
        return CenterHead(in_channels=in_channels, **kwargs)
