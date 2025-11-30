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
        has_object: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CENTER head.

        Args:
            features: Input features [B, C, H, W] or [B, N, C]
            target_boxes: Target boxes in cxcywh format [B, 4], optional
            target_labels: Not used, for API compatibility
            validity_mask: Validity mask [B], optional
            has_object: Has-object flags [B], optional

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

        score_map = torch.clamp(torch.sigmoid(score_map_logits), min=1e-4, max=1 - 1e-4)

        # Hard argmax for center location (OSTrack style)
        # Get flat index for gather operation
        score_flat = score_map.view(B, -1)
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
        # Gradient flows through offset addition (OSTrack style)
        pred_cx = (idx_x + offset_x) / feat_sz
        pred_cy = (idx_y + offset_y) / feat_sz

        pred_cx = torch.clamp(pred_cx, 0.0, 1.0)
        pred_cy = torch.clamp(pred_cy, 0.0, 1.0)
        pred_w = torch.clamp(pred_w, 1e-4, 1.0)
        pred_h = torch.clamp(pred_h, 1e-4, 1.0)

        pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
        pred_quality = score_map.view(B, -1).max(dim=-1)[0]

        outputs = {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_quality,
            'score_map': score_map,
            'size_map': size_map,
            'boxes': pred_boxes,
            'quality': pred_quality,
            'pred_confidence': pred_quality,
            'confidence': pred_quality,
        }

        if offset_map is not None:
            outputs['offset_map'] = offset_map

        if target_boxes is not None:
            losses = self._compute_losses(
                score_map_logits, size_map, offset_map,
                pred_boxes, target_boxes,
                validity_mask, feat_sz
            )
            outputs.update(losses)
            outputs['total_loss'] = losses.get('total_loss', torch.tensor(0.0, device=features.device))

        return outputs

    def _compute_losses(
        self,
        score_map_logits: torch.Tensor,
        size_map: torch.Tensor,
        offset_map: Optional[torch.Tensor],
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        validity_mask: Optional[torch.Tensor],
        feat_sz: int
    ) -> Dict[str, torch.Tensor]:
        """Compute CENTER head losses."""
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

        target_score_map = torch.exp(
            -((xx.unsqueeze(0) - target_idx_x.view(B, 1, 1)) ** 2 +
              (yy.unsqueeze(0) - target_idx_y.view(B, 1, 1)) ** 2) /
            (2 * sigma.view(B, 1, 1) ** 2)
        )
        target_score_map = target_score_map.unsqueeze(1)

        score_map_prob = torch.sigmoid(score_map_logits)

        pos_weight = target_score_map
        neg_weight = 1 - target_score_map

        pos_loss = -pos_weight * torch.log(score_map_prob.clamp(min=1e-6))
        neg_loss = -neg_weight * ((1 - score_map_prob) ** 2) * torch.log((1 - score_map_prob).clamp(min=1e-6))

        score_loss = (pos_loss + neg_loss).mean()

        target_size = torch.stack([target_w, target_h], dim=-1)
        pred_size = pred_boxes[:, 2:4]

        size_loss = F.l1_loss(pred_size, target_size, reduction='none').mean(dim=-1)

        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = compute_giou(pred_xyxy, target_xyxy)
        iou_loss = 1.0 - giou

        if validity_mask is not None:
            validity_mask = validity_mask.float()
            num_valid = validity_mask.sum().clamp(min=1.0)
            size_loss = (size_loss * validity_mask).sum() / num_valid
            iou_loss = (iou_loss * validity_mask).sum() / num_valid
        else:
            size_loss = size_loss.mean()
            iou_loss = iou_loss.mean()

        total_loss = (
            score_loss * self.loss_weight_score +
            size_loss * self.loss_weight_size +
            iou_loss * 2.0
        )

        return {
            'score_loss': score_loss,
            'size_loss': size_loss,
            'iou_loss': iou_loss,
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
