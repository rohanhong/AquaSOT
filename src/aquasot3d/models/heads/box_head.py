"""
2D Bounding Box Prediction Heads

This module implements 2D box prediction heads for RGB, Sonar, and Fusion branches.
Each head outputs classification scores (quality) and bounding box coordinates.

Architecture:
- Classification: Binary classification with quality estimation (BCE/Focal Loss)
- Regression: Box coordinates (L1 loss) + IoU-based losses (GIoU/CIoU)
- Separate heads for RGB, Sonar, and Fusion to leverage modality-specific features

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal, Tuple
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

    Example:
        >>> boxes_xywh = torch.tensor([[100, 200, 50, 80]])  # x=100, y=200, w=50, h=80
        >>> boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
        >>> # Result: [[125, 240, 50, 80]]  # cx=125, cy=240, w=50, h=80
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
    # Calculate intersection
    x1_max = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1_max = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2_min = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2_min = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2_min - x1_max).clamp(min=0)
    inter_h = (y2_min - y1_max).clamp(min=0)
    inter_area = inter_w * inter_h

    # Calculate union
    # Clamp box dimensions to prevent negative areas
    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6)

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    # Compute IoU with epsilon for numerical stability
    iou = inter_area / (union_area + 1e-7)

    # Clamp to valid IoU range [0, 1]
    # This is critical for preventing BCE assertion failures when IoU is used as quality target
    # IoU can theoretically exceed 1.0 due to floating point precision issues
    iou = torch.clamp(iou, min=0.0, max=1.0)

    # Replace any NaN/inf values with 0 (no overlap)
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
    # IoU (now returns values clamped to [0, 1])
    iou = compute_iou(boxes1, boxes2)

    # Enclosing box
    x1_min = torch.min(boxes1[:, 0], boxes2[:, 0])
    y1_min = torch.min(boxes1[:, 1], boxes2[:, 1])
    x2_max = torch.max(boxes1[:, 2], boxes2[:, 2])
    y2_max = torch.max(boxes1[:, 3], boxes2[:, 3])

    # Clamp enclosing dimensions to prevent negative areas
    enclosing_w = (x2_max - x1_min).clamp(min=1e-6)
    enclosing_h = (y2_max - y1_min).clamp(min=1e-6)
    enclosing_area = enclosing_w * enclosing_h

    # Union area (use clamped dimensions)
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

    # GIoU computation
    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)

    # Clamp to theoretical range [-1, 1]
    giou = torch.clamp(giou, min=-1.0, max=1.0)

    # Replace NaN/inf with -1 (worst case)
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
    # IoU (now returns values clamped to [0, 1])
    iou = compute_iou(boxes1, boxes2)

    # Convert to cxcywh for center distance and aspect ratio
    boxes1_cxcywh = box_xyxy_to_cxcywh(boxes1)
    boxes2_cxcywh = box_xyxy_to_cxcywh(boxes2)

    # Center distance
    cx1, cy1, w1, h1 = boxes1_cxcywh.unbind(-1)
    cx2, cy2, w2, h2 = boxes2_cxcywh.unbind(-1)
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Diagonal of enclosing box
    x1_min = torch.min(boxes1[:, 0], boxes2[:, 0])
    y1_min = torch.min(boxes1[:, 1], boxes2[:, 1])
    x2_max = torch.max(boxes1[:, 2], boxes2[:, 2])
    y2_max = torch.max(boxes1[:, 3], boxes2[:, 3])

    # Clamp diagonal dimensions to prevent division by zero
    diagonal_w = (x2_max - x1_min).clamp(min=1e-6)
    diagonal_h = (y2_max - y1_min).clamp(min=1e-6)
    diagonal_sq = diagonal_w ** 2 + diagonal_h ** 2

    # Aspect ratio consistency (clamp w and h to prevent division by zero)
    w1_safe = w1.clamp(min=1e-6)
    h1_safe = h1.clamp(min=1e-6)
    w2_safe = w2.clamp(min=1e-6)
    h2_safe = h2.clamp(min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2_safe / h2_safe) - torch.atan(w1_safe / h1_safe), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    # CIoU computation
    ciou = iou - (center_dist_sq / (diagonal_sq + 1e-7) + alpha * v)

    # Clamp to theoretical range [-1, 1]
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    # Replace NaN/inf with -1 (worst case)
    ciou = torch.where(torch.isfinite(ciou), ciou, torch.full_like(ciou, -1.0))

    return ciou


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss.

    Loss = 1 - GIoU
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute GIoU loss.

        Args:
            pred_boxes: Predicted boxes in cxcywh format, shape (N, 4)
            target_boxes: Target boxes in cxcywh format, shape (N, 4)

        Returns:
            GIoU loss
        """
        # Convert to xyxy
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)

        giou = compute_giou(pred_xyxy, target_xyxy)

        # Clamp GIoU to prevent numerical instability
        # Theoretical range is [-1, 1], but we add margin for safety
        giou = torch.clamp(giou, min=-1.0, max=1.0)

        loss = 1 - giou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CIoULoss(nn.Module):
    """
    Complete IoU Loss.

    Loss = 1 - CIoU
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute CIoU loss.

        Args:
            pred_boxes: Predicted boxes in cxcywh format, shape (N, 4)
            target_boxes: Target boxes in cxcywh format, shape (N, 4)

        Returns:
            CIoU loss
        """
        # Convert to xyxy
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)

        ciou = compute_ciou(pred_xyxy, target_xyxy)

        # Clamp CIoU to prevent numerical instability
        # Theoretical range is [-1, 1], but we add margin for safety
        ciou = torch.clamp(ciou, min=-1.0, max=1.0)

        loss = 1 - ciou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss for joint classification and quality estimation.

    QFL paper: https://arxiv.org/abs/2006.04388

    Combines classification with IoU-based quality score in a single loss.
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
        """
        Compute Quality Focal Loss.

        Args:
            pred_scores: Predicted quality scores, shape (N,)
            target_labels: Target labels (0 or 1), shape (N,)
            target_quality: Target quality (IoU), shape (N,)

        Returns:
            QFL loss
        """
        # Scale factor
        scale_factor = pred_scores - target_labels * target_quality

        # Focal weight
        focal_weight = torch.abs(scale_factor) ** self.beta

        # BCE loss (disable autocast for AMP compatibility)
        # Note: pred_scores are already sigmoid outputs, so we use binary_cross_entropy
        with torch.cuda.amp.autocast(enabled=False):
            pred_scores_fp32 = pred_scores.float()
            target_fp32 = (target_labels * target_quality).float()

            # Clamp to [0, 1] to prevent numerical instability
            # Sigmoid can occasionally produce values slightly outside [0, 1] due to FP precision
            # IoU-based quality scores should also be clamped for safety
            pred_scores_fp32 = torch.clamp(pred_scores_fp32, min=0.0, max=1.0)
            target_fp32 = torch.clamp(target_fp32, min=0.0, max=1.0)

            bce_loss = F.binary_cross_entropy(
                pred_scores_fp32,
                target_fp32,
                reduction='none'
            )

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


@PREDICTION_HEAD_REGISTRY.register("BoxHead")
class BoxHead(BaseHead):
    """
    2D Bounding Box Prediction Head.

    Predicts classification scores (quality) and bounding box coordinates
    from input features. Supports multiple loss functions and output formats.

    Args:
        in_channels: Input feature dimension
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of MLP layers (default: 3)
        box_format: Output box format, 'cxcywh' or 'xyxy' (default: 'cxcywh')
        use_quality_focal: Use Quality Focal Loss instead of BCE (default: True)
        iou_loss_type: Type of IoU loss, 'giou' or 'ciou' (default: 'giou')
        loss_weight_cls: Classification loss weight (default: 1.0)
        loss_weight_l1: L1 loss weight (default: 5.0)
        loss_weight_iou: IoU loss weight (default: 2.0)

    Returns:
        Dictionary containing:
            - 'pred_boxes': Predicted boxes in specified format, shape (B, 4)
            - 'pred_scores': Predicted quality scores, shape (B,)
            - 'cls_loss': Classification loss (if targets provided)
            - 'l1_loss': L1 regression loss (if targets provided)
            - 'iou_loss': IoU-based loss (if targets provided)

    Example:
        >>> head = BoxHead(in_channels=768, hidden_dim=256)
        >>> features = torch.randn(4, 768)
        >>> outputs = head(features)
        >>> print(outputs['pred_boxes'].shape)  # (4, 4)
        >>> print(outputs['pred_scores'].shape)  # (4,)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        box_format: Literal['cxcywh', 'xyxy'] = 'cxcywh',
        use_quality_focal: bool = True,
        iou_loss_type: Literal['giou', 'ciou'] = 'giou',
        loss_weight_cls: float = 1.0,
        loss_weight_l1: float = 1.0,  # Reduced from 5.0 for numerical stability
        loss_weight_iou: float = 2.0,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.box_format = box_format
        self.use_quality_focal = use_quality_focal
        self.iou_loss_type = iou_loss_type
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_l1 = loss_weight_l1
        self.loss_weight_iou = loss_weight_iou

        # Build MLP layers for shared feature processing
        layers = []
        current_dim = in_channels
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            current_dim = hidden_dim
        self.shared_mlp = nn.Sequential(*layers)

        # Classification head (quality estimation)
        self.cls_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Regression head (box coordinates with constrained output)
        # Critical fix: Add Sigmoid activation to constrain output to [0, 1]
        # This ensures predictions are in normalized coordinate space matching targets
        self.reg_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # Constrain to [0, 1] for normalized coordinates
        )

        # Loss functions
        if use_quality_focal:
            self.cls_loss_fn = QualityFocalLoss(beta=2.0)
        else:
            self.cls_loss_fn = nn.BCELoss()

        self.l1_loss_fn = nn.L1Loss(reduction='mean')

        if iou_loss_type == 'giou':
            self.iou_loss_fn = GIoULoss(reduction='mean')
        elif iou_loss_type == 'ciou':
            self.iou_loss_fn = CIoULoss(reduction='mean')
        else:
            raise ValueError(f"Unknown IoU loss type: {iou_loss_type}")

        # Initialize regression head with small weights for stable initial predictions
        # This ensures initial box predictions are centered around [0.5, 0.5, 0.3, 0.3]
        self._init_weights()

        self.logger.info(
            f"Initialized BoxHead: in_channels={in_channels}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"box_format={box_format}, use_quality_focal={use_quality_focal}, "
            f"iou_loss={iou_loss_type}"
        )

    def _init_weights(self):
        """
        Initialize weights for stable training.

        Uses Xavier/Kaiming initialization for linear layers and small bias
        initialization for regression output to center initial predictions.
        """
        # Initialize all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Special initialization for regression head output layer
        # Set bias to produce initial predictions around [0.5, 0.5, 0.3, 0.3]
        # After sigmoid: logit(0.5)=0, logit(0.3)≈-0.85
        reg_output_layer = self.reg_head[-2]  # Last linear before sigmoid
        if isinstance(reg_output_layer, nn.Linear):
            nn.init.zeros_(reg_output_layer.weight)
            if reg_output_layer.bias is not None:
                # Bias values to initialize around center with moderate size
                reg_output_layer.bias.data[0] = 0.0  # cx ≈ 0.5 after sigmoid
                reg_output_layer.bias.data[1] = 0.0  # cy ≈ 0.5 after sigmoid
                reg_output_layer.bias.data[2] = -0.85  # w ≈ 0.3 after sigmoid
                reg_output_layer.bias.data[3] = -0.85  # h ≈ 0.3 after sigmoid

    def forward(
        self,
        features: torch.Tensor,
        target_boxes: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for box prediction.

        Args:
            features: Input features, shape (B, C) or (B, C, H, W)
            target_boxes: Target boxes in cxcywh format, shape (B, 4), optional
            target_labels: Target labels (0 or 1), shape (B,), optional

        Returns:
            Dictionary with predictions and losses
        """
        # Handle spatial features
        if features.dim() == 4:
            # Global average pooling
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        elif features.dim() == 3:
            # Average over sequence dimension
            features = features.mean(dim=1)

        batch_size = features.shape[0]

        # Check if input dimension matches expected dimension
        actual_dim = features.shape[1]
        if actual_dim != self.in_channels:
            # Create dynamic projection layer to adapt input dimension
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(actual_dim, self.in_channels).to(features.device)
            features = self.input_proj(features)

        # Shared feature processing
        shared_feat = self.shared_mlp(features)  # (B, hidden_dim)

        # Classification (quality score)
        pred_scores = self.cls_head(shared_feat).squeeze(-1)  # (B,)

        # CRITICAL: Clamp pred_scores to [0, 1] for numerical stability
        # Sigmoid in FP16/AMP mode can occasionally produce values slightly outside [0, 1]
        # This prevents CUDA assertion failures in BCE loss computation
        pred_scores = torch.clamp(pred_scores, min=0.0, max=1.0)

        # Regression (box coordinates in cxcywh)
        pred_boxes_cxcywh = self.reg_head(shared_feat)  # (B, 4)

        # Convert to requested format
        if self.box_format == 'xyxy':
            pred_boxes = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
        else:
            pred_boxes = pred_boxes_cxcywh

        outputs = {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
        }

        # Compute losses if targets are provided
        if target_boxes is not None:
            # Convert target boxes from xywh to cxcywh format for loss computation
            if target_boxes.shape[-1] != 4:
                raise ValueError(f"Invalid target_boxes shape: {target_boxes.shape}")

            # CRITICAL: Dataset provides GT in xywh format (top-left + size)
            # But loss functions expect cxcywh format (center + size)
            # Must convert here to ensure correct loss computation
            target_boxes_cxcywh = box_xywh_to_cxcywh(target_boxes)

            # L1 loss
            l1_loss = self.l1_loss_fn(pred_boxes_cxcywh, target_boxes_cxcywh)

            # IoU loss
            iou_loss = self.iou_loss_fn(pred_boxes_cxcywh, target_boxes_cxcywh)

            # Classification loss
            if target_labels is not None:
                if self.use_quality_focal:
                    # Compute IoU as quality target
                    with torch.no_grad():
                        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
                        target_xyxy = box_cxcywh_to_xyxy(target_boxes_cxcywh)  # Use converted cxcywh
                        target_quality = compute_iou(pred_xyxy, target_xyxy)
                    cls_loss = self.cls_loss_fn(pred_scores, target_labels, target_quality)
                else:
                    cls_loss = self.cls_loss_fn(pred_scores, target_labels.float())
            else:
                # Default to positive labels if not provided
                target_labels = torch.ones_like(pred_scores)
                if self.use_quality_focal:
                    with torch.no_grad():
                        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
                        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
                        target_quality = compute_iou(pred_xyxy, target_xyxy)
                    cls_loss = self.cls_loss_fn(pred_scores, target_labels, target_quality)
                else:
                    cls_loss = self.cls_loss_fn(pred_scores, target_labels)

            outputs.update({
                'cls_loss': cls_loss * self.loss_weight_cls,
                'l1_loss': l1_loss * self.loss_weight_l1,
                'iou_loss': iou_loss * self.loss_weight_iou,
                'total_loss': (
                    cls_loss * self.loss_weight_cls +
                    l1_loss * self.loss_weight_l1 +
                    iou_loss * self.loss_weight_iou
                )
            })

        return outputs

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        target_boxes: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute box prediction loss from predictions and targets.

        Args:
            predictions: Dictionary from forward() containing pred_boxes and pred_scores
            target_boxes: Target boxes in cxcywh format, shape (B, 4)
            target_labels: Optional target labels (default: all positive)

        Returns:
            Total loss (scalar)
        """
        # Extract predictions
        pred_boxes = predictions['pred_boxes']
        pred_scores = predictions['pred_scores']

        # Convert boxes to cxcywh if needed
        if self.box_format == 'xyxy':
            pred_boxes_cxcywh = box_xyxy_to_cxcywh(pred_boxes)
        else:
            pred_boxes_cxcywh = pred_boxes

        # L1 loss
        l1_loss = self.l1_loss_fn(pred_boxes_cxcywh, target_boxes)

        # IoU loss
        iou_loss = self.iou_loss_fn(pred_boxes_cxcywh, target_boxes)

        # Classification loss
        if target_labels is None:
            target_labels = torch.ones_like(pred_scores)

        if self.use_quality_focal:
            # Compute IoU as quality target
            with torch.no_grad():
                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
                target_xyxy = box_cxcywh_to_xyxy(target_boxes)
                target_quality = compute_iou(pred_xyxy, target_xyxy)
            cls_loss = self.cls_loss_fn(pred_scores, target_labels, target_quality)
        else:
            cls_loss = self.cls_loss_fn(pred_scores, target_labels.float())

        # Total weighted loss
        total_loss = (
            cls_loss * self.loss_weight_cls +
            l1_loss * self.loss_weight_l1 +
            iou_loss * self.loss_weight_iou
        )

        return total_loss


@PREDICTION_HEAD_REGISTRY.register("RGBBoxHead")
class RGBBoxHead(BoxHead):
    """
    RGB-specific 2D Box Prediction Head.

    Specialized version of BoxHead optimized for RGB features.
    Uses same architecture but can have RGB-specific configurations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("Initialized RGB-specific BoxHead")


@PREDICTION_HEAD_REGISTRY.register("SonarBoxHead")
class SonarBoxHead(BoxHead):
    """
    Sonar-specific 2D Box Prediction Head.

    Specialized version of BoxHead optimized for Sonar features.
    Uses same architecture but can have sonar-specific configurations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("Initialized Sonar-specific BoxHead")


@PREDICTION_HEAD_REGISTRY.register("FusionBoxHead")
class FusionBoxHead(BoxHead):
    """
    Fusion 2D Box Prediction Head.

    Specialized version of BoxHead for fused multi-modal features.
    This head produces the primary tracking output.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("Initialized Fusion BoxHead (primary output)")


def build_box_head(
    head_type: str = 'fusion',
    in_channels: int = 768,
    **kwargs
) -> BoxHead:
    """
    Factory function to build box prediction head.

    Args:
        head_type: Type of head ('rgb', 'sonar', 'fusion', or 'base')
        in_channels: Input feature dimension
        **kwargs: Additional arguments for BoxHead

    Returns:
        BoxHead instance

    Example:
        >>> head = build_box_head('fusion', in_channels=768, hidden_dim=256)
    """
    if head_type == 'rgb':
        return RGBBoxHead(in_channels=in_channels, **kwargs)
    elif head_type == 'sonar':
        return SonarBoxHead(in_channels=in_channels, **kwargs)
    elif head_type == 'fusion':
        return FusionBoxHead(in_channels=in_channels, **kwargs)
    else:
        return BoxHead(in_channels=in_channels, **kwargs)
