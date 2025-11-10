"""
Multi-Task Loss Manager for AquaSOT-3D

This module implements a comprehensive loss manager for multi-modal tracking,
handling multiple task-specific losses with dynamic weighting strategies.

Loss Components:
- RGB Branch: classification + bbox regression (GIoU/CIoU)
- Sonar Branch: classification + bbox regression (GIoU/CIoU)
- Fusion Branch: classification + bbox regression (GIoU/CIoU)
- Cross-Modal Consistency: component + representation (InfoNCE)
- Weak 3D Center: cyclic consistency loss
- Temporal: smoothness loss (optional)

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal, Callable, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class LossManager(nn.Module):
    """
    Manages multi-task losses for AquaSOT-3D tracker.

    This class orchestrates the computation and weighting of all loss components,
    supporting both static and dynamic weight scheduling.

    Args:
        loss_weights: Dictionary of static loss weights for each component
        weight_schedule: Optional schedule type for dynamic loss weighting
        schedule_config: Configuration for weight scheduling
        reduction: Loss reduction method ('mean' or 'sum')

    Example:
        >>> loss_weights = {
        ...     'rgb_cls': 1.0, 'rgb_box': 1.0,
        ...     'sonar_cls': 1.0, 'sonar_box': 1.0,
        ...     'fusion_cls': 1.5, 'fusion_box': 1.5,
        ...     'component_consistency': 0.3,
        ...     'representation_consistency': 0.15,
        ...     'weak3d_cycle': 0.8,
        ...     'temporal_smooth': 0.1
        ... }
        >>> manager = LossManager(loss_weights)
        >>> total_loss, loss_dict = manager.compute_losses(predictions, targets)
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        weight_schedule: Optional[Literal['linear', 'step', 'cosine']] = None,
        schedule_config: Optional[Dict[str, Any]] = None,
        reduction: Literal['mean', 'sum'] = 'mean',
        **kwargs
    ):
        super().__init__()

        # Default loss weights (as specified in Development_Planning.md)
        self.loss_weights = loss_weights or {
            'rgb_cls': 1.0,
            'rgb_box': 1.0,
            'sonar_cls': 1.0,
            'sonar_box': 1.0,
            'fusion_cls': 1.0,
            'fusion_box': 1.0,
            'component_consistency': 0.2,
            'representation_consistency': 0.1,
            'weak3d_cycle': 0.6,
            'temporal_smooth': 0.1
        }

        self.weight_schedule = weight_schedule
        self.schedule_config = schedule_config or {}
        self.reduction = reduction

        # Track current epoch for scheduling
        self.current_epoch = 0

        # Loss history for logging
        self.loss_history = []

        logger.info(f"LossManager initialized with weights: {self.loss_weights}")
        if weight_schedule:
            logger.info(f"Using dynamic weight schedule: {weight_schedule}")

    def compute_losses(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all losses and combine them with appropriate weights.

        Args:
            predictions: Dictionary containing all model predictions
                - 'rgb_box': RGB bounding box predictions
                - 'sonar_box': Sonar bounding box predictions
                - 'fusion_box': Fusion bounding box predictions
                - 'rgb_cls': RGB classification scores (optional)
                - 'sonar_cls': Sonar classification scores (optional)
                - 'fusion_cls': Fusion classification scores (optional)
                - 'weak3d': Weak 3D center predictions (optional)
                - 'consistency': Consistency losses (optional)
                - 'temporal': Temporal smoothness losses (optional)
            targets: Dictionary containing ground truth
                - 'rgb_box': RGB target boxes
                - 'sonar_box': Sonar target boxes
                - 'rgb_cls': RGB classification targets (optional)
                - 'sonar_cls': Sonar classification targets (optional)
                - 'weak3d': 3D center targets (optional)
            epoch: Current training epoch (for weight scheduling)

        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Dictionary containing individual loss components
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Update weights if using scheduling
        weights = self._get_current_weights()

        loss_dict = {}

        # ============================================
        # 1. RGB Branch Losses
        # ============================================
        # Support both singular and plural forms (avoid tensor boolean ambiguity)
        rgb_box_pred = predictions.get('rgb_box') if 'rgb_box' in predictions else predictions.get('rgb_boxes')
        rgb_box_target = targets.get('rgb_box') if 'rgb_box' in targets else targets.get('rgb_boxes')
        if rgb_box_pred is not None and rgb_box_target is not None:
            rgb_box_loss = self._compute_box_loss(rgb_box_pred, rgb_box_target)
            loss_dict['rgb_box'] = rgb_box_loss

        if 'rgb_cls' in predictions and 'rgb_cls' in targets:
            rgb_cls_loss = self._compute_classification_loss(
                predictions['rgb_cls'],
                targets['rgb_cls']
            )
            loss_dict['rgb_cls'] = rgb_cls_loss

        # ============================================
        # 2. Sonar Branch Losses
        # ============================================
        sonar_box_pred = predictions.get('sonar_box') if 'sonar_box' in predictions else predictions.get('sonar_boxes')
        sonar_box_target = targets.get('sonar_box') if 'sonar_box' in targets else targets.get('sonar_boxes')
        if sonar_box_pred is not None and sonar_box_target is not None:
            sonar_box_loss = self._compute_box_loss(sonar_box_pred, sonar_box_target)
            loss_dict['sonar_box'] = sonar_box_loss

        if 'sonar_cls' in predictions and 'sonar_cls' in targets:
            sonar_cls_loss = self._compute_classification_loss(
                predictions['sonar_cls'],
                targets['sonar_cls']
            )
            loss_dict['sonar_cls'] = sonar_cls_loss

        # ============================================
        # 3. Fusion Branch Losses
        # ============================================
        fusion_box_pred = predictions.get('fusion_box') if 'fusion_box' in predictions else predictions.get('fusion_boxes')
        fusion_box_target = targets.get('fusion_box') if 'fusion_box' in targets else targets.get('fusion_boxes')
        if fusion_box_pred is not None and fusion_box_target is not None:
            fusion_box_loss = self._compute_box_loss(fusion_box_pred, fusion_box_target)
            loss_dict['fusion_box'] = fusion_box_loss

        if 'fusion_cls' in predictions and 'fusion_cls' in targets:
            fusion_cls_loss = self._compute_classification_loss(
                predictions['fusion_cls'],
                targets['fusion_cls']
            )
            loss_dict['fusion_cls'] = fusion_cls_loss

        # ============================================
        # 4. Consistency Losses
        # ============================================
        if 'consistency' in predictions:
            consistency_dict = predictions['consistency']

            if 'component' in consistency_dict:
                loss_dict['component_consistency'] = consistency_dict['component']

            if 'representation' in consistency_dict:
                loss_dict['representation_consistency'] = consistency_dict['representation']

        # ============================================
        # 5. Weak 3D Cycle Loss
        # ============================================
        if 'weak3d' in predictions and 'cyclic_loss' in predictions['weak3d']:
            loss_dict['weak3d_cycle'] = predictions['weak3d']['cyclic_loss']

        # ============================================
        # 6. Temporal Smoothness Loss
        # ============================================
        if 'temporal' in predictions and 'smoothness_loss' in predictions['temporal']:
            loss_dict['temporal_smooth'] = predictions['temporal']['smoothness_loss']

        # ============================================
        # 7. Combine All Losses
        # ============================================
        # Collect weighted losses first to ensure proper gradient tracking
        weighted_losses = {}
        loss_components = []

        for loss_name, loss_value in loss_dict.items():
            if loss_name in weights and loss_value is not None:
                weighted_loss = weights[loss_name] * loss_value
                loss_components.append(weighted_loss)
                weighted_losses[f'{loss_name}_weighted'] = weighted_loss

        # Compute total loss with proper gradient tracking
        if len(loss_components) > 0:
            total_loss = sum(loss_components)
        else:
            # Fallback: create zero tensor with gradient tracking
            device = self._get_device(predictions)
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Second pass: add weighted losses to dict
        loss_dict.update(weighted_losses)
        loss_dict['total'] = total_loss

        # Store for logging
        self.loss_history.append({
            'epoch': self.current_epoch,
            'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            **{k: v.item() if isinstance(v, torch.Tensor) else v
               for k, v in loss_dict.items() if not k.endswith('_weighted')}
        })

        return total_loss, loss_dict

    def _compute_box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        loss_type: Literal['l1', 'giou', 'ciou'] = 'giou'
    ) -> torch.Tensor:
        """
        Compute bounding box regression loss.

        Args:
            pred_boxes: Predicted boxes [B, 4] in cxcywh format
            target_boxes: Target boxes [B, 4] in cxcywh format
            loss_type: Type of loss to use

        Returns:
            Box regression loss
        """
        # L1 loss component
        l1_loss = F.l1_loss(pred_boxes, target_boxes, reduction=self.reduction)

        if loss_type == 'l1':
            return l1_loss

        # GIoU or CIoU loss
        # Convert to xyxy format for IoU computation
        from aquasot3d.models.heads.box_head import box_cxcywh_to_xyxy, compute_giou, compute_ciou

        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

        if loss_type == 'giou':
            giou = compute_giou(pred_boxes_xyxy, target_boxes_xyxy)
            giou_loss = 1.0 - giou.mean() if self.reduction == 'mean' else (1.0 - giou).sum()
            return l1_loss + giou_loss

        elif loss_type == 'ciou':
            ciou = compute_ciou(pred_boxes_xyxy, target_boxes_xyxy)
            ciou_loss = 1.0 - ciou.mean() if self.reduction == 'mean' else (1.0 - ciou).sum()
            return l1_loss + ciou_loss

        return l1_loss

    def _compute_classification_loss(
        self,
        pred_scores: torch.Tensor,
        target_labels: torch.Tensor,
        loss_type: Literal['bce', 'focal'] = 'bce'
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            pred_scores: Predicted classification scores [B, ...]
            target_labels: Target labels [B, ...]
            loss_type: Type of classification loss

        Returns:
            Classification loss
        """
        if loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(
                pred_scores,
                target_labels.float(),
                reduction=self.reduction
            )

        elif loss_type == 'focal':
            # Focal loss implementation
            from aquasot3d.models.heads.box_head import QualityFocalLoss
            focal_loss = QualityFocalLoss(reduction=self.reduction)
            return focal_loss(pred_scores, target_labels)

        return F.binary_cross_entropy_with_logits(
            pred_scores,
            target_labels.float(),
            reduction=self.reduction
        )

    def _get_current_weights(self) -> Dict[str, float]:
        """
        Get current loss weights based on schedule.

        Returns:
            Current loss weights
        """
        if self.weight_schedule is None:
            return self.loss_weights

        # Implement weight scheduling
        if self.weight_schedule == 'linear':
            return self._linear_schedule()
        elif self.weight_schedule == 'step':
            return self._step_schedule()
        elif self.weight_schedule == 'cosine':
            return self._cosine_schedule()

        return self.loss_weights

    def _linear_schedule(self) -> Dict[str, float]:
        """Linear weight scheduling."""
        max_epoch = self.schedule_config.get('max_epoch', 60)
        target_weights = self.schedule_config.get('target_weights', {})

        weights = self.loss_weights.copy()
        progress = min(self.current_epoch / max_epoch, 1.0)

        for key, target_weight in target_weights.items():
            if key in weights:
                initial_weight = self.loss_weights[key]
                weights[key] = initial_weight + progress * (target_weight - initial_weight)

        return weights

    def _step_schedule(self) -> Dict[str, float]:
        """Step-wise weight scheduling."""
        milestones = self.schedule_config.get('milestones', [20, 40])
        weight_changes = self.schedule_config.get('weight_changes', {})

        weights = self.loss_weights.copy()

        for milestone in sorted(milestones):
            if self.current_epoch >= milestone and milestone in weight_changes:
                for key, value in weight_changes[milestone].items():
                    if key in weights:
                        weights[key] = value

        return weights

    def _cosine_schedule(self) -> Dict[str, float]:
        """Cosine annealing weight scheduling."""
        import math

        max_epoch = self.schedule_config.get('max_epoch', 60)
        target_weights = self.schedule_config.get('target_weights', {})

        weights = self.loss_weights.copy()
        progress = min(self.current_epoch / max_epoch, 1.0)

        # Cosine annealing factor
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))

        for key, target_weight in target_weights.items():
            if key in weights:
                initial_weight = self.loss_weights[key]
                weights[key] = target_weight + (initial_weight - target_weight) * cos_factor

        return weights

    def _get_device(self, predictions: Dict[str, Any]) -> torch.device:
        """Helper to extract device from predictions."""
        for value in predictions.values():
            if isinstance(value, torch.Tensor):
                return value.device
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, torch.Tensor):
                        return v.device
        return torch.device('cpu')

    def get_loss_history(self) -> list:
        """Return loss history for logging/visualization."""
        return self.loss_history

    def reset_history(self):
        """Reset loss history."""
        self.loss_history = []

    def state_dict(self) -> Dict[str, Any]:
        """Save manager state."""
        return {
            'loss_weights': self.loss_weights,
            'current_epoch': self.current_epoch,
            'weight_schedule': self.weight_schedule,
            'schedule_config': self.schedule_config
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load manager state."""
        self.loss_weights = state_dict.get('loss_weights', self.loss_weights)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.weight_schedule = state_dict.get('weight_schedule', self.weight_schedule)
        self.schedule_config = state_dict.get('schedule_config', self.schedule_config)


def build_loss_manager(config: Optional[Dict[str, Any]] = None) -> LossManager:
    """
    Factory function to create LossManager from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized LossManager

    Example:
        >>> config = {
        ...     'loss_weights': {'rgb_cls': 1.0, 'rgb_box': 1.0},
        ...     'weight_schedule': 'linear',
        ...     'schedule_config': {'max_epoch': 60, 'target_weights': {'fusion_box': 1.5}}
        ... }
        >>> manager = build_loss_manager(config)
    """
    config = config or {}
    return LossManager(**config)