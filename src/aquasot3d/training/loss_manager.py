"""
Multi-Task Loss Manager for AquaSOT-3D

This module implements a comprehensive loss manager for multi-modal tracking,
handling multiple task-specific losses with dynamic weighting strategies.

Loss Components:
- RGB Branch: bbox regression (GIoU/CIoU) + confidence (objectness)
- Sonar Branch: bbox regression (GIoU/CIoU) + confidence (objectness)
- Auxiliary Heads: pre-fusion bbox regression (feature degradation prevention)
- Cross-Modal Consistency: component + representation (InfoNCE)
- Cross-View Consistency: dual-view fusion alignment

ARCHITECTURE UPDATE (Audit 2025112313):
Fusion branch removed (dual-view fusion architecture).

CONTINUOUS FLOW DESIGN (Audit 2025112306):
Confidence losses train the model to predict object presence/absence (has_object flag),
enabling proper handling of no-object cases in continuous data flow.

Author: AquaSOT-3D Team
Date: 2025-11-23
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
        ...     'rgb_box': 1.0, 'rgb_conf': 1.0,
        ...     'sonar_box': 1.0, 'sonar_conf': 1.0,
        ...     'aux_rgb': 0.5, 'aux_sonar': 0.5,
        ...     'geometric_consistency': 0.5,
        ...     'representation_consistency': 0.3,
        ...     'view_consistency': 0.3
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

        # Default loss weights
        # ARCHITECTURE UPDATE (Audit 2025112313): Fusion removed, auxiliary heads added
        # CONTINUOUS FLOW DESIGN (Audit 2025112306 Phase 5): Added confidence loss weights
        # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution): Added validity loss weights
        self.loss_weights = loss_weights or {
            'rgb_box': 1.0,
            'rgb_conf': 1.0,  # Objectness confidence for RGB
            'rgb_validity': 1.0,  # Head validity for RGB fusion routing
            'sonar_box': 1.0,
            'sonar_conf': 1.0,  # Objectness confidence for Sonar
            'sonar_validity': 1.0,  # Head validity for Sonar fusion routing
            'aux_rgb': 0.5,  # Auxiliary head for RGB (pre-fusion)
            'aux_sonar': 0.5,  # Auxiliary head for Sonar (pre-fusion)
            'geometric_consistency': 0.5,
            'representation_consistency': 0.3,
            'view_consistency': 0.3  # Cross-view consistency (dual-view fusion)
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

    def _get_batch_size(self, predictions: Dict[str, Any]) -> int:
        """
        Extract batch size from predictions.

        Args:
            predictions: Model predictions dictionary

        Returns:
            Batch size

        Raises:
            ValueError: If batch size cannot be determined
        """
        for key in ['rgb_box', 'rgb_boxes', 'sonar_box', 'sonar_boxes']:
            if key in predictions and predictions[key] is not None:
                return predictions[key].shape[0]

        raise ValueError("Cannot determine batch size from predictions")

    def _validate_validity_mask(
        self,
        mask: Optional[torch.Tensor],
        mask_name: str,
        batch_size: int
    ) -> Optional[torch.Tensor]:
        """
        Validate validity mask shape and dtype.

        CRITICAL FIX (Audit 2025112306 P0-3): Ensure validity masks have correct format
        to prevent silent failures in loss masking.

        Args:
            mask: Validity mask tensor or None
            mask_name: Name for error messages
            batch_size: Expected batch size

        Returns:
            Validated mask as float32 [B] tensor, or None

        Raises:
            ValueError: If mask has incorrect shape/dtype
        """
        if mask is None:
            return None

        # Convert to tensor if needed
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

        # Check shape
        if mask.ndim != 1:
            raise ValueError(
                f"{mask_name} must be 1D tensor [B], got shape {mask.shape}"
            )

        if mask.shape[0] != batch_size:
            raise ValueError(
                f"{mask_name} batch size {mask.shape[0]} != expected {batch_size}"
            )

        # Convert to float32 if needed (for masking operations)
        if mask.dtype != torch.float32:
            mask = mask.float()

        return mask

    def compute_losses(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all losses and combine them with appropriate weights.

        Priority 1 Enhancement: Support model-internal losses passthrough.
        If predictions contain 'losses' dict, use those directly; otherwise compute from predictions/targets.

        Args:
            predictions: Dictionary containing all model predictions
                - 'losses': Pre-computed loss dictionary from model (optional, Priority 1)
                - 'rgb_boxes': RGB bounding box predictions
                - 'sonar_boxes': Sonar bounding box predictions
                - 'rgb_confidence': RGB objectness scores (optional)
                - 'sonar_confidence': Sonar objectness scores (optional)
                - 'consistency': Consistency losses (optional)
            targets: Dictionary containing ground truth
                - 'rgb_box': RGB target boxes
                - 'sonar_box': Sonar target boxes
            epoch: Current training epoch (for weight scheduling)

        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Dictionary containing individual loss components
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Update weights if using scheduling
        weights = self._get_current_weights()

        # CRITICAL FIX (Audit 2025113001 Issue 5): Always compute box losses externally
        # Previous bug: When model provided internal losses (consistency), we returned early
        # and NEVER computed box losses, causing training to stagnate at ~5.0.
        #
        # New design: Merge model-internal losses with externally computed box losses.
        # Model provides: consistency losses (geometric, representation, view)
        # LossManager computes: box losses (rgb_box, sonar_box, rgb_conf, sonar_conf)

        loss_dict = {}

        # Priority 1.1: Extract model-internal losses
        # CenterHead returns total_loss which includes: score_loss + size_loss*5 + iou_loss*2
        # This is critical for training the center heatmap and size regression!
        model_losses = predictions.get('losses', {})
        if model_losses:
            # Map model loss keys to config weight keys
            model_loss_map = {
                # Box losses from CenterHead (includes score_loss on heatmap!)
                'rgb_box_loss': 'rgb_box',
                'sonar_box_loss': 'sonar_box',
                # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
                # Validity losses from CenterHead for calibration monitoring
                'rgb_validity_loss': 'rgb_validity',
                'sonar_validity_loss': 'sonar_validity',
                # Consistency losses
                'geometric_consistency': 'geometric_consistency',
                'representation_consistency': 'representation_consistency',
                'view_consistency': 'view_consistency',
                # Auxiliary losses
                'aux_rgb_loss': 'aux_rgb',
                'aux_sonar_loss': 'aux_sonar',
            }
            for model_key, config_key in model_loss_map.items():
                if model_key in model_losses and model_losses[model_key] is not None:
                    loss_value = model_losses[model_key]
                    if isinstance(loss_value, torch.Tensor):
                        # Store RAW value only - weighting happens in section 5
                        loss_dict[config_key] = loss_value

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Fusion GT fallback REMOVED. Fusion box head no longer exists.
        # RGB and Sonar BoxHeads now use dual-view fused features with separate GT.

        # CRITICAL (Audit 2025111903 Phase 4): Extract validity masks from targets
        # These masks propagate from dataset → augmentation → collate → here
        # CRITICAL FIX (Audit 2025112306 P0-3): Validate all masks
        batch_size = self._get_batch_size(predictions)
        is_valid = self._validate_validity_mask(
            targets.get('is_valid', None), 'is_valid', batch_size
        )
        is_valid_rgb = self._validate_validity_mask(
            targets.get('is_valid_rgb', None), 'is_valid_rgb', batch_size
        )
        is_valid_sonar = self._validate_validity_mask(
            targets.get('is_valid_sonar', None), 'is_valid_sonar', batch_size
        )

        # CRITICAL FIX (Audit 2025120403 Issue 15/18): Extract has_object masks
        # has_object indicates whether the target object is present in the frame
        # When has_object=False (occlusion, GT=[0,0,0,0]), box regression loss must be zero
        # This prevents training on invalid/occluded frames that would corrupt the model
        has_object_rgb = self._validate_validity_mask(
            targets.get('has_object_rgb', targets.get('search_has_object_rgb', None)),
            'has_object_rgb', batch_size
        )
        has_object_sonar = self._validate_validity_mask(
            targets.get('has_object_sonar', targets.get('search_has_object_sonar', None)),
            'has_object_sonar', batch_size
        )

        # CRITICAL FIX (Audit 2025120403 Issue 15/18): Combine validity and has_object masks
        # Final mask = is_valid AND has_object (both must be true for loss computation)
        if is_valid_rgb is not None and has_object_rgb is not None:
            rgb_loss_mask = is_valid_rgb * has_object_rgb  # Element-wise AND via multiply
        elif has_object_rgb is not None:
            rgb_loss_mask = has_object_rgb
        else:
            rgb_loss_mask = is_valid_rgb  # Fallback to validity only

        if is_valid_sonar is not None and has_object_sonar is not None:
            sonar_loss_mask = is_valid_sonar * has_object_sonar
        elif has_object_sonar is not None:
            sonar_loss_mask = has_object_sonar
        else:
            sonar_loss_mask = is_valid_sonar

        # ============================================
        # 1. RGB Branch Losses (fallback only if model didn't provide)
        # ============================================
        # Prefer model-provided loss (CenterHead total_loss with score+size+iou)
        # Only compute external GIoU loss as fallback for backward compatibility
        if 'rgb_box' not in loss_dict:
            rgb_box_pred = None
            for key in ['rgb_box', 'rgb_boxes']:
                if key in predictions and predictions[key] is not None:
                    rgb_box_pred = predictions[key]
                    break

            rgb_box_target = None
            for key in ['rgb_box', 'rgb_boxes', 'search_bbox_rgb']:
                if key in targets and targets[key] is not None:
                    rgb_box_target = targets[key]
                    break

            if rgb_box_pred is not None and rgb_box_target is not None:
                # CRITICAL FIX (Audit 2025120403 Issue 15/18): Use combined mask
                # rgb_loss_mask = is_valid AND has_object (zero loss for occluded frames)
                rgb_box_loss = self._compute_box_loss(
                    rgb_box_pred,
                    rgb_box_target,
                    validity_mask=rgb_loss_mask
                )
                loss_dict['rgb_box'] = rgb_box_loss

        # ============================================
        # 2. Sonar Branch Losses (fallback only if model didn't provide)
        # ============================================
        # Prefer model-provided loss (CenterHead total_loss with score+size+iou)
        # Only compute external GIoU loss as fallback for backward compatibility
        if 'sonar_box' not in loss_dict:
            sonar_box_pred = None
            for key in ['sonar_box', 'sonar_boxes']:
                if key in predictions and predictions[key] is not None:
                    sonar_box_pred = predictions[key]
                    break

            sonar_box_target = None
            for key in ['sonar_box', 'sonar_boxes', 'search_bbox_sonar']:
                if key in targets and targets[key] is not None:
                    sonar_box_target = targets[key]
                    break

            if sonar_box_pred is not None and sonar_box_target is not None:
                # CRITICAL FIX (Audit 2025120403 Issue 15/18): Use combined mask
                # sonar_loss_mask = is_valid AND has_object (zero loss for occluded frames)
                sonar_box_loss = self._compute_box_loss(
                    sonar_box_pred,
                    sonar_box_target,
                    validity_mask=sonar_loss_mask
                )
                loss_dict['sonar_box'] = sonar_box_loss

        # ============================================
        # 3. Consistency Losses
        # ============================================
        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Fusion box loss computation REMOVED. Fusion box head no longer exists.
        # RGB and Sonar BoxHeads now use dual-view fused features.
        # (If fusion predictions exist in old checkpoints, they will be ignored)

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
        # 4.5. Confidence Losses (Continuous Flow Design - Audit 2025112306 Phase 5)
        # ============================================
        # CONTINUOUS FLOW DESIGN: Extract confidence losses from predictions
        # These losses are computed by BoxHead when has_object targets are provided
        # Confidence loss trains the model to predict object presence/absence

        # RGB confidence loss
        if 'rgb_conf_loss' in predictions and predictions['rgb_conf_loss'] is not None:
            loss_dict['rgb_conf'] = predictions['rgb_conf_loss']

        # Sonar confidence loss
        if 'sonar_conf_loss' in predictions and predictions['sonar_conf_loss'] is not None:
            loss_dict['sonar_conf'] = predictions['sonar_conf_loss']

        # ============================================
        # 5. Combine All Losses
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

    def _compute_from_model_losses(
        self,
        model_losses: Dict[str, torch.Tensor],
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss from model-provided losses (Priority 1).

        This method handles model-internal losses with name mapping and weighting.

        Args:
            model_losses: Loss dictionary from model outputs['losses']
            epoch: Current epoch (for logging)

        Returns:
            (total_loss, loss_dict) where loss_dict contains all component losses and total
        """
        loss_dict = {}
        total_loss = 0.0

        # Get current weights (with scheduling)
        weights = self._get_current_weights()

        # Define loss name mapping from model internal names to config keys
        # ARCHITECTURE UPDATE (Audit 2025112313): Fusion removed, auxiliary + view consistency added
        loss_name_map = {
            'rgb_box_loss': 'rgb_box',
            'sonar_box_loss': 'sonar_box',
            'rgb_conf_loss': 'rgb_conf',
            'sonar_conf_loss': 'sonar_conf',
            'aux_rgb_loss': 'aux_rgb',
            'aux_sonar_loss': 'aux_sonar',
            'geometric_consistency': 'geometric_consistency',
            'representation_consistency': 'representation_consistency',
            'view_consistency': 'view_consistency',
        }

        # Process individual losses with name mapping
        for model_key, loss_value in model_losses.items():
            if loss_value is None or not isinstance(loss_value, torch.Tensor):
                continue

            # Map model loss name to config weight key
            config_key = loss_name_map.get(model_key, model_key)

            # Get weight for this loss component
            weight = weights.get(config_key, 0.0)

            # Weighted loss
            weighted_loss = weight * loss_value
            total_loss = total_loss + weighted_loss

            # Store both raw and weighted loss for logging
            loss_dict[f'{config_key}_raw'] = loss_value.detach()
            loss_dict[config_key] = weighted_loss.detach()

        # Add total loss to dict
        loss_dict['total'] = total_loss.detach()

        # Store for logging
        if epoch is not None:
            self.loss_history.append({
                'epoch': epoch,
                'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                **{k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in loss_dict.items() if not k.endswith('_weighted') and k != 'total'}
            })

        return total_loss, loss_dict

    def _compute_box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        loss_type: Literal['l1', 'giou', 'ciou'] = 'giou',
        validity_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute bounding box regression loss with enforced type-safe format checking.

        CRITICAL (Audit 2025111411 Finding 3): This function enforces cxcywh format
        using type-safe wrappers throughout computation. All inputs are automatically
        wrapped and validated to prevent silent format confusion between xywh/cxcywh.

        CRITICAL (Audit 2025111903 Phase 4): Supports validity masking to exclude
        invalid samples (occlusions, augmentation filtering) from loss computation.
        When validity_mask is provided, only valid samples contribute to the loss gradient.

        Args:
            pred_boxes: Predicted boxes [B, 4] in cxcywh format (normalized [0,1])
            target_boxes: Target boxes [B, 4] in cxcywh format (normalized [0,1])
            loss_type: Type of loss to use ('l1', 'giou', 'ciou')
            validity_mask: Optional validity mask [B] (1.0=valid, 0.0=invalid)

        Returns:
            Box regression loss (scalar)

        Raises:
            ValueError: If inputs are not [B, 4] tensors in cxcywh format
        """
        # CRITICAL: Type-safe format enforcement (Audit 2025111411 Finding 3)
        # Enforce bbox wrappers throughout computation to prevent silent format confusion
        from aquasot3d.utils.bbox import BBoxCXCYWH

        # Wrap inputs to enforce format contract and validate shape
        try:
            pred_bbox = BBoxCXCYWH(pred_boxes)
            target_bbox = BBoxCXCYWH(target_boxes)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Bbox format validation failed: {e}. "
                f"This function requires [B, 4] tensors in cxcywh format. "
                f"Got pred_boxes.shape={pred_boxes.shape}, target_boxes.shape={target_boxes.shape}"
            )

        # Extract validated tensors for computation
        pred_tensor = pred_bbox.tensor()
        target_tensor = target_bbox.tensor()

        # Warn if boxes appear to be in pixel coordinates instead of normalized
        if torch.any(pred_tensor > 2.0) or torch.any(target_tensor > 2.0):
            import warnings
            warnings.warn(
                "Box coordinates > 2.0 detected in loss computation. "
                "This function expects normalized [0,1] cxcywh format. "
                "Use normalize_bbox() before passing to loss computation.",
                RuntimeWarning
            )

        # L1 loss component using validated tensors
        # CRITICAL (Audit 2025111903 Phase 4): Compute per-sample loss (no reduction)
        l1_loss_per_sample = F.l1_loss(pred_tensor, target_tensor, reduction='none').mean(dim=-1)  # [B]

        if loss_type == 'l1':
            # Apply validity masking before reduction
            if validity_mask is not None:
                masked_loss = l1_loss_per_sample * validity_mask  # [B]
                num_valid = validity_mask.sum().clamp(min=1.0)
                return masked_loss.sum() / num_valid if self.reduction == 'mean' else masked_loss.sum()
            else:
                return l1_loss_per_sample.mean() if self.reduction == 'mean' else l1_loss_per_sample.sum()

        # GIoU or CIoU loss - use wrapper conversion methods for type safety
        # Convert to xyxy format using type-safe wrapper methods
        pred_bbox_xyxy = pred_bbox.to_xyxy()
        target_bbox_xyxy = target_bbox.to_xyxy()

        # Extract tensors for IoU computation
        from aquasot3d.models.heads.box_head import compute_giou, compute_ciou
        pred_boxes_xyxy = pred_bbox_xyxy.tensor()
        target_boxes_xyxy = target_bbox_xyxy.tensor()

        if loss_type == 'giou':
            giou = compute_giou(pred_boxes_xyxy, target_boxes_xyxy)  # [B]
            giou_loss_per_sample = 1.0 - giou  # [B]
            combined_loss = l1_loss_per_sample + giou_loss_per_sample  # [B]

            # Apply validity masking before reduction
            if validity_mask is not None:
                masked_loss = combined_loss * validity_mask  # [B]
                num_valid = validity_mask.sum().clamp(min=1.0)
                return masked_loss.sum() / num_valid if self.reduction == 'mean' else masked_loss.sum()
            else:
                return combined_loss.mean() if self.reduction == 'mean' else combined_loss.sum()

        elif loss_type == 'ciou':
            ciou = compute_ciou(pred_boxes_xyxy, target_boxes_xyxy)  # [B]
            ciou_loss_per_sample = 1.0 - ciou  # [B]
            combined_loss = l1_loss_per_sample + ciou_loss_per_sample  # [B]

            # Apply validity masking before reduction
            if validity_mask is not None:
                masked_loss = combined_loss * validity_mask  # [B]
                num_valid = validity_mask.sum().clamp(min=1.0)
                return masked_loss.sum() / num_valid if self.reduction == 'mean' else masked_loss.sum()
            else:
                return combined_loss.mean() if self.reduction == 'mean' else combined_loss.sum()

        # Fallback to L1 with masking
        if validity_mask is not None:
            masked_loss = l1_loss_per_sample * validity_mask
            num_valid = validity_mask.sum().clamp(min=1.0)
            return masked_loss.sum() / num_valid if self.reduction == 'mean' else masked_loss.sum()
        else:
            return l1_loss_per_sample.mean() if self.reduction == 'mean' else l1_loss_per_sample.sum()

    def _compute_classification_loss(
        self,
        pred_scores: torch.Tensor,
        target_labels: torch.Tensor,
        loss_type: Literal['bce', 'focal'] = 'bce',
        validity_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute classification loss.

        IMPORTANT: This method expects pred_scores to be already sigmoid-activated (range [0,1]),
        NOT raw logits. This aligns with BoxHead design which uses nn.Sigmoid() in cls_head.

        CRITICAL (Audit 2025111903 Phase 4): Supports validity masking to exclude
        invalid samples (occlusions, augmentation filtering) from loss computation.
        When validity_mask is provided, only valid samples contribute to the loss gradient.

        Args:
            pred_scores: Predicted classification scores [B, ...], already sigmoid-activated in [0,1]
            target_labels: Target labels [B, ...]
            loss_type: Type of classification loss
            validity_mask: Optional validity mask [B] (1.0=valid, 0.0=invalid)

        Returns:
            Classification loss
        """
        if loss_type == 'bce':
            # CRITICAL (Audit 2025112103): Disable AMP for BCE to avoid assertion failures
            # Convert to FP32 for numerical stability
            with torch.cuda.amp.autocast(enabled=False):
                pred_scores_fp32 = pred_scores.float()
                target_fp32 = target_labels.float()

                # Clamp both predictions and targets to [0, 1] for numerical stability
                pred_scores_clamped = torch.clamp(pred_scores_fp32, min=0.0, max=1.0)
                target_clamped = torch.clamp(target_fp32, min=0.0, max=1.0)

                # CRITICAL (Audit 2025111903 Phase 4): Compute per-sample loss for masking
                loss_per_sample = F.binary_cross_entropy(
                    pred_scores_clamped,
                    target_clamped,
                    reduction='none'
                )

            # Apply validity masking before reduction
            if validity_mask is not None:
                # Expand mask to match loss dimensions if needed
                mask_expanded = validity_mask.view(-1, *([1] * (loss_per_sample.ndim - 1)))
                masked_loss = loss_per_sample * mask_expanded
                num_valid = validity_mask.sum().clamp(min=1.0)
                return masked_loss.sum() / num_valid if self.reduction == 'mean' else masked_loss.sum()
            else:
                return loss_per_sample.mean() if self.reduction == 'mean' else loss_per_sample.sum()

        elif loss_type == 'focal':
            # Focal loss implementation
            from aquasot3d.models.heads.box_head import QualityFocalLoss
            # Note: QualityFocalLoss doesn't support per-sample masking natively
            # We compute with 'none' reduction and apply masking manually
            focal_loss_fn = QualityFocalLoss(reduction='none')
            loss_per_sample = focal_loss_fn(pred_scores, target_labels)

            # Apply validity masking before reduction
            if validity_mask is not None:
                mask_expanded = validity_mask.view(-1, *([1] * (loss_per_sample.ndim - 1)))
                masked_loss = loss_per_sample * mask_expanded
                num_valid = validity_mask.sum().clamp(min=1.0)
                return masked_loss.sum() / num_valid if self.reduction == 'mean' else masked_loss.sum()
            else:
                return loss_per_sample.mean() if self.reduction == 'mean' else loss_per_sample.sum()

        # Default fallback to BCE with masking
        # CRITICAL (Audit 2025112103): Disable AMP for BCE to avoid assertion failures
        with torch.cuda.amp.autocast(enabled=False):
            pred_scores_fp32 = pred_scores.float()
            target_fp32 = target_labels.float()

            pred_scores_clamped = torch.clamp(pred_scores_fp32, min=0.0, max=1.0)
            target_clamped = torch.clamp(target_fp32, min=0.0, max=1.0)
            loss_per_sample = F.binary_cross_entropy(
                pred_scores_clamped,
                target_clamped,
                reduction='none'
            )

        # Apply validity masking before reduction
        if validity_mask is not None:
            mask_expanded = validity_mask.view(-1, *([1] * (loss_per_sample.ndim - 1)))
            masked_loss = loss_per_sample * mask_expanded
            num_valid = validity_mask.sum().clamp(min=1.0)
            return masked_loss.sum() / num_valid if self.reduction == 'mean' else masked_loss.sum()
        else:
            return loss_per_sample.mean() if self.reduction == 'mean' else loss_per_sample.sum()

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
        ...     'loss_weights': {'rgb_box': 1.0, 'sonar_box': 1.0, 'aux_rgb': 0.5},
        ...     'weight_schedule': 'linear',
        ...     'schedule_config': {'max_epoch': 60, 'target_weights': {'aux_rgb': 1.0}}
        ... }
        >>> manager = build_loss_manager(config)
    """
    config = config or {}
    return LossManager(**config)
