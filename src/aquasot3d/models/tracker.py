"""
AquaSOT-3D Tracker - Template-Search Multi-Modal Tracking System

This module implements the complete tracking pipeline integrating:
- Multi-modal backbones (RGB + Sonar) with OSTrack one-stream architecture
- G2-DCMA cross-modal fusion with learnable confidence weighting
- Multi-task prediction heads for 2D bounding box detection

Architecture:
    Input: RGB template/search + Sonar template/search
    -> Backbones: OSTrack one-stream ViT for RGB (3-channel) and Sonar (1-channel)
    -> Fusion: ConfidenceAwareG2DCMA with Dual-View Fusion (rgb_view_fused, sonar_view_fused)
    -> Heads: BoxHead (RGB uses rgb_view_fused, Sonar uses sonar_view_fused)
    Output: 2D boxes (RGB/Sonar) + losses

    ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
    Fusion BoxHead removed. RGB and Sonar BoxHeads now use dual-view fused features,
    enabling end-to-end learning of RGB-Sonar correspondence without calibration.

Author: AquaSOT-3D Team
Date: 2025-11-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import logging

from aquasot3d.core import BaseModule, TRACKER_REGISTRY
from aquasot3d.models.backbones import BackboneFactory, MultiModalBackbone
from aquasot3d.models.fusion import build_g2_dcma, build_g2_dcma_lite, build_conditional_g2_dcma
from aquasot3d.models.heads import (
    build_box_head,
)


logger = logging.getLogger(__name__)


@TRACKER_REGISTRY.register("AquaSOT3DTracker")
class AquaSOT3DTracker(BaseModule):
    """
    AquaSOT-3D Template-Search Tracker with Multi-Modal Fusion.

    This is the main tracking model that integrates:
    - OSTrack one-stream ViT backbones for RGB and Sonar
    - ConfidenceAwareG2DCMA cross-modal fusion with learnable confidence
    - Multi-task prediction heads for 2D bounding boxes
    - Uncertainty-aware fusion for robust multi-modal tracking

    Args:
        rgb_backbone_config: Configuration for RGB backbone (must include template_size=256, search_size=512)
        sonar_backbone_config: Configuration for Sonar backbone (must include template_size=256, search_size=512)
        fusion_config: Configuration for G2-DCMA fusion module
        box_head_config: Configuration for 2D box prediction heads

    Example:
        >>> config = {
        ...     'rgb_backbone': {'type': 'vit_rgb', 'model_name': 'vit_base_patch16_224', 'template_size': 256, 'search_size': 512},
        ...     'sonar_backbone': {'type': 'vit_sonar', 'model_name': 'vit_base_patch16_224', 'template_size': 256, 'search_size': 512},
        ...     'fusion': {'type': 'g2_dcma', 'rgb_dim': 768, 'sonar_dim': 768},
        ... }
        >>> tracker = build_tracker('AquaSOT3DTracker', **config)
        >>> outputs = tracker(rgb_template, rgb_search, sonar_template, sonar_search)
    """

    def __init__(
        self,
        rgb_backbone_config: Dict[str, Any],
        sonar_backbone_config: Dict[str, Any],
        fusion_config: Dict[str, Any],
        box_head_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()

        # =====================================================
        # 1. Multi-Modal Backbones
        # =====================================================
        self.backbone = MultiModalBackbone(
            rgb_config=rgb_backbone_config,
            sonar_config=sonar_backbone_config
        )

        # Get feature dimensions from actual backbones (not config, to handle timm model variations)
        # Priority: actual model > config > default
        if self.backbone.rgb_backbone is not None and hasattr(self.backbone.rgb_backbone, 'get_feature_dim'):
            self.rgb_feat_dim = self.backbone.rgb_backbone.get_feature_dim()
        elif rgb_backbone_config:
            self.rgb_feat_dim = rgb_backbone_config.get('output_dim', 768)
        else:
            self.rgb_feat_dim = 768

        if self.backbone.sonar_encoder is not None and hasattr(self.backbone.sonar_encoder, 'get_feature_dim'):
            self.sonar_feat_dim = self.backbone.sonar_encoder.get_feature_dim()
        elif sonar_backbone_config:
            self.sonar_feat_dim = sonar_backbone_config.get('output_dim', 384)
        else:
            self.sonar_feat_dim = 384

        # =====================================================
        # 2. Cross-Modal Fusion (G2-DCMA)
        # =====================================================
        self.fusion = None

        # Only build fusion if config is provided and type is not null (for multi-modal training)
        if fusion_config and fusion_config.get('type') not in [None, 'null']:
            # Make a copy to avoid modifying the original config (high-generalization fix)
            fusion_config_copy = fusion_config.copy()
            fusion_type = fusion_config_copy.pop('type', 'g2_dcma')
            if fusion_type == 'g2_dcma':
                self.fusion = build_g2_dcma(fusion_config_copy)
            elif fusion_type == 'g2_dcma_lite':
                self.fusion = build_g2_dcma_lite(fusion_config_copy)
            elif fusion_type == 'conditional_g2_dcma':
                # CRITICAL FIX (Audit 2025112101 Phase 2.2):
                # ConditionalG2DCMA requires rgb_channels and sonar_channels
                self.fusion = build_conditional_g2_dcma(
                    rgb_channels=self.rgb_feat_dim,
                    sonar_channels=self.sonar_feat_dim,
                    **fusion_config_copy
                )
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")

            self.fused_feat_dim = fusion_config.get('output_dim', self.rgb_feat_dim)
            logger.info(f"Initialized fusion module: {fusion_type}")
        else:
            # Single-modal training: use active modality's feature dimension
            if rgb_backbone_config and rgb_backbone_config.get('type') not in [None, 'null']:
                self.fused_feat_dim = self.rgb_feat_dim
            elif sonar_backbone_config and sonar_backbone_config.get('type') not in [None, 'null']:
                self.fused_feat_dim = self.sonar_feat_dim
            else:
                # Fallback: use RGB dimension if both are None
                self.fused_feat_dim = self.rgb_feat_dim
            logger.info("Fusion module disabled (single-modal training)")


        # =====================================================
        # 4. Prediction Heads
        # =====================================================

        # 4.1 2D Box Prediction Heads
        if box_head_config is None:
            box_head_config = {}

        # CRITICAL FIX: Conditional initialization for all box heads
        # Only create heads for active modalities to avoid memory waste and maintain symmetry

        # RGB box head (only if RGB backbone exists)
        self.rgb_box_head = None
        if self.backbone.rgb_backbone is not None:
            self.rgb_box_head = build_box_head(
                'rgb',
                in_channels=self.rgb_feat_dim,
                **box_head_config
            )
            logger.info("Initialized RGB box head")
        else:
            logger.info("RGB box head disabled (no RGB backbone)")

        # Sonar box head (only if Sonar encoder exists)
        self.sonar_box_head = None
        if self.backbone.sonar_encoder is not None:
            self.sonar_box_head = build_box_head(
                'sonar',
                in_channels=self.sonar_feat_dim,
                **box_head_config
            )
            logger.info("Initialized Sonar box head")
        else:
            logger.info("Sonar box head disabled (no Sonar encoder)")

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Fusion box head REMOVED. RGB and Sonar BoxHeads now use dual-view fused features.
        # This simplifies architecture and strengthens G2-DCMA supervision through dual gradients.

        # 4.3 Auxiliary Heads for Feature Degradation Prevention
        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Add lightweight auxiliary heads that predict from original features (pre-fusion).
        # This prevents backbone feature degradation by providing direct gradient flow.
        # These heads are ONLY used during training, not inference.
        self.aux_rgb_head = None
        self.aux_sonar_head = None
        if self.backbone.rgb_backbone is not None and self.backbone.sonar_encoder is not None:
            # Very lightweight: just adaptive pooling + linear layers
            self.aux_rgb_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
                nn.Flatten(),  # [B, C, 1, 1] -> [B, C]
                nn.Linear(self.rgb_feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 4)  # Predict bbox [cx, cy, w, h]
            )
            self.aux_sonar_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.sonar_feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 4)
            )
            logger.info("Initialized auxiliary heads for feature degradation prevention")
        else:
            logger.info("Auxiliary heads disabled (single-modal training)")

        # =====================================================
        # 6. Template Update State Management
        # =====================================================
        # Template-search paradigm has no internal state.
        # Template updates are managed externally via OnlineTracker.

        self.logger.info(
            f"Initialized AquaSOT3DTracker: "
            f"rgb_dim={self.rgb_feat_dim}, sonar_dim={self.sonar_feat_dim}, "
            f"fused_dim={self.fused_feat_dim}"
        )


    def detect_modality_availability(
        self,
        is_rgb_valid: torch.Tensor,
        is_sonar_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect modality availability mode for each sample.

        CRITICAL FIX (Audit 2025112101): Enable single/zero-modality handling.
        Returns masks indicating which samples belong to which modality configuration.

        Args:
            is_rgb_valid: [B] or [B, T] boolean tensor, True if RGB has valid target
            is_sonar_valid: [B] or [B, T] boolean tensor, True if Sonar has valid target

        Returns:
            Tuple of four boolean masks:
            - zero_modality_mask: [B] or [B, T], both modalities invalid
            - single_rgb_mask: [B] or [B, T], only RGB valid
            - single_sonar_mask: [B] or [B, T], only Sonar valid
            - dual_modality_mask: [B] or [B, T], both valid

        Example:
            >>> is_rgb_valid = torch.tensor([True, False, True, False])
            >>> is_sonar_valid = torch.tensor([True, True, False, False])
            >>> zero, rgb, sonar, dual = tracker.detect_modality_availability(
            ...     is_rgb_valid, is_sonar_valid
            ... )
            >>> # dual = [True, False, False, False]  # Sample 0: both valid
            >>> # single_sonar = [False, True, False, False]  # Sample 1: only Sonar
            >>> # single_rgb = [False, False, True, False]  # Sample 2: only RGB
            >>> # zero = [False, False, False, True]  # Sample 3: neither valid
        """
        # Compute modality masks
        zero_modality_mask = (~is_rgb_valid) & (~is_sonar_valid)
        single_rgb_mask = is_rgb_valid & (~is_sonar_valid)
        single_sonar_mask = (~is_rgb_valid) & is_sonar_valid
        dual_modality_mask = is_rgb_valid & is_sonar_valid

        # Log distribution for debugging (only if logger is in debug mode)
        if self.logger.isEnabledFor(logging.DEBUG):
            batch_size = is_rgb_valid.shape[0]
            n_zero = zero_modality_mask.sum().item()
            n_rgb = single_rgb_mask.sum().item()
            n_sonar = single_sonar_mask.sum().item()
            n_dual = dual_modality_mask.sum().item()

            self.logger.debug(
                f"Modality distribution: "
                f"dual={n_dual}/{batch_size} ({n_dual/batch_size*100:.1f}%), "
                f"RGB-only={n_rgb}/{batch_size} ({n_rgb/batch_size*100:.1f}%), "
                f"Sonar-only={n_sonar}/{batch_size} ({n_sonar/batch_size*100:.1f}%), "
                f"zero={n_zero}/{batch_size} ({n_zero/batch_size*100:.1f}%)"
            )

        return zero_modality_mask, single_rgb_mask, single_sonar_mask, dual_modality_mask


    def extract_backbone_features(
        self,
        rgb_template: torch.Tensor,
        rgb_search: torch.Tensor,
        sonar_template: torch.Tensor,
        sonar_search: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from RGB and Sonar backbones using template-search paradigm.

        ARCHITECTURE CHANGE (Audit 2025112104):
        Migrated to template-search dual input (OSTrack-style).

        Args:
            rgb_template: RGB template frame, shape (B, 3, H_t, W_t)
            rgb_search: RGB search frame, shape (B, 3, H_s, W_s)
            sonar_template: Sonar template frame, shape (B, 1, H_t, W_t)
            sonar_search: Sonar search frame, shape (B, 1, H_s, W_s)

        Returns:
            Dictionary containing extracted features:
            {
                'rgb_template_features': [B, N_z, D],
                'rgb_search_features': [B, N_x, D],
                'sonar_template_features': [B, N_z, D],
                'sonar_search_features': [B, N_x, D]
            }
        """
        # Extract multi-modal features using template-search paradigm
        backbone_outputs = self.backbone(
            rgb_template=rgb_template,
            rgb_search=rgb_search,
            sonar_template=sonar_template,
            sonar_search=sonar_search
        )

        # Extract features from nested structure
        # backbone_outputs = {
        #     'rgb': {'template_features': ..., 'search_features': ...},
        #     'sonar': {'template_features': ..., 'search_features': ...}
        # }
        return {
            'rgb_template_features': backbone_outputs['rgb']['template_features'],
            'rgb_search_features': backbone_outputs['rgb']['search_features'],
            'sonar_template_features': backbone_outputs['sonar']['template_features'],
            'sonar_search_features': backbone_outputs['sonar']['search_features'],
        }

    def fuse_features(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse RGB and Sonar features using G2-DCMA or ConditionalG2DCMA.

        CRITICAL FIX (Audit 2025112101 Phase 2.2):
        Now accepts is_rgb_valid and is_sonar_valid for conditional fusion.
        When using ConditionalG2DCMA, these flags enable single-modality bypass.

        Args:
            rgb_features: RGB features, shape (B, C_rgb, H, W) or (B, N, C_rgb)
            sonar_features: Sonar features, shape (B, C_sonar, H, W)
            is_rgb_valid: Optional [B] boolean tensor, True if RGB has valid target
            is_sonar_valid: Optional [B] boolean tensor, True if Sonar has valid target
            return_consistency: Whether to compute consistency losses

        Returns:
            Dictionary containing fused features and losses
        """
        # Note: Both RGB and Sonar features are required (fusion is always enabled)

        # Handle RGB ViT sequence features: reshape (B, N, C) -> (B, C, H, W)
        if rgb_features.dim() == 3:
            B, N, C = rgb_features.shape
            # Check if class token is present (N = H*W + 1)
            if int((N - 1) ** 0.5) ** 2 == N - 1:
                # Has class token, remove it
                rgb_features = rgb_features[:, 1:, :]  # Remove class token
                N = N - 1
            # Assume square spatial layout
            H = W = int(N ** 0.5)
            rgb_features = rgb_features.permute(0, 2, 1).reshape(B, C, H, W)

        # Handle Sonar ViT sequence features: reshape (B, N, C) -> (B, C, H, W)
        # Ensure sonar_features is a tensor, not a list
        if isinstance(sonar_features, list):
            sonar_features = sonar_features[0]

        # Reshape sequence format to spatial format (symmetric to RGB handling)
        if sonar_features.dim() == 3:
            B, N, C = sonar_features.shape
            # Check if class token is present (N = H*W + 1)
            if int((N - 1) ** 0.5) ** 2 == N - 1:
                # Has class token, remove it
                sonar_features = sonar_features[:, 1:, :]  # Remove class token
                N = N - 1
            # Assume square spatial layout
            H = W = int(N ** 0.5)
            sonar_features = sonar_features.permute(0, 2, 1).reshape(B, C, H, W)

        # CRITICAL FIX (Audit 2025112101 Phase 2.2):
        # Pass is_rgb_valid and is_sonar_valid to ConditionalG2DCMA for modality-aware routing
        # For standard G2DCMA, these parameters will be ignored
        fusion_kwargs = {
            'rgb_features': rgb_features,
            'sonar_features': sonar_features,
            'return_consistency': return_consistency
        }

        # Check if fusion module is ConditionalG2DCMA (has forward signature with is_rgb_valid)
        from aquasot3d.models.fusion import ConditionalG2DCMA
        if isinstance(self.fusion, ConditionalG2DCMA):
            # ConditionalG2DCMA requires is_rgb_valid and is_sonar_valid
            fusion_kwargs['is_rgb_valid'] = is_rgb_valid
            fusion_kwargs['is_sonar_valid'] = is_sonar_valid

        fusion_outputs = self.fusion(**fusion_kwargs)

        return fusion_outputs


    def predict_boxes(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        fused_features: torch.Tensor,
        gt_boxes: Optional[Dict[str, torch.Tensor]] = None,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        has_object_rgb: Optional[torch.Tensor] = None,
        has_object_sonar: Optional[torch.Tensor] = None,
        has_object: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict 2D bounding boxes for all modalities using template-search paradigm.

        CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX):
        Accepts has_object flags for confidence head training.

        Args:
            rgb_features: RGB features, shape (B, C_rgb, H, W)
            sonar_features: Sonar features, shape (B, C_sonar, H, W)
            fused_features: Fused features, shape (B, C_fused, H, W)
            gt_boxes: Optional ground truth boxes for loss computation
            is_rgb_valid: Optional RGB validity flags, shape (B,)
            is_sonar_valid: Optional Sonar validity flags, shape (B,)
            has_object_rgb: Optional RGB has_object flags, shape (B,)
            has_object_sonar: Optional Sonar has_object flags, shape (B,)
            has_object: Optional combined has_object flags, shape (B,)

        Returns:
            Dictionary containing box predictions and losses
        """
        # Ensure features are tensors (extract from list if needed)
        if rgb_features is not None and isinstance(rgb_features, list):
            rgb_features = rgb_features[0]
        if sonar_features is not None and isinstance(sonar_features, list):
            sonar_features = sonar_features[0]
        if fused_features is not None and isinstance(fused_features, list):
            fused_features = fused_features[0]

        outputs = {}

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # RGB box prediction now uses RGB-view fused features instead of original features
        # This forces G2-DCMA to learn RGB-Sonar correspondence through dual supervision
        if self.rgb_box_head is not None and rgb_features is not None:
            # Extract RGB-view fused features from dual-view fusion output
            if isinstance(fused_features, dict) and 'rgb_view_fused' in fused_features:
                rgb_input_features = fused_features['rgb_view_fused']
            elif isinstance(fused_features, dict) and 'fused_features' in fused_features:
                # Fallback: use 'fused_features' key (backward compatibility)
                rgb_input_features = fused_features['fused_features']
            else:
                # Legacy: fused_features is a tensor, not a dict
                rgb_input_features = fused_features

            rgb_box_outputs = self.rgb_box_head(
                rgb_input_features,  # ← CHANGED: Now uses fused features instead of rgb_features
                target_boxes=gt_boxes.get('rgb') if gt_boxes and 'rgb' in gt_boxes else None,
                target_labels=None,
                validity_mask=is_rgb_valid,
                has_object=has_object_rgb
            )
            outputs['rgb_boxes'] = rgb_box_outputs['pred_boxes']
            outputs['rgb_quality'] = rgb_box_outputs['pred_scores']

            # Add confidence prediction if available
            if 'pred_confidence' in rgb_box_outputs:
                outputs['rgb_confidence'] = rgb_box_outputs['pred_confidence']

            # Add confidence loss if available
            if 'conf_loss' in rgb_box_outputs:
                outputs['rgb_conf_loss'] = rgb_box_outputs['conf_loss']

            # Legacy loss handling (if compute_loss was used separately)
            if gt_boxes is not None and 'rgb' in gt_boxes and 'total_loss' not in rgb_box_outputs:
                rgb_loss = self.rgb_box_head.compute_loss(
                    rgb_box_outputs,
                    gt_boxes['rgb']
                )
                outputs['rgb_box_loss'] = rgb_loss

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Sonar box prediction now uses Sonar-view fused features instead of original features
        if self.sonar_box_head is not None and sonar_features is not None:
            # Extract Sonar-view fused features from dual-view fusion output
            if isinstance(fused_features, dict) and 'sonar_view_fused' in fused_features:
                sonar_input_features = fused_features['sonar_view_fused']
            elif isinstance(fused_features, dict) and 'fused_features' in fused_features:
                # Fallback: use 'fused_features' key (backward compatibility)
                sonar_input_features = fused_features['fused_features']
            else:
                # Legacy: fused_features is a tensor, not a dict
                sonar_input_features = fused_features

            sonar_box_outputs = self.sonar_box_head(
                sonar_input_features,  # ← CHANGED: Now uses fused features instead of sonar_features
                target_boxes=gt_boxes.get('sonar') if gt_boxes and 'sonar' in gt_boxes else None,
                target_labels=None,
                validity_mask=is_sonar_valid,
                has_object=has_object_sonar
            )
            outputs['sonar_boxes'] = sonar_box_outputs['pred_boxes']
            outputs['sonar_quality'] = sonar_box_outputs['pred_scores']

            # Add confidence prediction if available
            if 'pred_confidence' in sonar_box_outputs:
                outputs['sonar_confidence'] = sonar_box_outputs['pred_confidence']

            # Add confidence loss if available
            if 'conf_loss' in sonar_box_outputs:
                outputs['sonar_conf_loss'] = sonar_box_outputs['conf_loss']

            # Legacy loss handling
            if gt_boxes is not None and 'sonar' in gt_boxes and 'total_loss' not in sonar_box_outputs:
                sonar_loss = self.sonar_box_head.compute_loss(
                    sonar_box_outputs,
                    gt_boxes['sonar']
                )
                outputs['sonar_box_loss'] = sonar_loss

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Fusion box prediction REMOVED. Primary tracking output now comes from:
        # - RGB BoxHead (using rgb_view_fused features)
        # - Sonar BoxHead (using sonar_view_fused features)
        # During inference, use RGB boxes as primary output (or Sonar if RGB is invalid)

        # Always output modality validity for evaluation/analysis
        if is_rgb_valid is not None:
            outputs['is_rgb_valid'] = is_rgb_valid
        if is_sonar_valid is not None:
            outputs['is_sonar_valid'] = is_sonar_valid

        return outputs

    def forward(
        self,
        rgb_template: torch.Tensor,
        rgb_search: torch.Tensor,
        sonar_template: torch.Tensor,
        sonar_search: torch.Tensor,
        gt_boxes: Optional[Dict[str, torch.Tensor]] = None,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        has_object_rgb: Optional[torch.Tensor] = None,
        has_object_sonar: Optional[torch.Tensor] = None,
        has_object: Optional[torch.Tensor] = None,
        return_losses: bool = True,
        return_intermediate: bool = False,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass of AquaSOT-3D tracker using template-search paradigm.

        CRITICAL FIX (Audit 2025112101 Phase 2.2):
        Accepts is_rgb_valid and is_sonar_valid for conditional fusion.
        These flags enable single-modality bypass in ConditionalG2DCMA.

        CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX):
        Accepts has_object flags for confidence head training.
        These flags distinguish "no object" from "object at [0,0,0,0]".

        Args:
            rgb_template: RGB template frame, shape (B, 3, 256, 256)
            rgb_search: RGB search frame, shape (B, 3, 512, 512)
            sonar_template: Sonar template frame, shape (B, 1, 256, 256)
            sonar_search: Sonar search frame, shape (B, 1, 512, 512)
            gt_boxes: Optional ground truth boxes for loss computation
                Expected format: {'rgb': (B, 4), 'sonar': (B, 4)}
            is_rgb_valid: Optional [B] boolean tensor, True if RGB has valid target
            is_sonar_valid: Optional [B] boolean tensor, True if Sonar has valid target
            has_object_rgb: Optional [B] float32 tensor, 1.0 if RGB has object, 0.0 if no object
            has_object_sonar: Optional [B] float32 tensor, 1.0 if Sonar has object, 0.0 if no object
            has_object: Optional [B] float32 tensor, combined has_object flag for fusion
            return_losses: Whether to compute and return losses
            return_intermediate: Whether to return intermediate features
            metadata: Optional metadata dict containing:
                - 'rgb_search_metadata': List of dicts with scale/padding info
                - 'sonar_search_metadata': List of dicts with scale/padding info
                (CRITICAL FIX for Audit 2025111411 Finding 4)

        Returns:
            Dictionary containing:
                - 'rgb_boxes': RGB box predictions (B, 4) [PRIMARY OUTPUT - uses rgb_view_fused]
                - 'sonar_boxes': Sonar box predictions (B, 4) [PRIMARY OUTPUT - uses sonar_view_fused]
                - 'rgb_confidence': RGB objectness scores (B,) [if confidence head enabled]
                - 'sonar_confidence': Sonar objectness scores (B,)
                - Various losses [if return_losses=True]
                - Intermediate features [if return_intermediate=True]

            ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
            Fusion boxes removed. RGB and Sonar BoxHeads now use dual-view fused features,
            forcing G2-DCMA to learn RGB-Sonar correspondence through dual supervision.
        """
        # Get batch size from template (template-search paradigm)
        batch_size = rgb_template.shape[0]

        # Validate metadata list lengths match batch size (Audit 2025111412 Finding 4)
        if metadata is not None:
            for key in ['rgb_search_metadata', 'sonar_search_metadata',
                       'template_rgb_metadata', 'template_sonar_metadata']:
                if key in metadata and metadata[key] is not None:
                    meta_list = metadata[key]
                    # Handle both list and dict (legacy) formats
                    if isinstance(meta_list, dict):
                        meta_list = [meta_list]

                    if len(meta_list) != batch_size:
                        raise ValueError(
                            f"Metadata length mismatch at tracker forward: "
                            f"{key} has {len(meta_list)} entries but batch_size={batch_size}. "
                            f"This indicates a loader bug. Check collate function."
                        )

        # =====================================================
        # Stage 1: Feature Extraction (Template-Search Paradigm)
        # =====================================================
        # ARCHITECTURE CHANGE (Audit 2025112104):
        # Extract template and search features. Template-search interaction
        # happens via self-attention in ViT backbone (one-stream processing).
        backbone_outputs = self.extract_backbone_features(
            rgb_template, rgb_search, sonar_template, sonar_search
        )

        # Use search features for prediction (template info already encoded via attention)
        # Shape: [B, N_x, D] where N_x = num_search_patches
        rgb_features = backbone_outputs['rgb_search_features']
        sonar_features = backbone_outputs['sonar_search_features']

        # =====================================================
        # Stage 1.5: Auxiliary Predictions (Pre-Fusion)
        # =====================================================
        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Compute auxiliary predictions from original features to prevent feature degradation.
        # These predictions provide direct gradient flow to backbones.
        aux_outputs = {}
        if self.training and return_losses and self.aux_rgb_head is not None:
            # Prepare features for auxiliary heads (need spatial format [B, C, H, W])
            # RGB features
            if rgb_features.dim() == 3:  # [B, N, C]
                B, N, C = rgb_features.shape
                # Remove class token if present
                if int((N - 1) ** 0.5) ** 2 == N - 1:
                    rgb_features_spatial = rgb_features[:, 1:, :]
                    N = N - 1
                else:
                    rgb_features_spatial = rgb_features
                H = W = int(N ** 0.5)
                rgb_features_spatial = rgb_features_spatial.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                rgb_features_spatial = rgb_features

            # Sonar features
            if sonar_features.dim() == 3:  # [B, N, C]
                B, N, C = sonar_features.shape
                if int((N - 1) ** 0.5) ** 2 == N - 1:
                    sonar_features_spatial = sonar_features[:, 1:, :]
                    N = N - 1
                else:
                    sonar_features_spatial = sonar_features
                H = W = int(N ** 0.5)
                sonar_features_spatial = sonar_features_spatial.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                sonar_features_spatial = sonar_features

            # Auxiliary predictions (very simple, just for gradient flow)
            aux_rgb_boxes = self.aux_rgb_head(rgb_features_spatial)  # [B, 4]
            aux_sonar_boxes = self.aux_sonar_head(sonar_features_spatial)  # [B, 4]

            aux_outputs['aux_rgb_boxes'] = aux_rgb_boxes
            aux_outputs['aux_sonar_boxes'] = aux_sonar_boxes

            # Compute auxiliary losses if GT is available
            if gt_boxes is not None:
                # Simple L1 loss for auxiliary predictions (lightweight supervision)
                if 'rgb' in gt_boxes:
                    aux_rgb_loss = F.l1_loss(aux_rgb_boxes, gt_boxes['rgb'], reduction='mean')
                    aux_outputs['aux_rgb_loss'] = aux_rgb_loss * 0.3  # Lower weight
                if 'sonar' in gt_boxes:
                    aux_sonar_loss = F.l1_loss(aux_sonar_boxes, gt_boxes['sonar'], reduction='mean')
                    aux_outputs['aux_sonar_loss'] = aux_sonar_loss * 0.3  # Lower weight

        # =====================================================
        # Stage 2: Cross-Modal Fusion
        # =====================================================
        # CRITICAL FIX (Audit 2025112101 Phase 2.2):
        # If no validity flags provided, default to all valid (dual-modality)
        if is_rgb_valid is None:
            is_rgb_valid = torch.ones(batch_size, dtype=torch.bool, device=rgb_features.device)
        if is_sonar_valid is None:
            is_sonar_valid = torch.ones(batch_size, dtype=torch.bool, device=sonar_features.device)

        fusion_outputs = self.fuse_features(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=return_losses
        )

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # fusion_outputs now contains both rgb_view_fused and sonar_view_fused
        # Keep the full dict for passing to predict_boxes
        fused_features = fusion_outputs  # Keep as dict to access both views


        # =====================================================
        # Stage 4: 2D Box Prediction
        # =====================================================
        # CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX): Pass has_object flags
        box_outputs = self.predict_boxes(
            rgb_features, sonar_features, fused_features,
            gt_boxes=gt_boxes if return_losses else None,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            has_object_rgb=has_object_rgb,
            has_object_sonar=has_object_sonar,
            has_object=has_object
        )

        # =====================================================
        # Stage 6: Output Organization
        # =====================================================
        outputs = {}

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Fusion boxes removed. Primary output now from RGB/Sonar BoxHeads on fused features.
        # Add predictions conditionally based on available modalities
        if 'rgb_boxes' in box_outputs:
            outputs['rgb_boxes'] = box_outputs['rgb_boxes']
            outputs['rgb_quality'] = box_outputs.get('rgb_quality', None)
            # CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX): Add confidence predictions
            outputs['rgb_confidence'] = box_outputs.get('rgb_confidence', None)
        if 'sonar_boxes' in box_outputs:
            outputs['sonar_boxes'] = box_outputs['sonar_boxes']
            outputs['sonar_quality'] = box_outputs.get('sonar_quality', None)
            # CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX): Add confidence predictions
            outputs['sonar_confidence'] = box_outputs.get('sonar_confidence', None)

        # Add losses if requested
        if return_losses:
            losses = {}

            # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
            # Fusion losses removed. Only RGB and Sonar losses now.
            # 2D box losses
            if 'rgb_box_loss' in box_outputs:
                losses['rgb_box_loss'] = box_outputs['rgb_box_loss']
            if 'sonar_box_loss' in box_outputs:
                losses['sonar_box_loss'] = box_outputs['sonar_box_loss']

            # CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX):
            # Extract confidence losses for objectness prediction
            if 'rgb_conf_loss' in box_outputs:
                losses['rgb_conf_loss'] = box_outputs['rgb_conf_loss']
            if 'sonar_conf_loss' in box_outputs:
                losses['sonar_conf_loss'] = box_outputs['sonar_conf_loss']

            # Consistency losses
            if 'consistency_losses' in fusion_outputs:
                # Geometric consistency: compares geometric encodings (scale vs radial)
                losses['geometric_consistency'] = fusion_outputs['consistency_losses']['component_loss']
                losses['representation_consistency'] = fusion_outputs['consistency_losses']['infonce_loss']

            # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
            # Add cross-view consistency loss
            if 'view_consistency_loss' in fusion_outputs:
                losses['view_consistency'] = fusion_outputs['view_consistency_loss']

            # ARCHITECTURE UPDATE (Audit 2025112313 - Auxiliary Heads):
            # Add auxiliary losses for feature degradation prevention
            if 'aux_rgb_loss' in aux_outputs:
                losses['aux_rgb_loss'] = aux_outputs['aux_rgb_loss']
            if 'aux_sonar_loss' in aux_outputs:
                losses['aux_sonar_loss'] = aux_outputs['aux_sonar_loss']

            outputs['losses'] = losses

        # Add intermediate features if requested
        if return_intermediate:
            outputs['intermediate'] = {
                'rgb_features': rgb_features,
                'sonar_features': sonar_features,
                'fused_features': fused_features,
                'geometric_info': fusion_outputs.get('geometric_info', {}),
                'deformable_info': fusion_outputs.get('deformable_info', {}),
            }

        return outputs


def build_tracker(
    tracker_type: str = 'AquaSOT3DTracker',
    **kwargs
) -> AquaSOT3DTracker:
    """
    Factory function to build tracker from configuration.

    Args:
        tracker_type: Type of tracker to build
        **kwargs: Configuration parameters for the tracker

    Returns:
        Tracker instance

    Example:
        >>> config = {
        ...     'rgb_backbone_config': {'type': 'vit_rgb', 'model_name': 'vit_base_patch16_224'},
        ...     'sonar_backbone_config': {'type': 'sonar', 'encoder_type': 'vit'},
        ...     'fusion_config': {'type': 'g2_dcma', 'output_dim': 768},
        ... }
        >>> tracker = build_tracker('AquaSOT3DTracker', **config)
    """
    if tracker_type in TRACKER_REGISTRY:
        return TRACKER_REGISTRY.get(tracker_type)(**kwargs)
    else:
        raise ValueError(
            f"Unknown tracker type: {tracker_type}. "
            f"Available: {list(TRACKER_REGISTRY.keys())}"
        )
