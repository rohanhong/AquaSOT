"""
AquaSOT-3D Tracker - Template-Search Multi-Modal Tracking System

This module implements the complete tracking pipeline integrating:
- Multi-modal backbones (RGB + Sonar) with OSTrack one-stream architecture
- G2-DCMA cross-modal fusion
- Multi-task prediction heads (2D boxes + 3D center)
- Geometric cyclic projection

Architecture:
    Input: RGB template/search + Sonar template/search
    -> Backbones: OSTrack one-stream ViT for RGB (3-channel) and Sonar (1-channel)
    -> Fusion: G2-DCMA (geometry-gated deformable cross-modal attention)
    -> Heads: BoxHead (RGB/Sonar/Fusion) + Weak3DHead + GeometricProjector
    Output: 2D boxes (RGB/Sonar/Fusion) + 3D center (X,Y,Z) + losses

Author: AquaSOT-3D Team
Date: 2025-11-21
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
    build_fusion_head,
    build_weak3d_head,
    extract_radial_activation
)
from aquasot3d.models.projector import build_geometric_projector
from aquasot3d.models.quality import (
    create_rgb_quality_estimator,
    create_sonar_quality_estimator,
    create_adaptive_weighting
)


logger = logging.getLogger(__name__)


@TRACKER_REGISTRY.register("AquaSOT3DTracker")
class AquaSOT3DTracker(BaseModule):
    """
    AquaSOT-3D Template-Search Tracker with Multi-Modal Fusion and 3D Estimation.

    This is the main tracking model that integrates:
    - OSTrack one-stream ViT backbones for RGB and Sonar
    - G2-DCMA cross-modal fusion
    - Multi-task prediction heads (2D boxes + 3D center)
    - Quality-aware gating for robust fusion

    Args:
        rgb_backbone_config: Configuration for RGB backbone (must include template_size=256, search_size=512)
        sonar_backbone_config: Configuration for Sonar backbone (must include template_size=256, search_size=512)
        fusion_config: Configuration for G2-DCMA fusion module
        box_head_config: Configuration for 2D box prediction heads
        fusion_head_config: Configuration for multi-modal fusion head
        weak3d_config: Configuration for weak 3D estimation head
        projector_config: Configuration for geometric projector
        quality_config: Configuration for quality estimators
        use_weak3d: Whether to use weak 3D estimation (default: True)
        use_quality_gating: Whether to use quality-aware fusion (default: True)
        confidence_threshold: Threshold for quality gating (default: 0.3)

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
        fusion_head_config: Optional[Dict[str, Any]] = None,
        weak3d_config: Optional[Dict[str, Any]] = None,
        projector_config: Optional[Dict[str, Any]] = None,
        quality_config: Optional[Dict[str, Any]] = None,
        use_weak3d: bool = True,
        use_quality_gating: bool = True,
        confidence_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__()

        self.use_weak3d = use_weak3d
        self.use_quality_gating = use_quality_gating
        self.confidence_threshold = confidence_threshold

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

        # Fusion box head (only if both modalities exist)
        self.fusion_box_head = None
        if self.backbone.rgb_backbone is not None and self.backbone.sonar_encoder is not None:
            self.fusion_box_head = build_box_head(
                'fusion',
                in_channels=self.fused_feat_dim,
                **box_head_config
            )
            logger.info("Initialized Fusion box head (multi-modal training)")
        else:
            logger.info("Fusion box head disabled (single-modal training)")

        # 4.2 Multi-Modal Fusion Head (for quality prediction)
        # Only build fusion_head if both modalities are available (multi-modal training)
        self.fusion_head = None
        if self.backbone.rgb_backbone is not None and self.backbone.sonar_encoder is not None:
            if fusion_head_config is None:
                fusion_head_config = {}

            self.fusion_head = build_fusion_head(
                rgb_feat_dim=self.rgb_feat_dim,
                sonar_feat_dim=self.sonar_feat_dim,
                **fusion_head_config
            )
            logger.info("Initialized fusion_head for multi-modal quality estimation")
        else:
            logger.info("Fusion head disabled (single-modal training)")

        # 4.3 Weak 3D Center Estimation (Optional)
        if self.use_weak3d:
            if weak3d_config is None:
                weak3d_config = {}

            self.weak3d_head = build_weak3d_head(
                rgb_feat_dim=self.rgb_feat_dim,
                sonar_feat_dim=self.sonar_feat_dim,
                **weak3d_config
            )

            # Geometric Cyclic Projector for 3D→2D projection
            if projector_config is None:
                projector_config = {}

            self.rgb_projector = build_geometric_projector('rgb', **projector_config)
            self.sonar_projector = build_geometric_projector('sonar', **projector_config)
        else:
            self.weak3d_head = None
            self.rgb_projector = None
            self.sonar_projector = None

        # =====================================================
        # 5. Quality Estimation (SNR/Sharpness-based)
        # =====================================================
        if self.use_quality_gating:
            if quality_config is None:
                quality_config = {}

            # RGB quality estimator (sharpness/blur detection)
            rgb_quality_cfg = quality_config.get('rgb', {}).copy()
            # Extract method/learnable to avoid duplicate kwargs
            rgb_method = rgb_quality_cfg.pop('method', 'combined')
            rgb_learnable = rgb_quality_cfg.pop('learnable', True)
            self.rgb_quality_estimator = create_rgb_quality_estimator(
                method=rgb_method,
                learnable=rgb_learnable,
                **rgb_quality_cfg
            )

            # Sonar quality estimator (SNR/speckle noise)
            sonar_quality_cfg = quality_config.get('sonar', {}).copy()
            # Extract method/learnable to avoid duplicate kwargs
            sonar_method = sonar_quality_cfg.pop('method', 'combined')
            sonar_learnable = sonar_quality_cfg.pop('learnable', True)
            self.sonar_quality_estimator = create_sonar_quality_estimator(
                method=sonar_method,
                learnable=sonar_learnable,
                **sonar_quality_cfg
            )

            # Adaptive weighting module
            adaptive_cfg = quality_config.get('adaptive_weighting', {}).copy()
            # Extract parameters to avoid duplicate kwargs
            method = adaptive_cfg.pop('method', 'softmax')
            quality_threshold = adaptive_cfg.pop('threshold', 0.3)
            self.adaptive_weighting = create_adaptive_weighting(
                weighting_method=method,
                quality_threshold=quality_threshold,
                **adaptive_cfg
            )

            # Projection layers for adaptive weighting (registered for state dict)
            self.quality_projection_adapters = nn.ModuleDict()

            logger.info("Initialized quality-adaptive gating: RGB + Sonar estimators + Adaptive weighting")
        else:
            self.rgb_quality_estimator = None
            self.sonar_quality_estimator = None
            self.adaptive_weighting = None
            self.quality_projection_adapters = None
            logger.info("Quality-adaptive gating disabled")

        # =====================================================
        # 6. Template Update State Management
        # =====================================================
        # Template-search paradigm has no internal state.
        # Template updates are managed externally via OnlineTracker.

        self.logger.info(
            f"Initialized AquaSOT3DTracker: "
            f"rgb_dim={self.rgb_feat_dim}, sonar_dim={self.sonar_feat_dim}, "
            f"fused_dim={self.fused_feat_dim}, "
            f"use_weak3d={use_weak3d}, use_quality_gating={use_quality_gating}"
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

    def detect_modality_by_confidence(
        self,
        rgb_confidence: Optional[torch.Tensor] = None,
        sonar_confidence: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect modality validity based on confidence thresholding.

        CRITICAL FIX (Audit 2025112101 Phase 4.1):
        Enables automatic zero-modality detection when prediction quality is low.
        Even if data exists, low confidence indicates unreliable predictions.

        Args:
            rgb_confidence: [B] confidence scores from RGB box head (0-1 range)
            sonar_confidence: [B] confidence scores from Sonar box head (0-1 range)
            threshold: Confidence threshold (default: self.confidence_threshold)

        Returns:
            Tuple of boolean tensors:
            - is_rgb_valid: [B] True if RGB confidence >= threshold
            - is_sonar_valid: [B] True if Sonar confidence >= threshold

        Example:
            >>> rgb_conf = torch.tensor([0.8, 0.2, 0.5, 0.1])  # threshold=0.3
            >>> sonar_conf = torch.tensor([0.9, 0.4, 0.2, 0.05])
            >>> is_rgb, is_sonar = tracker.detect_modality_by_confidence(rgb_conf, sonar_conf)
            >>> # is_rgb = [True, False, True, False]
            >>> # is_sonar = [True, True, False, False]
        """
        if threshold is None:
            threshold = self.confidence_threshold

        # Default to all valid if confidence not provided
        device = rgb_confidence.device if rgb_confidence is not None else 'cpu'
        batch_size = rgb_confidence.shape[0] if rgb_confidence is not None else 1

        if rgb_confidence is None:
            is_rgb_valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            is_rgb_valid = rgb_confidence >= threshold

        if sonar_confidence is None:
            is_sonar_valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            is_sonar_valid = sonar_confidence >= threshold

        # Log confidence-based filtering
        if self.logger.isEnabledFor(logging.DEBUG):
            if rgb_confidence is not None:
                n_rgb_filtered = (~is_rgb_valid).sum().item()
                self.logger.debug(
                    f"RGB confidence filtering: {n_rgb_filtered}/{batch_size} samples below threshold {threshold:.2f}"
                )
            if sonar_confidence is not None:
                n_sonar_filtered = (~is_sonar_valid).sum().item()
                self.logger.debug(
                    f"Sonar confidence filtering: {n_sonar_filtered}/{batch_size} samples below threshold {threshold:.2f}"
                )

        return is_rgb_valid, is_sonar_valid

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
        is_sonar_valid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict 2D bounding boxes for all modalities using template-search paradigm.

        Args:
            rgb_features: RGB features, shape (B, C_rgb, H, W)
            sonar_features: Sonar features, shape (B, C_sonar, H, W)
            fused_features: Fused features, shape (B, C_fused, H, W)
            gt_boxes: Optional ground truth boxes for loss computation
            is_rgb_valid: Optional RGB validity flags, shape (B,)
            is_sonar_valid: Optional Sonar validity flags, shape (B,)

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

        # RGB box prediction
        if self.rgb_box_head is not None and rgb_features is not None:
            rgb_box_outputs = self.rgb_box_head(rgb_features)
            outputs['rgb_boxes'] = rgb_box_outputs['pred_boxes']
            outputs['rgb_quality'] = rgb_box_outputs['pred_scores']

            if gt_boxes is not None and 'rgb' in gt_boxes:
                rgb_loss = self.rgb_box_head.compute_loss(
                    rgb_box_outputs,
                    gt_boxes['rgb']
                )
                outputs['rgb_box_loss'] = rgb_loss

        # Sonar box prediction
        if self.sonar_box_head is not None and sonar_features is not None:
            sonar_box_outputs = self.sonar_box_head(sonar_features)
            outputs['sonar_boxes'] = sonar_box_outputs['pred_boxes']
            outputs['sonar_quality'] = sonar_box_outputs['pred_scores']

            if gt_boxes is not None and 'sonar' in gt_boxes:
                sonar_loss = self.sonar_box_head.compute_loss(
                    sonar_box_outputs,
                    gt_boxes['sonar']
                )
                outputs['sonar_box_loss'] = sonar_loss

        # Fusion box prediction
        if self.fusion_box_head is not None and fused_features is not None:
            fusion_box_outputs = self.fusion_box_head(fused_features)
            outputs['fusion_boxes'] = fusion_box_outputs['pred_boxes']
            outputs['fusion_quality'] = fusion_box_outputs['pred_scores']

            if gt_boxes is not None and 'fusion' in gt_boxes:
                fusion_loss = self.fusion_box_head.compute_loss(
                    fusion_box_outputs,
                    gt_boxes['fusion']
                )
                outputs['fusion_box_loss'] = fusion_loss

        # CRITICAL FIX (Audit 2025112101 Phase 4.2):
        # Automatic modality detection from confidence scores for evaluation
        # If modality validity not provided, detect from box prediction confidence
        if is_rgb_valid is None or is_sonar_valid is None:
            rgb_conf = outputs.get('rgb_quality', None)  # [B] confidence scores
            sonar_conf = outputs.get('sonar_quality', None)

            # Auto-detect validity from confidence thresholding
            is_rgb_valid_computed, is_sonar_valid_computed = self.detect_modality_by_confidence(
                rgb_confidence=rgb_conf,
                sonar_confidence=sonar_conf
            )

            # Use computed validity if not provided
            if is_rgb_valid is None:
                is_rgb_valid = is_rgb_valid_computed
            if is_sonar_valid is None:
                is_sonar_valid = is_sonar_valid_computed

        # Always output modality validity for evaluation/analysis
        outputs['is_rgb_valid'] = is_rgb_valid
        outputs['is_sonar_valid'] = is_sonar_valid

        return outputs

    def predict_3d_center(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        sonar_radial_activation: torch.Tensor,
        gt_boxes: Optional[Dict[str, torch.Tensor]] = None,
        compute_cyclic_loss: bool = True,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict weak 3D center and compute cyclic consistency loss.

        CRITICAL FIX (Audit 2025111412 Finding 2):
        Accepts metadata for calibrated geometry. Projectors receive scale/padding
        info to adjust log-polar assumptions for resized data, ensuring 3D projections
        match true crop scale across sensor modalities.

        Args:
            rgb_features: RGB features
            sonar_features: Sonar features
            sonar_radial_activation: Sonar radial activation profile
            gt_boxes: Optional ground truth boxes for cyclic loss
            compute_cyclic_loss: Whether to compute cyclic consistency loss
            metadata: Optional dict with template/search metadata for sensor calibration

        Returns:
            Dictionary containing 3D predictions and losses
        """
        if not self.use_weak3d:
            return {}

        # Ensure features are tensors (extract from list if needed)
        if isinstance(rgb_features, list):
            rgb_features = rgb_features[0]
        if isinstance(sonar_features, list):
            sonar_features = sonar_features[0]

        # Handle ViT sequence features: Pool (B, N, C) -> (B, C) for weak3d_head
        # weak3d_head expects pooled features [B, C], not sequence features [B, N, C]
        if rgb_features.dim() == 3:
            # Global average pooling over sequence dimension
            rgb_features = rgb_features.mean(dim=1)  # [B, N, C] -> [B, C]
        if sonar_features.dim() == 3:
            sonar_features = sonar_features.mean(dim=1)  # [B, N, C] -> [B, C]

        # Predict 3D center
        weak3d_outputs = self.weak3d_head(
            rgb_features=rgb_features,
            sonar_features=sonar_features,
            sonar_radial_activation=sonar_radial_activation,
            return_intermediate=False
        )

        center_3d = weak3d_outputs['center_3d']  # (B, 3) - (X, Y, Z)

        outputs = {
            'center_3d': center_3d,
            'X': weak3d_outputs['X'],
            'Y': weak3d_outputs['Y'],
            'Z': weak3d_outputs['Z'],
            'routing_info': weak3d_outputs.get('routing_info', None)
        }

        # Compute cyclic consistency loss if ground truth boxes are provided
        if compute_cyclic_loss and gt_boxes is not None:
            # CRITICAL FIX (Audit 2025111413 Finding 2):
            # Projectors now receive metadata for calibrated geometry, reconciling
            # resized crops with true sensor geometry for accurate cyclic losses

            # Extract relevant metadata for projectors
            rgb_metadata = metadata.get('rgb_search_metadata') if metadata else None
            sonar_metadata = metadata.get('sonar_search_metadata') if metadata else None

            # Project 3D → 2D with metadata-aware geometry
            # CRITICAL FIX (Audit 2025111413 Finding 2): Pass metadata to projectors
            rgb_proj_outputs = self.rgb_projector(
                center_3d,
                compute_cyclic_loss=True,
                metadata=rgb_metadata
            )
            sonar_proj_outputs = self.sonar_projector(
                center_3d,
                compute_cyclic_loss=True,
                metadata=sonar_metadata
            )

            outputs['rgb_proj_boxes'] = rgb_proj_outputs['bbox_2d']
            outputs['sonar_proj_boxes'] = sonar_proj_outputs['bbox_2d']
            outputs['rgb_cyclic_loss'] = rgb_proj_outputs['cyclic_loss']
            outputs['sonar_cyclic_loss'] = sonar_proj_outputs['cyclic_loss']

            # Compute projection loss against ground truth
            if 'rgb' in gt_boxes:
                rgb_proj_loss = self.rgb_projector.compute_projection_loss(
                    rgb_proj_outputs['bbox_2d'],
                    gt_boxes['rgb'],
                    loss_type='l1'
                )
                outputs['rgb_proj_loss'] = rgb_proj_loss

            if 'sonar' in gt_boxes:
                sonar_proj_loss = self.sonar_projector.compute_projection_loss(
                    sonar_proj_outputs['bbox_2d'],
                    gt_boxes['sonar'],
                    loss_type='l1'
                )
                outputs['sonar_proj_loss'] = sonar_proj_loss

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
        return_losses: bool = True,
        return_intermediate: bool = False,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass of AquaSOT-3D tracker using template-search paradigm.

        CRITICAL FIX (Audit 2025112101 Phase 2.2):
        Accepts is_rgb_valid and is_sonar_valid for conditional fusion.
        These flags enable single-modality bypass in ConditionalG2DCMA.

        Args:
            rgb_template: RGB template frame, shape (B, 3, 256, 256)
            rgb_search: RGB search frame, shape (B, 3, 512, 512)
            sonar_template: Sonar template frame, shape (B, 1, 256, 256)
            sonar_search: Sonar search frame, shape (B, 1, 512, 512)
            gt_boxes: Optional ground truth boxes for loss computation
                Expected format: {'rgb': (B, 4), 'sonar': (B, 4), 'fusion': (B, 4)}
            is_rgb_valid: Optional [B] boolean tensor, True if RGB has valid target
            is_sonar_valid: Optional [B] boolean tensor, True if Sonar has valid target
            return_losses: Whether to compute and return losses
            return_intermediate: Whether to return intermediate features
            metadata: Optional metadata dict containing:
                - 'rgb_search_metadata': List of dicts with scale/padding info
                - 'sonar_search_metadata': List of dicts with scale/padding info
                (CRITICAL FIX for Audit 2025111411 Finding 4)

        Returns:
            Dictionary containing:
                - 'rgb_boxes': RGB box predictions (B, 4)
                - 'sonar_boxes': Sonar box predictions (B, 4)
                - 'fusion_boxes': Fusion box predictions (B, 4) [PRIMARY OUTPUT]
                - 'center_3d': 3D center (X, Y, Z) (B, 3) [if use_weak3d=True]
                - Various losses [if return_losses=True]
                - Intermediate features [if return_intermediate=True]
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

        fused_features = fusion_outputs['fused_features']  # (B, C_fused, H, W)

        # =====================================================
        # Stage 3: Quality Estimation (Physical + Learned)
        # =====================================================
        B = fused_features.shape[0]

        # 3.1 Physical quality estimation (SNR/Sharpness)
        # CRITICAL FIX (Audit 2025111408 Finding 3):
        # Quality estimators need pixel coordinates, not normalized coordinates
        if self.use_quality_gating and self.rgb_quality_estimator is not None:
            # Get pixel bboxes if available, otherwise use None (full frame quality)
            rgb_bbox_pixels = gt_boxes.get('rgb_pixels') if gt_boxes is not None else None
            sonar_bbox_pixels = gt_boxes.get('sonar_pixels') if gt_boxes is not None else None

            # CRITICAL FIX (Audit 2025111411 Finding 4):
            # Extract per-sample metadata for padding-aware quality estimation
            rgb_search_metadata = metadata.get('rgb_search_metadata') if metadata else None
            sonar_search_metadata = metadata.get('sonar_search_metadata') if metadata else None

            # Estimate RGB quality based on sharpness/blur
            rgb_quality = self.rgb_quality_estimator(
                rgb_frame=rgb_search,
                bbox=rgb_bbox_pixels,
                metadata=rgb_search_metadata
            )  # (B, 1)

            # Estimate Sonar quality based on SNR/speckle
            sonar_quality = self.sonar_quality_estimator(
                sonar_frame=sonar_search,
                bbox=sonar_bbox_pixels,
                metadata=sonar_search_metadata
            )  # (B, 1)
        else:
            # No quality gating: use default high quality scores
            rgb_quality = torch.ones(B, 1, device=fused_features.device)
            sonar_quality = torch.ones(B, 1, device=fused_features.device)

        # 3.2 Learned quality estimation (from fusion_head)
        quality_fusion_loss = None
        if self.fusion_head is not None:
            rgb_features_for_quality = rgb_features
            sonar_features_for_quality = sonar_features
            if rgb_features_for_quality is not None and isinstance(rgb_features_for_quality, list):
                rgb_features_for_quality = rgb_features_for_quality[0]
            if sonar_features_for_quality is not None and isinstance(sonar_features_for_quality, list):
                sonar_features_for_quality = sonar_features_for_quality[0]

            # CRITICAL FIX (Audit 2025111412 Finding 1):
            # Pass metadata to fusion_head for physical alignment awareness
            quality_outputs = self.fusion_head(
                rgb_features_for_quality, sonar_features_for_quality,
                metadata=metadata
            )
            fusion_quality_learned = quality_outputs['quality_score']  # (B,)

            # Combine physical and learned quality scores
            # Use learned quality as primary, physical quality as regularization
            fusion_quality = fusion_quality_learned

            # Compute quality-consistency loss during training
            # Aligns learned quality with physical quality (SNR/sharpness)
            if return_losses and self.use_quality_gating:
                # FusionHead returns fusion_weights directly (already normalized)
                # It also optionally returns rgb_confidence/sonar_confidence
                fusion_weights_learned = quality_outputs.get('fusion_weights')  # (B, 2)

                if fusion_weights_learned is not None:
                    # Use fusion_weights as proxy for learned quality
                    # fusion_weights[:, 0] = RGB weight (higher when RGB is better)
                    # fusion_weights[:, 1] = Sonar weight (higher when Sonar is better)

                    # Compute quality-consistency loss
                    # Supervises learned fusion weights with physical quality (SNR/sharpness)
                    quality_fusion_loss = self.fusion_head.compute_fusion_loss(
                        rgb_quality=rgb_quality.squeeze(1),     # Physical quality (sharpness)
                        sonar_quality=sonar_quality.squeeze(1), # Physical quality (SNR)
                        fusion_weights=fusion_weights_learned,   # Learned weights from FusionHead
                        target_quality=None  # Use agreement-based loss (no ground-truth quality)
                    )
        else:
            # Single-modal training: use physical quality or default
            if self.use_quality_gating:
                # Average RGB and Sonar quality
                fusion_quality = (rgb_quality.squeeze(1) + sonar_quality.squeeze(1)) / 2.0
            else:
                fusion_quality = torch.ones(B, device=fused_features.device)

        # 3.3 Apply adaptive quality-based reweighting to fused features
        # This modulates the fused output based on physical quality cues (SNR/sharpness)
        quality_weights_dict = None
        if self.use_quality_gating and self.adaptive_weighting is not None:
            # Reshape features if needed for adaptive weighting
            # AdaptiveWeighting expects (B, C, H, W) spatial features
            if fused_features.dim() == 3:
                # Sequence format (B, N, C) -> (B, C, H, W)
                B_seq, N_seq, C_seq = fused_features.shape
                H_seq = W_seq = int(N_seq ** 0.5)
                fused_features_spatial = fused_features.permute(0, 2, 1).reshape(B_seq, C_seq, H_seq, W_seq)
            else:
                fused_features_spatial = fused_features

            # Prepare modality-specific features for reweighting
            # Use raw backbone features (before G2-DCMA fusion) for quality-based modulation
            rgb_feat_for_adaptive = rgb_features if rgb_features is not None else fused_features_spatial
            sonar_feat_for_adaptive = sonar_features if sonar_features is not None else fused_features_spatial

            # Handle list format
            if isinstance(rgb_feat_for_adaptive, list):
                rgb_feat_for_adaptive = rgb_feat_for_adaptive[0]
            if isinstance(sonar_feat_for_adaptive, list):
                sonar_feat_for_adaptive = sonar_feat_for_adaptive[0]

            # Reshape if needed
            if rgb_feat_for_adaptive.dim() == 3:
                B_rgb, N_rgb, C_rgb = rgb_feat_for_adaptive.shape
                # Check if class token is present (ViT outputs [B, N+1, C] with cls_token)
                if int((N_rgb - 1) ** 0.5) ** 2 == N_rgb - 1:
                    # Remove cls_token (first token)
                    rgb_feat_for_adaptive = rgb_feat_for_adaptive[:, 1:, :]
                    N_rgb = N_rgb - 1
                H_rgb = W_rgb = int(N_rgb ** 0.5)
                rgb_feat_for_adaptive = rgb_feat_for_adaptive.permute(0, 2, 1).reshape(B_rgb, C_rgb, H_rgb, W_rgb)
            if sonar_feat_for_adaptive.dim() == 3:
                B_sonar, N_sonar, C_sonar = sonar_feat_for_adaptive.shape
                # Check if class token is present
                if int((N_sonar - 1) ** 0.5) ** 2 == N_sonar - 1:
                    # Remove cls_token (first token)
                    sonar_feat_for_adaptive = sonar_feat_for_adaptive[:, 1:, :]
                    N_sonar = N_sonar - 1
                H_sonar = W_sonar = int(N_sonar ** 0.5)
                sonar_feat_for_adaptive = sonar_feat_for_adaptive.permute(0, 2, 1).reshape(B_sonar, C_sonar, H_sonar, W_sonar)

            # Resize features to match if needed (for quality-based fusion)
            if rgb_feat_for_adaptive.shape[2:] != sonar_feat_for_adaptive.shape[2:]:
                target_size = fused_features_spatial.shape[2:]
                rgb_feat_for_adaptive = F.interpolate(rgb_feat_for_adaptive, size=target_size, mode='bilinear', align_corners=False)
                sonar_feat_for_adaptive = F.interpolate(sonar_feat_for_adaptive, size=target_size, mode='bilinear', align_corners=False)

            # Match channel dimensions to fused_features using registered adapters
            if rgb_feat_for_adaptive.shape[1] != fused_features_spatial.shape[1]:
                # Get or create RGB projection adapter
                proj_key = f'rgb_{rgb_feat_for_adaptive.shape[1]}_to_{fused_features_spatial.shape[1]}'
                if proj_key not in self.quality_projection_adapters:
                    self.quality_projection_adapters[proj_key] = nn.Conv2d(
                        rgb_feat_for_adaptive.shape[1], fused_features_spatial.shape[1], kernel_size=1
                    ).to(rgb_feat_for_adaptive.device)
                rgb_feat_for_adaptive = self.quality_projection_adapters[proj_key](rgb_feat_for_adaptive)

            if sonar_feat_for_adaptive.shape[1] != fused_features_spatial.shape[1]:
                # Get or create Sonar projection adapter
                proj_key = f'sonar_{sonar_feat_for_adaptive.shape[1]}_to_{fused_features_spatial.shape[1]}'
                if proj_key not in self.quality_projection_adapters:
                    self.quality_projection_adapters[proj_key] = nn.Conv2d(
                        sonar_feat_for_adaptive.shape[1], fused_features_spatial.shape[1], kernel_size=1
                    ).to(sonar_feat_for_adaptive.device)
                sonar_feat_for_adaptive = self.quality_projection_adapters[proj_key](sonar_feat_for_adaptive)

            # Apply adaptive weighting based on physical quality
            fused_features_quality_weighted, quality_weights_dict = self.adaptive_weighting(
                rgb_feat=rgb_feat_for_adaptive,
                sonar_feat=sonar_feat_for_adaptive,
                q_rgb=rgb_quality,
                q_sonar=sonar_quality
            )

            # Blend quality-weighted features with geometric-fused features
            # Quality-weighted: emphasizes modality with better physical quality
            # Geometric-fused: emphasizes geometric consistency
            # Combine both: alpha * geometric + (1-alpha) * quality
            blend_alpha = 0.7  # Favor geometric fusion, but modulate with quality
            fused_features = blend_alpha * fused_features_spatial + (1 - blend_alpha) * fused_features_quality_weighted

            # Restore original shape if needed
            if fused_features.dim() == 4 and fused_features_spatial.shape != fused_features.shape:
                # Was sequence format originally, restore it
                pass  # Keep spatial for now, heads expect spatial

        # =====================================================
        # Stage 4: 2D Box Prediction
        # =====================================================
        box_outputs = self.predict_boxes(
            rgb_features, sonar_features, fused_features,
            gt_boxes=gt_boxes if return_losses else None,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid
        )

        # =====================================================
        # Stage 6: 3D Center Prediction (Optional)
        # =====================================================
        if self.use_weak3d:
            # Extract sonar radial activation for depth proxy
            # Ensure sonar_features is a tensor
            sonar_feat_for_3d = sonar_features
            if isinstance(sonar_feat_for_3d, list):
                sonar_feat_for_3d = sonar_feat_for_3d[0]

            # Handle ViT sequence features: reshape (B, N, C) -> (B, C, H, W) for weak3d
            if sonar_feat_for_3d.dim() == 3:
                B, N, C = sonar_feat_for_3d.shape
                # Check if class token is present (N = H*W + 1)
                if int((N - 1) ** 0.5) ** 2 == N - 1:
                    # Has class token, remove it
                    sonar_feat_for_3d = sonar_feat_for_3d[:, 1:, :]  # Remove class token
                    N = N - 1
                # Assume square spatial layout
                H = W = int(N ** 0.5)
                sonar_feat_for_3d = sonar_feat_for_3d.permute(0, 2, 1).reshape(B, C, H, W)

            sonar_radial = extract_radial_activation(
                sonar_features=sonar_feat_for_3d,
                num_radial_bins=50
            )

            # CRITICAL FIX (Audit 2025111412 Finding 2):
            # Pass metadata to weak-3D for calibrated geometry
            weak3d_outputs = self.predict_3d_center(
                rgb_features, sonar_features, sonar_radial,
                gt_boxes=gt_boxes if return_losses else None,
                compute_cyclic_loss=return_losses,
                metadata=metadata
            )
        else:
            weak3d_outputs = {}

        # =====================================================
        # Stage 7: Output Organization
        # =====================================================
        outputs = {
            # Quality scores (always present)
            'fusion_quality': fusion_quality,
            'rgb_quality_physical': rgb_quality.squeeze(1) if self.use_quality_gating else None,
            'sonar_quality_physical': sonar_quality.squeeze(1) if self.use_quality_gating else None,
        }

        # Add adaptive quality weights if computed
        if quality_weights_dict is not None:
            outputs['quality_weights'] = quality_weights_dict

        # Add predictions conditionally based on available modalities
        if 'fusion_boxes' in box_outputs:
            outputs['fusion_boxes'] = box_outputs['fusion_boxes']
        if 'rgb_boxes' in box_outputs:
            outputs['rgb_boxes'] = box_outputs['rgb_boxes']
            outputs['rgb_quality'] = box_outputs.get('rgb_quality', None)
        if 'sonar_boxes' in box_outputs:
            outputs['sonar_boxes'] = box_outputs['sonar_boxes']
            outputs['sonar_quality'] = box_outputs.get('sonar_quality', None)

        # Add 3D predictions if available
        if 'center_3d' in weak3d_outputs:
            outputs['center_3d'] = weak3d_outputs['center_3d']

        # Add losses if requested
        if return_losses:
            losses = {}

            # 2D box losses
            if 'rgb_box_loss' in box_outputs:
                losses['rgb_box_loss'] = box_outputs['rgb_box_loss']
            if 'sonar_box_loss' in box_outputs:
                losses['sonar_box_loss'] = box_outputs['sonar_box_loss']
            if 'fusion_box_loss' in box_outputs:
                losses['fusion_box_loss'] = box_outputs['fusion_box_loss']

            # Consistency losses
            if 'consistency_losses' in fusion_outputs:
                # Geometric consistency: compares geometric encodings (scale vs radial)
                losses['geometric_consistency'] = fusion_outputs['consistency_losses']['component_loss']
                losses['representation_consistency'] = fusion_outputs['consistency_losses']['infonce_loss']

            # 3D losses
            if 'X_consistency_loss' in weak3d_outputs:
                # Component consistency: true X coordinate agreement between RGB and Sonar
                losses['component_consistency'] = weak3d_outputs['X_consistency_loss']
            if 'rgb_cyclic_loss' in weak3d_outputs:
                losses['rgb_cyclic_loss'] = weak3d_outputs['rgb_cyclic_loss']
            if 'sonar_cyclic_loss' in weak3d_outputs:
                losses['sonar_cyclic_loss'] = weak3d_outputs['sonar_cyclic_loss']
            if 'rgb_proj_loss' in weak3d_outputs:
                losses['rgb_proj_loss'] = weak3d_outputs['rgb_proj_loss']
            if 'sonar_proj_loss' in weak3d_outputs:
                losses['sonar_proj_loss'] = weak3d_outputs['sonar_proj_loss']

            # Quality-consistency loss (learned quality vs physical quality)
            if quality_fusion_loss is not None:
                losses['quality_fusion_loss'] = quality_fusion_loss

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
            if 'X_rgb' in weak3d_outputs:
                outputs['intermediate']['X_rgb'] = weak3d_outputs['X_rgb']
                outputs['intermediate']['X_sonar'] = weak3d_outputs['X_sonar']

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
