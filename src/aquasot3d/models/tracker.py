"""
AquaSOT-3D Tracker - Complete End-to-End Multi-Modal Tracking System

This module implements the complete tracking pipeline integrating:
- Multi-modal backbones (RGB + Sonar)
- G2-DCMA cross-modal fusion
- Temporal modeling (Mamba/Transformer/ARTrack)
- Multi-task prediction heads (2D boxes + 3D center)
- Geometric cyclic projection

Architecture:
    Input: RGB template/search + Sonar template/search
    -> Backbones: ViT/ConvNeXt (RGB) + SonarEncoder (Sonar)
    -> Fusion: G2-DCMA (geometry-gated deformable cross-modal attention)
    -> Temporal: Vision-Mamba / Transformer / ARTrack
    -> Heads: BoxHead (RGB/Sonar/Fusion) + Weak3DHead + GeometricProjector
    Output: 2D boxes (RGB/Sonar/Fusion) + 3D center (X,Y,Z) + losses

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import logging

from aquasot3d.core import BaseModule, TRACKER_REGISTRY
from aquasot3d.models.backbones import BackboneFactory, MultiModalBackbone
from aquasot3d.models.fusion import build_g2_dcma, build_g2_dcma_lite
from aquasot3d.models.temporal import (
    build_vision_mamba_head,
    build_transformer_head,
    build_artrack_head,
    build_temporal_aggregator
)
from aquasot3d.models.heads import (
    build_box_head,
    build_fusion_head,
    build_weak3d_head,
    extract_radial_activation
)
from aquasot3d.models.projector import build_geometric_projector


logger = logging.getLogger(__name__)


@TRACKER_REGISTRY.register("AquaSOT3DTracker")
class AquaSOT3DTracker(BaseModule):
    """
    Complete AquaSOT-3D tracking system with multi-modal fusion and 3D estimation.

    This is the main tracking model that integrates all components for end-to-end
    training and inference. It follows the OSTrack template-search paradigm and
    adds multi-modal fusion, temporal modeling, and weak 3D estimation.

    Args:
        rgb_backbone_config: Configuration for RGB backbone
        sonar_backbone_config: Configuration for Sonar backbone
        fusion_config: Configuration for G2-DCMA fusion module
        temporal_config: Configuration for temporal head
        box_head_config: Configuration for 2D box prediction heads
        fusion_head_config: Configuration for multi-modal fusion head
        weak3d_config: Configuration for weak 3D estimation head
        projector_config: Configuration for geometric projector
        use_temporal: Whether to use temporal modeling (default: True)
        use_weak3d: Whether to use weak 3D estimation (default: True)
        temporal_aggregation: Whether to use temporal aggregation (default: False)

    Example:
        >>> config = {
        ...     'rgb_backbone': {'type': 'vit_rgb', 'model_name': 'vit_base_patch16_224'},
        ...     'sonar_backbone': {'type': 'sonar', 'encoder_type': 'vit'},
        ...     'fusion': {'type': 'g2_dcma', 'rgb_dim': 768, 'sonar_dim': 384},
        ...     'temporal': {'type': 'mamba', 'feature_dim': 768},
        ... }
        >>> tracker = build_tracker('AquaSOT3DTracker', **config)
        >>> outputs = tracker(rgb_template, rgb_search, sonar_template, sonar_search)
    """

    def __init__(
        self,
        rgb_backbone_config: Dict[str, Any],
        sonar_backbone_config: Dict[str, Any],
        fusion_config: Dict[str, Any],
        temporal_config: Optional[Dict[str, Any]] = None,
        box_head_config: Optional[Dict[str, Any]] = None,
        fusion_head_config: Optional[Dict[str, Any]] = None,
        weak3d_config: Optional[Dict[str, Any]] = None,
        projector_config: Optional[Dict[str, Any]] = None,
        use_temporal: bool = True,
        use_weak3d: bool = True,
        temporal_aggregation: bool = False,
        **kwargs
    ):
        super().__init__()

        self.use_temporal = use_temporal
        self.use_weak3d = use_weak3d
        self.temporal_aggregation = temporal_aggregation

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
        # 3. Temporal Modeling (Optional)
        # =====================================================
        self.temporal_head = None
        self.temporal_aggregator = None

        if self.use_temporal and temporal_config is not None and temporal_config.get('enabled', True):
            # Make a copy to avoid modifying the original config (high-generalization fix)
            temporal_config_copy = temporal_config.copy()
            temporal_type = temporal_config_copy.pop('type', 'mamba')
            # Remove control parameters and feature_dim (we override it)
            temporal_config_copy.pop('enabled', None)
            temporal_config_copy.pop('feature_dim', None)  # Remove to avoid conflict

            if temporal_type == 'mamba':
                self.temporal_head = build_vision_mamba_head(
                    feature_dim=self.fused_feat_dim,
                    **temporal_config_copy
                )
            elif temporal_type == 'transformer':
                self.temporal_head = build_transformer_head(
                    feature_dim=self.fused_feat_dim,
                    **temporal_config_copy
                )
            elif temporal_type == 'artrack':
                self.temporal_head = build_artrack_head(
                    feature_dim=self.fused_feat_dim,
                    **temporal_config_copy
                )
            else:
                raise ValueError(f"Unknown temporal type: {temporal_type}")

            logger.info(f"Initialized temporal head: {temporal_type}")
        else:
            logger.info("Temporal modeling disabled")

            # Optional temporal aggregation
            if self.temporal_aggregation:
                self.temporal_aggregator = build_temporal_aggregator(
                    feature_dim=self.fused_feat_dim,
                    **temporal_config.get('aggregator', {})
                )

        # =====================================================
        # 4. Prediction Heads
        # =====================================================

        # 4.1 2D Box Prediction Heads
        if box_head_config is None:
            box_head_config = {}

        self.rgb_box_head = build_box_head(
            'rgb',
            in_channels=self.rgb_feat_dim,
            **box_head_config
        )

        self.sonar_box_head = build_box_head(
            'sonar',
            in_channels=self.sonar_feat_dim,
            **box_head_config
        )

        self.fusion_box_head = build_box_head(
            'fusion',
            in_channels=self.fused_feat_dim,
            **box_head_config
        )

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
        # 5. Temporal State Management
        # =====================================================
        self.temporal_state = None
        self.reset_temporal_state()

        self.logger.info(
            f"Initialized AquaSOT3DTracker: "
            f"rgb_dim={self.rgb_feat_dim}, sonar_dim={self.sonar_feat_dim}, "
            f"fused_dim={self.fused_feat_dim}, use_temporal={use_temporal}, "
            f"use_weak3d={use_weak3d}"
        )

    def reset_temporal_state(self) -> None:
        """
        Reset temporal state for new sequence.

        Call this when starting tracking on a new video sequence.
        """
        self.temporal_state = None
        if self.temporal_head is not None and hasattr(self.temporal_head, 'reset_state'):
            self.temporal_head.reset_state()
        if self.temporal_aggregator is not None and hasattr(self.temporal_aggregator, 'reset'):
            self.temporal_aggregator.reset()

    def extract_backbone_features(
        self,
        rgb_template: torch.Tensor,
        rgb_search: torch.Tensor,
        sonar_template: torch.Tensor,
        sonar_search: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from RGB and Sonar backbones.

        Args:
            rgb_template: RGB template frame, shape (B, 3, H_t, W_t)
            rgb_search: RGB search frame, shape (B, 3, H_s, W_s)
            sonar_template: Sonar template frame, shape (B, 1, H_t, W_t)
            sonar_search: Sonar search frame, shape (B, 1, H_s, W_s)

        Returns:
            Dictionary containing extracted features
        """
        # Extract multi-modal features
        backbone_outputs = self.backbone(
            rgb_template=rgb_template,
            rgb_search=rgb_search,
            sonar_template=sonar_template,
            sonar_search=sonar_search,
            return_separate=True
        )

        # Extract features from nested structure (handle None for single-modal training)
        # backbone_outputs = {'rgb': {...} or None, 'sonar': {...} or None}
        rgb_features = backbone_outputs['rgb']['search_features'] if backbone_outputs['rgb'] is not None else None
        sonar_features = backbone_outputs['sonar']['search_features'] if backbone_outputs['sonar'] is not None else None

        return {
            'rgb_features': rgb_features,
            'sonar_features': sonar_features,
        }

    def fuse_features(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse RGB and Sonar features using G2-DCMA.

        Args:
            rgb_features: RGB features, shape (B, C_rgb, H, W) or (B, N, C_rgb)
            sonar_features: Sonar features, shape (B, C_sonar, H, W)
            return_consistency: Whether to compute consistency losses

        Returns:
            Dictionary containing fused features and losses
        """
        # Handle single-modal case (fusion module disabled)
        if self.fusion is None:
            # Return available modality features as fused features
            if rgb_features is not None:
                # Reshape RGB features if needed
                if rgb_features.dim() == 3:
                    B, N, C = rgb_features.shape
                    # Check if class token is present
                    if int((N - 1) ** 0.5) ** 2 == N - 1:
                        rgb_features = rgb_features[:, 1:, :]
                        N = N - 1
                    H = W = int(N ** 0.5)
                    rgb_features = rgb_features.permute(0, 2, 1).reshape(B, C, H, W)
                return {'fused_features': rgb_features}
            elif sonar_features is not None:
                if isinstance(sonar_features, list):
                    sonar_features = sonar_features[0]
                # Reshape Sonar features if needed (ViT outputs tokens)
                if sonar_features.dim() == 3:
                    B, N, C = sonar_features.shape
                    # Check if class token is present (N = H*W + 1)
                    if int((N - 1) ** 0.5) ** 2 == N - 1:
                        sonar_features = sonar_features[:, 1:, :]
                        N = N - 1
                    H = W = int(N ** 0.5)
                    sonar_features = sonar_features.permute(0, 2, 1).reshape(B, C, H, W)
                return {'fused_features': sonar_features}
            else:
                raise ValueError("At least one modality must be available")

        # Handle ViT sequence features: reshape (B, N, C) -> (B, C, H, W)
        if rgb_features is not None and rgb_features.dim() == 3:
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
        if sonar_features is not None:
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

        fusion_outputs = self.fusion(
            rgb_features=rgb_features,
            sonar_features=sonar_features,
            return_consistency=return_consistency
        )

        return fusion_outputs

    def apply_temporal_modeling(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Apply temporal modeling to features.

        Args:
            features: Input features, shape (B, C, H, W) or (B, T, C, H, W)
            quality_scores: Optional quality scores, shape (B,) or (B, T)

        Returns:
            Tuple of (aggregated_features, new_temporal_state)
        """
        if self.temporal_head is None:
            return features, None

        # Apply temporal head
        aggregated_features, new_state = self.temporal_head(
            features=features,
            quality_scores=quality_scores,
            temporal_state=self.temporal_state
        )

        # Apply temporal aggregation if enabled
        if self.temporal_aggregation and self.temporal_aggregator is not None:
            aggregated_features = self.temporal_aggregator(
                current_features=aggregated_features,
                quality_score=quality_scores
            )['aggregated_features']

        return aggregated_features, new_state

    def predict_boxes(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        fused_features: torch.Tensor,
        gt_boxes: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict 2D bounding boxes for all modalities.

        Args:
            rgb_features: RGB features, shape (B, C_rgb, H, W)
            sonar_features: Sonar features, shape (B, C_sonar, H, W)
            fused_features: Fused features, shape (B, C_fused, H, W)
            gt_boxes: Optional ground truth boxes for loss computation

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

        return outputs

    def predict_3d_center(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        sonar_radial_activation: torch.Tensor,
        gt_boxes: Optional[Dict[str, torch.Tensor]] = None,
        compute_cyclic_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Predict weak 3D center and compute cyclic consistency loss.

        Args:
            rgb_features: RGB features
            sonar_features: Sonar features
            sonar_radial_activation: Sonar radial activation profile
            gt_boxes: Optional ground truth boxes for cyclic loss
            compute_cyclic_loss: Whether to compute cyclic consistency loss

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

        # Predict 3D center
        weak3d_outputs = self.weak3d_head(
            rgb_features=rgb_features,
            sonar_features=sonar_features,
            sonar_radial_activation=sonar_radial_activation,
            compute_consistency_loss=True
        )

        center_3d = weak3d_outputs['center_3d']  # (B, 3) - (X, Y, Z)

        outputs = {
            'center_3d': center_3d,
            'X_rgb': weak3d_outputs['X_rgb'],
            'X_sonar': weak3d_outputs['X_sonar'],
            'X_consistency_loss': weak3d_outputs['consistency_loss']
        }

        # Compute cyclic consistency loss if ground truth boxes are provided
        if compute_cyclic_loss and gt_boxes is not None:
            # Project 3D → 2D
            rgb_proj_outputs = self.rgb_projector(
                center_3d,
                compute_cyclic_loss=True
            )
            sonar_proj_outputs = self.sonar_projector(
                center_3d,
                compute_cyclic_loss=True
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
        return_losses: bool = True,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass of AquaSOT-3D tracker.

        Args:
            rgb_template: RGB template frame, shape (B, 3, H_t, W_t)
            rgb_search: RGB search frame, shape (B, 3, H_s, W_s)
            sonar_template: Sonar template frame, shape (B, 1, H_t, W_t)
            sonar_search: Sonar search frame, shape (B, 1, H_s, W_s)
            gt_boxes: Optional ground truth boxes for loss computation
                Expected format: {'rgb': (B, 4), 'sonar': (B, 4), 'fusion': (B, 4)}
            return_losses: Whether to compute and return losses
            return_intermediate: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - 'rgb_boxes': RGB box predictions (B, 4)
                - 'sonar_boxes': Sonar box predictions (B, 4)
                - 'fusion_boxes': Fusion box predictions (B, 4) [PRIMARY OUTPUT]
                - 'center_3d': 3D center (X, Y, Z) (B, 3) [if use_weak3d=True]
                - Various losses [if return_losses=True]
                - Intermediate features [if return_intermediate=True]
        """
        # =====================================================
        # Stage 1: Feature Extraction
        # =====================================================
        backbone_outputs = self.extract_backbone_features(
            rgb_template, rgb_search, sonar_template, sonar_search
        )

        rgb_features = backbone_outputs['rgb_features']  # (B, C_rgb, H, W)
        sonar_features = backbone_outputs['sonar_features']  # (B, C_sonar, H, W)

        # =====================================================
        # Stage 2: Cross-Modal Fusion
        # =====================================================
        fusion_outputs = self.fuse_features(
            rgb_features, sonar_features,
            return_consistency=return_losses
        )

        fused_features = fusion_outputs['fused_features']  # (B, C_fused, H, W)

        # =====================================================
        # Stage 3: Quality Estimation
        # =====================================================
        # Ensure features are tensors for fusion_head
        if self.fusion_head is not None:
            rgb_features_for_quality = rgb_features
            sonar_features_for_quality = sonar_features
            if rgb_features_for_quality is not None and isinstance(rgb_features_for_quality, list):
                rgb_features_for_quality = rgb_features_for_quality[0]
            if sonar_features_for_quality is not None and isinstance(sonar_features_for_quality, list):
                sonar_features_for_quality = sonar_features_for_quality[0]

            quality_outputs = self.fusion_head(
                rgb_features_for_quality, sonar_features_for_quality
            )
            fusion_quality = quality_outputs['quality_score']  # (B,)
        else:
            # Single-modal training: use default quality score
            B = fused_features.shape[0]
            fusion_quality = torch.ones(B, device=fused_features.device)

        # =====================================================
        # Stage 4: Temporal Modeling (Optional)
        # =====================================================
        if self.use_temporal:
            fused_features, new_temporal_state = self.apply_temporal_modeling(
                fused_features,
                quality_scores=fusion_quality
            )
            self.temporal_state = new_temporal_state

        # =====================================================
        # Stage 5: 2D Box Prediction
        # =====================================================
        box_outputs = self.predict_boxes(
            rgb_features, sonar_features, fused_features,
            gt_boxes=gt_boxes if return_losses else None
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
                sonar_img=sonar_search,
                sonar_feat_map=sonar_feat_for_3d,
                num_bins=50
            )

            weak3d_outputs = self.predict_3d_center(
                rgb_features, sonar_features, sonar_radial,
                gt_boxes=gt_boxes if return_losses else None,
                compute_cyclic_loss=return_losses
            )
        else:
            weak3d_outputs = {}

        # =====================================================
        # Stage 7: Output Organization
        # =====================================================
        outputs = {
            # Quality scores (always present)
            'fusion_quality': fusion_quality,
        }

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
                losses['component_consistency'] = fusion_outputs['consistency_losses']['component_loss']
                losses['representation_consistency'] = fusion_outputs['consistency_losses']['infonce_loss']

            # 3D losses
            if 'X_consistency_loss' in weak3d_outputs:
                losses['X_consistency_loss'] = weak3d_outputs['X_consistency_loss']
            if 'rgb_cyclic_loss' in weak3d_outputs:
                losses['rgb_cyclic_loss'] = weak3d_outputs['rgb_cyclic_loss']
            if 'sonar_cyclic_loss' in weak3d_outputs:
                losses['sonar_cyclic_loss'] = weak3d_outputs['sonar_cyclic_loss']
            if 'rgb_proj_loss' in weak3d_outputs:
                losses['rgb_proj_loss'] = weak3d_outputs['rgb_proj_loss']
            if 'sonar_proj_loss' in weak3d_outputs:
                losses['sonar_proj_loss'] = weak3d_outputs['sonar_proj_loss']

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
        ...     'temporal_config': {'type': 'mamba', 'num_blocks': 4},
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
