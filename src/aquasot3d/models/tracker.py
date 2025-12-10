"""
AquaSOT-3D Tracker - Template-Search Multi-Modal Tracking System

This module implements the complete tracking pipeline integrating:
- Multi-modal backbones (RGB + Sonar) with OSTrack one-stream architecture
- DCMA cross-modal fusion with learnable confidence weighting
- Multi-task prediction heads for 2D bounding box detection

Architecture:
    Input: RGB template/search + Sonar template/search
    -> Backbones: OSTrack one-stream ViT for RGB (3-channel) and Sonar (1-channel)
    -> Fusion: ConditionalDCMA with Dual-View Fusion (rgb_view_fused, sonar_view_fused)
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
from aquasot3d.models.fusion import (
    build_dcma, build_dcma_lite, build_conditional_dcma, ConditionalDCMA,
    # CRITICAL FIX (Audit 2025121001 Issue 8): IdentityFusion for single-modality ablation
    build_identity_fusion, IdentityFusion
)
from aquasot3d.models.heads import (
    build_box_head,
)
# CRITICAL FIX (Audit 2025121001 Issue 13): ModalityDropout for robustness training
from aquasot3d.dataloader.transforms.modality_dropout import ModalityDropout, AdaptiveModalityDropout


logger = logging.getLogger(__name__)


@TRACKER_REGISTRY.register("AquaSOT3DTracker")
class AquaSOT3DTracker(BaseModule):
    """
    AquaSOT-3D Template-Search Tracker with Multi-Modal Fusion.

    This is the main tracking model that integrates:
    - OSTrack one-stream ViT backbones for RGB and Sonar
    - ConditionalDCMA cross-modal fusion with learnable confidence
    - Multi-task prediction heads for 2D bounding boxes
    - Uncertainty-aware fusion for robust multi-modal tracking

    Args:
        rgb_backbone_config: Configuration for RGB backbone (template_size=128, search_size=256 for OSTrack-256)
        sonar_backbone_config: Configuration for Sonar backbone (template_size=128, search_size=256 for OSTrack-256)
        fusion_config: Configuration for DCMA fusion module
        box_head_config: Configuration for 2D box prediction heads

    Example:
        >>> config = {
        ...     'rgb_backbone': {'type': 'vit_rgb', 'model_name': 'vit_base_patch16_224', 'template_size': 128, 'search_size': 256},
        ...     'sonar_backbone': {'type': 'vit_sonar', 'model_name': 'vit_base_patch16_224', 'template_size': 128, 'search_size': 256},
        ...     'fusion': {'type': 'dcma', 'rgb_dim': 768, 'sonar_dim': 768},
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
        # CRITICAL FIX (Audit 2025112503 Finding 4 Follow-up):
        # Expose allow_single_modality parameter from config for ablation experiments
        self.backbone = MultiModalBackbone(
            rgb_config=rgb_backbone_config,
            sonar_config=sonar_backbone_config,
            allow_single_modality=kwargs.get('allow_single_modality', False)
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
            # CRITICAL FIX (Audit 2025120102): Default to 768 for ViT-Base, not 384
            self.sonar_feat_dim = sonar_backbone_config.get('output_dim', 768)
        else:
            self.sonar_feat_dim = 768  # ViT-Base default

        # =====================================================
        # 2. Cross-Modal Fusion (DCMA)
        # =====================================================
        self.fusion = None

        # Only build fusion if config is provided and type is not null (for multi-modal training)
        if fusion_config and fusion_config.get('type') not in [None, 'null']:
            # Make a copy to avoid modifying the original config (high-generalization fix)
            fusion_config_copy = fusion_config.copy()
            fusion_type = fusion_config_copy.pop('type', 'dcma')
            # Support both old (g2_*) and new (*) names for backward compatibility
            if fusion_type in ('dcma', 'g2_dcma'):
                self.fusion = build_dcma(fusion_config_copy)
            elif fusion_type in ('dcma_lite', 'g2_dcma_lite'):
                self.fusion = build_dcma_lite(fusion_config_copy)
            elif fusion_type in ('conditional_dcma', 'conditional_g2_dcma'):
                # CRITICAL FIX (Audit 2025112101 Phase 2.2):
                # ConditionalDCMA requires rgb_channels and sonar_channels
                # CRITICAL FIX (Audit 2025112505 Issue 5):
                # Remove rgb_channels and sonar_channels from config to avoid duplicate keyword arguments
                fusion_config_copy.pop('rgb_channels', None)
                fusion_config_copy.pop('sonar_channels', None)
                self.fusion = build_conditional_dcma(
                    rgb_channels=self.rgb_feat_dim,
                    sonar_channels=self.sonar_feat_dim,
                    **fusion_config_copy
                )
            elif fusion_type == 'identity_fusion':
                # CRITICAL FIX (Audit 2025121001 Issue 8): IdentityFusion for single-modality ablation
                # IdentityFusion passes through single-modality features without cross-modal fusion.
                # Use active_modality config to select which modality to use ('rgb' or 'sonar').
                fusion_config_copy.pop('rgb_channels', None)
                fusion_config_copy.pop('sonar_channels', None)
                self.fusion = build_identity_fusion(
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
        # This simplifies architecture and strengthens DCMA supervision through dual gradients.

        # 4.3 Auxiliary Heads for Feature Degradation Prevention
        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Add lightweight auxiliary heads that predict from original features (pre-fusion).
        # This prevents backbone feature degradation by providing direct gradient flow.
        # These heads are ONLY used during training, not inference.
        #
        # CRITICAL FIX (Audit 2025113001): Add Sigmoid activation to constrain output to [0,1]
        # Without Sigmoid, outputs are unbounded while GT is normalized to [0,1], causing
        # large loss values and inconsistent gradients.
        self.aux_rgb_head = None
        self.aux_sonar_head = None
        if self.backbone.rgb_backbone is not None and self.backbone.sonar_encoder is not None:
            # Very lightweight: adaptive pooling + MLP + Sigmoid
            self.aux_rgb_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
                nn.Flatten(),  # [B, C, 1, 1] -> [B, C]
                nn.Linear(self.rgb_feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 4),  # Predict bbox [cx, cy, w, h]
                nn.Sigmoid()  # CRITICAL: Constrain to [0, 1] for normalized coords
            )
            self.aux_sonar_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.sonar_feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 4),
                nn.Sigmoid()  # CRITICAL: Constrain to [0, 1] for normalized coords
            )
            # CRITICAL FIX (Audit 2025113001): Initialize auxiliary heads properly
            # Following BoxHead conventions for input sensitivity and centered initial predictions
            self._init_aux_heads()
            logger.info("Initialized auxiliary heads for feature degradation prevention")
        else:
            logger.info("Auxiliary heads disabled (single-modal training)")

        # =====================================================
        # 6. Template Update State Management
        # =====================================================
        # Template-search paradigm has no internal state.
        # Template updates are managed externally via OnlineTracker.

        # =====================================================
        # 7. ModalityDropout for Robustness Training
        # =====================================================
        # CRITICAL FIX (Audit 2025121001 Issue 13): ModalityDropout for robustness training
        # Randomly drops modalities during training to train single/zero-modality operation
        modality_dropout_config = kwargs.get('modality_dropout', {})
        if modality_dropout_config.get('enabled', False):
            if modality_dropout_config.get('adaptive', False):
                # Adaptive dropout with curriculum learning
                self.modality_dropout = AdaptiveModalityDropout(
                    p_drop_rgb_start=modality_dropout_config.get('p_drop_rgb', 0.05),
                    p_drop_rgb_end=modality_dropout_config.get('p_drop_rgb', 0.1) * 2,
                    p_drop_sonar_start=modality_dropout_config.get('p_drop_sonar', 0.05),
                    p_drop_sonar_end=modality_dropout_config.get('p_drop_sonar', 0.1) * 2,
                    p_drop_both_start=modality_dropout_config.get('p_drop_both', 0.01),
                    p_drop_both_end=modality_dropout_config.get('p_drop_both', 0.02) * 2,
                    warmup_epochs=modality_dropout_config.get('warmup_epochs', 50),
                    mode=modality_dropout_config.get('mode', 'zero')
                )
                self.logger.info(f"Initialized AdaptiveModalityDropout: {self.modality_dropout}")
            else:
                # Standard modality dropout
                self.modality_dropout = ModalityDropout(
                    p_drop_rgb=modality_dropout_config.get('p_drop_rgb', 0.1),
                    p_drop_sonar=modality_dropout_config.get('p_drop_sonar', 0.1),
                    p_drop_both=modality_dropout_config.get('p_drop_both', 0.02),
                    mode=modality_dropout_config.get('mode', 'zero')
                )
                self.logger.info(f"Initialized ModalityDropout: {self.modality_dropout}")
        else:
            self.modality_dropout = None

        self.logger.info(
            f"Initialized AquaSOT3DTracker: "
            f"rgb_dim={self.rgb_feat_dim}, sonar_dim={self.sonar_feat_dim}, "
            f"fused_dim={self.fused_feat_dim}"
        )

    def _init_aux_heads(self):
        """
        Initialize auxiliary heads following BoxHead conventions.

        CRITICAL FIX (Audit 2025113001): Proper initialization is essential for:
        1. Input sensitivity: Network must respond to different inputs
        2. Centered initial predictions: Start with [0.5, 0.5, 0.3, 0.3]
        3. Consistent gradient magnitudes: Avoid exploding/vanishing gradients

        Reference: OSTrack/MixFormer weight initialization standards.
        """
        for head in [self.aux_rgb_head, self.aux_sonar_head]:
            if head is None:
                continue

            # Initialize all Linear layers with standard Xavier
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            # Special initialization for output layer (Linear before Sigmoid)
            # This produces initial predictions around [0.5, 0.5, 0.3, 0.3]
            output_layer = head[-2]  # Linear layer before Sigmoid
            if isinstance(output_layer, nn.Linear):
                # std=0.1 for input sensitivity (following BoxHead fix)
                nn.init.normal_(output_layer.weight, mean=0.0, std=0.1)
                if output_layer.bias is not None:
                    # Initialize bias to produce centered predictions after Sigmoid
                    # sigmoid(0) = 0.5, sigmoid(-0.85) ~ 0.3
                    output_layer.bias.data[0] = 0.0   # cx ~ 0.5 after sigmoid
                    output_layer.bias.data[1] = 0.0   # cy ~ 0.5 after sigmoid
                    output_layer.bias.data[2] = -0.85  # w ~ 0.3 after sigmoid
                    output_layer.bias.data[3] = -0.85  # h ~ 0.3 after sigmoid

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
        sonar_search: torch.Tensor,
        # CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised): Attention masks
        rgb_search_att_mask: Optional[torch.Tensor] = None,
        sonar_search_att_mask: Optional[torch.Tensor] = None,
        rgb_template_att_mask: Optional[torch.Tensor] = None,
        sonar_template_att_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from RGB and Sonar backbones using template-search paradigm.

        ARCHITECTURE CHANGE (Audit 2025112104):
        Migrated to template-search dual input (OSTrack-style).

        CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised):
        Now accepts attention masks to properly handle padded regions in backbone.
        Both template and search masks are supported.

        Args:
            rgb_template: RGB template frame, shape (B, 3, H_t, W_t)
            rgb_search: RGB search frame, shape (B, 3, H_s, W_s)
            sonar_template: Sonar template frame, shape (B, 1, H_t, W_t)
            sonar_search: Sonar search frame, shape (B, 1, H_s, W_s)
            rgb_search_att_mask: Optional attention mask for RGB search [B, H, W], 1=valid, 0=padded
            sonar_search_att_mask: Optional attention mask for Sonar search [B, H, W], 1=valid, 0=padded
            rgb_template_att_mask: Optional attention mask for RGB template [B, H, W], 1=valid, 0=padded
            sonar_template_att_mask: Optional attention mask for Sonar template [B, H, W], 1=valid, 0=padded

        Returns:
            Dictionary containing extracted features:
            {
                'rgb_template_features': [B, N_z, D],
                'rgb_search_features': [B, N_x, D],
                'sonar_template_features': [B, N_z, D],
                'sonar_search_features': [B, N_x, D]
            }
        """
        # CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised):
        # Pass attention masks to backbone for proper padding handling
        # Both template and search masks are passed
        backbone_outputs = self.backbone(
            rgb_template=rgb_template,
            rgb_search=rgb_search,
            sonar_template=sonar_template,
            sonar_search=sonar_search,
            rgb_search_att_mask=rgb_search_att_mask,
            sonar_search_att_mask=sonar_search_att_mask,
            rgb_template_att_mask=rgb_template_att_mask,
            sonar_template_att_mask=sonar_template_att_mask
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

    def extract_backbone_features_dual_template(
        self,
        rgb_template_init: torch.Tensor,
        rgb_template_dyn: torch.Tensor,
        rgb_search: torch.Tensor,
        sonar_template_init: torch.Tensor,
        sonar_template_dyn: torch.Tensor,
        sonar_search: torch.Tensor,
        # Attention masks
        rgb_template_init_att_mask: Optional[torch.Tensor] = None,
        rgb_template_dyn_att_mask: Optional[torch.Tensor] = None,
        rgb_search_att_mask: Optional[torch.Tensor] = None,
        sonar_template_init_att_mask: Optional[torch.Tensor] = None,
        sonar_template_dyn_att_mask: Optional[torch.Tensor] = None,
        sonar_search_att_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features using MixFormer-style dual-template processing.

        Args:
            rgb_template_init: RGB init template [B, 3, H, W]
            rgb_template_dyn: RGB dynamic template [B, 3, H, W]
            rgb_search: RGB search region [B, 3, H, W]
            sonar_template_init: Sonar init template [B, 1, H, W]
            sonar_template_dyn: Sonar dynamic template [B, 1, H, W]
            sonar_search: Sonar search region [B, 1, H, W]
            *_att_mask: Attention masks [B, H, W], 1=valid, 0=padded

        Returns:
            Dictionary containing extracted features:
            {
                'rgb_template_init_features': [B, N_z, D],
                'rgb_template_dyn_features': [B, N_z, D],
                'rgb_search_features': [B, N_x, D],
                'sonar_template_init_features': [B, N_z, D],
                'sonar_template_dyn_features': [B, N_z, D],
                'sonar_search_features': [B, N_x, D]
            }
        """
        # Use dual-template forward on backbone
        backbone_outputs = self.backbone.forward_dual_template(
            rgb_template_init=rgb_template_init,
            rgb_template_dyn=rgb_template_dyn,
            rgb_search=rgb_search,
            sonar_template_init=sonar_template_init,
            sonar_template_dyn=sonar_template_dyn,
            sonar_search=sonar_search,
            rgb_template_init_att_mask=rgb_template_init_att_mask,
            rgb_template_dyn_att_mask=rgb_template_dyn_att_mask,
            rgb_search_att_mask=rgb_search_att_mask,
            sonar_template_init_att_mask=sonar_template_init_att_mask,
            sonar_template_dyn_att_mask=sonar_template_dyn_att_mask,
            sonar_search_att_mask=sonar_search_att_mask
        )

        # Extract features from nested structure
        return {
            'rgb_template_init_features': backbone_outputs['rgb']['template_init_features'],
            'rgb_template_dyn_features': backbone_outputs['rgb']['template_dyn_features'],
            'rgb_search_features': backbone_outputs['rgb']['search_features'],
            'sonar_template_init_features': backbone_outputs['sonar']['template_init_features'],
            'sonar_template_dyn_features': backbone_outputs['sonar']['template_dyn_features'],
            'sonar_search_features': backbone_outputs['sonar']['search_features'],
        }

    def fuse_features(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        return_consistency: bool = True,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse RGB and Sonar features using DCMA or ConditionalDCMA.

        CRITICAL FIX (Audit 2025112101 Phase 2.2):
        Now accepts is_rgb_valid and is_sonar_valid for conditional fusion.
        When using ConditionalDCMA, these flags enable single-modality bypass.

        ENHANCED FIX (Audit 2025112403 Follow-up):
        Now accepts metadata for per-sample letterbox parameter support.

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
        # Pass is_rgb_valid and is_sonar_valid to ConditionalDCMA for modality-aware routing
        # For standard DCMA, these parameters will be ignored
        fusion_kwargs = {
            'rgb_features': rgb_features,
            'sonar_features': sonar_features,
            'return_consistency': return_consistency
        }

        # CRITICAL FIX (Audit 2025120401 Finding 9-10): Unified metadata conversion
        # Use crop_metadata_to_letterbox to convert crop-style metadata to letterbox format
        # This ensures consistent metadata format across training/validation/online evaluation
        # SYMMETRIC PASSING: Both RGB and Sonar metadata are converted and passed to fusion
        if metadata is not None:
            from aquasot3d.utils.preprocessing import crop_metadata_to_letterbox

            def _convert_metadata_list(metadata_list, modality_name):
                """Convert a list of crop metadata to letterbox format."""
                if not metadata_list or len(metadata_list) == 0:
                    return None, True

                converted_list = []
                for meta in metadata_list:
                    if meta is None:
                        converted_list.append(None)
                        continue

                    # Check if already in letterbox format (has 'scale' key)
                    # or crop format (has 'resize_factor' key)
                    if 'scale' in meta and 'original_height' in meta:
                        # Already letterbox format
                        converted_list.append(meta)
                    elif 'resize_factor' in meta or 'crop_sz' in meta:
                        # Crop format - convert to letterbox
                        output_size = meta.get('output_size', 256)
                        converted = crop_metadata_to_letterbox(meta, output_size)
                        converted_list.append(converted)
                    else:
                        # Unknown format - pass through
                        converted_list.append(meta)

                # Check consistency
                all_consistent = True
                first_meta = converted_list[0] if converted_list else None

                if first_meta and len(converted_list) > 1:
                    for meta in converted_list[1:]:
                        if meta is None:
                            all_consistent = False
                            break
                        if (meta.get('scale') != first_meta.get('scale') or
                            meta.get('pad_top') != first_meta.get('pad_top') or
                            meta.get('original_height') != first_meta.get('original_height')):
                            all_consistent = False
                            break

                return converted_list, all_consistent

            # === Process Sonar metadata ===
            sonar_metadata_list = (
                metadata.get('sonar_metadata') or
                metadata.get('sonar_search_metadata') or
                metadata.get('search_sonar_metadata')
            )

            if sonar_metadata_list:
                converted_sonar, sonar_consistent = _convert_metadata_list(sonar_metadata_list, 'sonar')
                if converted_sonar:
                    first_sonar = converted_sonar[0] if converted_sonar else None
                    if sonar_consistent and first_sonar and 'scale' in first_sonar:
                        fusion_kwargs['sonar_letterbox_metadata'] = first_sonar
                    elif not sonar_consistent:
                        fusion_kwargs['sonar_letterbox_metadata'] = converted_sonar

            # === CRITICAL FIX (Audit 2025120401 Finding 9): Symmetric RGB metadata ===
            # When dynamic search expansion or non-uniform scaling is used, RGB crop
            # physical scale also varies. Pass RGB letterbox metadata symmetrically
            # to ensure cross-modal physical baseline consistency.
            rgb_metadata_list = (
                metadata.get('rgb_metadata') or
                metadata.get('rgb_search_metadata') or
                metadata.get('search_rgb_metadata')
            )

            if rgb_metadata_list:
                converted_rgb, rgb_consistent = _convert_metadata_list(rgb_metadata_list, 'rgb')
                if converted_rgb:
                    first_rgb = converted_rgb[0] if converted_rgb else None
                    if rgb_consistent and first_rgb and 'scale' in first_rgb:
                        fusion_kwargs['rgb_letterbox_metadata'] = first_rgb
                    elif not rgb_consistent:
                        fusion_kwargs['rgb_letterbox_metadata'] = converted_rgb

        # Check if fusion module is ConditionalDCMA (has forward signature with is_rgb_valid)
        from aquasot3d.models.fusion import ConditionalDCMA
        if isinstance(self.fusion, ConditionalDCMA):
            # ConditionalDCMA requires is_rgb_valid and is_sonar_valid
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
        has_object: Optional[torch.Tensor] = None,
        # CRITICAL FIX (Audit 2025120902): Attention masks for head
        rgb_att_mask: Optional[torch.Tensor] = None,
        sonar_att_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict 2D bounding boxes for all modalities using template-search paradigm.

        CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX):
        Accepts has_object flags for confidence head training.

        CRITICAL FIX (Audit 2025120902): Now accepts attention masks
        to pass to box heads for proper padding handling.

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
            rgb_att_mask: Optional attention mask for RGB [B, H, W], 1=valid, 0=padded
            sonar_att_mask: Optional attention mask for Sonar [B, H, W], 1=valid, 0=padded

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
        # This forces DCMA to learn RGB-Sonar correspondence through dual supervision
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
                has_object=has_object_rgb,
                attention_mask=rgb_att_mask  # CRITICAL FIX (Audit 2025120902): Pass attention mask
            )
            outputs['rgb_boxes'] = rgb_box_outputs['pred_boxes']
            outputs['rgb_quality'] = rgb_box_outputs['pred_scores']

            # Add confidence prediction if available
            if 'pred_confidence' in rgb_box_outputs:
                outputs['rgb_confidence'] = rgb_box_outputs['pred_confidence']

            # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
            # Add validity prediction for fusion routing
            if 'validity_final' in rgb_box_outputs:
                outputs['rgb_validity'] = rgb_box_outputs['validity_final']
            if 'validity_logits' in rgb_box_outputs:
                outputs['rgb_validity_logits'] = rgb_box_outputs['validity_logits']

            # Add confidence loss if available
            if 'conf_loss' in rgb_box_outputs:
                outputs['rgb_conf_loss'] = rgb_box_outputs['conf_loss']

            # Extract total loss from CenterHead (includes score + size + iou losses)
            if gt_boxes is not None and 'rgb' in gt_boxes:
                if 'total_loss' in rgb_box_outputs:
                    outputs['rgb_box_loss'] = rgb_box_outputs['total_loss']
                # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
                # Extract validity loss separately for calibration logging
                if 'validity_loss' in rgb_box_outputs:
                    outputs['rgb_validity_loss'] = rgb_box_outputs['validity_loss']

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
                has_object=has_object_sonar,
                attention_mask=sonar_att_mask  # CRITICAL FIX (Audit 2025120902): Pass attention mask
            )
            outputs['sonar_boxes'] = sonar_box_outputs['pred_boxes']
            outputs['sonar_quality'] = sonar_box_outputs['pred_scores']

            # Add confidence prediction if available
            if 'pred_confidence' in sonar_box_outputs:
                outputs['sonar_confidence'] = sonar_box_outputs['pred_confidence']

            # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
            # Add validity prediction for fusion routing
            if 'validity_final' in sonar_box_outputs:
                outputs['sonar_validity'] = sonar_box_outputs['validity_final']
            if 'validity_logits' in sonar_box_outputs:
                outputs['sonar_validity_logits'] = sonar_box_outputs['validity_logits']

            # Add confidence loss if available
            if 'conf_loss' in sonar_box_outputs:
                outputs['sonar_conf_loss'] = sonar_box_outputs['conf_loss']

            # Extract total loss from CenterHead (includes score + size + iou losses)
            if gt_boxes is not None and 'sonar' in gt_boxes:
                if 'total_loss' in sonar_box_outputs:
                    outputs['sonar_box_loss'] = sonar_box_outputs['total_loss']
                # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
                # Extract validity loss separately for calibration logging
                if 'validity_loss' in sonar_box_outputs:
                    outputs['sonar_validity_loss'] = sonar_box_outputs['validity_loss']

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
        metadata: Optional[Dict] = None,
        # CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised): Attention masks for padded regions
        # Search region masks
        rgb_att_mask: Optional[torch.Tensor] = None,
        sonar_att_mask: Optional[torch.Tensor] = None,
        # Template region masks (for edge cases where template has padding)
        rgb_template_att_mask: Optional[torch.Tensor] = None,
        sonar_template_att_mask: Optional[torch.Tensor] = None,
        # Internal: pre-computed backbone outputs (used by forward_dual_template)
        _precomputed_backbone_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass of AquaSOT-3D tracker using template-search paradigm.

        CRITICAL FIX (Audit 2025112101 Phase 2.2):
        Accepts is_rgb_valid and is_sonar_valid for conditional fusion.
        These flags enable single-modality bypass in ConditionalDCMA.

        CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX):
        Accepts has_object flags for confidence head training.
        These flags distinguish "no object" from "object at [0,0,0,0]".

        Args:
            rgb_template: RGB template frame, shape (B, 3, 128, 128) OSTrack-256
            rgb_search: RGB search frame, shape (B, 3, 256, 256) OSTrack-256
            sonar_template: Sonar template frame, shape (B, 1, 128, 128) OSTrack-256
            sonar_search: Sonar search frame, shape (B, 1, 256, 256) OSTrack-256
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
            rgb_att_mask: Optional attention mask for RGB search [B, H, W], 1=valid, 0=padded
                (CRITICAL FIX for Audit 2025120902 Issue 15, 17 Revised)
            sonar_att_mask: Optional attention mask for Sonar search [B, H, W], 1=valid, 0=padded
                (CRITICAL FIX for Audit 2025120902 Issue 15, 17 Revised)
            rgb_template_att_mask: Optional attention mask for RGB template [B, H, W], 1=valid, 0=padded
            sonar_template_att_mask: Optional attention mask for Sonar template [B, H, W], 1=valid, 0=padded

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
            forcing DCMA to learn RGB-Sonar correspondence through dual supervision.
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
        # CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised):
        # Pass attention masks to backbone for proper padding handling
        # Both template and search masks are passed to handle edge cases

        # DUAL-TEMPLATE SUPPORT: Use pre-computed outputs if provided
        if _precomputed_backbone_outputs is not None:
            backbone_outputs = _precomputed_backbone_outputs
        else:
            backbone_outputs = self.extract_backbone_features(
                rgb_template, rgb_search, sonar_template, sonar_search,
                rgb_search_att_mask=rgb_att_mask,
                sonar_search_att_mask=sonar_att_mask,
                rgb_template_att_mask=rgb_template_att_mask,
                sonar_template_att_mask=sonar_template_att_mask
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
            # CRITICAL FIX (Audit 2025112509 Bug B6): Add validity masking to auxiliary losses
            # Without masking, invalid samples (is_rgb_valid=0) corrupt backbone features
            # by forcing regression to zero bbox, learning meaningless spatial patterns
            if gt_boxes is not None:
                # RGB auxiliary loss with validity masking
                if 'rgb' in gt_boxes:
                    # Compute per-sample L1 loss
                    aux_rgb_loss_per_sample = F.l1_loss(
                        aux_rgb_boxes, gt_boxes['rgb'], reduction='none'
                    ).mean(dim=-1)  # [B]

                    # Apply validity masking if available
                    if is_rgb_valid is not None:
                        validity_mask = is_rgb_valid.float() if is_rgb_valid.dtype != torch.float32 else is_rgb_valid
                        masked_loss = aux_rgb_loss_per_sample * validity_mask
                        num_valid = validity_mask.sum().clamp(min=1.0)
                        aux_rgb_loss = masked_loss.sum() / num_valid
                    else:
                        aux_rgb_loss = aux_rgb_loss_per_sample.mean()

                    aux_outputs['aux_rgb_loss'] = aux_rgb_loss * 0.3  # Lower weight

                # Sonar auxiliary loss with validity masking
                if 'sonar' in gt_boxes:
                    # Compute per-sample L1 loss
                    aux_sonar_loss_per_sample = F.l1_loss(
                        aux_sonar_boxes, gt_boxes['sonar'], reduction='none'
                    ).mean(dim=-1)  # [B]

                    # Apply validity masking if available
                    if is_sonar_valid is not None:
                        validity_mask = is_sonar_valid.float() if is_sonar_valid.dtype != torch.float32 else is_sonar_valid
                        masked_loss = aux_sonar_loss_per_sample * validity_mask
                        num_valid = validity_mask.sum().clamp(min=1.0)
                        aux_sonar_loss = masked_loss.sum() / num_valid
                    else:
                        aux_sonar_loss = aux_sonar_loss_per_sample.mean()

                    aux_outputs['aux_sonar_loss'] = aux_sonar_loss * 0.3  # Lower weight

        # =====================================================
        # Stage 1.75: Modality Dropout (Training Only)
        # =====================================================
        # CRITICAL FIX (Audit 2025121001 Issue 13): Apply modality dropout
        # Randomly drops modalities to train single/zero-modality robustness.
        # Applied AFTER auxiliary heads (so they always see original features)
        # but BEFORE fusion (so fusion learns to handle missing modalities).
        if self.training and self.modality_dropout is not None:
            # Get current epoch for adaptive dropout (if applicable)
            current_epoch = getattr(self, '_current_epoch', 0)

            # Apply modality dropout to features
            rgb_features, sonar_features, dropout_is_rgb_valid, dropout_is_sonar_valid = \
                self.modality_dropout(rgb_features, sonar_features, current_epoch)

            # Update validity flags based on dropout
            # If a modality was dropped, mark it as invalid for fusion routing
            if is_rgb_valid is not None:
                is_rgb_valid = is_rgb_valid & dropout_is_rgb_valid
            else:
                is_rgb_valid = dropout_is_rgb_valid.expand(rgb_features.shape[0])

            if is_sonar_valid is not None:
                is_sonar_valid = is_sonar_valid & dropout_is_sonar_valid
            else:
                is_sonar_valid = dropout_is_sonar_valid.expand(sonar_features.shape[0])

        # =====================================================
        # Stage 2: Cross-Modal Fusion
        # =====================================================
        # CRITICAL FIX (Audit 2025112508 Finding 2):
        # DO NOT force None to all-true here! Allow ConditionalDCMA to infer validity
        # from confidence scores. This enables single/zero-modality routing at inference.
        #
        # Previous behavior (INCORRECT):
        # if is_rgb_valid is None:
        #     is_rgb_valid = torch.ones(...)  # ❌ Forces dual-modality always
        #
        # New behavior (CORRECT):
        # Pass None through to fusion layer, let ConditionalDCMA auto-infer
        # based on learned confidence estimators.

        fusion_outputs = self.fuse_features(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,  # Can be None - fusion will infer
            is_sonar_valid=is_sonar_valid,  # Can be None - fusion will infer
            return_consistency=return_losses,
            metadata=metadata
        )

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # fusion_outputs now contains both rgb_view_fused and sonar_view_fused
        # Keep the full dict for passing to predict_boxes
        fused_features = fusion_outputs  # Keep as dict to access both views

        # CRITICAL FIX (Audit 2025112508 Finding 2):
        # Extract inferred validity from fusion layer (if available)
        # This ensures predict_boxes uses the same validity that fusion used for routing
        if 'inferred_is_rgb_valid' in fusion_outputs:
            is_rgb_valid = fusion_outputs['inferred_is_rgb_valid']
        if 'inferred_is_sonar_valid' in fusion_outputs:
            is_sonar_valid = fusion_outputs['inferred_is_sonar_valid']

        # =====================================================
        # Stage 4: 2D Box Prediction
        # =====================================================
        # CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX): Pass has_object flags
        # CRITICAL FIX (Audit 2025112508 Finding 2): Now uses inferred validity
        # CRITICAL FIX (Audit 2025120902): Pass attention masks to heads
        box_outputs = self.predict_boxes(
            rgb_features, sonar_features, fused_features,
            gt_boxes=gt_boxes if return_losses else None,
            is_rgb_valid=is_rgb_valid,  # May be inferred from fusion
            is_sonar_valid=is_sonar_valid,  # May be inferred from fusion
            has_object_rgb=has_object_rgb,
            has_object_sonar=has_object_sonar,
            has_object=has_object,
            rgb_att_mask=rgb_att_mask,  # Pass attention masks to heads
            sonar_att_mask=sonar_att_mask
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
                # CRITICAL FIX (Audit 2025112510): Use correct keys from ConditionalDCMA
                # ConditionalDCMA renames keys: component_loss -> component_consistency, infonce_loss -> representation_consistency
                consistency = fusion_outputs['consistency_losses']
                _device = rgb_template.device  # Use input tensor device
                losses['geometric_consistency'] = consistency.get('component_consistency', consistency.get('component_loss', torch.tensor(0.0, device=_device)))
                losses['representation_consistency'] = consistency.get('representation_consistency', consistency.get('infonce_loss', torch.tensor(0.0, device=_device)))

            # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
            # Add cross-view consistency loss
            if 'view_consistency_loss' in fusion_outputs:
                losses['view_consistency'] = fusion_outputs['view_consistency_loss']

            # CRITICAL FIX (Audit 2025121001 Issue 4): Routing validity loss from ConditionalDCMA
            # This supervises the confidence estimators for automatic modality routing
            if 'routing_validity_loss' in fusion_outputs:
                losses['routing_validity_loss'] = fusion_outputs['routing_validity_loss']

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

    def forward_dual_template(
        self,
        rgb_template_init: torch.Tensor,
        rgb_template_dyn: torch.Tensor,
        rgb_search: torch.Tensor,
        sonar_template_init: torch.Tensor,
        sonar_template_dyn: torch.Tensor,
        sonar_search: torch.Tensor,
        gt_boxes: Optional[Dict[str, torch.Tensor]] = None,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        has_object_rgb: Optional[torch.Tensor] = None,
        has_object_sonar: Optional[torch.Tensor] = None,
        has_object: Optional[torch.Tensor] = None,
        return_losses: bool = True,
        return_intermediate: bool = False,
        metadata: Optional[Dict] = None,
        # Attention masks
        rgb_att_mask: Optional[torch.Tensor] = None,
        sonar_att_mask: Optional[torch.Tensor] = None,
        rgb_template_init_att_mask: Optional[torch.Tensor] = None,
        rgb_template_dyn_att_mask: Optional[torch.Tensor] = None,
        sonar_template_init_att_mask: Optional[torch.Tensor] = None,
        sonar_template_dyn_att_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        MixFormer-style dual-template forward pass.

        Processes both init (anchor) and dynamic templates through the backbone,
        then combines features for fusion and prediction.

        Args:
            rgb_template_init: RGB init template [B, 3, H, W]
            rgb_template_dyn: RGB dynamic template [B, 3, H, W]
            rgb_search: RGB search region [B, 3, H, W]
            sonar_template_init: Sonar init template [B, 1, H, W]
            sonar_template_dyn: Sonar dynamic template [B, 1, H, W]
            sonar_search: Sonar search region [B, 1, H, W]
            (other args same as forward)

        Returns:
            Same as forward method
        """
        batch_size = rgb_template_init.shape[0]

        # =====================================================
        # Stage 1: Dual-Template Feature Extraction
        # =====================================================
        backbone_outputs = self.extract_backbone_features_dual_template(
            rgb_template_init=rgb_template_init,
            rgb_template_dyn=rgb_template_dyn,
            rgb_search=rgb_search,
            sonar_template_init=sonar_template_init,
            sonar_template_dyn=sonar_template_dyn,
            sonar_search=sonar_search,
            rgb_template_init_att_mask=rgb_template_init_att_mask,
            rgb_template_dyn_att_mask=rgb_template_dyn_att_mask,
            rgb_search_att_mask=rgb_att_mask,
            sonar_template_init_att_mask=sonar_template_init_att_mask,
            sonar_template_dyn_att_mask=sonar_template_dyn_att_mask,
            sonar_search_att_mask=sonar_att_mask
        )

        # Use search features for prediction
        # Template info is already encoded via joint attention in backbone
        rgb_features = backbone_outputs['rgb_search_features']
        sonar_features = backbone_outputs['sonar_search_features']

        # Delegate to standard forward for fusion and prediction
        # (after backbone extraction, the flow is identical)
        return self.forward(
            rgb_template=rgb_template_init,  # Not used, just for signature
            rgb_search=rgb_search,
            sonar_template=sonar_template_init,  # Not used
            sonar_search=sonar_search,
            gt_boxes=gt_boxes,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            has_object_rgb=has_object_rgb,
            has_object_sonar=has_object_sonar,
            has_object=has_object,
            return_losses=return_losses,
            return_intermediate=return_intermediate,
            metadata=metadata,
            rgb_att_mask=rgb_att_mask,
            sonar_att_mask=sonar_att_mask,
            rgb_template_att_mask=rgb_template_init_att_mask,
            sonar_template_att_mask=sonar_template_init_att_mask,
            # Override backbone extraction by passing pre-computed features
            _precomputed_backbone_outputs={
                'rgb_search_features': rgb_features,
                'sonar_search_features': sonar_features,
                'rgb_template_init_features': backbone_outputs['rgb_template_init_features'],
                'rgb_template_dyn_features': backbone_outputs['rgb_template_dyn_features'],
                'sonar_template_init_features': backbone_outputs['sonar_template_init_features'],
                'sonar_template_dyn_features': backbone_outputs['sonar_template_dyn_features'],
            }
        )


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
