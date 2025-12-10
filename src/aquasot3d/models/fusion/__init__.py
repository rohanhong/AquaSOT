"""
Fusion Module for Cross-Modal RGB-Sonar Fusion

This package implements the DCMA (Deformable Cross-Modal Attention)
fusion framework for AquaSOT-3D.

Components:
- deformable_attention: Deformable cross-attention with learnable sampling offsets
- consistency_loss: Dual consistency (component + representation) constraints
- dcma: Complete DCMA fusion module integrating all components

Usage:
    from aquasot3d.models.fusion import build_dcma, DCMA

    # Build from config
    config = {'embed_dim': 256, 'num_heads': 8, ...}
    fusion_module = build_dcma(config)

    # Or instantiate directly
    fusion_module = DCMA(embed_dim=256, num_heads=8)
"""

# Deformable Attention
from aquasot3d.models.fusion.deformable_attention import (
    DeformableOffsetNetwork,
    BilinearSampler,
    DeformableCrossAttention,
    MultiScaleDeformableAttention,
    build_deformable_cross_attention,
    build_multi_scale_deformable_attention
)

# Consistency Loss
from aquasot3d.models.fusion.consistency_loss import (
    ComponentConsistencyLoss,
    InfoNCELoss,
    DualConsistencyLoss,
    ProjectionHead,
    build_dual_consistency_loss,
    build_projection_head
)

# DCMA (including ConditionalDCMA)
from aquasot3d.models.fusion.dcma import (
    ModalityConfidenceEstimator,
    DCMA,
    DCMALite,
    ConditionalDCMA,
    build_dcma,
    build_dcma_lite,
    build_conditional_dcma,
    # CRITICAL FIX (Audit 2025121001 Issue 8): IdentityFusion for single-modality ablation
    IdentityFusion,
    build_identity_fusion,
    # Backward compatibility aliases
    G2DCMA,
    G2DCMALite,
    ConditionalG2DCMA,
    build_g2_dcma,
    build_g2_dcma_lite,
    build_conditional_g2_dcma
)

__all__ = [
    # Deformable Attention
    'DeformableOffsetNetwork',
    'BilinearSampler',
    'DeformableCrossAttention',
    'MultiScaleDeformableAttention',
    'build_deformable_cross_attention',
    'build_multi_scale_deformable_attention',

    # Consistency Loss
    'ComponentConsistencyLoss',
    'InfoNCELoss',
    'DualConsistencyLoss',
    'ProjectionHead',
    'build_dual_consistency_loss',
    'build_projection_head',

    # DCMA (primary names)
    'ModalityConfidenceEstimator',
    'DCMA',
    'DCMALite',
    'build_dcma',
    'build_dcma_lite',

    # Conditional DCMA
    'ConditionalDCMA',
    'build_conditional_dcma',

    # IdentityFusion for single-modality ablation
    'IdentityFusion',
    'build_identity_fusion',

    # Backward compatibility aliases (deprecated, use new names)
    'G2DCMA',
    'G2DCMALite',
    'ConditionalG2DCMA',
    'build_g2_dcma',
    'build_g2_dcma_lite',
    'build_conditional_g2_dcma'
]
