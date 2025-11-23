"""
Fusion Module for Cross-Modal RGB-Sonar Fusion

This package implements the G2-DCMA (Geometry-Gated Deformable Cross-Modal Attention)
fusion framework for AquaSOT-3D.

Components:
- geometric_gating: Geometry-aware attention with radial and scale priors
- deformable_attention: Deformable cross-attention with learnable sampling offsets
- consistency_loss: Dual consistency (component + representation) constraints
- g2_dcma: Complete G2-DCMA fusion module integrating all components

Usage:
    from aquasot3d.models.fusion import build_g2_dcma, G2DCMA

    # Build from config
    config = {'embed_dim': 256, 'num_heads': 8, ...}
    fusion_module = build_g2_dcma(config)

    # Or instantiate directly
    fusion_module = G2DCMA(embed_dim=256, num_heads=8)
"""

# Geometric Gating
from aquasot3d.models.fusion.geometric_gating import (
    RadialProfileExtractor,
    ScalePyramidExtractor,
    GeometricGatingFunction,
    GeometryGatedAttention,
    build_geometry_gated_attention
)

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

# G2-DCMA (including ConditionalG2DCMA - consolidated in Audit 2025112309)
from aquasot3d.models.fusion.g2_dcma import (
    G2DCMA,
    G2DCMALite,
    ConditionalG2DCMA,
    build_g2_dcma,
    build_g2_dcma_lite,
    build_conditional_g2_dcma
)

__all__ = [
    # Geometric Gating
    'RadialProfileExtractor',
    'ScalePyramidExtractor',
    'GeometricGatingFunction',
    'GeometryGatedAttention',
    'build_geometry_gated_attention',

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

    # G2-DCMA
    'G2DCMA',
    'G2DCMALite',
    'build_g2_dcma',
    'build_g2_dcma_lite',

    # Conditional G2-DCMA (Audit 2025112101 Phase 2.1)
    'ConditionalG2DCMA',
    'build_conditional_g2_dcma'
]
