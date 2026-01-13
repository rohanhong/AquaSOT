"""
AquaSOT-3D Model Module.

Main components:
- AquaSOT3D: Dual-stream RGB-Sonar tracker
- VisionTransformer: ViT backbone with variable input channels
- DCMA: Deformable Cross-Modal Attention fusion
- CenterPredictor: OSTrack-style box prediction head
"""

from .aquasot3d import AquaSOT3D, build_aquasot3d
from .vit import VisionTransformer, vit_base_patch16_224, build_vit_rgb, build_vit_sonar
from .fusion import DCMA, IdentityFusion, build_fusion
from .utils import combine_tokens, recover_tokens
from .base_backbone import BaseBackbone

__all__ = [
    'AquaSOT3D',
    'build_aquasot3d',
    'VisionTransformer',
    'vit_base_patch16_224',
    'build_vit_rgb',
    'build_vit_sonar',
    'DCMA',
    'IdentityFusion',
    'build_fusion',
    'combine_tokens',
    'recover_tokens',
    'BaseBackbone',
]
