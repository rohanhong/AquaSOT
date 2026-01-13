"""
AquaSOT-3D Models Module.

Contains:
- aquasot3d: Main dual-stream tracker
- layers: Shared model components (head, patch_embed, etc.)
"""

from .aquasot3d import AquaSOT3D, build_aquasot3d
from .layers import CenterPredictor, PatchEmbed, build_box_head

__all__ = [
    'AquaSOT3D',
    'build_aquasot3d',
    'CenterPredictor',
    'PatchEmbed',
    'build_box_head',
]
