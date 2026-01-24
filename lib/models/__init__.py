"""
AquaSOT Models Module.

Contains:
- aquasot: Main dual-stream tracker
- layers: Shared model components (head, patch_embed, etc.)
"""

from .aquasot import AquaSOT, build_aquasot
from .layers import CenterPredictor, PatchEmbed, build_box_head

__all__ = [
    'AquaSOT',
    'build_aquasot',
    'CenterPredictor',
    'PatchEmbed',
    'build_box_head',
]
