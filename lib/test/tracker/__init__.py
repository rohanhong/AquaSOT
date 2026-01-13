"""
Tracker interfaces for AquaSOT-3D evaluation.

Based on got10k-toolkit tracker interface.
"""

from .base_tracker import Tracker
from .aquasot3d_tracker import AquaSOT3DTracker

__all__ = [
    'Tracker',
    'AquaSOT3DTracker',
]
