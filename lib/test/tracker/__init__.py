"""
Tracker interfaces for AquaSOT evaluation.

Based on got10k-toolkit tracker interface.
"""

from .base_tracker import Tracker
from .aquasot_tracker import AquaSOTTracker

__all__ = [
    'Tracker',
    'AquaSOTTracker',
]
