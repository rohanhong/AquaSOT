"""
Evaluation and testing utilities for AquaSOT.
"""

from .tracker import Tracker, AquaSOTTracker
from .evaluation import (
    rect_iou,
    center_error,
    success_curve,
    precision_curve,
    ExperimentAquaSOT,
)

__all__ = [
    'Tracker',
    'AquaSOTTracker',
    'rect_iou',
    'center_error',
    'success_curve',
    'precision_curve',
    'ExperimentAquaSOT',
]
