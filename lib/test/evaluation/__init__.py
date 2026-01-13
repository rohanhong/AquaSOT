"""
Evaluation utilities for AquaSOT-3D.

Based on got10k-toolkit evaluation framework.
"""

from .metrics import rect_iou, center_error, success_curve, precision_curve
from .experiment import ExperimentAquaSOT3D

__all__ = [
    'rect_iou',
    'center_error',
    'success_curve',
    'precision_curve',
    'ExperimentAquaSOT3D',
]
