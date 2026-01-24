"""
Evaluation utilities for AquaSOT.

Based on got10k-toolkit evaluation framework.
"""

from .metrics import rect_iou, center_error, success_curve, precision_curve
from .experiment import ExperimentAquaSOT

__all__ = [
    'rect_iou',
    'center_error',
    'success_curve',
    'precision_curve',
    'ExperimentAquaSOT',
]
