"""
Evaluation and Metrics Module for AquaSOT-3D

This module provides comprehensive evaluation tools for tracking performance
assessment, including metrics computation, evaluation pipeline, and visualization.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

from aquasot3d.evaluation.metrics import (
    compute_iou,
    compute_center_error,
    compute_success_curve,
    compute_success_auc,
    compute_precision_curve,
    compute_precision_at_threshold,
    compute_sequence_metrics,
    aggregate_metrics,
    format_metrics_report
)

from aquasot3d.evaluation.evaluator import (
    TrackingEvaluator,
    evaluate_model
)

from aquasot3d.evaluation.online_evaluator import (
    OnlineTrackingEvaluator
)

from aquasot3d.evaluation.visualizer import (
    TrackingVisualizer
)

__all__ = [
    # Metrics functions
    'compute_iou',
    'compute_center_error',
    'compute_success_curve',
    'compute_success_auc',
    'compute_precision_curve',
    'compute_precision_at_threshold',
    'compute_sequence_metrics',
    'aggregate_metrics',
    'format_metrics_report',

    # Evaluator (offline - uses GT for search center, for debugging)
    'TrackingEvaluator',
    'evaluate_model',

    # Online Evaluator (standard protocol - uses prediction for search center)
    'OnlineTrackingEvaluator',

    # Visualizer
    'TrackingVisualizer'
]
