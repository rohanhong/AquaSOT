"""
Tracking Metrics for AquaSOT-3D

This module implements standard tracking evaluation metrics following the
RGBS50 benchmark protocol, including Success (AUC), Precision, and Center Error.

Metrics are computed separately for RGB and Sonar predictions.

ARCHITECTURE UPDATE (Audit 2025112313): Fusion removed (dual-view fusion architecture).

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_iou(
    bbox1: np.ndarray,
    bbox2: np.ndarray,
    format: str = 'cxcywh'
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    CRITICAL FIX (Audit 2025111409 Finding 2): Removed format guessing heuristics.
    Format must be explicitly specified to avoid mis-parsing positive xyxy boxes as cxcywh.

    BEST PRACTICE (Audit 2025111411 Finding 3): Use type-safe bbox wrappers:
        from aquasot3d.utils import BBoxXYWH, BBoxCXCYWH
        bbox1 = BBoxXYWH(np.array([10, 20, 40, 50]))  # Explicit xywh
        bbox2_cxcywh = BBoxCXCYWH(pred_boxes[i])      # Explicit cxcywh
        bbox2 = bbox2_cxcywh.to_xywh()                # Type-safe conversion
        iou = compute_iou(bbox1.numpy(), bbox2.numpy(), format='xywh')

    Args:
        bbox1: First bounding box [4,] array
        bbox2: Second bounding box [4,] array
        format: Box format - 'xywh' (x,y,w,h), 'cxcywh' (cx,cy,w,h), or 'xyxy' (x1,y1,x2,y2)

    Returns:
        IoU value in [0, 1]

    Example:
        >>> bbox1 = np.array([50, 50, 40, 40])  # cx, cy, w, h
        >>> bbox2 = np.array([55, 55, 40, 40])
        >>> iou = compute_iou(bbox1, bbox2, format='cxcywh')
    """
    # CRITICAL VALIDATION (Audit 2025111411 Finding 3):
    # Validate bbox shape and format to prevent silent errors
    if bbox1.shape != (4,) or bbox2.shape != (4,):
        raise ValueError(
            f"Invalid bbox shapes: bbox1={bbox1.shape}, bbox2={bbox2.shape}. "
            f"Expected (4,) arrays in format '{format}'."
        )

    # Validate format is recognized
    if format not in ['xywh', 'cxcywh', 'xyxy']:
        raise ValueError(
            f"Unknown bbox format: '{format}'. "
            f"Use 'xywh', 'cxcywh', or 'xyxy'. "
            f"Prefer type-safe wrappers (BBoxXYWH, BBoxCXCYWH) to avoid confusion."
        )

    # Warn if boxes appear to be normalized [0,1] when pixel coords expected
    if format in ['xywh', 'xyxy'] and np.all(bbox1 <= 1.0) and np.all(bbox2 <= 1.0):
        import warnings
        warnings.warn(
            f"Boxes in range [0,1] detected for format='{format}'. "
            f"Evaluation metrics expect pixel coordinates. "
            f"Use denormalize_bbox() to convert from normalized [0,1] to pixels.",
            RuntimeWarning
        )

    # Convert to xyxy format if needed
    if format == 'cxcywh':
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        bbox1_xyxy = np.array([
            bbox1[0] - bbox1[2] / 2,
            bbox1[1] - bbox1[3] / 2,
            bbox1[0] + bbox1[2] / 2,
            bbox1[1] + bbox1[3] / 2
        ])
        bbox2_xyxy = np.array([
            bbox2[0] - bbox2[2] / 2,
            bbox2[1] - bbox2[3] / 2,
            bbox2[0] + bbox2[2] / 2,
            bbox2[1] + bbox2[3] / 2
        ])
    elif format == 'xywh':
        # Convert from [x, y, w, h] (top-left corner) to [x1, y1, x2, y2]
        bbox1_xyxy = np.array([
            bbox1[0],
            bbox1[1],
            bbox1[0] + bbox1[2],
            bbox1[1] + bbox1[3]
        ])
        bbox2_xyxy = np.array([
            bbox2[0],
            bbox2[1],
            bbox2[0] + bbox2[2],
            bbox2[1] + bbox2[3]
        ])
    elif format == 'xyxy':
        bbox1_xyxy = bbox1
        bbox2_xyxy = bbox2
    else:
        raise ValueError(f"Unknown format: {format}. Must be 'cxcywh', 'xywh', or 'xyxy'")

    # Compute intersection
    x1_inter = max(bbox1_xyxy[0], bbox2_xyxy[0])
    y1_inter = max(bbox1_xyxy[1], bbox2_xyxy[1])
    x2_inter = min(bbox1_xyxy[2], bbox2_xyxy[2])
    y2_inter = min(bbox1_xyxy[3], bbox2_xyxy[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Compute union
    bbox1_area = (bbox1_xyxy[2] - bbox1_xyxy[0]) * (bbox1_xyxy[3] - bbox1_xyxy[1])
    bbox2_area = (bbox2_xyxy[2] - bbox2_xyxy[0]) * (bbox2_xyxy[3] - bbox2_xyxy[1])
    union_area = bbox1_area + bbox2_area - inter_area

    # Compute IoU
    if union_area <= 0:
        return 0.0

    iou = inter_area / union_area
    return float(iou)


def compute_center_error(
    bbox1: np.ndarray,
    bbox2: np.ndarray,
    format: str = 'cxcywh'
) -> float:
    """
    Compute center location error between two bounding boxes.

    CRITICAL FIX (Audit 2025111409 Finding 2): Removed format guessing heuristics.
    Format must be explicitly specified.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        format: Box format - 'cxcywh' or 'xyxy'

    Returns:
        Euclidean distance between centers

    Example:
        >>> bbox1 = np.array([50, 50, 40, 40])  # cx, cy, w, h
        >>> bbox2 = np.array([55, 55, 40, 40])
        >>> error = compute_center_error(bbox1, bbox2, format='cxcywh')
    """
    # Extract centers based on format
    if format == 'cxcywh':
        # Centers are already at indices 0, 1
        center1 = bbox1[:2]
        center2 = bbox2[:2]
    elif format == 'xywh':
        # Compute centers from top-left corner + dimensions
        center1 = np.array([bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2])
        center2 = np.array([bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2])
    elif format == 'xyxy':
        # Compute centers from corners
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
    else:
        raise ValueError(f"Unknown format: {format}. Must be 'cxcywh', 'xywh', or 'xyxy'")

    # Compute Euclidean distance
    distance = np.sqrt(np.sum((center1 - center2) ** 2))
    return float(distance)


def _is_valid_bbox(bbox: np.ndarray) -> bool:
    """
    Check if a bounding box is valid (non-zero and finite).

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    Determines whether a bbox should be included in metrics computation.
    Filters out occluded frames (0,0,0,0) and invalid predictions.

    Args:
        bbox: Bounding box array [4] or None

    Returns:
        True if bbox is valid for metrics computation
    """
    if bbox is None:
        return False
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)
    if bbox.shape != (4,):
        return False
    # Check for NaN/Inf
    if not np.all(np.isfinite(bbox)):
        return False
    # Check for zero bbox (occlusion marker)
    if np.sum(np.abs(bbox)) == 0:
        return False
    # Check for degenerate bbox (zero or negative width/height)
    if len(bbox) >= 4 and (bbox[2] <= 0 or bbox[3] <= 0):
        return False
    return True


def compute_success_curve(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    thresholds: Optional[np.ndarray] = None,
    visibility_mask: Optional[List[bool]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute success curve (overlap precision) at different IoU thresholds.

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    - Added visibility_mask parameter to explicitly skip frames
    - Automatically skips frames where GT is zero (occlusion) or pred is None
    - Follows LaSOT/OTB/GOT-10k standard evaluation protocol

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        thresholds: IoU thresholds (default: 0 to 1 with 0.05 step)
        visibility_mask: Optional list of booleans indicating valid frames
                        (True=include in metrics, False=skip)

    Returns:
        thresholds: IoU thresholds
        success_rates: Success rate at each threshold

    Example:
        >>> pred_boxes = [np.array([10, 10, 50, 50]) for _ in range(100)]
        >>> gt_boxes = [np.array([12, 12, 52, 52]) for _ in range(100)]
        >>> thresholds, success = compute_success_curve(pred_boxes, gt_boxes)
    """
    if thresholds is None:
        thresholds = np.arange(0, 1.05, 0.05)

    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return thresholds, np.zeros_like(thresholds)

    # Compute IoU for each frame
    # CRITICAL FIX (Audit 2025111409 Finding 2): Explicitly specify bbox format
    # Dataset returns [x, y, w, h] (xywh format), not [cx, cy, w, h]
    # CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21): Skip invalid frames
    ious = []
    for i, (pred_box, gt_box) in enumerate(zip(pred_bboxes, gt_bboxes)):
        # Check explicit visibility mask first
        if visibility_mask is not None and not visibility_mask[i]:
            continue
        # Check if GT is valid (non-zero, non-occluded)
        if not _is_valid_bbox(gt_box):
            continue
        # Check if prediction is valid
        if not _is_valid_bbox(pred_box):
            continue
        iou = compute_iou(pred_box, gt_box, format='xywh')
        ious.append(iou)

    if len(ious) == 0:
        return thresholds, np.zeros_like(thresholds)

    ious = np.array(ious)

    # Compute success rate at each threshold
    success_rates = np.array([np.mean(ious > thr) for thr in thresholds])

    return thresholds, success_rates


def compute_success_auc(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    thresholds: Optional[np.ndarray] = None,
    visibility_mask: Optional[List[bool]] = None
) -> float:
    """
    Compute Area Under Curve (AUC) of success plot.

    This is the primary metric in VOT and RGBS50 benchmarks.

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    - Added visibility_mask parameter to explicitly skip frames
    - Passes mask through to compute_success_curve for consistent filtering

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        thresholds: IoU thresholds (default: 0 to 1 with 0.05 step)
        visibility_mask: Optional list of booleans indicating valid frames

    Returns:
        AUC value in [0, 1]

    Example:
        >>> auc = compute_success_auc(pred_boxes, gt_boxes)
    """
    thresholds, success_rates = compute_success_curve(
        pred_bboxes, gt_bboxes, thresholds, visibility_mask
    )

    # Compute AUC using trapezoidal rule
    auc = np.trapz(success_rates, thresholds)

    return float(auc)


def compute_precision_curve(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    thresholds: Optional[np.ndarray] = None,
    visibility_mask: Optional[List[bool]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision curve at different center error thresholds.

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    - Added visibility_mask parameter to explicitly skip frames
    - Automatically skips frames where GT is zero (occlusion) or pred is None
    - Prevents inflated precision from zero-vs-zero center errors

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        thresholds: Center error thresholds in pixels (default: 0 to 50 with step 1)
        visibility_mask: Optional list of booleans indicating valid frames

    Returns:
        thresholds: Center error thresholds
        precisions: Precision at each threshold

    Example:
        >>> thresholds, precision = compute_precision_curve(pred_boxes, gt_boxes)
    """
    if thresholds is None:
        thresholds = np.arange(0, 51, 1)

    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return thresholds, np.zeros_like(thresholds, dtype=float)

    # Compute center error for each frame
    # CRITICAL FIX (Audit 2025111409 Finding 2): Explicitly specify bbox format
    # CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21): Skip invalid frames
    errors = []
    for i, (pred_box, gt_box) in enumerate(zip(pred_bboxes, gt_bboxes)):
        # Check explicit visibility mask first
        if visibility_mask is not None and not visibility_mask[i]:
            continue
        # Check if GT is valid (non-zero, non-occluded)
        if not _is_valid_bbox(gt_box):
            continue
        # Check if prediction is valid
        if not _is_valid_bbox(pred_box):
            continue
        error = compute_center_error(pred_box, gt_box, format='xywh')
        errors.append(error)

    if len(errors) == 0:
        return thresholds, np.zeros_like(thresholds, dtype=float)

    errors = np.array(errors)

    # Compute precision at each threshold
    precisions = np.array([np.mean(errors <= thr) for thr in thresholds], dtype=float)

    return thresholds, precisions


def compute_precision_at_threshold(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    threshold: float = 20.0,
    visibility_mask: Optional[List[bool]] = None
) -> float:
    """
    Compute precision at a specific center error threshold.

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    - Added visibility_mask parameter to explicitly skip frames

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        threshold: Center error threshold in pixels (default: 20)
        visibility_mask: Optional list of booleans indicating valid frames

    Returns:
        Precision value in [0, 1]

    Example:
        >>> precision = compute_precision_at_threshold(pred_boxes, gt_boxes, threshold=20)
    """
    _, precisions = compute_precision_curve(
        pred_bboxes, gt_bboxes, np.array([threshold]), visibility_mask
    )
    return float(precisions[0])


def compute_average_center_error(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    visibility_mask: Optional[List[bool]] = None
) -> float:
    """
    Compute average center location error across all frames.

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    - Added visibility_mask parameter to explicitly skip frames
    - Automatically skips frames where GT is zero (occlusion) or pred is None
    - Prevents inflated center error from zero-vs-prediction distances

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        visibility_mask: Optional list of booleans indicating valid frames

    Returns:
        Average center error in pixels

    Example:
        >>> ace = compute_average_center_error(pred_boxes, gt_boxes)
    """
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return float('inf')

    # CRITICAL FIX (Audit 2025111409 Finding 2): Explicitly specify bbox format
    # CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21): Skip invalid frames
    errors = []
    for i, (pred_box, gt_box) in enumerate(zip(pred_bboxes, gt_bboxes)):
        # Check explicit visibility mask first
        if visibility_mask is not None and not visibility_mask[i]:
            continue
        # Check if GT is valid (non-zero, non-occluded)
        if not _is_valid_bbox(gt_box):
            continue
        # Check if prediction is valid
        if not _is_valid_bbox(pred_box):
            continue
        error = compute_center_error(pred_box, gt_box, format='xywh')
        errors.append(error)

    if len(errors) == 0:
        return float('inf')

    return float(np.mean(errors))


def compute_success_at_threshold(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    threshold: float = 0.5,
    visibility_mask: Optional[List[bool]] = None
) -> float:
    """
    Compute success rate at a specific IoU threshold.

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    - Added visibility_mask parameter to explicitly skip frames
    - Automatically skips frames where GT is zero (occlusion) or pred is None

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        threshold: IoU threshold (default: 0.5)
        visibility_mask: Optional list of booleans indicating valid frames

    Returns:
        Success rate (proportion of frames with IoU > threshold)

    Example:
        >>> success = compute_success_at_threshold(pred_boxes, gt_boxes, threshold=0.7)
    """
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return 0.0

    # CRITICAL FIX (Audit 2025111409 Finding 2): Explicitly specify bbox format
    # CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21): Skip invalid frames
    ious = []
    for i, (pred_box, gt_box) in enumerate(zip(pred_bboxes, gt_bboxes)):
        # Check explicit visibility mask first
        if visibility_mask is not None and not visibility_mask[i]:
            continue
        # Check if GT is valid (non-zero, non-occluded)
        if not _is_valid_bbox(gt_box):
            continue
        # Check if prediction is valid
        if not _is_valid_bbox(pred_box):
            continue
        iou = compute_iou(pred_box, gt_box, format='xywh')
        ious.append(iou)

    if len(ious) == 0:
        return 0.0

    return float(np.mean([iou > threshold for iou in ious]))


def compute_sequence_metrics(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    modality: str = 'rgb',
    additional_metrics: Optional[List[str]] = None,
    visibility_mask: Optional[List[bool]] = None
) -> Dict[str, float]:
    """
    Compute all standard tracking metrics for a sequence.

    ARCHITECTURE UPDATE (Audit 2025112313): Default modality changed to 'rgb'.

    CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21):
    - Added visibility_mask parameter to explicitly skip frames
    - Passes mask through to all sub-metric functions
    - Automatically filters occluded (GT=0) and invalid prediction frames

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        modality: Modality name ('rgb' or 'sonar')
        additional_metrics: Optional list of additional metrics to compute.
            Supported values:
                - 'precision_N': Precision at N pixels (e.g., 'precision_10', 'precision_5')
                - 'success_X': Success at IoU threshold X (e.g., 'success_0.7', 'success_0.6')
        visibility_mask: Optional list of booleans indicating valid frames

    Returns:
        Dictionary containing all metrics

    Example:
        >>> metrics = compute_sequence_metrics(
        ...     pred_boxes, gt_boxes, 'rgb',
        ...     additional_metrics=['precision_10', 'precision_5', 'success_0.7']
        ... )
        >>> print(f"AUC: {metrics['success_auc']:.4f}")
        >>> print(f"Precision@10: {metrics['precision_10']:.4f}")
    """
    # CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21): Count valid frames properly
    num_valid_gt = sum(1 for gt in gt_bboxes if _is_valid_bbox(gt))
    num_valid_pred = sum(1 for pred in pred_bboxes if _is_valid_bbox(pred))

    metrics = {
        'modality': modality,
        'num_frames': len(gt_bboxes),
        'num_valid_gt_frames': num_valid_gt,
        'num_valid_predictions': num_valid_pred
    }

    # Success (AUC) - with visibility masking
    metrics['success_auc'] = compute_success_auc(
        pred_bboxes, gt_bboxes, visibility_mask=visibility_mask
    )

    # Precision at 20 pixels (standard metric) - with visibility masking
    metrics['precision_20'] = compute_precision_at_threshold(
        pred_bboxes, gt_bboxes, threshold=20.0, visibility_mask=visibility_mask
    )

    # Average center error - with visibility masking
    metrics['avg_center_error'] = compute_average_center_error(
        pred_bboxes, gt_bboxes, visibility_mask=visibility_mask
    )

    # Additional standard metrics
    # Mean IoU - with proper validity checking
    # CRITICAL FIX (Audit 2025111409 Finding 2): Explicitly specify bbox format
    # CRITICAL FIX (Audit 2025120403 Issue 1, 8, 21): Skip invalid frames
    ious = []
    for i, (pred, gt) in enumerate(zip(pred_bboxes, gt_bboxes)):
        if visibility_mask is not None and not visibility_mask[i]:
            continue
        if not _is_valid_bbox(gt) or not _is_valid_bbox(pred):
            continue
        ious.append(compute_iou(pred, gt, format='xywh'))
    metrics['mean_iou'] = float(np.mean(ious)) if len(ious) > 0 else 0.0

    # Success rate at 0.5 IoU threshold (PASCAL VOC metric)
    metrics['success_0.5'] = float(np.mean([iou > 0.5 for iou in ious])) if len(ious) > 0 else 0.0

    # Compute additional metrics if requested
    if additional_metrics:
        for metric_name in additional_metrics:
            # Parse precision metrics (e.g., 'precision_10' -> threshold=10)
            if metric_name.startswith('precision_'):
                try:
                    threshold = float(metric_name.split('_')[1])
                    metrics[metric_name] = compute_precision_at_threshold(
                        pred_bboxes, gt_bboxes, threshold=threshold,
                        visibility_mask=visibility_mask
                    )
                except (IndexError, ValueError):
                    logger.warning(f"Invalid metric name: {metric_name}, skipping")

            # Parse success metrics (e.g., 'success_0.7' -> threshold=0.7)
            elif metric_name.startswith('success_'):
                try:
                    threshold = float(metric_name.split('_')[1].replace('_', '.'))
                    metrics[metric_name] = compute_success_at_threshold(
                        pred_bboxes, gt_bboxes, threshold=threshold,
                        visibility_mask=visibility_mask
                    )
                except (IndexError, ValueError):
                    logger.warning(f"Invalid metric name: {metric_name}, skipping")

    return metrics


def aggregate_metrics(
    sequence_metrics: List[Dict[str, float]],
    modality: str = 'rgb'
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple sequences.

    ARCHITECTURE UPDATE (Audit 2025112313): Default modality changed to 'rgb'.

    Args:
        sequence_metrics: List of per-sequence metrics dictionaries
        modality: Modality name ('rgb' or 'sonar')

    Returns:
        Aggregated metrics dictionary

    Example:
        >>> seq_metrics = [compute_sequence_metrics(pred1, gt1), ...]
        >>> overall = aggregate_metrics(seq_metrics, 'rgb')
    """
    if len(sequence_metrics) == 0:
        return {'modality': modality, 'num_sequences': 0}

    aggregated = {
        'modality': modality,
        'num_sequences': len(sequence_metrics)
    }

    # Collect all metric names (standard + any additional metrics)
    metric_names = ['success_auc', 'precision_20', 'avg_center_error', 'mean_iou', 'success_0.5']

    # Keys to exclude from aggregation (non-numeric metadata)
    exclude_keys = {'modality', 'num_frames', 'num_valid_predictions', 'aggregation_strategy',
                    'rgb_weight', 'sonar_weight'}

    # Discover additional metrics from the sequence metrics
    if len(sequence_metrics) > 0:
        for key in sequence_metrics[0].keys():
            if key not in metric_names and key not in exclude_keys:
                # Only include numeric values
                sample_value = sequence_metrics[0][key]
                if isinstance(sample_value, (int, float, np.number)):
                    metric_names.append(key)

    for metric_name in metric_names:
        values = [m[metric_name] for m in sequence_metrics if metric_name in m]
        if len(values) > 0:
            aggregated[f'{metric_name}_mean'] = float(np.mean(values))
            aggregated[f'{metric_name}_median'] = float(np.median(values))
            aggregated[f'{metric_name}_std'] = float(np.std(values))

    # Total frames
    total_frames = sum(m.get('num_frames', 0) for m in sequence_metrics)
    total_valid = sum(m.get('num_valid_predictions', 0) for m in sequence_metrics)

    aggregated['total_frames'] = total_frames
    aggregated['total_valid_predictions'] = total_valid
    aggregated['valid_prediction_rate'] = total_valid / total_frames if total_frames > 0 else 0.0

    return aggregated


def format_metrics_report(
    metrics: Dict[str, Dict[str, float]],
    sequence_name: Optional[str] = None
) -> str:
    """
    Format metrics into a readable report.

    Args:
        metrics: Metrics dictionary
        sequence_name: Optional sequence name

    Returns:
        Formatted report string

    Example:
        >>> report = format_metrics_report(metrics, 'sequence_001')
        >>> print(report)
    """
    report = []

    if sequence_name:
        report.append(f"Metrics for Sequence: {sequence_name}")
    else:
        report.append("Tracking Metrics Report")

    report.append("=" * 60)

    # ARCHITECTURE UPDATE (Audit 2025112313): Fusion removed
    for modality in ['rgb', 'sonar']:
        if modality not in metrics:
            continue

        m = metrics[modality]
        report.append(f"\n{modality.upper()} Metrics:")
        report.append("-" * 40)
        report.append(f"  Success (AUC):        {m.get('success_auc', 0.0):.4f}")
        report.append(f"  Precision@20:         {m.get('precision_20', 0.0):.4f}")
        report.append(f"  Avg Center Error:     {m.get('avg_center_error', 0.0):.2f} px")
        report.append(f"  Mean IoU:             {m.get('mean_iou', 0.0):.4f}")
        report.append(f"  Success@0.5:          {m.get('success_0.5', 0.0):.4f}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def compute_scenario_metrics(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    scenario_labels: List[str],
    modality: str = 'rgb',
    additional_metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each modality availability scenario.

    CRITICAL FIX (Audit 2025112101 Phase 4.2):
    Enables per-scenario evaluation to diagnose single-modality performance
    and zero-modality handling.

    ARCHITECTURE UPDATE (Audit 2025112313): Default modality changed to 'rgb'.

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        scenario_labels: List of scenario labels for each frame
            - 'dual': Both RGB and Sonar valid
            - 'single_rgb': Only RGB valid
            - 'single_sonar': Only Sonar valid
            - 'zero': Neither modality valid
        modality: Modality name ('rgb' or 'sonar')
        additional_metrics: Optional list of additional metrics to compute

    Returns:
        Dictionary with metrics for each scenario:
        {
            'dual': {...metrics...},
            'single_rgb': {...metrics...},
            'single_sonar': {...metrics...},
            'zero': {...metrics...},
            'overall': {...metrics...}
        }

    Example:
        >>> scenario_labels = ['dual', 'dual', 'single_rgb', 'zero']
        >>> metrics = compute_scenario_metrics(pred_boxes, gt_boxes, scenario_labels, 'rgb')
        >>> print(f"Dual-modality AUC: {metrics['dual']['success_auc']:.4f}")
        >>> print(f"Single-RGB AUC: {metrics['single_rgb']['success_auc']:.4f}")
    """
    scenario_metrics = {}

    # Define all possible scenarios
    scenarios = ['dual', 'single_rgb', 'single_sonar', 'zero']

    # Split predictions by scenario
    for scenario in scenarios:
        # Extract predictions for this scenario
        scenario_pred = []
        scenario_gt = []

        for i, label in enumerate(scenario_labels):
            if label == scenario:
                if i < len(pred_bboxes) and pred_bboxes[i] is not None:
                    scenario_pred.append(pred_bboxes[i])
                    scenario_gt.append(gt_bboxes[i])

        # Compute metrics if we have samples
        if len(scenario_pred) > 0:
            scenario_metrics[scenario] = compute_sequence_metrics(
                scenario_pred,
                scenario_gt,
                modality=f"{modality}_{scenario}",
                additional_metrics=additional_metrics
            )
        else:
            # No samples in this scenario - return empty metrics
            scenario_metrics[scenario] = {
                'modality': f"{modality}_{scenario}",
                'num_frames': 0,
                'num_valid_predictions': 0,
                'success_auc': 0.0,
                'precision_20': 0.0,
                'avg_center_error': float('inf'),
                'mean_iou': 0.0,
                'success_0.5': 0.0
            }

    # Compute overall metrics (all scenarios combined)
    scenario_metrics['overall'] = compute_sequence_metrics(
        pred_bboxes,
        gt_bboxes,
        modality=modality,
        additional_metrics=additional_metrics
    )

    return scenario_metrics


def aggregate_scenario_metrics(
    sequence_scenario_metrics: List[Dict[str, Dict[str, float]]],
    modality: str = 'rgb'
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-scenario metrics across multiple sequences.

    CRITICAL FIX (Audit 2025112101 Phase 4.2):
    Enables dataset-level analysis of per-scenario performance.

    ARCHITECTURE UPDATE (Audit 2025112313): Default modality changed to 'rgb'.

    Args:
        sequence_scenario_metrics: List of per-sequence scenario metrics
        modality: Modality name ('rgb' or 'sonar')

    Returns:
        Aggregated metrics for each scenario

    Example:
        >>> seq_metrics = [compute_scenario_metrics(pred1, gt1, labels1), ...]
        >>> aggregated = aggregate_scenario_metrics(seq_metrics, 'rgb')
        >>> print(f"Avg Dual AUC: {aggregated['dual']['success_auc_mean']:.4f}")
    """
    aggregated = {}

    scenarios = ['dual', 'single_rgb', 'single_sonar', 'zero', 'overall']

    for scenario in scenarios:
        # Extract metrics for this scenario from all sequences
        scenario_metrics = []
        for seq_metrics in sequence_scenario_metrics:
            if scenario in seq_metrics:
                scenario_metrics.append(seq_metrics[scenario])

        # Aggregate
        if len(scenario_metrics) > 0:
            aggregated[scenario] = aggregate_metrics(
                scenario_metrics,
                modality=f"{modality}_{scenario}"
            )
        else:
            # No data for this scenario
            aggregated[scenario] = {
                'modality': f"{modality}_{scenario}",
                'num_sequences': 0
            }

    return aggregated


def format_scenario_metrics_report(
    scenario_metrics: Dict[str, Dict[str, float]],
    sequence_name: Optional[str] = None
) -> str:
    """
    Format per-scenario metrics into a readable report.

    CRITICAL FIX (Audit 2025112101 Phase 4.2):
    Provides detailed breakdown of performance by modality availability.

    Args:
        scenario_metrics: Scenario metrics dictionary
        sequence_name: Optional sequence name

    Returns:
        Formatted report string

    Example:
        >>> report = format_scenario_metrics_report(scenario_metrics, 'sequence_001')
        >>> print(report)
    """
    report = []

    if sequence_name:
        report.append(f"Per-Scenario Metrics for Sequence: {sequence_name}")
    else:
        report.append("Per-Scenario Tracking Metrics Report")

    report.append("=" * 70)

    # Display metrics for each scenario
    scenarios = [
        ('dual', 'Dual-Modality (RGB + Sonar)'),
        ('single_rgb', 'Single-Modality (RGB only)'),
        ('single_sonar', 'Single-Modality (Sonar only)'),
        ('zero', 'Zero-Modality (Low confidence)')
    ]

    for scenario_key, scenario_name in scenarios:
        if scenario_key not in scenario_metrics:
            continue

        m = scenario_metrics[scenario_key]
        num_frames = m.get('num_frames', 0)

        report.append(f"\n{scenario_name}:")
        report.append("-" * 50)

        if num_frames == 0:
            report.append("  No frames in this scenario")
            continue

        report.append(f"  Num Frames:           {num_frames}")
        report.append(f"  Success (AUC):        {m.get('success_auc', 0.0):.4f}")
        report.append(f"  Precision@20:         {m.get('precision_20', 0.0):.4f}")
        report.append(f"  Avg Center Error:     {m.get('avg_center_error', 0.0):.2f} px")
        report.append(f"  Mean IoU:             {m.get('mean_iou', 0.0):.4f}")
        report.append(f"  Success@0.5:          {m.get('success_0.5', 0.0):.4f}")

    # Overall metrics
    if 'overall' in scenario_metrics:
        report.append(f"\nOverall (All Scenarios):")
        report.append("-" * 50)
        m = scenario_metrics['overall']
        report.append(f"  Total Frames:         {m.get('num_frames', 0)}")
        report.append(f"  Success (AUC):        {m.get('success_auc', 0.0):.4f}")
        report.append(f"  Precision@20:         {m.get('precision_20', 0.0):.4f}")
        report.append(f"  Avg Center Error:     {m.get('avg_center_error', 0.0):.2f} px")
        report.append(f"  Mean IoU:             {m.get('mean_iou', 0.0):.4f}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)
