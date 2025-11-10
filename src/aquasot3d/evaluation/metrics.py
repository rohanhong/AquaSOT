"""
Tracking Metrics for AquaSOT-3D

This module implements standard tracking evaluation metrics following the
RGBS50 benchmark protocol, including Success (AUC), Precision, and Center Error.

Metrics are computed separately for RGB, Sonar, and Fusion predictions.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box [x1, y1, x2, y2] or [cx, cy, w, h]
        bbox2: Second bounding box [x1, y1, x2, y2] or [cx, cy, w, h]

    Returns:
        IoU value in [0, 1]

    Example:
        >>> bbox1 = np.array([10, 10, 50, 50])  # x1,y1,x2,y2
        >>> bbox2 = np.array([20, 20, 60, 60])
        >>> iou = compute_iou(bbox1, bbox2)
    """
    # Convert to xyxy format if needed
    if bbox1.shape[-1] == 4 and bbox2.shape[-1] == 4:
        # Check if format is cxcywh (values typically < image size)
        if np.all(bbox1[2:] > 0) and np.all(bbox2[2:] > 0):
            # Assume cxcywh, convert to xyxy
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
        else:
            bbox1_xyxy = bbox1
            bbox2_xyxy = bbox2
    else:
        bbox1_xyxy = bbox1
        bbox2_xyxy = bbox2

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


def compute_center_error(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute center location error between two bounding boxes.

    Args:
        bbox1: First bounding box [cx, cy, w, h] or [x1, y1, x2, y2]
        bbox2: Second bounding box [cx, cy, w, h] or [x1, y1, x2, y2]

    Returns:
        Euclidean distance between centers

    Example:
        >>> bbox1 = np.array([50, 50, 40, 40])  # cx, cy, w, h
        >>> bbox2 = np.array([55, 55, 40, 40])
        >>> error = compute_center_error(bbox1, bbox2)
    """
    # Extract centers
    if bbox1.shape[-1] == 4:
        # Check format
        if bbox1[2] > 0 and bbox1[3] > 0:  # Likely cxcywh
            center1 = bbox1[:2]
        else:  # xyxy format
            center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])

        if bbox2[2] > 0 and bbox2[3] > 0:
            center2 = bbox2[:2]
        else:
            center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
    else:
        center1 = bbox1[:2]
        center2 = bbox2[:2]

    # Compute Euclidean distance
    distance = np.sqrt(np.sum((center1 - center2) ** 2))
    return float(distance)


def compute_success_curve(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    thresholds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute success curve (overlap precision) at different IoU thresholds.

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        thresholds: IoU thresholds (default: 0 to 1 with 0.05 step)

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
    ious = []
    for pred_box, gt_box in zip(pred_bboxes, gt_bboxes):
        if pred_box is not None and gt_box is not None:
            iou = compute_iou(pred_box, gt_box)
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
    thresholds: Optional[np.ndarray] = None
) -> float:
    """
    Compute Area Under Curve (AUC) of success plot.

    This is the primary metric in VOT and RGBS50 benchmarks.

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        thresholds: IoU thresholds (default: 0 to 1 with 0.05 step)

    Returns:
        AUC value in [0, 1]

    Example:
        >>> auc = compute_success_auc(pred_boxes, gt_boxes)
    """
    thresholds, success_rates = compute_success_curve(pred_bboxes, gt_bboxes, thresholds)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(success_rates, thresholds)

    return float(auc)


def compute_precision_curve(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    thresholds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision curve at different center error thresholds.

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        thresholds: Center error thresholds in pixels (default: 0 to 50 with step 1)

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
    errors = []
    for pred_box, gt_box in zip(pred_bboxes, gt_bboxes):
        if pred_box is not None and gt_box is not None:
            error = compute_center_error(pred_box, gt_box)
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
    threshold: float = 20.0
) -> float:
    """
    Compute precision at a specific center error threshold.

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        threshold: Center error threshold in pixels (default: 20)

    Returns:
        Precision value in [0, 1]

    Example:
        >>> precision = compute_precision_at_threshold(pred_boxes, gt_boxes, threshold=20)
    """
    _, precisions = compute_precision_curve(pred_bboxes, gt_bboxes, np.array([threshold]))
    return float(precisions[0])


def compute_average_center_error(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray]
) -> float:
    """
    Compute average center location error across all frames.

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes

    Returns:
        Average center error in pixels

    Example:
        >>> ace = compute_average_center_error(pred_boxes, gt_boxes)
    """
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return float('inf')

    errors = []
    for pred_box, gt_box in zip(pred_bboxes, gt_bboxes):
        if pred_box is not None and gt_box is not None:
            error = compute_center_error(pred_box, gt_box)
            errors.append(error)

    if len(errors) == 0:
        return float('inf')

    return float(np.mean(errors))


def compute_sequence_metrics(
    pred_bboxes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    modality: str = 'fusion'
) -> Dict[str, float]:
    """
    Compute all standard tracking metrics for a sequence.

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        modality: Modality name ('rgb', 'sonar', or 'fusion')

    Returns:
        Dictionary containing all metrics

    Example:
        >>> metrics = compute_sequence_metrics(pred_boxes, gt_boxes, 'fusion')
        >>> print(f"AUC: {metrics['success_auc']:.4f}")
    """
    metrics = {
        'modality': modality,
        'num_frames': len(gt_bboxes),
        'num_valid_predictions': sum(1 for b in pred_bboxes if b is not None)
    }

    # Success (AUC)
    metrics['success_auc'] = compute_success_auc(pred_bboxes, gt_bboxes)

    # Precision at 20 pixels
    metrics['precision_20'] = compute_precision_at_threshold(pred_bboxes, gt_bboxes, threshold=20.0)

    # Average center error
    metrics['avg_center_error'] = compute_average_center_error(pred_bboxes, gt_bboxes)

    # Additional metrics
    # Mean IoU
    ious = [compute_iou(pred, gt) for pred, gt in zip(pred_bboxes, gt_bboxes)
            if pred is not None and gt is not None]
    metrics['mean_iou'] = float(np.mean(ious)) if len(ious) > 0 else 0.0

    # Success rate at 0.5 IoU threshold (PASCAL VOC metric)
    metrics['success_0.5'] = float(np.mean([iou > 0.5 for iou in ious])) if len(ious) > 0 else 0.0

    return metrics


def aggregate_metrics(
    sequence_metrics: List[Dict[str, float]],
    modality: str = 'fusion'
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple sequences.

    Args:
        sequence_metrics: List of per-sequence metrics dictionaries
        modality: Modality name

    Returns:
        Aggregated metrics dictionary

    Example:
        >>> seq_metrics = [compute_sequence_metrics(pred1, gt1), ...]
        >>> overall = aggregate_metrics(seq_metrics, 'fusion')
    """
    if len(sequence_metrics) == 0:
        return {'modality': modality, 'num_sequences': 0}

    aggregated = {
        'modality': modality,
        'num_sequences': len(sequence_metrics)
    }

    # Aggregate each metric
    metric_names = ['success_auc', 'precision_20', 'avg_center_error', 'mean_iou', 'success_0.5']

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


def compute_multi_modal_metrics(
    rgb_pred: List[np.ndarray],
    sonar_pred: List[np.ndarray],
    fusion_pred: List[np.ndarray],
    rgb_gt: List[np.ndarray],
    sonar_gt: List[np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for all three modalities (RGB, Sonar, Fusion).

    Args:
        rgb_pred: RGB predictions
        sonar_pred: Sonar predictions
        fusion_pred: Fusion predictions
        rgb_gt: RGB ground truth
        sonar_gt: Sonar ground truth

    Returns:
        Dictionary with metrics for each modality

    Example:
        >>> metrics = compute_multi_modal_metrics(
        ...     rgb_pred, sonar_pred, fusion_pred,
        ...     rgb_gt, sonar_gt
        ... )
        >>> print(f"Fusion AUC: {metrics['fusion']['success_auc']:.4f}")
    """
    metrics = {}

    # RGB metrics
    metrics['rgb'] = compute_sequence_metrics(rgb_pred, rgb_gt, 'rgb')

    # Sonar metrics
    metrics['sonar'] = compute_sequence_metrics(sonar_pred, sonar_gt, 'sonar')

    # Fusion metrics (compare against both RGB and Sonar GT, take average)
    fusion_rgb_metrics = compute_sequence_metrics(fusion_pred, rgb_gt, 'fusion')
    fusion_sonar_metrics = compute_sequence_metrics(fusion_pred, sonar_gt, 'fusion')

    # Average fusion metrics
    metrics['fusion'] = {
        'modality': 'fusion',
        'num_frames': fusion_rgb_metrics['num_frames'],
        'num_valid_predictions': fusion_rgb_metrics['num_valid_predictions']
    }

    for key in ['success_auc', 'precision_20', 'avg_center_error', 'mean_iou', 'success_0.5']:
        metrics['fusion'][key] = (fusion_rgb_metrics[key] + fusion_sonar_metrics[key]) / 2.0

    return metrics


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

    for modality in ['rgb', 'sonar', 'fusion']:
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
