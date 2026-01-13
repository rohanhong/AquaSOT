"""
Evaluation metrics for object tracking.

Based on got10k-toolkit metrics.py.
"""

import numpy as np


def rect_iou(rects1, rects2, bound=None):
    """Calculate intersection over union (IoU) between rectangles.

    Args:
        rects1: Array of boxes (N, 4) in [x, y, w, h] format
        rects2: Array of boxes (N, 4) in [x, y, w, h] format
        bound: Optional (H, W) tuple to clip boxes

    Returns:
        iou: Array of IoU values (N,)
    """
    rects1 = np.asarray(rects1)
    rects2 = np.asarray(rects2)

    # Handle single box case
    if rects1.ndim == 1:
        rects1 = rects1[np.newaxis, :]
    if rects2.ndim == 1:
        rects2 = rects2[np.newaxis, :]

    assert rects1.shape == rects2.shape

    if bound is not None:
        # Clip boxes to image bounds
        rects1 = _clip_boxes(rects1, bound)
        rects2 = _clip_boxes(rects2, bound)

    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    x1_1 = rects1[:, 0]
    y1_1 = rects1[:, 1]
    x2_1 = rects1[:, 0] + rects1[:, 2]
    y2_1 = rects1[:, 1] + rects1[:, 3]

    x1_2 = rects2[:, 0]
    y1_2 = rects2[:, 1]
    x2_2 = rects2[:, 0] + rects2[:, 2]
    y2_2 = rects2[:, 1] + rects2[:, 3]

    # Calculate intersection
    x1_i = np.maximum(x1_1, x1_2)
    y1_i = np.maximum(y1_1, y1_2)
    x2_i = np.minimum(x2_1, x2_2)
    y2_i = np.minimum(y2_1, y2_2)

    w_i = np.maximum(0, x2_i - x1_i)
    h_i = np.maximum(0, y2_i - y1_i)
    intersection = w_i * h_i

    # Calculate union
    area1 = rects1[:, 2] * rects1[:, 3]
    area2 = rects2[:, 2] * rects2[:, 3]
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = np.zeros(len(rects1))
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]

    return iou


def center_error(rects1, rects2, bound=None):
    """Calculate center error between rectangles.

    Args:
        rects1: Array of boxes (N, 4) in [x, y, w, h] format
        rects2: Array of boxes (N, 4) in [x, y, w, h] format
        bound: Optional (H, W) tuple (not used, for API compatibility)

    Returns:
        errors: Array of center errors in pixels (N,)
    """
    rects1 = np.asarray(rects1)
    rects2 = np.asarray(rects2)

    # Handle single box case
    if rects1.ndim == 1:
        rects1 = rects1[np.newaxis, :]
    if rects2.ndim == 1:
        rects2 = rects2[np.newaxis, :]

    assert rects1.shape == rects2.shape

    # Calculate centers
    cx1 = rects1[:, 0] + rects1[:, 2] / 2
    cy1 = rects1[:, 1] + rects1[:, 3] / 2
    cx2 = rects2[:, 0] + rects2[:, 2] / 2
    cy2 = rects2[:, 1] + rects2[:, 3] / 2

    # Calculate Euclidean distance
    errors = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    return errors


def success_curve(ious, thresholds=None):
    """Calculate success curve from IoU values.

    Args:
        ious: Array of IoU values (N,) or list of arrays
        thresholds: Array of threshold values (default: 0 to 1 in 101 steps)

    Returns:
        curve: Success rate at each threshold (len(thresholds),)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    # Handle list of sequences
    if isinstance(ious, list):
        ious = np.concatenate(ious)

    ious = np.asarray(ious)

    # Calculate success rate at each threshold
    curve = np.zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        curve[i] = np.mean(ious >= thr)

    return curve


def precision_curve(errors, thresholds=None):
    """Calculate precision curve from center errors.

    Args:
        errors: Array of center errors (N,) or list of arrays
        thresholds: Array of threshold values in pixels (default: 0 to 50 in 51 steps)

    Returns:
        curve: Precision rate at each threshold (len(thresholds),)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 50, 51)

    # Handle list of sequences
    if isinstance(errors, list):
        errors = np.concatenate(errors)

    errors = np.asarray(errors)

    # Calculate precision rate at each threshold
    curve = np.zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        curve[i] = np.mean(errors <= thr)

    return curve


def calculate_auc(curve, start=0.0, end=1.0):
    """Calculate area under curve (AUC).

    Args:
        curve: Curve values (N,)
        start: Start of x-axis (default: 0.0)
        end: End of x-axis (default: 1.0)

    Returns:
        auc: Area under curve
    """
    # Use trapezoidal integration
    x = np.linspace(start, end, len(curve))
    auc = np.trapz(curve, x) / (end - start)
    return auc


def _clip_boxes(boxes, bound):
    """Clip boxes to image bounds.

    Args:
        boxes: Array of boxes (N, 4) in [x, y, w, h] format
        bound: (H, W) tuple

    Returns:
        Clipped boxes
    """
    boxes = boxes.copy()
    H, W = bound

    # Clip x and y to [0, W-1] and [0, H-1]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, W - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, H - 1)

    # Clip x2 and y2
    x2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, W)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, H)

    # Update width and height
    boxes[:, 2] = x2 - boxes[:, 0]
    boxes[:, 3] = y2 - boxes[:, 1]

    return boxes
