"""
Bounding box operations from OSTrack.
"""

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        x: Boxes in cxcywh format [..., 4]

    Returns:
        Boxes in xyxy format [..., 4]
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        x: Boxes in xyxy format [..., 4]

    Returns:
        Boxes in cxcywh format [..., 4]
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    """Convert boxes from (x, y, w, h) to (x1, y1, x2, y2).

    Args:
        x: Boxes in xywh format [..., 4]

    Returns:
        Boxes in xyxy format [..., 4]
    """
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    """Convert boxes from (x1, y1, x2, y2) to (x, y, w, h).

    Args:
        x: Boxes in xyxy format [..., 4]

    Returns:
        Boxes in xywh format [..., 4]
    """
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [M, 4] in xyxy format

    Returns:
        iou: [N, M] IoU matrix
        union: [N, M] Union areas
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """Compute Generalized IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [N, 4] in xyxy format

    Returns:
        giou: [N] GIoU values
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)
    iou = torch.diag(iou)
    union = torch.diag(union)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]

    giou = iou - (area - union) / area
    return giou


def giou_loss(pred_boxes, gt_boxes):
    """Compute GIoU loss.

    Args:
        pred_boxes: [B, 4] in xyxy format
        gt_boxes: [B, 4] in xyxy format

    Returns:
        loss: Scalar GIoU loss
        iou: [B] IoU values
    """
    giou = generalized_box_iou(pred_boxes, gt_boxes)
    iou, _ = box_iou(pred_boxes, gt_boxes)
    iou = torch.diag(iou)

    loss = 1 - giou
    return loss.mean(), iou


def l1_loss(pred_boxes, gt_boxes):
    """Compute L1 loss.

    Args:
        pred_boxes: [B, 4]
        gt_boxes: [B, 4]

    Returns:
        loss: Scalar L1 loss
    """
    return torch.nn.functional.l1_loss(pred_boxes, gt_boxes, reduction='mean')
