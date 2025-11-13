"""
Bounding Box Utility Functions

Provides unified bbox format conversion and manipulation utilities.
Implements Audit 2025111405 recommendation for formal format contracts.

Supported Formats:
- 'xywh': [x, y, w, h] - top-left corner + width/height (dataset format)
- 'cxcywh': [cx, cy, w, h] - center point + width/height (model format)
- 'xyxy': [x1, y1, x2, y2] - top-left + bottom-right corners

Coordinate Spaces:
- 'pixel': Absolute pixel coordinates
- 'normalized': Normalized to [0, 1] relative to image size

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import torch
import numpy as np
from typing import Union, Literal, Tuple


def convert_bbox_format(
    boxes: Union[torch.Tensor, np.ndarray],
    src_format: Literal['xywh', 'cxcywh', 'xyxy'],
    dst_format: Literal['xywh', 'cxcywh', 'xyxy'],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert bounding boxes between different coordinate formats.

    Based on torchvision.ops.box_convert best practices (Context7 doc).

    Args:
        boxes: Bounding boxes, shape (..., 4)
        src_format: Source format ('xywh', 'cxcywh', or 'xyxy')
        dst_format: Destination format ('xywh', 'cxcywh', or 'xyxy')

    Returns:
        Converted boxes in destination format, same shape and type as input

    Examples:
        >>> boxes_xywh = np.array([[100, 200, 50, 80]])
        >>> boxes_cxcywh = convert_bbox_format(boxes_xywh, 'xywh', 'cxcywh')
        >>> # Result: [[125, 240, 50, 80]]  # cx=x+w/2, cy=y+h/2

        >>> boxes_xyxy = torch.tensor([[10, 20, 50, 80]])
        >>> boxes_xywh = convert_bbox_format(boxes_xyxy, 'xyxy', 'xywh')
        >>> # Result: [[10, 20, 40, 60]]  # w=x2-x1, h=y2-y1
    """
    if src_format == dst_format:
        return boxes

    # Handle numpy arrays
    is_numpy = isinstance(boxes, np.ndarray)
    if is_numpy:
        boxes = torch.from_numpy(boxes)

    # Use torchvision's box_convert if available, else implement manually
    try:
        from torchvision.ops import box_convert
        result = box_convert(boxes, in_fmt=src_format, out_fmt=dst_format)
    except ImportError:
        # Manual implementation
        result = _manual_box_convert(boxes, src_format, dst_format)

    if is_numpy:
        result = result.numpy()

    return result


def _manual_box_convert(
    boxes: torch.Tensor,
    src_format: str,
    dst_format: str
) -> torch.Tensor:
    """Manual bbox conversion fallback when torchvision not available."""
    # First convert to xyxy as intermediate format
    if src_format == 'xywh':
        x, y, w, h = boxes.unbind(-1)
        x2 = x + w
        y2 = y + h
        xyxy = torch.stack([x, y, x2, y2], dim=-1)
    elif src_format == 'cxcywh':
        cx, cy, w, h = boxes.unbind(-1)
        x = cx - w / 2
        y = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = torch.stack([x, y, x2, y2], dim=-1)
    else:  # src_format == 'xyxy'
        xyxy = boxes

    # Convert from xyxy to target format
    if dst_format == 'xyxy':
        return xyxy
    elif dst_format == 'xywh':
        x1, y1, x2, y2 = xyxy.unbind(-1)
        w = x2 - x1
        h = y2 - y1
        return torch.stack([x1, y1, w, h], dim=-1)
    else:  # dst_format == 'cxcywh'
        x1, y1, x2, y2 = xyxy.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)


def normalize_bbox(
    boxes: Union[torch.Tensor, np.ndarray],
    image_size: Tuple[int, int],
    format: Literal['xywh', 'cxcywh', 'xyxy'] = 'cxcywh'
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize bounding boxes from pixel coordinates to [0, 1] range.

    Args:
        boxes: Bounding boxes in pixel coordinates, shape (..., 4)
        image_size: (height, width) of the image
        format: Box format ('xywh', 'cxcywh', or 'xyxy')

    Returns:
        Normalized boxes in [0, 1] range

    Example:
        >>> boxes = np.array([[100, 200, 50, 80]])  # xywh in 1920x1080
        >>> normalized = normalize_bbox(boxes, (1080, 1920), format='xywh')
        >>> # Result: [[0.052, 0.185, 0.026, 0.074]]
    """
    h, w = image_size
    is_numpy = isinstance(boxes, np.ndarray)

    if is_numpy:
        boxes = boxes.astype(np.float32)
        if format in ['xywh', 'cxcywh']:
            boxes[:, [0, 2]] /= w  # x, width
            boxes[:, [1, 3]] /= h  # y, height
        else:  # xyxy
            boxes[:, [0, 2]] /= w  # x1, x2
            boxes[:, [1, 3]] /= h  # y1, y2
    else:
        boxes = boxes.float()
        if format in ['xywh', 'cxcywh']:
            boxes[..., [0, 2]] /= w
            boxes[..., [1, 3]] /= h
        else:  # xyxy
            boxes[..., [0, 2]] /= w
            boxes[..., [1, 3]] /= h

    return boxes


def denormalize_bbox(
    boxes: Union[torch.Tensor, np.ndarray],
    image_size: Tuple[int, int],
    format: Literal['xywh', 'cxcywh', 'xyxy'] = 'cxcywh'
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize bounding boxes from [0, 1] range to pixel coordinates.

    Args:
        boxes: Normalized bounding boxes in [0, 1], shape (..., 4)
        image_size: (height, width) of the image
        format: Box format ('xywh', 'cxcywh', or 'xyxy')

    Returns:
        Boxes in pixel coordinates

    Example:
        >>> boxes_norm = np.array([[0.5, 0.5, 0.2, 0.3]])  # cxcywh
        >>> pixel_boxes = denormalize_bbox(boxes_norm, (1080, 1920), format='cxcywh')
        >>> # Result: [[960, 540, 384, 324]]  # pixels
    """
    h, w = image_size
    is_numpy = isinstance(boxes, np.ndarray)

    if is_numpy:
        boxes = boxes.astype(np.float32).copy()
        if format in ['xywh', 'cxcywh']:
            boxes[:, [0, 2]] *= w  # x, width
            boxes[:, [1, 3]] *= h  # y, height
        else:  # xyxy
            boxes[:, [0, 2]] *= w  # x1, x2
            boxes[:, [1, 3]] *= h  # y1, y2
    else:
        boxes = boxes.float().clone()
        if format in ['xywh', 'cxcywh']:
            boxes[..., [0, 2]] *= w
            boxes[..., [1, 3]] *= h
        else:  # xyxy
            boxes[..., [0, 2]] *= w
            boxes[..., [1, 3]] *= h

    return boxes


def scale_bbox(
    boxes: Union[torch.Tensor, np.ndarray],
    scale_factor: Union[float, Tuple[float, float]],
    format: Literal['xywh', 'cxcywh', 'xyxy'] = 'xywh'
) -> Union[torch.Tensor, np.ndarray]:
    """
    Scale bounding box coordinates by a factor (for image resize).

    CRITICAL: Use this after resizing images to keep bboxes aligned.

    Args:
        boxes: Bounding boxes, shape (..., 4)
        scale_factor: Uniform scale or (scale_x, scale_y) tuple
        format: Box format (all coordinates scaled equally)

    Returns:
        Scaled boxes

    Example:
        >>> boxes = np.array([[1000, 500, 200, 100]])  # xywh at 1920x1080
        >>> # After resize to 255x255, scale = 255/1920 ≈ 0.133
        >>> scaled = scale_bbox(boxes, 0.133, format='xywh')
        >>> # Result: [[133, 66.5, 26.6, 13.3]]
    """
    is_numpy = isinstance(boxes, np.ndarray)

    if isinstance(scale_factor, (tuple, list)):
        scale_x, scale_y = scale_factor
        if is_numpy:
            boxes = boxes.astype(np.float32).copy()
            boxes[:, [0, 2]] *= scale_x  # x, width (or x1, x2)
            boxes[:, [1, 3]] *= scale_y  # y, height (or y1, y2)
        else:
            boxes = boxes.float().clone()
            boxes[..., [0, 2]] *= scale_x
            boxes[..., [1, 3]] *= scale_y
    else:
        # Uniform scaling
        if is_numpy:
            boxes = boxes.astype(np.float32) * scale_factor
        else:
            boxes = boxes.float() * scale_factor

    return boxes


def convert_boxes(
    boxes: Union[torch.Tensor, np.ndarray],
    src: Literal['cxcywh_norm', 'xywh_norm', 'xywh_px', 'cxcywh_px', 'xyxy_px'],
    dst: Literal['cxcywh_norm', 'xywh_norm', 'xywh_px', 'cxcywh_px', 'xyxy_px'],
    image_size: Tuple[int, int] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Unified bbox converter handling both format and coordinate space.

    This is the recommended high-level API for bbox conversion.
    Implements Audit 2025111405 contract formalization.

    Args:
        boxes: Bounding boxes, shape (..., 4)
        src: Source format+space (e.g., 'cxcywh_norm', 'xywh_px')
        dst: Destination format+space
        image_size: Required if converting between pixel and normalized spaces

    Returns:
        Converted boxes

    Examples:
        >>> # Dataset → Model (pixel xywh → normalized cxcywh)
        >>> boxes = np.array([[1000, 500, 200, 100]])
        >>> model_boxes = convert_boxes(boxes, 'xywh_px', 'cxcywh_norm', (1080, 1920))

        >>> # Model → Visualizer (normalized cxcywh → pixel xywh)
        >>> pred = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        >>> viz_boxes = convert_boxes(pred, 'cxcywh_norm', 'xywh_px', (1080, 1920))
    """
    src_format, src_space = src.rsplit('_', 1)
    dst_format, dst_space = dst.rsplit('_', 1)

    result = boxes

    # Step 1: Denormalize if source is normalized and destination is pixel
    if src_space == 'norm' and dst_space == 'px':
        if image_size is None:
            raise ValueError("image_size required for norm→px conversion")
        result = denormalize_bbox(result, image_size, format=src_format)

    # Step 2: Convert format
    if src_format != dst_format:
        # Use intermediate pixel space for format conversion
        result = convert_bbox_format(result, src_format, dst_format)

    # Step 3: Normalize if source is pixel and destination is normalized
    if src_space == 'px' and dst_space == 'norm':
        if image_size is None:
            raise ValueError("image_size required for px→norm conversion")
        result = normalize_bbox(result, image_size, format=dst_format)

    return result
