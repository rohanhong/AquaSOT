"""
Type-safe Bounding Box Representations

This module provides type-safe wrapper classes for different bbox formats
to prevent silent format confusion between xywh, cxcywh, and xyxy formats.

CRITICAL FIX (Audit 2025111411 Finding 3):
Previous code allowed implicit format mixing where xywh tensors could be
passed to functions expecting cxcywh, causing silent prediction errors.
These typed wrappers make all conversions explicit and validated.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

from dataclasses import dataclass
from typing import Union, Tuple
import torch
import numpy as np


@dataclass
class BBoxXYWH:
    """
    Bounding box in [x, y, w, h] format (top-left corner + size).

    Format:
        x: left edge coordinate
        y: top edge coordinate
        w: width
        h: height

    Used by:
        - Dataset ground truth boxes
        - Evaluation metrics (compute_iou, compute_center_error)
        - Final predictions after denormalization
    """
    data: Union[torch.Tensor, np.ndarray]

    def __post_init__(self):
        """Validate shape and format."""
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            if self.data.shape[-1] != 4:
                raise ValueError(f"XYWH bbox must have 4 coordinates, got shape {self.data.shape}")
        else:
            raise TypeError(f"BBox data must be tensor or ndarray, got {type(self.data)}")

    def to_cxcywh(self) -> 'BBoxCXCYWH':
        """Convert to center format [cx, cy, w, h]."""
        if isinstance(self.data, torch.Tensor):
            cx = self.data[..., 0] + self.data[..., 2] / 2
            cy = self.data[..., 1] + self.data[..., 3] / 2
            w = self.data[..., 2]
            h = self.data[..., 3]
            cxcywh_data = torch.stack([cx, cy, w, h], dim=-1)
        else:  # numpy
            cx = self.data[..., 0] + self.data[..., 2] / 2
            cy = self.data[..., 1] + self.data[..., 3] / 2
            w = self.data[..., 2]
            h = self.data[..., 3]
            cxcywh_data = np.stack([cx, cy, w, h], axis=-1)

        return BBoxCXCYWH(cxcywh_data)

    def to_xyxy(self) -> 'BBoxXYXY':
        """Convert to corner format [x1, y1, x2, y2]."""
        if isinstance(self.data, torch.Tensor):
            x1 = self.data[..., 0]
            y1 = self.data[..., 1]
            x2 = self.data[..., 0] + self.data[..., 2]
            y2 = self.data[..., 1] + self.data[..., 3]
            xyxy_data = torch.stack([x1, y1, x2, y2], dim=-1)
        else:  # numpy
            x1 = self.data[..., 0]
            y1 = self.data[..., 1]
            x2 = self.data[..., 0] + self.data[..., 2]
            y2 = self.data[..., 1] + self.data[..., 3]
            xyxy_data = np.stack([x1, y1, x2, y2], axis=-1)

        return BBoxXYXY(xyxy_data)

    def tensor(self) -> torch.Tensor:
        """Get underlying tensor (convert if numpy)."""
        if isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data)
        return self.data

    def numpy(self) -> np.ndarray:
        """Get underlying numpy array (convert if tensor)."""
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data


@dataclass
class BBoxCXCYWH:
    """
    Bounding box in [cx, cy, w, h] format (center + size).

    Format:
        cx: center x coordinate
        cy: center y coordinate
        w: width
        h: height

    Used by:
        - Model predictions (normalized [0,1] during training)
        - Loss functions (L1, GIoU)
        - Template/search boxes in Siamese networks
    """
    data: Union[torch.Tensor, np.ndarray]

    def __post_init__(self):
        """Validate shape and format."""
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            if self.data.shape[-1] != 4:
                raise ValueError(f"CXCYWH bbox must have 4 coordinates, got shape {self.data.shape}")
        else:
            raise TypeError(f"BBox data must be tensor or ndarray, got {type(self.data)}")

    def to_xywh(self) -> BBoxXYWH:
        """Convert to corner format [x, y, w, h]."""
        if isinstance(self.data, torch.Tensor):
            x = self.data[..., 0] - self.data[..., 2] / 2
            y = self.data[..., 1] - self.data[..., 3] / 2
            w = self.data[..., 2]
            h = self.data[..., 3]
            xywh_data = torch.stack([x, y, w, h], dim=-1)
        else:  # numpy
            x = self.data[..., 0] - self.data[..., 2] / 2
            y = self.data[..., 1] - self.data[..., 3] / 2
            w = self.data[..., 2]
            h = self.data[..., 3]
            xywh_data = np.stack([x, y, w, h], axis=-1)

        return BBoxXYWH(xywh_data)

    def to_xyxy(self) -> 'BBoxXYXY':
        """Convert to corner format [x1, y1, x2, y2]."""
        if isinstance(self.data, torch.Tensor):
            x1 = self.data[..., 0] - self.data[..., 2] / 2
            y1 = self.data[..., 1] - self.data[..., 3] / 2
            x2 = self.data[..., 0] + self.data[..., 2] / 2
            y2 = self.data[..., 1] + self.data[..., 3] / 2
            xyxy_data = torch.stack([x1, y1, x2, y2], dim=-1)
        else:  # numpy
            x1 = self.data[..., 0] - self.data[..., 2] / 2
            y1 = self.data[..., 1] - self.data[..., 3] / 2
            x2 = self.data[..., 0] + self.data[..., 2] / 2
            y2 = self.data[..., 1] + self.data[..., 3] / 2
            xyxy_data = np.stack([x1, y1, x2, y2], axis=-1)

        return BBoxXYXY(xyxy_data)

    def tensor(self) -> torch.Tensor:
        """Get underlying tensor (convert if numpy)."""
        if isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data)
        return self.data

    def numpy(self) -> np.ndarray:
        """Get underlying numpy array (convert if tensor)."""
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data


@dataclass
class BBoxXYXY:
    """
    Bounding box in [x1, y1, x2, y2] format (two corners).

    Format:
        x1: left edge coordinate
        y1: top edge coordinate
        x2: right edge coordinate
        y2: bottom edge coordinate

    Used by:
        - Some detection frameworks
        - IoU computation (internally)
    """
    data: Union[torch.Tensor, np.ndarray]

    def __post_init__(self):
        """Validate shape and format."""
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            if self.data.shape[-1] != 4:
                raise ValueError(f"XYXY bbox must have 4 coordinates, got shape {self.data.shape}")
        else:
            raise TypeError(f"BBox data must be tensor or ndarray, got {type(self.data)}")

    def to_xywh(self) -> BBoxXYWH:
        """Convert to [x, y, w, h] format."""
        if isinstance(self.data, torch.Tensor):
            x = self.data[..., 0]
            y = self.data[..., 1]
            w = self.data[..., 2] - self.data[..., 0]
            h = self.data[..., 3] - self.data[..., 1]
            xywh_data = torch.stack([x, y, w, h], dim=-1)
        else:  # numpy
            x = self.data[..., 0]
            y = self.data[..., 1]
            w = self.data[..., 2] - self.data[..., 0]
            h = self.data[..., 3] - self.data[..., 1]
            xywh_data = np.stack([x, y, w, h], axis=-1)

        return BBoxXYWH(xywh_data)

    def to_cxcywh(self) -> BBoxCXCYWH:
        """Convert to [cx, cy, w, h] format."""
        if isinstance(self.data, torch.Tensor):
            cx = (self.data[..., 0] + self.data[..., 2]) / 2
            cy = (self.data[..., 1] + self.data[..., 3]) / 2
            w = self.data[..., 2] - self.data[..., 0]
            h = self.data[..., 3] - self.data[..., 1]
            cxcywh_data = torch.stack([cx, cy, w, h], dim=-1)
        else:  # numpy
            cx = (self.data[..., 0] + self.data[..., 2]) / 2
            cy = (self.data[..., 1] + self.data[..., 3]) / 2
            w = self.data[..., 2] - self.data[..., 0]
            h = self.data[..., 3] - self.data[..., 1]
            cxcywh_data = np.stack([cx, cy, w, h], axis=-1)

        return BBoxCXCYWH(cxcywh_data)

    def tensor(self) -> torch.Tensor:
        """Get underlying tensor (convert if numpy)."""
        if isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data)
        return self.data

    def numpy(self) -> np.ndarray:
        """Get underlying numpy array (convert if tensor)."""
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data


# Convenience factory functions

def bbox_from_xywh(data: Union[torch.Tensor, np.ndarray]) -> BBoxXYWH:
    """Create BBoxXYWH from raw data."""
    return BBoxXYWH(data)


def bbox_from_cxcywh(data: Union[torch.Tensor, np.ndarray]) -> BBoxCXCYWH:
    """Create BBoxCXCYWH from raw data."""
    return BBoxCXCYWH(data)


def bbox_from_xyxy(data: Union[torch.Tensor, np.ndarray]) -> BBoxXYXY:
    """Create BBoxXYXY from raw data."""
    return BBoxXYXY(data)


def convert_bbox_format(
    bbox: Union[torch.Tensor, np.ndarray],
    src_format: str,
    dst_format: str
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert bbox between formats with explicit format specification.

    Args:
        bbox: Bounding box tensor/array
        src_format: Source format ('xywh', 'cxcywh', or 'xyxy')
        dst_format: Destination format ('xywh', 'cxcywh', or 'xyxy')

    Returns:
        Converted bbox in same type as input

    Example:
        >>> # Convert model prediction from cxcywh to xywh for metrics
        >>> pred_xywh = convert_bbox_format(pred_cxcywh, 'cxcywh', 'xywh')
    """
    # Create typed bbox
    if src_format == 'xywh':
        typed_bbox = BBoxXYWH(bbox)
    elif src_format == 'cxcywh':
        typed_bbox = BBoxCXCYWH(bbox)
    elif src_format == 'xyxy':
        typed_bbox = BBoxXYXY(bbox)
    else:
        raise ValueError(f"Unknown source format: {src_format}. Must be 'xywh', 'cxcywh', or 'xyxy'")

    # Convert to target format
    if dst_format == 'xywh':
        result = typed_bbox.to_xywh() if hasattr(typed_bbox, 'to_xywh') else typed_bbox
    elif dst_format == 'cxcywh':
        result = typed_bbox.to_cxcywh() if hasattr(typed_bbox, 'to_cxcywh') else typed_bbox
    elif dst_format == 'xyxy':
        result = typed_bbox.to_xyxy() if hasattr(typed_bbox, 'to_xyxy') else typed_bbox
    else:
        raise ValueError(f"Unknown destination format: {dst_format}. Must be 'xywh', 'cxcywh', or 'xyxy'")

    # Return in same type as input
    if isinstance(bbox, torch.Tensor):
        return result.tensor()
    else:
        return result.numpy()
