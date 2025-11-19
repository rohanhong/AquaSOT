"""
Bbox-Aware Transform Base Classes and Utilities

This module provides the foundation for coordinate-aware data augmentation,
addressing the critical bbox-image desynchronization bugs identified in
Audit 2025111801.

Key Features:
- Unified BboxAwareTransform interface for all geometric transforms
- BboxFormat dataclass for explicit coordinate format contracts
- Bbox validation and boundary clipping utilities
- LegacyTransformWrapper for backward compatibility with existing code

Reference:
    Albumentations library best practices for bbox-aware augmentation
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

Author: AquaSOT-3D Team
Date: 2025-11-18
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple, List, Any
import numpy as np
import torch


@dataclass
class BboxFormat:
    """
    Explicit bbox coordinate format specification.

    Solves Audit 2025111801 Finding 4: Ambiguous coordinate format contracts
    leading to silent bbox-image desynchronization during transforms.

    Attributes:
        name: Format name ('xywh', 'cxcywh', 'xyxy')
        description: Human-readable description

    Format Specifications:
        - 'xywh': [x, y, w, h] - Top-left corner (x, y) and size (w, h)
        - 'cxcywh': [cx, cy, w, h] - Center (cx, cy) and size (w, h)
        - 'xyxy': [x1, y1, x2, y2] - Two corners (top-left, bottom-right)

    Example:
        >>> bbox_format = BboxFormat.XYWH
        >>> print(bbox_format.name)  # 'xywh'
        >>> print(bbox_format.description)  # '[x, y, w, h] - Top-left corner + size'
    """
    name: str
    description: str

    # Standard format definitions
    XYWH = None  # Defined after class
    CXCYWH = None
    XYXY = None


# Initialize standard formats
BboxFormat.XYWH = BboxFormat('xywh', '[x, y, w, h] - Top-left corner + size')
BboxFormat.CXCYWH = BboxFormat('cxcywh', '[cx, cy, w, h] - Center + size')
BboxFormat.XYXY = BboxFormat('xyxy', '[x1, y1, x2, y2] - Two corners')


def validate_bboxes(
    bboxes: Union[np.ndarray, torch.Tensor],
    image_shape: Tuple[int, int],
    bbox_format: str = 'xywh',
    clip: bool = True,
    min_area: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Validate and optionally clip bounding boxes to image boundaries.

    Ensures bboxes remain valid after geometric transformations, preventing
    out-of-bounds coordinates that cause training instabilities.

    Args:
        bboxes: Bounding boxes (N, 4) in specified format
        image_shape: Image dimensions (height, width)
        bbox_format: Bbox format ('xywh', 'cxcywh', 'xyxy')
        clip: If True, clip bboxes to image boundaries
        min_area: Minimum bbox area (pixels²) for validity

    Returns:
        Validated bboxes in same format and type as input

    Raises:
        ValueError: If bboxes have invalid shape or format

    Example:
        >>> bboxes = np.array([[100, 200, 50, -10]])  # Invalid: negative height
        >>> image_shape = (376, 682)
        >>> valid_bboxes = validate_bboxes(bboxes, image_shape, clip=True)
    """
    if bboxes is None or len(bboxes) == 0:
        return bboxes

    is_torch = isinstance(bboxes, torch.Tensor)
    device = bboxes.device if is_torch else None

    # Convert to numpy for processing
    if is_torch:
        bboxes_np = bboxes.detach().cpu().numpy()
    else:
        bboxes_np = bboxes.copy()

    if bboxes_np.ndim != 2 or bboxes_np.shape[1] != 4:
        raise ValueError(f"Bboxes must have shape (N, 4), got {bboxes_np.shape}")

    H, W = image_shape

    # Convert to xyxy for clipping
    if bbox_format == 'xywh':
        x1 = bboxes_np[:, 0]
        y1 = bboxes_np[:, 1]
        x2 = x1 + bboxes_np[:, 2]
        y2 = y1 + bboxes_np[:, 3]
    elif bbox_format == 'cxcywh':
        cx, cy, w, h = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    elif bbox_format == 'xyxy':
        x1, y1, x2, y2 = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
    else:
        raise ValueError(f"Unknown bbox format: {bbox_format}")

    # Clip to image boundaries
    if clip:
        x1 = np.clip(x1, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        x2 = np.clip(x2, 0, W)
        y2 = np.clip(y2, 0, H)

    # Ensure valid boxes (x2 > x1, y2 > y1)
    w_clipped = np.maximum(x2 - x1, 1.0)
    h_clipped = np.maximum(y2 - y1, 1.0)

    # Filter by minimum area
    area = w_clipped * h_clipped
    valid_mask = area >= min_area

    # Convert back to original format
    if bbox_format == 'xywh':
        result = np.stack([x1, y1, w_clipped, h_clipped], axis=1)
    elif bbox_format == 'cxcywh':
        cx_new = x1 + w_clipped / 2
        cy_new = y1 + h_clipped / 2
        result = np.stack([cx_new, cy_new, w_clipped, h_clipped], axis=1)
    elif bbox_format == 'xyxy':
        result = np.stack([x1, y1, x1 + w_clipped, y1 + h_clipped], axis=1)

    # Apply valid mask
    result = result[valid_mask]

    # Convert back to original type
    if is_torch:
        result = torch.from_numpy(result).to(dtype=bboxes.dtype, device=device)

    return result


def transform_bboxes_affine(
    bboxes: Union[np.ndarray, torch.Tensor],
    affine_matrix: np.ndarray,
    image_shape: Tuple[int, int],
    bbox_format: str = 'xywh'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform bounding boxes using affine transformation matrix.

    Critical fix for Audit 2025111801 Finding 2: RadialScalingJitter applies
    cv2.warpAffine to images without updating bboxes, causing ~19% IoU loss.

    Strategy: Transform all 4 corners and recompute axis-aligned bbox.

    Args:
        bboxes: Bounding boxes (N, 4)
        affine_matrix: 2x3 affine transformation matrix from cv2.getRotationMatrix2D
        image_shape: Image dimensions (height, width)
        bbox_format: Bbox format ('xywh', 'cxcywh', 'xyxy')

    Returns:
        Transformed bboxes in same format and type

    Mathematical Formulation:
        Given affine matrix M = [[a, b, tx], [c, d, ty]]:
            [x']   [a  b] [x]   [tx]
            [y'] = [c  d] [y] + [ty]

        For each bbox, transform all 4 corners and recompute axis-aligned box:
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            corners' = M @ corners
            bbox' = axis_aligned_bbox(corners')

    Example:
        >>> bboxes = np.array([[100, 200, 50, 80]])
        >>> M = cv2.getRotationMatrix2D((341, 375), 0, 1.05)  # Scale 1.05x
        >>> bboxes_transformed = transform_bboxes_affine(bboxes, M, (376, 682))
        >>> # Bbox now synchronized with cv2.warpAffine(image, M, ...)
    """
    if bboxes is None or len(bboxes) == 0:
        return bboxes

    is_torch = isinstance(bboxes, torch.Tensor)
    device = bboxes.device if is_torch else None

    # Convert to numpy
    if is_torch:
        bboxes_np = bboxes.detach().cpu().numpy()
    else:
        bboxes_np = bboxes.copy()

    # Handle 1D bbox (single bbox without batch dimension)
    squeeze_output = False
    if bboxes_np.ndim == 1:
        bboxes_np = bboxes_np.reshape(1, 4)
        squeeze_output = True

    # Convert to xyxy for corner transformation
    if bbox_format == 'xywh':
        x1 = bboxes_np[:, 0]
        y1 = bboxes_np[:, 1]
        x2 = x1 + bboxes_np[:, 2]
        y2 = y1 + bboxes_np[:, 3]
    elif bbox_format == 'cxcywh':
        cx, cy, w, h = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    elif bbox_format == 'xyxy':
        x1, y1, x2, y2 = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
    else:
        raise ValueError(f"Unknown bbox format: {bbox_format}")

    # Transform all 4 corners for each bbox
    N = len(bboxes_np)
    corners = np.zeros((N, 4, 2))  # (N, 4 corners, 2 coords)
    corners[:, 0, :] = np.stack([x1, y1], axis=1)  # Top-left
    corners[:, 1, :] = np.stack([x2, y1], axis=1)  # Top-right
    corners[:, 2, :] = np.stack([x1, y2], axis=1)  # Bottom-left
    corners[:, 3, :] = np.stack([x2, y2], axis=1)  # Bottom-right

    # Apply affine transformation: [x', y'] = M @ [x, y, 1]
    # M shape: (2, 3), corners shape: (N, 4, 2)
    corners_homogeneous = np.concatenate([
        corners,
        np.ones((N, 4, 1))
    ], axis=2)  # (N, 4, 3)

    # Reshape for batch matrix multiplication
    corners_flat = corners_homogeneous.reshape(-1, 3)  # (N*4, 3)
    corners_transformed_flat = corners_flat @ affine_matrix.T  # (N*4, 2)
    corners_transformed = corners_transformed_flat.reshape(N, 4, 2)  # (N, 4, 2)

    # Compute axis-aligned bounding box from transformed corners
    x1_new = np.min(corners_transformed[:, :, 0], axis=1)
    y1_new = np.min(corners_transformed[:, :, 1], axis=1)
    x2_new = np.max(corners_transformed[:, :, 0], axis=1)
    y2_new = np.max(corners_transformed[:, :, 1], axis=1)

    # Convert back to original format
    if bbox_format == 'xywh':
        result = np.stack([
            x1_new,
            y1_new,
            x2_new - x1_new,  # width
            y2_new - y1_new   # height
        ], axis=1)
    elif bbox_format == 'cxcywh':
        w_new = x2_new - x1_new
        h_new = y2_new - y1_new
        cx_new = x1_new + w_new / 2
        cy_new = y1_new + h_new / 2
        result = np.stack([cx_new, cy_new, w_new, h_new], axis=1)
    elif bbox_format == 'xyxy':
        result = np.stack([x1_new, y1_new, x2_new, y2_new], axis=1)

    # Validate and clip to image boundaries
    result = validate_bboxes(result, image_shape, bbox_format=bbox_format, clip=True)

    # Squeeze if input was 1D
    if squeeze_output and len(result) == 1:
        result = result[0]

    # Convert back to original type
    if is_torch:
        result = torch.from_numpy(result).to(dtype=bboxes.dtype, device=device)

    return result


class BboxAwareTransform(ABC):
    """
    Abstract base class for bbox-aware data augmentation transforms.

    This class enforces the coordinate synchronization contract between image
    transformations and bounding box annotations, solving the critical
    desynchronization bugs identified in Audit 2025111801.

    Subclasses must implement:
        - apply_to_image(): Image transformation logic
        - apply_to_bboxes(): Corresponding bbox transformation logic

    The __call__ method orchestrates both transformations and returns a dict
    with synchronized results, following Albumentations best practices.

    Attributes:
        bbox_format: Expected bbox format ('xywh', 'cxcywh', 'xyxy')
        p: Probability of applying transform (default 1.0)

    Returns:
        Dictionary with keys:
            - 'image': Transformed image
            - 'bboxes': Transformed bboxes (or None if not provided)
            - **kwargs: Pass-through metadata (e.g., 'metadata', 'frame_id')

    Example:
        >>> class MyGeometricTransform(BboxAwareTransform):
        ...     def apply_to_image(self, image):
        ...         return cv2.warpAffine(image, M, (W, H))
        ...
        ...     def apply_to_bboxes(self, bboxes, image_shape):
        ...         return transform_bboxes_affine(bboxes, M, image_shape, self.bbox_format)
        ...
        >>> transform = MyGeometricTransform(bbox_format='xywh', p=0.5)
        >>> result = transform(image=img, bboxes=boxes)
        >>> transformed_img = result['image']
        >>> transformed_boxes = result['bboxes']
    """

    def __init__(self, bbox_format: str = 'xywh', p: float = 1.0, **kwargs):
        """
        Initialize bbox-aware transform.

        Args:
            bbox_format: Expected bbox format ('xywh', 'cxcywh', 'xyxy')
            p: Probability of applying transform
            **kwargs: Additional subclass-specific parameters
        """
        self.bbox_format = bbox_format
        self.p = p

        # Validate bbox format
        valid_formats = ['xywh', 'cxcywh', 'xyxy']
        if bbox_format not in valid_formats:
            raise ValueError(
                f"bbox_format must be one of {valid_formats}, got '{bbox_format}'"
            )

    @abstractmethod
    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply transformation to image.

        Args:
            image: Input image (H, W, C) as numpy array

        Returns:
            Transformed image with same shape
        """
        pass

    @abstractmethod
    def apply_to_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor, None],
        image_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor, None]:
        """
        Apply transformation to bounding boxes.

        CRITICAL: This method MUST maintain coordinate synchronization with
        apply_to_image(). For geometric transforms, use transform_bboxes_affine()
        or similar utilities to ensure consistency.

        Args:
            bboxes: Bounding boxes (N, 4) in self.bbox_format, or None
            image_shape: Original image dimensions (height, width)

        Returns:
            Transformed bboxes in same format, or None if input is None
        """
        pass

    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply transform to image and bboxes with probabilistic execution.

        Args:
            image: Input image (H, W, C)
            bboxes: Bounding boxes (N, 4) or None
            **kwargs: Additional metadata to pass through (e.g., 'metadata', 'frame_id')

        Returns:
            Dictionary with 'image', 'bboxes', and pass-through kwargs
        """
        # Probabilistic execution
        if np.random.rand() > self.p:
            return {'image': image, 'bboxes': bboxes, **kwargs}

        # Get image shape before transformation
        image_shape = image.shape[:2]  # (H, W)

        # Apply transformations
        transformed_image = self.apply_to_image(image)
        transformed_bboxes = self.apply_to_bboxes(bboxes, image_shape)

        return {
            'image': transformed_image,
            'bboxes': transformed_bboxes,
            **kwargs
        }


class PhotometricTransform(BboxAwareTransform):
    """
    Base class for photometric (color/intensity) transforms that do not affect bboxes.

    Photometric transforms modify pixel values without changing spatial coordinates,
    so bboxes can be passed through unchanged. Subclasses only need to implement
    apply_to_image().

    Examples: SpeckleNoise, IntensityNormalization, ColorJitter, Blur

    Example:
        >>> class MyPhotometricTransform(PhotometricTransform):
        ...     def apply_to_image(self, image):
        ...         return image * 1.2  # Brightness adjustment
        ...
        >>> transform = MyPhotometricTransform()
        >>> result = transform(image=img, bboxes=boxes)
        >>> # result['bboxes'] == boxes (unchanged)
    """

    def apply_to_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor, None],
        image_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor, None]:
        """
        Photometric transforms do not affect bbox coordinates.

        Returns bboxes unchanged.
        """
        return bboxes


class LegacyTransformWrapper(BboxAwareTransform):
    """
    Backward compatibility wrapper for existing single-argument transforms.

    Solves Audit 2025111801 Finding 4: Enable incremental migration from
    legacy `__call__(img) -> img` transforms to bbox-aware interface without
    breaking existing code.

    This wrapper allows legacy transforms to work within the new bbox-aware
    pipeline by:
    1. Calling the legacy transform's __call__(image) method
    2. Passing bboxes through unchanged (assumes photometric-only transform)
    3. Warning users if geometric transforms are detected

    Args:
        legacy_transform: Existing transform with `__call__(img) -> img` signature
        is_photometric: If True, assume photometric (bbox-safe). If False, warn about potential desync.

    Example:
        >>> from sonar_transforms import SpeckleNoise  # Legacy transform
        >>> legacy_speckle = SpeckleNoise(noise_level=0.1)
        >>> wrapped = LegacyTransformWrapper(legacy_speckle, is_photometric=True)
        >>> result = wrapped(image=img, bboxes=boxes)  # Works in new pipeline
    """

    def __init__(
        self,
        legacy_transform: Any,
        is_photometric: bool = True,
        bbox_format: str = 'xywh',
        **kwargs
    ):
        super().__init__(bbox_format=bbox_format, **kwargs)
        self.legacy_transform = legacy_transform
        self.is_photometric = is_photometric

        if not is_photometric:
            import warnings
            warnings.warn(
                f"LegacyTransformWrapper wrapping geometric transform "
                f"'{type(legacy_transform).__name__}'. Bboxes will NOT be updated. "
                f"Consider migrating to BboxAwareTransform for correct behavior.",
                UserWarning
            )

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """Call legacy transform's __call__ method."""
        return self.legacy_transform(image)

    def apply_to_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor, None],
        image_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor, None]:
        """
        Pass bboxes through unchanged.

        WARNING: If legacy_transform is geometric, this will cause desynchronization!
        """
        return bboxes


def compose_transforms(transforms: List[BboxAwareTransform]) -> 'ComposeTransforms':
    """
    Factory function to create transform composition.

    Args:
        transforms: List of BboxAwareTransform instances

    Returns:
        ComposeTransforms instance that applies transforms sequentially

    Example:
        >>> transforms = [
        ...     SpeckleNoiseBboxAware(noise_level=0.1),
        ...     IntensityNormalizationBboxAware(),
        ...     RandomOcclusionBboxAware(min_size=0.05, max_size=0.3)
        ... ]
        >>> pipeline = compose_transforms(transforms)
        >>> result = pipeline(image=img, bboxes=boxes)
    """
    return ComposeTransforms(transforms)


class ComposeTransforms:
    """
    Compose multiple bbox-aware transforms into a single pipeline.

    Applies transforms sequentially, threading image and bbox results through
    each transform. Follows Albumentations Compose pattern.

    Args:
        transforms: List of BboxAwareTransform instances

    Example:
        >>> pipeline = ComposeTransforms([
        ...     SpeckleNoiseBboxAware(p=0.5),
        ...     RadialScalingJitterBboxAware(p=0.3),
        ... ])
        >>> result = pipeline(image=img, bboxes=boxes)
    """

    def __init__(self, transforms: List[BboxAwareTransform]):
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply all transforms sequentially.

        Args:
            image: Input image
            bboxes: Input bboxes
            **kwargs: Metadata to pass through

        Returns:
            Dictionary with final transformed image and bboxes
        """
        current_image = image
        current_bboxes = bboxes
        current_kwargs = kwargs.copy()

        for transform in self.transforms:
            result = transform(
                image=current_image,
                bboxes=current_bboxes,
                **current_kwargs
            )
            current_image = result['image']
            current_bboxes = result['bboxes']
            # Update metadata if transforms add new keys
            current_kwargs.update({
                k: v for k, v in result.items()
                if k not in ['image', 'bboxes']
            })

        return {
            'image': current_image,
            'bboxes': current_bboxes,
            **current_kwargs
        }
