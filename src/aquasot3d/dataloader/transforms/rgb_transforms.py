"""
Bbox-Aware RGB Transforms

Refactored RGB augmentation transforms with explicit bbox coordinate
synchronization, following the same architecture as Sonar transforms.

This module implements bbox-aware versions of RGB transforms:
- Photometric: ColorJitter, RandomBlur, Normalize
- Geometric: RandomHorizontalFlip, RandomOcclusion, RandomVerticalFlip

CRITICAL (Audit 2025111901):
- Unified architecture with Sonar transforms (same base classes)
- Strict bbox format discipline (xywh throughout, converted to cxcywh only in trainer)
- Dictionary interface: {'image': ..., 'bboxes': ...}
- Perfect migration of git augmentation methods

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

from typing import Optional, Tuple, Union
import random
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .base import (
    PhotometricTransform,
    BboxAwareTransform,
    validate_bboxes,
)


# ==============================================================================
# Photometric Transforms (Bbox-Preserving)
# ==============================================================================

class ColorJitterBboxAware(PhotometricTransform):
    """
    Bbox-aware color jittering for RGB images.

    Randomly adjusts brightness, contrast, saturation, and hue to
    simulate underwater lighting variations. Preserves bbox coordinates
    as this is a photometric-only transform.

    Args:
        brightness: Brightness factor range (default 0.2)
        contrast: Contrast factor range (default 0.2)
        saturation: Saturation factor range (default 0.2)
        hue: Hue factor range (default 0.1)
        prob: Probability of applying jittering (default 0.5)
        bbox_format: Bbox format

    Example:
        >>> transform = ColorJitterBboxAware(brightness=0.2, prob=0.5)
        >>> result = transform(image=rgb_img, bboxes=boxes)
        >>> # result['bboxes'] unchanged, result['image'] has color jitter
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        prob: float = 0.5,
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=prob)
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color jittering to RGB image.

        Args:
            image: RGB image (H, W, 3) in range [0, 255]

        Returns:
            Jittered image (H, W, 3)
        """
        # Convert to PIL for torchvision transforms
        img_pil = TF.to_pil_image(image.astype(np.uint8))
        img_pil = self.jitter(img_pil)
        jittered = np.array(img_pil, dtype=image.dtype)

        return jittered


class RandomBlurBboxAware(PhotometricTransform):
    """
    Bbox-aware random Gaussian blur for RGB images.

    Simulates water turbidity and motion blur. Photometric-only transform
    that preserves bbox coordinates.

    Args:
        kernel_sizes: List of possible kernel sizes (default [3, 5, 7])
        sigma_range: Gaussian sigma range (default (0.1, 2.0))
        prob: Probability of applying blur (default 0.3)
        bbox_format: Bbox format

    Example:
        >>> transform = RandomBlurBboxAware(kernel_sizes=[3, 5], prob=0.3)
        >>> result = transform(image=rgb_img, bboxes=boxes)
    """

    def __init__(
        self,
        kernel_sizes: list = [3, 5, 7],
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        prob: float = 0.3,
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=prob)
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random Gaussian blur.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Blurred image (H, W, 3)
        """
        kernel_size = random.choice(self.kernel_sizes)
        sigma = random.uniform(*self.sigma_range)
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        return blurred


class NormalizeBboxAware(PhotometricTransform):
    """
    Bbox-aware normalization to ImageNet statistics.

    Normalizes RGB images to zero mean and unit variance using ImageNet
    statistics. Does not affect bbox coordinates.

    Args:
        mean: Mean values for RGB channels (default ImageNet mean)
        std: Std values for RGB channels (default ImageNet std)
        bbox_format: Bbox format

    Example:
        >>> transform = NormalizeBboxAware()
        >>> result = transform(image=rgb_img, bboxes=boxes)
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=1.0)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to ImageNet statistics.

        Args:
            image: RGB image (H, W, 3) in range [0, 255]

        Returns:
            Normalized image (H, W, 3)
        """
        # Convert to [0, 1] range
        img_float = image.astype(np.float32) / 255.0

        # Normalize
        normalized = (img_float - self.mean) / self.std

        return normalized


# ==============================================================================
# Geometric Transforms (Bbox-Aware)
# ==============================================================================

class RandomHorizontalFlipBboxAware(BboxAwareTransform):
    """
    Bbox-aware random horizontal flip for RGB images.

    Flips both image and bbox coordinates consistently, ensuring
    bbox-image synchronization for underwater tracking.

    Args:
        prob: Probability of flipping (default 0.5)
        bbox_format: Bbox format

    Example:
        >>> transform = RandomHorizontalFlipBboxAware(prob=0.5)
        >>> result = transform(image=rgb_img, bboxes=boxes)
        >>> # Both image and bboxes are flipped consistently
    """

    def __init__(
        self,
        prob: float = 0.5,
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=prob)

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply horizontal flip to image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Flipped image (H, W, 3)
        """
        flipped = cv2.flip(image, 1)  # Horizontal flip
        return flipped

    def apply_to_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor, None],
        image_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor, None]:
        """
        Apply horizontal flip to bboxes.

        CRITICAL: This method ensures bbox coordinates are transformed
        consistently with the image.

        Args:
            bboxes: Bounding boxes (N, 4)
            image_shape: Image dimensions (H, W)

        Returns:
            Flipped bboxes (N, 4)
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

        H, W = image_shape

        # Flip bboxes based on format
        if self.bbox_format == 'xywh':
            # [x, y, w, h] format
            x, y, w, h = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
            # Flip x coordinate: new_x = W - old_x - w
            x_flipped = W - x - w
            flipped_bboxes = np.stack([x_flipped, y, w, h], axis=1)

        elif self.bbox_format == 'cxcywh':
            # [cx, cy, w, h] format
            cx, cy, w, h = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
            # Flip center x: new_cx = W - old_cx
            cx_flipped = W - cx
            flipped_bboxes = np.stack([cx_flipped, cy, w, h], axis=1)

        elif self.bbox_format == 'xyxy':
            # [x1, y1, x2, y2] format
            x1, y1, x2, y2 = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
            # Flip: new_x1 = W - old_x2, new_x2 = W - old_x1
            x1_flipped = W - x2
            x2_flipped = W - x1
            flipped_bboxes = np.stack([x1_flipped, y1, x2_flipped, y2], axis=1)

        else:
            raise ValueError(f"Unknown bbox format: {self.bbox_format}")

        # Convert back to original type
        if is_torch:
            flipped_bboxes = torch.from_numpy(flipped_bboxes).to(dtype=bboxes.dtype, device=device)

        return flipped_bboxes


class RandomVerticalFlipBboxAware(BboxAwareTransform):
    """
    Bbox-aware random vertical flip for RGB images.

    Flips both image and bbox coordinates consistently. Less common than
    horizontal flip but useful for underwater scenes.

    Args:
        prob: Probability of flipping (default 0.1)
        bbox_format: Bbox format

    Example:
        >>> transform = RandomVerticalFlipBboxAware(prob=0.1)
        >>> result = transform(image=rgb_img, bboxes=boxes)
    """

    def __init__(
        self,
        prob: float = 0.1,
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=prob)

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply vertical flip to image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Flipped image (H, W, 3)
        """
        flipped = cv2.flip(image, 0)  # Vertical flip
        return flipped

    def apply_to_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor, None],
        image_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor, None]:
        """
        Apply vertical flip to bboxes.

        Args:
            bboxes: Bounding boxes (N, 4)
            image_shape: Image dimensions (H, W)

        Returns:
            Flipped bboxes (N, 4)
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

        H, W = image_shape

        # Flip bboxes based on format
        if self.bbox_format == 'xywh':
            x, y, w, h = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
            y_flipped = H - y - h
            flipped_bboxes = np.stack([x, y_flipped, w, h], axis=1)

        elif self.bbox_format == 'cxcywh':
            cx, cy, w, h = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
            cy_flipped = H - cy
            flipped_bboxes = np.stack([cx, cy_flipped, w, h], axis=1)

        elif self.bbox_format == 'xyxy':
            x1, y1, x2, y2 = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
            y1_flipped = H - y2
            y2_flipped = H - y1
            flipped_bboxes = np.stack([x1, y1_flipped, x2, y2_flipped], axis=1)

        else:
            raise ValueError(f"Unknown bbox format: {self.bbox_format}")

        # Convert back to original type
        if is_torch:
            flipped_bboxes = torch.from_numpy(flipped_bboxes).to(dtype=bboxes.dtype, device=device)

        return flipped_bboxes


class RandomOcclusionBboxAware(BboxAwareTransform):
    """
    Bbox-aware random occlusion augmentation (CutMix-style).

    Randomly masks rectangular regions to simulate temporary occlusions.
    Filters out bboxes that are heavily occluded by the masked region.

    Args:
        min_size: Minimum occlusion size ratio (default 0.05)
        max_size: Maximum occlusion size ratio (default 0.3)
        prob: Probability of applying occlusion (default 0.2)
        fill_mode: Fill mode ('zero', 'random', 'mean')
        occlusion_threshold: Min bbox occlusion ratio to filter (default 0.5)
        bbox_format: Bbox format

    Example:
        >>> transform = RandomOcclusionBboxAware(min_size=0.05, max_size=0.3)
        >>> result = transform(image=rgb_img, bboxes=boxes)
        >>> # result['bboxes'] excludes heavily occluded boxes
    """

    def __init__(
        self,
        min_size: float = 0.05,
        max_size: float = 0.3,
        prob: float = 0.2,
        fill_mode: str = 'zero',
        occlusion_threshold: float = 0.5,
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=prob)
        self.min_size = min_size
        self.max_size = max_size
        self.fill_mode = fill_mode
        self.occlusion_threshold = occlusion_threshold
        self._occlusion_mask = None  # Store for bbox filtering

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random occlusion to image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Occluded image (H, W, 3)
        """
        H, W = image.shape[:2]

        # Random occlusion size
        size_ratio = random.uniform(self.min_size, self.max_size)
        occ_h = int(H * size_ratio)
        occ_w = int(W * size_ratio)

        # Random position
        x = random.randint(0, max(W - occ_w, 1))
        y = random.randint(0, max(H - occ_h, 1))

        # Create occlusion mask for bbox filtering
        self._occlusion_mask = np.zeros((H, W), dtype=bool)
        self._occlusion_mask[y:y+occ_h, x:x+occ_w] = True

        # Apply occlusion
        occluded = image.copy()
        if self.fill_mode == 'zero':
            occluded[y:y+occ_h, x:x+occ_w] = 0
        elif self.fill_mode == 'mean':
            mean_color = image.mean(axis=(0, 1))
            occluded[y:y+occ_h, x:x+occ_w] = mean_color
        elif self.fill_mode == 'random':
            occluded[y:y+occ_h, x:x+occ_w] = np.random.randint(0, 256, (occ_h, occ_w, 3))

        return occluded

    def apply_to_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor, None],
        image_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor, None]:
        """
        Filter bboxes that are heavily occluded.

        CRITICAL: This method prevents training on invisible targets.

        Args:
            bboxes: Bounding boxes (N, 4)
            image_shape: Image dimensions (H, W)

        Returns:
            Filtered bboxes (M, 4) where M <= N
        """
        if bboxes is None or len(bboxes) == 0 or self._occlusion_mask is None:
            return bboxes

        is_torch = isinstance(bboxes, torch.Tensor)
        device = bboxes.device if is_torch else None

        # Convert to numpy
        if is_torch:
            bboxes_np = bboxes.detach().cpu().numpy()
        else:
            bboxes_np = bboxes.copy()

        # Convert to xyxy for IoU computation
        if self.bbox_format == 'xywh':
            x1 = bboxes_np[:, 0]
            y1 = bboxes_np[:, 1]
            x2 = x1 + bboxes_np[:, 2]
            y2 = y1 + bboxes_np[:, 3]
        elif self.bbox_format == 'cxcywh':
            cx, cy, w, h = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
        elif self.bbox_format == 'xyxy':
            x1, y1, x2, y2 = bboxes_np[:, 0], bboxes_np[:, 1], bboxes_np[:, 2], bboxes_np[:, 3]
        else:
            raise ValueError(f"Unknown bbox format: {self.bbox_format}")

        # Compute occlusion ratio for each bbox
        H, W = image_shape
        valid_mask = np.ones(len(bboxes_np), dtype=bool)

        for i in range(len(bboxes_np)):
            # Clamp bbox to image boundaries
            x1_clip = int(np.clip(x1[i], 0, W - 1))
            y1_clip = int(np.clip(y1[i], 0, H - 1))
            x2_clip = int(np.clip(x2[i], 1, W))
            y2_clip = int(np.clip(y2[i], 1, H))

            # Extract bbox region from occlusion mask
            bbox_region = self._occlusion_mask[y1_clip:y2_clip, x1_clip:x2_clip]

            if bbox_region.size == 0:
                continue

            # Compute occlusion ratio
            occlusion_ratio = np.mean(bbox_region)

            # Filter if heavily occluded
            if occlusion_ratio >= self.occlusion_threshold:
                valid_mask[i] = False

        # Filter bboxes
        filtered_bboxes = bboxes_np[valid_mask]

        # Convert back to original type
        if is_torch:
            filtered_bboxes = torch.from_numpy(filtered_bboxes).to(dtype=bboxes.dtype, device=device)

        return filtered_bboxes


# ==============================================================================
# Utility Classes
# ==============================================================================

class ToTensor:
    """
    Convert numpy array to PyTorch tensor.

    Converts (H, W, C) to (C, H, W) format.
    """

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert to tensor.

        Args:
            img: Image array (H, W, C)

        Returns:
            Tensor (C, H, W)
        """
        # Handle grayscale
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        # Transpose to (C, H, W)
        img = img.transpose(2, 0, 1)

        # Convert to tensor
        return torch.from_numpy(img.copy())


# ==============================================================================
# Backward Compatibility Aliases
# ==============================================================================

# Legacy names without BboxAware suffix for backward compatibility with old imports
ColorJitter = ColorJitterBboxAware
RandomBlur = RandomBlurBboxAware
Normalize = NormalizeBboxAware
RandomHorizontalFlip = RandomHorizontalFlipBboxAware
RandomOcclusion = RandomOcclusionBboxAware
