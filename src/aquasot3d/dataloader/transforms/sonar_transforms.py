"""
Sonar-Specific Data Augmentation Transforms

This module implements data augmentation techniques specifically designed
for Forward-Looking Sonar (FLS) images:
- Speckle noise simulation (multiplicative acoustic noise)
- Radial scaling jitter (range variation)
- Sector occlusion (angular region masking)
- Intensity normalization (acoustic return strength)
- Fan-shape aware geometric transforms

These augmentations preserve the physical characteristics of sonar imagery
while introducing training variability.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Optional, Tuple
import random
import numpy as np
import cv2
import torch


class SpeckleNoise:
    """
    Add multiplicative speckle noise to simulate sonar acoustic artifacts.

    Speckle is a multiplicative noise characteristic of coherent imaging
    systems like sonar. It appears as granular patterns due to interference.

    Args:
        noise_level: Noise strength (default: 0.1)
        prob: Probability of applying noise (default: 0.5)

    Formula:
        noisy_img = img * (1 + noise_level * randn)
    """

    def __init__(self, noise_level: float = 0.1, prob: float = 0.5):
        self.noise_level = noise_level
        self.prob = prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply speckle noise.

        Args:
            img: Sonar image (H, W, C) in range [0, 255]

        Returns:
            Noisy image (H, W, C)
        """
        if random.random() < self.prob:
            # Generate multiplicative noise
            noise = np.random.randn(*img.shape) * self.noise_level

            # Apply multiplicative speckle
            img_float = img.astype(np.float32)
            noisy = img_float * (1 + noise)

            # Clip to valid range
            noisy = np.clip(noisy, 0, 255)

            return noisy.astype(img.dtype)

        return img


class RadialScalingJitter:
    """
    Random radial scaling to simulate range variation.

    Applies radial-dependent scaling to simulate varying distances
    or sensor configurations.

    Args:
        scale_range: Scaling factor range (default: (0.9, 1.1))
        prob: Probability of applying jitter (default: 0.3)
        sensor_pos: Sensor position (cx, cy), None = auto-compute
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.3,
        sensor_pos: Optional[Tuple[float, float]] = None,
    ):
        self.scale_range = scale_range
        self.prob = prob
        self.sensor_pos = sensor_pos

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply radial scaling jitter.

        Args:
            img: Sonar image (H, W, C)

        Returns:
            Scaled image (H, W, C)
        """
        if random.random() >= self.prob:
            return img

        H, W = img.shape[:2]

        # Determine sensor position
        if self.sensor_pos is None:
            cx, cy = W / 2, H - 1
        else:
            cx, cy = self.sensor_pos

        # Random scale factor
        scale = random.uniform(*self.scale_range)

        # Create scaling transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)

        # Apply transformation
        scaled = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return scaled


class SectorOcclusion:
    """
    Random angular sector occlusion to simulate beam occlusion.

    Masks wedge-shaped regions emanating from sensor to simulate
    temporary acoustic shadows or beam blockages.

    Args:
        min_angle: Minimum occlusion angle in degrees (default: 10)
        max_angle: Maximum occlusion angle in degrees (default: 40)
        prob: Probability of applying occlusion (default: 0.2)
        sensor_pos: Sensor position (cx, cy), None = auto-compute
    """

    def __init__(
        self,
        min_angle: float = 10.0,
        max_angle: float = 40.0,
        prob: float = 0.2,
        sensor_pos: Optional[Tuple[float, float]] = None,
    ):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.prob = prob
        self.sensor_pos = sensor_pos

    def __call__(self, img: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply sector occlusion.

        Args:
            img: Sonar image (H, W, C)
            bbox: Optional bounding box to avoid occluding target [x, y, w, h]

        Returns:
            Occluded image (H, W, C)
        """
        if random.random() >= self.prob:
            return img

        H, W = img.shape[:2]

        # Determine sensor position
        if self.sensor_pos is None:
            cx, cy = int(W / 2), H - 1
        else:
            cx, cy = int(self.sensor_pos[0]), int(self.sensor_pos[1])

        # Random occlusion angle
        angle_width = random.uniform(self.min_angle, self.max_angle)

        # Random center angle (in degrees, 0 = upward)
        center_angle = random.uniform(-65, 65)  # Within typical FOV

        # Calculate sector boundaries
        start_angle = center_angle - angle_width / 2
        end_angle = center_angle + angle_width / 2

        # Check if sector overlaps with target bbox
        if bbox is not None and not np.all(bbox == 0):
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            bbox_cx = bbox_x + bbox_w / 2
            bbox_cy = bbox_y + bbox_h / 2

            # Calculate angle to bbox center
            dx = bbox_cx - cx
            dy = cy - bbox_cy  # Invert y for upward-facing
            bbox_angle = np.degrees(np.arctan2(dx, dy))

            # Check if bbox is in occlusion sector
            if start_angle <= bbox_angle <= end_angle:
                # Skip occlusion if it would block target
                return img

        # Create sector mask
        mask = np.ones((H, W), dtype=np.uint8) * 255

        # Define sector polygon points
        max_range = int(np.sqrt((W/2)**2 + H**2))  # Diagonal distance
        pt1 = (cx, cy)
        pt2 = (
            int(cx + max_range * np.sin(np.radians(start_angle))),
            int(cy - max_range * np.cos(np.radians(start_angle)))
        )
        pt3 = (
            int(cx + max_range * np.sin(np.radians(end_angle))),
            int(cy - max_range * np.cos(np.radians(end_angle)))
        )

        # Fill sector polygon (black = occluded)
        cv2.fillPoly(mask, [np.array([pt1, pt2, pt3])], 0)

        # Apply mask
        if img.ndim == 3:
            mask = mask[:, :, np.newaxis]

        img_masked = img * (mask / 255.0)

        return img_masked.astype(img.dtype)


class IntensityNormalization:
    """
    Normalize sonar intensity to handle varying acoustic return strengths.

    Args:
        mode: Normalization mode ('minmax', 'zscore', 'adaptive')
        clip_percentile: Percentile clipping for adaptive mode (default: (2, 98))
    """

    def __init__(
        self,
        mode: str = 'minmax',
        clip_percentile: Tuple[float, float] = (2, 98),
    ):
        if mode not in ['minmax', 'zscore', 'adaptive']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'minmax', 'zscore', or 'adaptive'")

        self.mode = mode
        self.clip_percentile = clip_percentile

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize intensity.

        Args:
            img: Sonar image (H, W, C) in range [0, 255]

        Returns:
            Normalized image (H, W, C) in range [0, 255]
        """
        img_float = img.astype(np.float32)

        if self.mode == 'minmax':
            # Min-max normalization
            img_min = img_float.min()
            img_max = img_float.max()
            if img_max > img_min:
                img_norm = (img_float - img_min) / (img_max - img_min) * 255
            else:
                img_norm = img_float

        elif self.mode == 'zscore':
            # Z-score normalization
            mean = img_float.mean()
            std = img_float.std()
            if std > 0:
                img_norm = (img_float - mean) / std
                img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min()) * 255
            else:
                img_norm = img_float

        else:  # adaptive
            # Adaptive percentile clipping
            low = np.percentile(img_float, self.clip_percentile[0])
            high = np.percentile(img_float, self.clip_percentile[1])

            img_clipped = np.clip(img_float, low, high)

            if high > low:
                img_norm = (img_clipped - low) / (high - low) * 255
            else:
                img_norm = img_float

        return img_norm.astype(img.dtype)


class FanMaskApplication:
    """
    Apply fan-shape mask to sonar image.

    Args:
        fov_angle: Field of view angle in degrees (default: 130)
        sensor_pos: Sensor position (cx, cy), None = auto-compute
    """

    def __init__(
        self,
        fov_angle: float = 130.0,
        sensor_pos: Optional[Tuple[float, float]] = None,
    ):
        self.fov_angle = fov_angle
        self.sensor_pos = sensor_pos
        self._mask_cache = {}

    def _get_mask(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get or create fan mask for given image shape.

        Args:
            img_shape: (H, W)

        Returns:
            Binary mask (H, W) with 1=valid, 0=background
        """
        if img_shape in self._mask_cache:
            return self._mask_cache[img_shape]

        H, W = img_shape

        # Determine sensor position
        if self.sensor_pos is None:
            cx, cy = W / 2, H - 1
        else:
            cx, cy = self.sensor_pos

        # Create coordinate grids
        y, x = np.ogrid[:H, :W]

        # Calculate radial distance
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Calculate angle from vertical axis
        theta = np.arctan2(x - cx, cy - y)

        # Convert FOV to radians
        fov_rad = np.deg2rad(self.fov_angle)

        # Apply fan constraints
        mask = (r <= H) & (np.abs(theta) <= fov_rad / 2)

        # Cache mask
        self._mask_cache[img_shape] = mask.astype(np.float32)

        return self._mask_cache[img_shape]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply fan mask.

        Args:
            img: Sonar image (H, W, C)

        Returns:
            Masked image (H, W, C)
        """
        H, W = img.shape[:2]
        mask = self._get_mask((H, W))

        if img.ndim == 3:
            mask = mask[:, :, np.newaxis]

        return (img * mask).astype(img.dtype)


class SonarTransformPipeline:
    """
    Complete sonar transformation pipeline for training.

    Args:
        enable_speckle: Enable speckle noise
        enable_radial_jitter: Enable radial scaling jitter
        enable_sector_occlusion: Enable sector occlusion
        enable_intensity_norm: Enable intensity normalization
        apply_fan_mask: Apply fan-shape mask
        to_tensor: Convert to tensor
    """

    def __init__(
        self,
        enable_speckle: bool = True,
        enable_radial_jitter: bool = True,
        enable_sector_occlusion: bool = True,
        enable_intensity_norm: bool = True,
        apply_fan_mask: bool = True,
        to_tensor: bool = True,
    ):
        self.transforms = []

        if enable_speckle:
            self.transforms.append(SpeckleNoise())

        if enable_radial_jitter:
            self.transforms.append(RadialScalingJitter())

        if enable_sector_occlusion:
            self.sector_occlusion = SectorOcclusion()
            self.enable_sector_occlusion = True
        else:
            self.enable_sector_occlusion = False

        if enable_intensity_norm:
            self.intensity_norm = IntensityNormalization(mode='adaptive')
            self.enable_intensity_norm = True
        else:
            self.enable_intensity_norm = False

        if apply_fan_mask:
            self.fan_mask = FanMaskApplication()
            self.enable_fan_mask = True
        else:
            self.enable_fan_mask = False

        self.to_tensor = to_tensor

    def __call__(
        self,
        img: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply transformation pipeline.

        Args:
            img: Sonar image (H, W, C)
            bbox: Optional bounding box [x, y, w, h]

        Returns:
            Transformed image (torch.Tensor if to_tensor=True)
        """
        # Apply transforms that don't need bbox
        for transform in self.transforms:
            img = transform(img)

        # Apply sector occlusion (may need bbox to avoid target)
        if self.enable_sector_occlusion:
            img = self.sector_occlusion(img, bbox)

        # Apply intensity normalization
        if self.enable_intensity_norm:
            img = self.intensity_norm(img)

        # Apply fan mask
        if self.enable_fan_mask:
            img = self.fan_mask(img)

        # Convert to tensor
        if self.to_tensor:
            # Normalize to [0, 1] range
            img = img.astype(np.float32) / 255.0

            # Transpose to (C, H, W)
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            else:
                img = img.transpose(2, 0, 1)

            img = torch.from_numpy(img.copy())

        return img


def letterbox_resize(
    img: np.ndarray,
    target_size: Tuple[int, int],
    fill_value: int = 0
) -> Tuple[np.ndarray, Tuple[float, int, int]]:
    """
    Resize image with aspect ratio preservation using letterbox padding.

    CRITICAL FIX (Audit 2025111408 Finding 2):
    Sonar images have specific aspect ratios (682x376 ≈ 1.81:1) that encode
    different spatial units along range and azimuth. Warping to squares breaks
    the log-polar transform assumption of isotropic pixels.

    This function preserves aspect ratio by:
    1. Scaling to fit within target size
    2. Padding the remainder to reach exact target size

    Args:
        img: Input image (H, W) or (H, W, C)
        target_size: Desired output size (H, W)
        fill_value: Padding fill value (default: 0 for black)

    Returns:
        Tuple of:
            - Padded image (target_H, target_W, C)
            - Metadata (scale, pad_top, pad_left) for coordinate adjustment

    Example:
        >>> img = np.random.rand(376, 682)  # Original sonar
        >>> resized, (scale, pad_y, pad_x) = letterbox_resize(img, (256, 256))
        >>> resized.shape
        (256, 256)
    """
    H, W = img.shape[:2]
    target_H, target_W = target_size

    # Calculate scale factor to fit within target (preserve aspect ratio)
    scale = min(target_H / H, target_W / W)

    # New dimensions after scaling
    new_H = int(H * scale)
    new_W = int(W * scale)

    # Resize image maintaining aspect ratio
    resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # CRITICAL FIX (Audit 2025111408 Finding 1 remediation):
    # For sonar images, the sensor origin MUST remain at bottom-center (cx=W/2, cy=H-1)
    # Therefore, we place ALL vertical padding on TOP (not centered)
    # This ensures the sensor position stays at the bottom edge after padding

    # Horizontal padding: center the image (sensor stays at horizontal center)
    pad_left = (target_W - new_W) // 2
    pad_right = target_W - new_W - pad_left

    # Vertical padding: ALL on top to keep sensor at bottom
    pad_top = target_H - new_H  # All remaining space goes to top
    pad_bottom = 0              # No padding at bottom (sensor stays at cy=H-1)

    # Pad to target size
    if resized.ndim == 2:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=fill_value
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[fill_value] * resized.shape[2]
        )

    # Return image and metadata for coordinate adjustment
    # Note: pad_top is now the TOTAL vertical padding (all on top)
    return padded, (scale, pad_top, pad_left)