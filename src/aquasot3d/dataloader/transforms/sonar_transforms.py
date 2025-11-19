"""
Bbox-Aware Sonar Transforms

Refactored sonar augmentation transforms with explicit bbox coordinate
synchronization for preprocessed rectangular sonar images.

This module implements bbox-aware transforms for rectangular coordinates:
- Photometric: SpeckleNoise, IntensityNormalization

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

from typing import Optional, Tuple, Union
import random
import numpy as np
import torch
import cv2

from .base import (
    PhotometricTransform,
    BboxAwareTransform,
    transform_bboxes_affine,
    validate_bboxes,
)


# ==============================================================================
# Phase 2: Photometric Transforms (Bbox-Preserving)
# ==============================================================================

class SpeckleNoiseBboxAware(PhotometricTransform):
    """
    Bbox-aware speckle noise augmentation for sonar imagery.

    Speckle is multiplicative noise characteristic of coherent imaging systems
    like sonar and SAR. This transform preserves bbox coordinates as it only
    modifies pixel intensities.

    Args:
        noise_level: Standard deviation of multiplicative noise (default 0.1)
        prob: Probability of applying noise (default 0.5)
        bbox_format: Bbox format ('xywh', 'cxcywh', 'xyxy')

    Example:
        >>> transform = SpeckleNoiseBboxAware(noise_level=0.1, prob=0.5)
        >>> result = transform(image=sonar_img, bboxes=boxes)
        >>> # result['bboxes'] unchanged, result['image'] has speckle noise
    """

    def __init__(
        self,
        noise_level: float = 0.1,
        prob: float = 0.5,
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=prob)
        self.noise_level = noise_level

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multiplicative speckle noise to sonar image.

        Args:
            image: Sonar image (H, W, C) in range [0, 255]

        Returns:
            Noisy image (H, W, C) in range [0, 255]
        """
        # Generate multiplicative noise: I_noisy = I * (1 + noise)
        noise = np.random.randn(*image.shape) * self.noise_level

        # Apply speckle
        img_float = image.astype(np.float32)
        noisy = img_float * (1 + noise)

        # Clip to valid range
        noisy = np.clip(noisy, 0, 255)

        return noisy.astype(image.dtype)


class IntensityNormalizationBboxAware(PhotometricTransform):
    """
    Bbox-aware intensity normalization for sonar imagery.

    Normalizes sonar intensity to improve feature extraction consistency
    across varying water conditions and target materials. Does not affect
    bbox coordinates.

    Args:
        mode: Normalization mode ('global', 'adaptive', 'none')
        target_mean: Target mean intensity (default 128.0)
        target_std: Target standard deviation (default 50.0)
        bbox_format: Bbox format

    Example:
        >>> transform = IntensityNormalizationBboxAware(mode='global')
        >>> result = transform(image=sonar_img, bboxes=boxes)
    """

    def __init__(
        self,
        mode: str = 'global',
        target_mean: float = 128.0,
        target_std: float = 50.0,
        bbox_format: str = 'xywh'
    ):
        super().__init__(bbox_format=bbox_format, p=1.0)
        self.mode = mode
        self.target_mean = target_mean
        self.target_std = target_std

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply intensity normalization.

        Args:
            image: Sonar image (H, W, C) in range [0, 255]

        Returns:
            Normalized image (H, W, C) in range [0, 255]
        """
        if self.mode == 'none':
            return image

        img_float = image.astype(np.float32)

        if self.mode == 'adaptive':
            # Adaptive normalization per image
            mean = np.mean(img_float)
            std = np.std(img_float) + 1e-6

            # Standardize and rescale
            normalized = (img_float - mean) / std
            normalized = normalized * self.target_std + self.target_mean

        elif self.mode == 'global':
            # Global normalization using target parameters
            normalized = (img_float / 255.0) * self.target_std + self.target_mean

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

        # Clip to valid range
        normalized = np.clip(normalized, 0, 255)

        return normalized.astype(image.dtype)


# ==============================================================================
# Backward Compatibility Aliases
# ==============================================================================

# Legacy names without BboxAware suffix for backward compatibility with old imports
SpeckleNoise = SpeckleNoiseBboxAware
IntensityNormalization = IntensityNormalizationBboxAware
