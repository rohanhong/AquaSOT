"""
Data transforms for AquaSOT-3D.

This module contains transformation and augmentation functions for
RGB and Sonar images (preprocessed to rectangular coordinates).
"""

from .rgb_transforms import (
    ColorJitter,
    RandomBlur,
    RandomOcclusion,
    RandomHorizontalFlip,
    Normalize,
    ToTensor,
    # Bbox-aware versions
    ColorJitterBboxAware,
    RandomBlurBboxAware,
    RandomOcclusionBboxAware,
    RandomHorizontalFlipBboxAware,
    NormalizeBboxAware,
)

from .sonar_transforms import (
    SpeckleNoise,
    IntensityNormalization,
    # Note: letterbox_resize removed - unified with RGB strategy (2025-11-19)
    # Bbox-aware versions
    SpeckleNoiseBboxAware,
    IntensityNormalizationBboxAware,
)

__all__ = [
    # RGB transforms (legacy names)
    'ColorJitter',
    'RandomBlur',
    'RandomOcclusion',
    'RandomHorizontalFlip',
    'Normalize',
    'ToTensor',
    # RGB transforms (bbox-aware)
    'ColorJitterBboxAware',
    'RandomBlurBboxAware',
    'RandomOcclusionBboxAware',
    'RandomHorizontalFlipBboxAware',
    'NormalizeBboxAware',
    # Sonar transforms (legacy names)
    'SpeckleNoise',
    'IntensityNormalization',
    # Note: letterbox_resize removed - unified with RGB strategy (2025-11-19)
    # Sonar transforms (bbox-aware)
    'SpeckleNoiseBboxAware',
    'IntensityNormalizationBboxAware',
]
