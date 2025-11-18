"""
Data transforms for AquaSOT-3D.

This module contains transformation and augmentation functions for
RGB and Sonar images.
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
    RadialScalingJitter,
    SectorOcclusion,
    IntensityNormalization,
    FanMaskApplication,
    LogPolarTransform,
    letterbox_resize,
    # Bbox-aware versions
    SpeckleNoiseBboxAware,
    RadialScalingJitterBboxAware,
    SectorOcclusionBboxAware,
    IntensityNormalizationBboxAware,
    FanMaskApplicationBboxAware,
    LogPolarTransformBboxAware,
)

from .sonar_geometry import (
    generate_fan_mask,
    compute_radial_weights,
    compute_angular_resolution,
    SonarGeometryHandler,
)

from .logpolar import (
    LogPolarTransform,
    LogPolarPreprocessor,
    transform_bboxes_logpolar,
    transform_bboxes_logpolar_inverse,
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
    'RadialScalingJitter',
    'SectorOcclusion',
    'IntensityNormalization',
    'FanMaskApplication',
    'LogPolarTransform',
    'letterbox_resize',
    # Sonar transforms (bbox-aware)
    'SpeckleNoiseBboxAware',
    'RadialScalingJitterBboxAware',
    'SectorOcclusionBboxAware',
    'IntensityNormalizationBboxAware',
    'FanMaskApplicationBboxAware',
    'LogPolarTransformBboxAware',
    # Sonar geometry
    'generate_fan_mask',
    'compute_radial_weights',
    'compute_angular_resolution',
    'SonarGeometryHandler',
    # Log-polar
    'LogPolarPreprocessor',
    'transform_bboxes_logpolar',
    'transform_bboxes_logpolar_inverse',
]
