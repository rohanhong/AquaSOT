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
    RGBTransformPipeline,
)

from .sonar_transforms import (
    SpeckleNoise,
    RadialScalingJitter,
    SectorOcclusion,
    IntensityNormalization,
    FanMaskApplication,
    SonarTransformPipeline,
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
)

__all__ = [
    # RGB transforms
    'ColorJitter',
    'RandomBlur',
    'RandomOcclusion',
    'RandomHorizontalFlip',
    'Normalize',
    'ToTensor',
    'RGBTransformPipeline',
    # Sonar transforms
    'SpeckleNoise',
    'RadialScalingJitter',
    'SectorOcclusion',
    'IntensityNormalization',
    'FanMaskApplication',
    'SonarTransformPipeline',
    # Sonar geometry
    'generate_fan_mask',
    'compute_radial_weights',
    'compute_angular_resolution',
    'SonarGeometryHandler',
    # Log-polar
    'LogPolarTransform',
    'LogPolarPreprocessor',
]
