"""
Quality Estimation Modules

RGB and Sonar quality estimators, and adaptive weighting for multi-modal fusion.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from .rgb_quality_estimator import (
    RGBQualityEstimator,
    create_rgb_quality_estimator,
)
from .sonar_quality_estimator import (
    SonarQualityEstimator,
    create_sonar_quality_estimator,
)
from .adaptive_weighting import (
    AdaptiveWeighting,
    create_adaptive_weighting,
)

__all__ = [
    # RGB Quality Estimator
    'RGBQualityEstimator',
    'create_rgb_quality_estimator',

    # Sonar Quality Estimator
    'SonarQualityEstimator',
    'create_sonar_quality_estimator',

    # Adaptive Weighting
    'AdaptiveWeighting',
    'create_adaptive_weighting',
]
