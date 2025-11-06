"""
Dataset loaders for AquaSOT-3D.

This module contains dataset classes for loading RGB-Sonar tracking sequences.
"""

from .rgb_sonar_dataset import RGBSonarDataset, is_occluded, parse_groundtruth
from .sequence_sampler import (
    BaseSampler,
    UniformSampler,
    RandomSampler,
    OcclusionAwareSampler,
    AdaptiveSampler,
    create_sampler,
)
from .template_search_sampler import TemplateSearchSampler
from .augmentation import (
    BaseAugmentation,
    TrackingAugmentation,
    TemplateSearchAugmentation,
    NoAugmentation,
    build_augmentation,
)

__all__ = [
    'RGBSonarDataset',
    'is_occluded',
    'parse_groundtruth',
    'BaseSampler',
    'UniformSampler',
    'RandomSampler',
    'OcclusionAwareSampler',
    'AdaptiveSampler',
    'create_sampler',
    'TemplateSearchSampler',
    'BaseAugmentation',
    'TrackingAugmentation',
    'TemplateSearchAugmentation',
    'NoAugmentation',
    'build_augmentation',
]
