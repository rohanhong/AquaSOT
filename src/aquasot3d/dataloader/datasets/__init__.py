"""
Dataset loaders for AquaSOT-3D.

This module contains dataset classes for loading RGB-Sonar tracking sequences.
"""

from .rgb_sonar_dataset import (
    RGBSonarDataset,
    MultiSequenceRGBSonarDataset,
    SequenceMetadata,
    is_occluded,
    parse_groundtruth,
)
from .tracking_dataset import TrackingDataset
from .evaluation_dataset import EvaluationSequenceDataset
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
    'MultiSequenceRGBSonarDataset',
    'SequenceMetadata',
    'TrackingDataset',
    'EvaluationSequenceDataset',
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
