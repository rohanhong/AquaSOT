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
from .sequence_dataset import FrameSequenceDataset
from .evaluation_dataset import EvaluationSequenceDataset
from .sequence_sampler import (
    BaseSampler,
    UniformSampler,
    RandomSampler,
    OcclusionAwareSampler,
    AdaptiveSampler,
    create_sampler,
)
from .augmentation import (
    BaseAugmentation,
    TrackingAugmentation,
    TemplateSearchAugmentation,
    NoAugmentation,
    build_augmentation,
    # Backward compatibility aliases
    BaseAugmentationBboxAware,
    TrackingAugmentationBboxAware,
    TemplateSearchAugmentationBboxAware,
    NoAugmentationBboxAware,
    build_augmentation_bbox_aware,
)

__all__ = [
    'RGBSonarDataset',
    'MultiSequenceRGBSonarDataset',
    'SequenceMetadata',
    'FrameSequenceDataset',
    'EvaluationSequenceDataset',
    'is_occluded',
    'parse_groundtruth',
    'BaseSampler',
    'UniformSampler',
    'RandomSampler',
    'OcclusionAwareSampler',
    'AdaptiveSampler',
    'create_sampler',
    # 'TemplateSearchSampler',  # Deprecated: full-image approach used instead
    'BaseAugmentation',
    'TrackingAugmentation',
    'TemplateSearchAugmentation',
    'NoAugmentation',
    'build_augmentation',
    # Backward compatibility aliases
    'BaseAugmentationBboxAware',
    'TrackingAugmentationBboxAware',
    'TemplateSearchAugmentationBboxAware',
    'NoAugmentationBboxAware',
    'build_augmentation_bbox_aware',
]
