"""
Dataset loaders for AquaSOT-3D.

ARCHITECTURE CHANGE (Audit 2025112104):
Migrated from frame-sequence to template-search paradigm.
Removed sequence_dataset and sequence_sampler modules.

This module contains dataset classes for template-search tracking.
"""

from .rgb_sonar_dataset import (
    RGBSonarDataset,
    MultiSequenceRGBSonarDataset,
    SequenceMetadata,
    is_occluded,
    parse_groundtruth,
)
from .template_search_dataset import TemplateSearchDataset, template_search_collate_fn
from .evaluation_dataset import EvaluationSequenceDataset
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
    # Base datasets
    'RGBSonarDataset',
    'MultiSequenceRGBSonarDataset',
    'SequenceMetadata',
    # Template-search training dataset
    'TemplateSearchDataset',
    'template_search_collate_fn',
    # Evaluation dataset
    'EvaluationSequenceDataset',
    # Utilities
    'is_occluded',
    'parse_groundtruth',
    # Augmentation
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
