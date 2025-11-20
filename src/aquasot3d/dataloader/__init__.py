"""
Data loading and preprocessing for AquaSOT-3D.

ARCHITECTURE CHANGE (Audit 2025112104):
Migrated from frame-sequence to template-search paradigm.
Removed FrameSequenceDataset, using TemplateSearchDataset instead.

This module provides dataset loaders, data transforms, and utilities for
template-search tracking.
"""

from . import datasets
from . import transforms
from .datasets import (
    RGBSonarDataset,
    MultiSequenceRGBSonarDataset,
    TemplateSearchDataset,
    template_search_collate_fn,
    EvaluationSequenceDataset,
)
from .loader import (
    collate_fn_template_search,
    create_dataloader,
    InfiniteDataLoader,
    PrefetchLoader,
    create_training_loaders,
)

__all__ = [
    'datasets',
    'transforms',
    # Base datasets
    'RGBSonarDataset',
    'MultiSequenceRGBSonarDataset',
    # Template-search dataset (training)
    'TemplateSearchDataset',
    'template_search_collate_fn',
    # Evaluation dataset
    'EvaluationSequenceDataset',
    # Dataloader utilities
    'collate_fn_template_search',
    'create_dataloader',
    'InfiniteDataLoader',
    'PrefetchLoader',
    'create_training_loaders',
]
