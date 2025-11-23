"""
Core module for AquaSOT-3D.

This module provides base classes, registry system, configuration management,
and online tracking interface for all components in the tracking system.

ARCHITECTURE CHANGE (Audit 2025112104):
Added OnlineTracker for template-search inference with template update strategies.
"""

from .base_module import BaseModule, BaseBackbone, BaseFusion, BaseHead
from .online_tracker import OnlineTracker, TemplateBuffer
from .registry import (
    Registry,
    BACKBONES,
    FUSION_MODULES,
    PREDICTION_HEADS,
    TRACKERS,
    DATASETS,
    TRANSFORMS,
    SAMPLERS,
    AUGMENTATIONS,
    OPTIMIZERS,
    SCHEDULERS,
    LOSSES,
    METRICS,
    EVALUATORS,
)
from .config import Config, ConfigDict

# Aliases for backward compatibility (using _REGISTRY naming convention)
BACKBONE_REGISTRY = BACKBONES
FUSION_REGISTRY = FUSION_MODULES
PREDICTION_HEAD_REGISTRY = PREDICTION_HEADS
TRACKER_REGISTRY = TRACKERS
DATASET_REGISTRY = DATASETS
TRANSFORM_REGISTRY = TRANSFORMS
SAMPLER_REGISTRY = SAMPLERS
AUGMENTATION_REGISTRY = AUGMENTATIONS
OPTIMIZER_REGISTRY = OPTIMIZERS
SCHEDULER_REGISTRY = SCHEDULERS
LOSS_REGISTRY = LOSSES
METRIC_REGISTRY = METRICS
EVALUATOR_REGISTRY = EVALUATORS

__all__ = [
    'BaseModule',
    'BaseBackbone',
    'BaseFusion',
    'BaseHead',
    'OnlineTracker',
    'TemplateBuffer',
    'Registry',
    # Plural forms
    'BACKBONES',
    'FUSION_MODULES',
    'PREDICTION_HEADS',
    'TRACKERS',
    'DATASETS',
    'TRANSFORMS',
    'SAMPLERS',
    'AUGMENTATIONS',
    'OPTIMIZERS',
    'SCHEDULERS',
    'LOSSES',
    'METRICS',
    'EVALUATORS',
    # _REGISTRY aliases
    'BACKBONE_REGISTRY',
    'FUSION_REGISTRY',
    'PREDICTION_HEAD_REGISTRY',
    'TRACKER_REGISTRY',
    'DATASET_REGISTRY',
    'TRANSFORM_REGISTRY',
    'SAMPLER_REGISTRY',
    'AUGMENTATION_REGISTRY',
    'OPTIMIZER_REGISTRY',
    'SCHEDULER_REGISTRY',
    'LOSS_REGISTRY',
    'METRIC_REGISTRY',
    'EVALUATOR_REGISTRY',
    'Config',
    'ConfigDict',
]
