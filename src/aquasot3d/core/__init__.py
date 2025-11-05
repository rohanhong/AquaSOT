"""
Core module for AquaSOT-3D.

This module provides base classes, registry system, and configuration management
for all components in the tracking system.
"""

from .base_module import BaseModule, BaseBackbone, BaseFusion, BaseHead
from .registry import Registry, BACKBONES, FUSION_MODULES, TEMPORAL_HEADS, PREDICTION_HEADS
from .config import Config, ConfigDict

__all__ = [
    'BaseModule',
    'BaseBackbone',
    'BaseFusion',
    'BaseHead',
    'Registry',
    'BACKBONES',
    'FUSION_MODULES',
    'TEMPORAL_HEADS',
    'PREDICTION_HEADS',
    'Config',
    'ConfigDict',
]
