"""
Projector Module

This module contains geometric projectors for 3D-2D transformations:
- Geometric Cyclic Projector: Learnable 3D->2D projection with cyclic consistency

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

from .geom_cycle import (
    GeometricCyclicProjector,
    RGBProjector,
    SonarProjector,
    build_geometric_projector,
)

__all__ = [
    'GeometricCyclicProjector',
    'RGBProjector',
    'SonarProjector',
    'build_geometric_projector',
]
