"""
AquaSOT-3D: End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation

A unified end-to-end multi-modal single-object tracking system that fuses RGB camera
and Forward-Looking Sonar (FLS) data for underwater object tracking, with weak 3D
center estimation (X, Y, Z) without requiring extrinsic calibration.

Key Features:
- Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)
- Differentiable Log-Polar preprocessing for sonar data
- Template-search paradigm for single-object tracking
- Weak 3D center estimation with cyclic consistency
- Calibration-free multi-modal fusion

Base Framework: OSTrack (ECCV 2022)
Target Dataset: RGBS50-style paired RGB+Sonar sequences
Coordinate System: Fusion Frame (X→right, Y→down, Z→forward)
"""

__version__ = '0.1.0'
__author__ = 'AquaSOT-3D Team'
__license__ = 'MIT'

# Import core modules when needed
# Lazy imports to avoid circular dependencies and speed up initial loading
