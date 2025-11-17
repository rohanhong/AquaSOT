"""
Unit Tests for Quality Estimation and Geometry Utilities

Tests for RGB Quality Estimator, Sonar Quality Estimator, Adaptive Weighting,
and Geometry Utilities (Phase 3.5).

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# Import quality estimation modules
from aquasot3d.models.quality.rgb_quality_estimator import (
    RGBQualityEstimator,
    create_rgb_quality_estimator,
)
from aquasot3d.models.quality.sonar_quality_estimator import (
    SonarQualityEstimator,
    create_sonar_quality_estimator,
)
from aquasot3d.models.quality.adaptive_weighting import (
    AdaptiveWeighting,
    create_adaptive_weighting,
)

# Import geometry utilities
from aquasot3d.utils.geometry import (
    CoordinateTransform,
    project_3d_to_2d_rgb,
    project_3d_to_2d_sonar,
    bbox_to_center,
    center_to_bbox,
    CyclicConsistencyLoss,
    compute_reprojection_error,
    create_coordinate_transform,
    create_cyclic_consistency_loss,
)


class TestRGBQualityEstimator:
    """Tests for RGB Quality Estimator."""

    @pytest.fixture
    def rgb_quality_estimator(self):
        """Create RGB quality estimator for testing."""
        return RGBQualityEstimator(
            method='combined',
            learnable=False,
            sharpness_threshold=100.0,
            temperature=50.0,
        )

    @pytest.fixture
    def rgb_quality_learnable(self):
        """Create learnable RGB quality estimator."""
        return RGBQualityEstimator(
            method='combined',
            learnable=True,
        )

    def test_rgb_initialization(self, rgb_quality_estimator):
        """Test RGB quality estimator initialization."""
        assert isinstance(rgb_quality_estimator, RGBQualityEstimator)
        assert rgb_quality_estimator.method == 'combined'
        assert not rgb_quality_estimator.learnable
        assert hasattr(rgb_quality_estimator, 'laplacian_kernel')
        assert hasattr(rgb_quality_estimator, 'sobel_x')
        assert hasattr(rgb_quality_estimator, 'sobel_y')

    def test_rgb_laplacian_variance(self, rgb_quality_estimator):
        """Test Laplacian variance computation."""
        rgb_frame = torch.randn(4, 3, 256, 256)
        variance = rgb_quality_estimator.compute_laplacian_variance(rgb_frame)

        assert variance.shape == (4, 1)
        assert torch.all(variance >= 0)

    def test_rgb_gradient_magnitude(self, rgb_quality_estimator):
        """Test gradient magnitude computation."""
        rgb_frame = torch.randn(4, 3, 256, 256)
        magnitude = rgb_quality_estimator.compute_gradient_magnitude(rgb_frame)

        assert magnitude.shape == (4, 1)
        assert torch.all(magnitude >= 0)

    def test_rgb_frequency_quality(self, rgb_quality_estimator):
        """Test frequency domain quality computation."""
        rgb_frame = torch.randn(4, 3, 256, 256)
        quality = rgb_quality_estimator.compute_frequency_quality(rgb_frame)

        assert quality.shape == (4, 1)
        assert torch.all(quality >= 0)
        assert torch.all(quality <= 1)

    def test_rgb_forward_without_bbox(self, rgb_quality_estimator):
        """Test RGB quality estimation without bbox."""
        rgb_frame = torch.randn(4, 3, 256, 256)
        quality = rgb_quality_estimator(rgb_frame)

        assert quality.shape == (4, 1)
        assert torch.all(quality >= 0)
        assert torch.all(quality <= 1)

    def test_rgb_forward_with_bbox(self, rgb_quality_estimator):
        """Test RGB quality estimation with bbox."""
        rgb_frame = torch.randn(4, 3, 256, 256)
        bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)

        quality = rgb_quality_estimator(rgb_frame, bbox)

        assert quality.shape == (4, 1)
        assert torch.all(quality >= 0)
        assert torch.all(quality <= 1)

    def test_rgb_learnable_forward(self, rgb_quality_learnable):
        """Test learnable RGB quality estimation."""
        rgb_frame = torch.randn(4, 3, 256, 256)
        bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)

        quality = rgb_quality_learnable(rgb_frame, bbox)

        assert quality.shape == (4, 1)
        assert torch.all(quality >= 0)
        assert torch.all(quality <= 1)

    def test_rgb_crop_roi(self, rgb_quality_estimator):
        """Test ROI cropping."""
        rgb_frame = torch.randn(4, 3, 256, 256)
        bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)

        roi = rgb_quality_estimator.crop_roi(rgb_frame, bbox)

        assert roi.shape == (4, 3, 64, 64)

    def test_rgb_factory_function(self):
        """Test factory function."""
        estimator = create_rgb_quality_estimator(method='laplacian', learnable=False)
        assert isinstance(estimator, RGBQualityEstimator)
        assert estimator.method == 'laplacian'

    def test_rgb_forward_with_metadata(self, rgb_quality_estimator):
        """
        Test RGB quality estimation with metadata parameter.

        CRITICAL FIX (Audit 2025111706 Finding 1 & 3):
        Unit test that RGBQualityEstimator accepts metadata in forward().
        """
        rgb_frame = torch.randn(2, 3, 256, 256)
        bbox = torch.tensor([[64, 64, 128, 128], [32, 32, 64, 64]], dtype=torch.float32)

        # Test with None metadata
        quality_none = rgb_quality_estimator(rgb_frame, bbox, metadata=None)
        assert quality_none.shape == (2, 1)
        assert torch.all((quality_none >= 0) & (quality_none <= 1))

        # Test with dict metadata (legacy)
        metadata_dict = {'pad_x': 10, 'pad_y': 20, 'scale_x': 0.5, 'scale_y': 0.5}
        quality_dict = rgb_quality_estimator(rgb_frame, bbox, metadata=metadata_dict)
        assert quality_dict.shape == (2, 1)

        # Test with list metadata (per-sample)
        metadata_list = [
            {'pad_x': 10, 'pad_y': 20},
            {'pad_x': 5, 'pad_y': 10}
        ]
        quality_list = rgb_quality_estimator(rgb_frame, bbox, metadata=metadata_list)
        assert quality_list.shape == (2, 1)

    def test_rgb_crop_roi_with_metadata(self, rgb_quality_estimator):
        """
        Test ROI cropping with padding metadata.

        CRITICAL FIX (Audit 2025111706 Finding 2):
        Verify that crop_roi() respects padding information.
        """
        rgb_frame = torch.randn(2, 3, 256, 256)
        bbox = torch.tensor([[64, 64, 128, 128], [32, 32, 64, 64]], dtype=torch.float32)

        # Test without metadata
        roi_none = rgb_quality_estimator.crop_roi(rgb_frame, bbox, metadata=None)
        assert roi_none.shape == (2, 3, 64, 64)

        # Test with metadata
        metadata = [
            {'pad_x': 10, 'pad_y': 20},
            {'pad_x': 5, 'pad_y': 10}
        ]
        roi_meta = rgb_quality_estimator.crop_roi(rgb_frame, bbox, metadata=metadata)
        assert roi_meta.shape == (2, 3, 64, 64)

        # Both should return valid ROIs (no exceptions)
        assert torch.all(torch.isfinite(roi_none))
        assert torch.all(torch.isfinite(roi_meta))


class TestSonarQualityEstimator:
    """Tests for Sonar Quality Estimator."""

    @pytest.fixture
    def sonar_quality_estimator(self):
        """Create Sonar quality estimator for testing."""
        return SonarQualityEstimator(
            method='combined',
            learnable=False,
            snr_threshold=5.0,
            temperature=2.0,
        )

    @pytest.fixture
    def sonar_quality_learnable(self):
        """Create learnable Sonar quality estimator."""
        return SonarQualityEstimator(
            method='combined',
            learnable=True,
        )

    def test_sonar_initialization(self, sonar_quality_estimator):
        """Test Sonar quality estimator initialization."""
        assert isinstance(sonar_quality_estimator, SonarQualityEstimator)
        assert sonar_quality_estimator.method == 'combined'
        assert not sonar_quality_estimator.learnable

    def test_sonar_compute_snr(self, sonar_quality_estimator):
        """Test SNR computation."""
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)

        snr = sonar_quality_estimator.compute_snr(sonar_frame, bbox)

        assert snr.shape == (4, 1)
        assert torch.all(snr >= 0)

    def test_sonar_compute_speckle_noise(self, sonar_quality_estimator):
        """Test speckle noise computation."""
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)

        noise_level = sonar_quality_estimator.compute_speckle_noise(sonar_frame, bbox)

        assert noise_level.shape == (4, 1)
        assert torch.all(noise_level >= 0)
        assert torch.all(noise_level <= 1)

    def test_sonar_compute_echo_intensity(self, sonar_quality_estimator):
        """Test echo intensity computation."""
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)

        intensity = sonar_quality_estimator.compute_echo_intensity(sonar_frame, bbox)

        assert intensity.shape == (4, 1)
        assert torch.all(intensity >= 0)

    def test_sonar_forward_without_bbox(self, sonar_quality_estimator):
        """Test Sonar quality estimation without bbox."""
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        quality = sonar_quality_estimator(sonar_frame)

        assert quality.shape == (4, 1)
        assert torch.all(quality >= 0)
        assert torch.all(quality <= 1)

    def test_sonar_forward_with_bbox(self, sonar_quality_estimator):
        """Test Sonar quality estimation with bbox."""
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)

        quality = sonar_quality_estimator(sonar_frame, bbox)

        assert quality.shape == (4, 1)
        assert torch.all(quality >= 0)
        assert torch.all(quality <= 1)

    def test_sonar_learnable_forward(self, sonar_quality_learnable):
        """Test learnable Sonar quality estimation."""
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)

        quality = sonar_quality_learnable(sonar_frame, bbox)

        assert quality.shape == (4, 1)
        assert torch.all(quality >= 0)
        assert torch.all(quality <= 1)

    def test_sonar_crop_roi(self, sonar_quality_estimator):
        """Test ROI cropping."""
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)

        roi = sonar_quality_estimator.crop_roi(sonar_frame, bbox)

        assert roi.shape == (4, 1, 64, 64)

    def test_sonar_factory_function(self):
        """Test factory function."""
        estimator = create_sonar_quality_estimator(method='snr', learnable=False)
        assert isinstance(estimator, SonarQualityEstimator)
        assert estimator.method == 'snr'


class TestAdaptiveWeighting:
    """Tests for Adaptive Weighting."""

    @pytest.fixture
    def adaptive_weighting_softmax(self):
        """Create adaptive weighting with softmax."""
        return AdaptiveWeighting(
            weighting_method='softmax',
            temporal_smoothing=False,
            use_gating=False,
        )

    @pytest.fixture
    def adaptive_weighting_temporal(self):
        """Create adaptive weighting with temporal smoothing."""
        return AdaptiveWeighting(
            weighting_method='softmax',
            temporal_smoothing=True,
            ema_alpha=0.9,
            use_gating=True,
            quality_threshold=0.3,
        )

    @pytest.fixture
    def adaptive_weighting_learned(self):
        """Create adaptive weighting with learned fusion."""
        return AdaptiveWeighting(
            weighting_method='learned',
            learnable_fusion=True,
            temporal_smoothing=False,
        )

    def test_weighting_initialization(self, adaptive_weighting_softmax):
        """Test adaptive weighting initialization."""
        assert isinstance(adaptive_weighting_softmax, AdaptiveWeighting)
        assert adaptive_weighting_softmax.weighting_method == 'softmax'
        assert not adaptive_weighting_softmax.temporal_smoothing

    def test_compute_weights_softmax(self, adaptive_weighting_softmax):
        """Test softmax weight computation."""
        q_rgb = torch.tensor([[0.8], [0.9], [0.3], [0.7]])
        q_sonar = torch.tensor([[0.6], [0.4], [0.85], [0.5]])

        w_rgb, w_sonar = adaptive_weighting_softmax.compute_weights_softmax(q_rgb, q_sonar)

        assert w_rgb.shape == (4, 1)
        assert w_sonar.shape == (4, 1)
        assert torch.allclose(w_rgb + w_sonar, torch.ones(4, 1), atol=1e-5)

    def test_compute_weights_sigmoid(self, adaptive_weighting_softmax):
        """Test sigmoid weight computation."""
        q_rgb = torch.tensor([[0.8], [0.9], [0.3], [0.7]])
        q_sonar = torch.tensor([[0.6], [0.4], [0.85], [0.5]])

        w_rgb, w_sonar = adaptive_weighting_softmax.compute_weights_sigmoid(q_rgb, q_sonar)

        assert w_rgb.shape == (4, 1)
        assert w_sonar.shape == (4, 1)
        assert torch.allclose(w_rgb + w_sonar, torch.ones(4, 1), atol=1e-5)

    def test_forward_spatial_features(self, adaptive_weighting_softmax):
        """Test forward pass with spatial features."""
        rgb_feat = torch.randn(4, 256, 14, 14)
        sonar_feat = torch.randn(4, 256, 14, 14)
        q_rgb = torch.tensor([[0.8], [0.9], [0.3], [0.7]])
        q_sonar = torch.tensor([[0.6], [0.4], [0.85], [0.5]])

        fused, weights = adaptive_weighting_softmax(rgb_feat, sonar_feat, q_rgb, q_sonar)

        assert fused.shape == (4, 256, 14, 14)
        assert 'w_rgb' in weights
        assert 'w_sonar' in weights
        assert weights['w_rgb'].shape == (4, 1)

    def test_forward_vector_features(self, adaptive_weighting_softmax):
        """Test forward pass with vector features."""
        rgb_feat = torch.randn(4, 256)
        sonar_feat = torch.randn(4, 256)
        q_rgb = torch.tensor([[0.8], [0.9], [0.3], [0.7]])
        q_sonar = torch.tensor([[0.6], [0.4], [0.85], [0.5]])

        fused, weights = adaptive_weighting_softmax(rgb_feat, sonar_feat, q_rgb, q_sonar)

        assert fused.shape == (4, 256)
        assert 'w_rgb' in weights
        assert 'w_sonar' in weights

    def test_temporal_smoothing(self, adaptive_weighting_temporal):
        """Test temporal smoothing."""
        rgb_feat = torch.randn(4, 256)
        sonar_feat = torch.randn(4, 256)
        q_rgb = torch.tensor([[0.8], [0.9], [0.3], [0.7]])
        q_sonar = torch.tensor([[0.6], [0.4], [0.85], [0.5]])

        # Set to eval mode to enable smoothing
        adaptive_weighting_temporal.eval()

        fused1, weights1 = adaptive_weighting_temporal(rgb_feat, sonar_feat, q_rgb, q_sonar)
        fused2, weights2 = adaptive_weighting_temporal(rgb_feat, sonar_feat, q_rgb, q_sonar)

        assert 'q_rgb_smoothed' in weights1
        assert 'q_sonar_smoothed' in weights1

    def test_quality_gating(self, adaptive_weighting_temporal):
        """Test quality-based gating."""
        w_rgb = torch.tensor([[0.6], [0.7], [0.5], [0.8]])
        w_sonar = torch.tensor([[0.4], [0.3], [0.5], [0.2]])
        q_rgb = torch.tensor([[0.8], [0.2], [0.5], [0.9]])  # Sample 1 below threshold
        q_sonar = torch.tensor([[0.6], [0.1], [0.2], [0.4]])  # Samples 1,2 below threshold

        w_rgb_gated, w_sonar_gated = adaptive_weighting_temporal.apply_quality_gating(
            w_rgb, w_sonar, q_rgb, q_sonar
        )

        assert w_rgb_gated.shape == (4, 1)
        assert w_sonar_gated.shape == (4, 1)
        # Check normalization
        assert torch.allclose(w_rgb_gated + w_sonar_gated, torch.ones(4, 1), atol=1e-5)

    def test_learned_weighting(self, adaptive_weighting_learned):
        """Test learned fusion weights."""
        rgb_feat = torch.randn(4, 256)
        sonar_feat = torch.randn(4, 256)
        q_rgb = torch.tensor([[0.8], [0.9], [0.3], [0.7]])
        q_sonar = torch.tensor([[0.6], [0.4], [0.85], [0.5]])

        fused, weights = adaptive_weighting_learned(rgb_feat, sonar_feat, q_rgb, q_sonar)

        assert fused.shape == (4, 256)
        assert 'w_rgb' in weights
        assert torch.allclose(weights['w_rgb'] + weights['w_sonar'], torch.ones(4, 1), atol=1e-5)

    def test_factory_function(self):
        """Test factory function."""
        weighting = create_adaptive_weighting('sigmoid', temporal_smoothing=False)
        assert isinstance(weighting, AdaptiveWeighting)
        assert weighting.weighting_method == 'sigmoid'


class TestCoordinateTransform:
    """Tests for Coordinate Transform."""

    @pytest.fixture
    def transform_identity(self):
        """Create identity transform."""
        return CoordinateTransform(init_identity=True, learnable=False)

    @pytest.fixture
    def transform_learnable(self):
        """Create learnable transform."""
        return CoordinateTransform(init_identity=True, learnable=True)

    def test_transform_initialization(self, transform_identity):
        """Test coordinate transform initialization."""
        assert isinstance(transform_identity, CoordinateTransform)
        assert not transform_identity.learnable
        assert transform_identity.transform_matrix.shape == (3, 4)

    def test_transform_forward_2d(self, transform_identity):
        """Test transform with 2D input."""
        points_3d = torch.randn(4, 3)
        transformed = transform_identity(points_3d)

        assert transformed.shape == (4, 3)

    def test_transform_forward_3d(self, transform_identity):
        """Test transform with 3D input."""
        points_3d = torch.randn(4, 10, 3)
        transformed = transform_identity(points_3d)

        assert transformed.shape == (4, 10, 3)

    def test_transform_learnable(self, transform_learnable):
        """Test learnable transform."""
        assert transform_learnable.learnable
        assert transform_learnable.transform_matrix.requires_grad

    def test_transform_factory_function(self):
        """Test factory function."""
        transform = create_coordinate_transform(init_identity=False, learnable=True)
        assert isinstance(transform, CoordinateTransform)
        assert transform.learnable


class TestProjectionFunctions:
    """Tests for 3D-2D projection functions."""

    def test_project_3d_to_2d_rgb(self):
        """Test RGB weak perspective projection."""
        center_3d = torch.tensor([[100.0, 200.0, 50.0], [150.0, 250.0, 60.0]])
        center_2d = project_3d_to_2d_rgb(center_3d)

        assert center_2d.shape == (2, 2)
        # Weak perspective: cx = X, cy = Y
        assert torch.allclose(center_2d[:, 0], center_3d[:, 0])
        assert torch.allclose(center_2d[:, 1], center_3d[:, 1])

    def test_project_3d_to_2d_rgb_with_principal_point(self):
        """Test RGB projection with principal point."""
        center_3d = torch.tensor([[100.0, 200.0, 50.0]])
        center_2d = project_3d_to_2d_rgb(center_3d, principal_point=(320.0, 240.0))

        assert center_2d.shape == (1, 2)
        assert center_2d[0, 0] == 100.0 + 320.0
        assert center_2d[0, 1] == 200.0 + 240.0

    def test_project_3d_to_2d_sonar(self):
        """Test Sonar learned projection."""
        center_3d = torch.tensor([[100.0, 200.0, 50.0], [150.0, 250.0, 60.0]])
        transform_matrix = torch.eye(3, 4)  # Identity transform

        center_2d = project_3d_to_2d_sonar(center_3d, transform_matrix)

        assert center_2d.shape == (2, 2)

    def test_bbox_to_center(self):
        """Test bbox to center conversion."""
        bbox = torch.tensor([[64.0, 64.0, 128.0, 128.0], [100.0, 200.0, 150.0, 120.0]])
        center = bbox_to_center(bbox)

        assert center.shape == (2, 2)
        assert center[0, 0] == 64.0 + 128.0 / 2.0
        assert center[0, 1] == 64.0 + 128.0 / 2.0
        assert center[1, 0] == 100.0 + 150.0 / 2.0
        assert center[1, 1] == 200.0 + 120.0 / 2.0

    def test_center_to_bbox(self):
        """Test center to bbox conversion."""
        center = torch.tensor([[128.0, 128.0], [175.0, 260.0]])
        size = torch.tensor([[128.0, 128.0], [150.0, 120.0]])

        bbox = center_to_bbox(center, size)

        assert bbox.shape == (2, 4)
        assert bbox[0, 0] == 128.0 - 128.0 / 2.0
        assert bbox[0, 1] == 128.0 - 128.0 / 2.0
        assert bbox[0, 2] == 128.0
        assert bbox[0, 3] == 128.0


class TestCyclicConsistencyLoss:
    """Tests for Cyclic Consistency Loss."""

    @pytest.fixture
    def cyclic_loss(self):
        """Create cyclic consistency loss."""
        return CyclicConsistencyLoss(
            loss_type='l2',
            weight_rgb=1.0,
            weight_sonar=1.0,
            normalize=False,
        )

    def test_cyclic_loss_initialization(self, cyclic_loss):
        """Test cyclic loss initialization."""
        assert isinstance(cyclic_loss, CyclicConsistencyLoss)
        assert cyclic_loss.loss_type == 'l2'
        assert cyclic_loss.weight_rgb == 1.0
        assert cyclic_loss.weight_sonar == 1.0

    def test_cyclic_loss_forward(self, cyclic_loss):
        """Test cyclic loss forward pass."""
        pred_3d = torch.randn(4, 3)
        gt_rgb_bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)
        gt_sonar_bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)
        transform = torch.eye(3, 4)

        loss, loss_dict = cyclic_loss(pred_3d, gt_rgb_bbox, gt_sonar_bbox, transform)

        assert loss.item() >= 0
        assert 'cyclic_total' in loss_dict
        assert 'cyclic_rgb' in loss_dict
        assert 'cyclic_sonar' in loss_dict
        assert 'proj_rgb_2d' in loss_dict
        assert 'proj_sonar_2d' in loss_dict

    def test_cyclic_loss_with_normalization(self):
        """Test cyclic loss with normalization."""
        cyclic_loss = CyclicConsistencyLoss(
            loss_type='l2',
            normalize=True,
        )

        pred_3d = torch.randn(4, 3)
        gt_rgb_bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)
        gt_sonar_bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)
        transform = torch.eye(3, 4)

        loss, loss_dict = cyclic_loss(
            pred_3d, gt_rgb_bbox, gt_sonar_bbox, transform,
            rgb_img_size=(256, 256),
            sonar_img_size=(682, 376)
        )

        assert loss.item() >= 0

    def test_cyclic_loss_l1(self):
        """Test cyclic loss with L1 loss type."""
        cyclic_loss = CyclicConsistencyLoss(loss_type='l1')

        pred_3d = torch.randn(4, 3)
        gt_rgb_bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)
        gt_sonar_bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)
        transform = torch.eye(3, 4)

        loss, loss_dict = cyclic_loss(pred_3d, gt_rgb_bbox, gt_sonar_bbox, transform)

        assert loss.item() >= 0

    def test_cyclic_loss_factory_function(self):
        """Test factory function."""
        cyclic_loss = create_cyclic_consistency_loss('smooth_l1', weight_rgb=0.5, weight_sonar=1.5)
        assert isinstance(cyclic_loss, CyclicConsistencyLoss)
        assert cyclic_loss.loss_type == 'smooth_l1'


class TestReprojectionError:
    """Tests for reprojection error computation."""

    def test_compute_reprojection_error(self):
        """Test reprojection error computation."""
        pred_3d = torch.tensor([[100.0, 200.0, 50.0], [150.0, 250.0, 60.0]])
        gt_rgb_bbox = torch.tensor([[64, 64, 128, 128], [100, 200, 150, 120]], dtype=torch.float32)
        gt_sonar_bbox = torch.tensor([[100, 200, 150, 120], [150, 250, 180, 140]], dtype=torch.float32)
        transform = torch.eye(3, 4)

        errors = compute_reprojection_error(pred_3d, gt_rgb_bbox, gt_sonar_bbox, transform)

        assert 'error_rgb' in errors
        assert 'error_sonar' in errors
        assert 'error_mean' in errors
        assert errors['error_rgb'].shape == (2,)
        assert errors['error_sonar'].shape == (2,)
        assert errors['error_mean'].shape == (2,)
        assert torch.all(errors['error_rgb'] >= 0)
        assert torch.all(errors['error_sonar'] >= 0)


# Integration tests
class TestQualityGeometryIntegration:
    """Integration tests for quality estimation and geometry utilities."""

    def test_quality_to_weighting_pipeline(self):
        """Test quality estimation to adaptive weighting pipeline."""
        # Create estimators
        rgb_estimator = create_rgb_quality_estimator(method='combined', learnable=False)
        sonar_estimator = create_sonar_quality_estimator(method='combined', learnable=False)
        weighting = create_adaptive_weighting('softmax', temporal_smoothing=False)

        # Generate inputs
        rgb_frame = torch.randn(4, 3, 256, 256)
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        rgb_bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)
        sonar_bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)
        rgb_feat = torch.randn(4, 256, 14, 14)
        sonar_feat = torch.randn(4, 256, 14, 14)

        # Pipeline: quality estimation -> adaptive weighting
        q_rgb = rgb_estimator(rgb_frame, rgb_bbox)
        q_sonar = sonar_estimator(sonar_frame, sonar_bbox)
        fused, weights = weighting(rgb_feat, sonar_feat, q_rgb, q_sonar)

        assert fused.shape == (4, 256, 14, 14)
        assert weights['w_rgb'].shape == (4, 1)
        assert torch.allclose(weights['w_rgb'] + weights['w_sonar'], torch.ones(4, 1), atol=1e-5)

    def test_geometry_cyclic_consistency_pipeline(self):
        """Test geometry utilities with cyclic consistency loss."""
        # Create coordinate transform and loss
        transform = create_coordinate_transform(init_identity=True, learnable=True)
        cyclic_loss = create_cyclic_consistency_loss('l2', weight_rgb=1.0, weight_sonar=1.0)

        # Generate inputs
        pred_3d = torch.randn(4, 3, requires_grad=True)
        gt_rgb_bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)
        gt_sonar_bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)

        # Transform and compute loss
        transformed_3d = transform(pred_3d)
        loss, loss_dict = cyclic_loss(
            transformed_3d,
            gt_rgb_bbox,
            gt_sonar_bbox,
            transform.transform_matrix
        )

        assert loss.requires_grad
        assert loss.item() >= 0

        # Test backpropagation
        loss.backward()
        assert pred_3d.grad is not None

    def test_end_to_end_quality_geometry_fusion(self):
        """Test end-to-end pipeline with quality, geometry, and fusion."""
        # Create all modules
        rgb_estimator = create_rgb_quality_estimator(method='combined', learnable=False)
        sonar_estimator = create_sonar_quality_estimator(method='combined', learnable=False)
        weighting = create_adaptive_weighting('softmax', temporal_smoothing=False)
        transform = create_coordinate_transform(init_identity=True, learnable=True)
        cyclic_loss = create_cyclic_consistency_loss('l2')

        # Generate inputs
        rgb_frame = torch.randn(4, 3, 256, 256)
        sonar_frame = torch.randn(4, 1, 682, 376).abs()
        rgb_bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)
        sonar_bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)
        rgb_feat = torch.randn(4, 256)
        sonar_feat = torch.randn(4, 256)
        pred_3d = torch.randn(4, 3, requires_grad=True)

        # Full pipeline
        q_rgb = rgb_estimator(rgb_frame, rgb_bbox)
        q_sonar = sonar_estimator(sonar_frame, sonar_bbox)
        fused_feat, weights = weighting(rgb_feat, sonar_feat, q_rgb, q_sonar)
        transformed_3d = transform(pred_3d)
        loss, loss_dict = cyclic_loss(
            transformed_3d,
            rgb_bbox,
            sonar_bbox,
            transform.transform_matrix
        )

        # Verify outputs
        assert fused_feat.shape == (4, 256)
        assert q_rgb.shape == (4, 1)
        assert q_sonar.shape == (4, 1)
        assert loss.item() >= 0
        assert 'proj_rgb_2d' in loss_dict
        assert 'proj_sonar_2d' in loss_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
