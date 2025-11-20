"""
Unit Tests for Quality Estimation

Tests for RGB Quality Estimator, Sonar Quality Estimator, and Adaptive Weighting.

Author: AquaSOT-3D Team
Date: 2025-11-20
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


# Integration tests
class TestQualityIntegration:
    """Integration tests for quality estimation."""

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
