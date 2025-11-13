"""
Test Suite for Quality Gating Integration

Tests the integration of quality estimators into the tracker.

Based on audit report: 2025111403_Comprehensive_Algo_Audit.md
Issue: RGBQualityEstimator, SonarQualityEstimator, and AdaptiveWeighting
were only referenced in unit tests, not integrated into the tracker.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.tracker import AquaSOT3DTracker, build_tracker
from aquasot3d.models.quality import (
    RGBQualityEstimator,
    SonarQualityEstimator,
    AdaptiveWeighting,
)


class TestQualityEstimatorIntegration:
    """Test that quality estimators are properly integrated into tracker."""

    def test_quality_estimators_instantiated(self):
        """
        Test that tracker instantiates quality estimators when use_quality_gating=True.

        Verifies fix for: Quality estimators were never instantiated in tracker.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'quality_config': {
                'rgb': {'method': 'combined', 'learnable': True},
                'sonar': {'method': 'combined', 'learnable': True},
                'adaptive_weighting': {'method': 'softmax'}
            }
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        # Check quality estimators are instantiated
        assert hasattr(tracker, 'rgb_quality_estimator'), \
            "RGB quality estimator not instantiated"
        assert hasattr(tracker, 'sonar_quality_estimator'), \
            "Sonar quality estimator not instantiated"
        assert hasattr(tracker, 'adaptive_weighting'), \
            "Adaptive weighting not instantiated"

        assert tracker.rgb_quality_estimator is not None
        assert tracker.sonar_quality_estimator is not None
        assert tracker.adaptive_weighting is not None

        # Check types
        assert isinstance(tracker.rgb_quality_estimator, RGBQualityEstimator)
        assert isinstance(tracker.sonar_quality_estimator, SonarQualityEstimator)
        assert isinstance(tracker.adaptive_weighting, AdaptiveWeighting)

    def test_quality_estimators_not_instantiated_when_disabled(self):
        """
        Test that quality estimators are None when use_quality_gating=False.

        Ensures no unnecessary computation when feature is disabled.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': False
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        assert tracker.rgb_quality_estimator is None
        assert tracker.sonar_quality_estimator is None
        assert tracker.adaptive_weighting is None

    def test_quality_scores_computed_in_forward(self):
        """
        Test that quality scores are computed during forward pass.

        Verifies that quality estimators are actually called, not just instantiated.
        """
        # Minimal tracker config
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'use_temporal': False,
            'use_weak3d': False,
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        # Create dummy inputs
        B = 2
        rgb_template = torch.randn(B, 3, 128, 128)
        rgb_search = torch.randn(B, 3, 256, 256)
        sonar_template = torch.randn(B, 1, 128, 128)
        sonar_search = torch.randn(B, 1, 256, 256)

        # Forward pass
        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

        # Check quality scores are in output
        assert 'rgb_quality_physical' in outputs, \
            "RGB quality scores not computed"
        assert 'sonar_quality_physical' in outputs, \
            "Sonar quality scores not computed"

        # Check quality scores have correct shape
        rgb_quality = outputs['rgb_quality_physical']
        sonar_quality = outputs['sonar_quality_physical']

        assert rgb_quality is not None
        assert sonar_quality is not None
        assert rgb_quality.shape == (B,), \
            f"Expected RGB quality shape (2,), got {rgb_quality.shape}"
        assert sonar_quality.shape == (B,), \
            f"Expected Sonar quality shape (2,), got {sonar_quality.shape}"

        # Quality scores should be in [0, 1] range
        assert torch.all(rgb_quality >= 0) and torch.all(rgb_quality <= 1), \
            "RGB quality scores out of [0, 1] range"
        assert torch.all(sonar_quality >= 0) and torch.all(sonar_quality <= 1), \
            "Sonar quality scores out of [0, 1] range"

    def test_adaptive_weighting_called_in_forward(self):
        """
        Test that adaptive_weighting module is called during forward pass.

        Verifies that quality-adaptive fusion is active, not just quality estimation.
        This addresses audit finding #4: quality scores must modulate fusion.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'use_temporal': False,
            'use_weak3d': False,
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        # Create dummy inputs
        B = 2
        rgb_template = torch.randn(B, 3, 128, 128)
        rgb_search = torch.randn(B, 3, 256, 256)
        sonar_template = torch.randn(B, 1, 128, 128)
        sonar_search = torch.randn(B, 1, 256, 256)

        # Forward pass
        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

        # Check that quality_weights are in output (proof that adaptive_weighting was called)
        assert 'quality_weights' in outputs, \
            "Quality weights not in output: adaptive_weighting not called"

        quality_weights = outputs['quality_weights']
        assert quality_weights is not None

        # Check that weights contain expected keys from AdaptiveWeighting
        assert 'w_rgb' in quality_weights, \
            "RGB weight not in quality_weights"
        assert 'w_sonar' in quality_weights, \
            "Sonar weight not in quality_weights"

        # Check weight shapes
        w_rgb = quality_weights['w_rgb']
        w_sonar = quality_weights['w_sonar']

        assert w_rgb.shape == (B, 1), \
            f"Expected w_rgb shape (2, 1), got {w_rgb.shape}"
        assert w_sonar.shape == (B, 1), \
            f"Expected w_sonar shape (2, 1), got {w_sonar.shape}"

        # Weights should sum to ~1.0 (softmax normalization)
        weight_sum = (w_rgb + w_sonar).squeeze()
        assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), \
            f"Weights don't sum to 1.0: {weight_sum}"

    def test_adaptive_weights_react_to_quality_differences(self):
        """
        Test that adaptive weights change based on quality score differences.

        Verifies that the quality-adaptive gating mechanism is working end-to-end.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'use_temporal': False,
            'use_weak3d': False,
            'quality_config': {
                'rgb': {'method': 'laplacian', 'learnable': False},
                'sonar': {'method': 'snr', 'learnable': False}
            }
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        B = 1

        # Scenario 1: High RGB quality, Low Sonar quality
        rgb_search_sharp = torch.zeros(B, 3, 256, 256)
        rgb_search_sharp[:, :, 100:150, 100:150] = 1.0  # Sharp edge

        sonar_search_weak = torch.randn(B, 1, 256, 256) * 0.1  # Weak, noisy

        rgb_template = torch.randn(B, 3, 128, 128)
        sonar_template = torch.randn(B, 1, 128, 128)

        with torch.no_grad():
            out1 = tracker(
                rgb_template, rgb_search_sharp,
                sonar_template, sonar_search_weak,
                return_losses=False
            )

        # Scenario 2: Low RGB quality, High Sonar quality
        rgb_search_blur = torch.ones(B, 3, 256, 256) * 0.5  # Uniform, no edges

        sonar_search_strong = torch.zeros(B, 1, 256, 256)
        sonar_search_strong[:, :, 100:150, 100:150] = 1.0  # Strong echo

        with torch.no_grad():
            out2 = tracker(
                rgb_template, rgb_search_blur,
                sonar_template, sonar_search_strong,
                return_losses=False
            )

        # Check that weights shifted based on quality
        w_rgb_1 = out1['quality_weights']['w_rgb']
        w_sonar_1 = out1['quality_weights']['w_sonar']

        w_rgb_2 = out2['quality_weights']['w_rgb']
        w_sonar_2 = out2['quality_weights']['w_sonar']

        # Scenario 1: RGB weight should be higher (RGB is sharper)
        # Scenario 2: Sonar weight should be higher (Sonar has stronger signal)
        assert w_rgb_1 > w_rgb_2, \
            "RGB weight didn't increase when RGB quality was higher"
        assert w_sonar_2 > w_sonar_1, \
            "Sonar weight didn't increase when Sonar quality was higher"

    def test_quality_scores_react_to_degradation(self):
        """
        Test that quality estimators react to image degradation.

        Verifies that SNR/sharpness estimation is working as intended.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'use_temporal': False,
            'use_weak3d': False,
            'quality_config': {
                'rgb': {'method': 'laplacian', 'learnable': False},
                'sonar': {'method': 'snr', 'learnable': False}
            }
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        B = 1

        # High-quality RGB image (sharp edges)
        rgb_search_sharp = torch.zeros(B, 3, 256, 256)
        rgb_search_sharp[:, :, 100:150, 100:150] = 1.0  # Sharp edge

        # Low-quality RGB image (blurred)
        rgb_search_blur = torch.ones(B, 3, 256, 256) * 0.5  # Uniform, no edges

        # High-quality Sonar (strong signal)
        sonar_search_strong = torch.zeros(B, 1, 256, 256)
        sonar_search_strong[:, :, 100:150, 100:150] = 1.0  # Strong echo

        # Low-quality Sonar (weak signal, high noise)
        sonar_search_weak = torch.randn(B, 1, 256, 256) * 0.1  # Noisy

        # Dummy template
        rgb_template = torch.randn(B, 3, 128, 128)
        sonar_template = torch.randn(B, 1, 128, 128)

        # Test RGB quality difference
        with torch.no_grad():
            out_sharp = tracker(
                rgb_template, rgb_search_sharp,
                sonar_template, sonar_search_strong,
                return_losses=False
            )
            out_blur = tracker(
                rgb_template, rgb_search_blur,
                sonar_template, sonar_search_strong,
                return_losses=False
            )

        rgb_quality_sharp = out_sharp['rgb_quality_physical']
        rgb_quality_blur = out_blur['rgb_quality_physical']

        # Sharp image should have higher quality
        assert rgb_quality_sharp.mean() > rgb_quality_blur.mean(), \
            "RGB quality estimator not responding to sharpness"

        # Test Sonar quality difference
        with torch.no_grad():
            out_strong = tracker(
                rgb_template, rgb_search_sharp,
                sonar_template, sonar_search_strong,
                return_losses=False
            )
            out_weak = tracker(
                rgb_template, rgb_search_sharp,
                sonar_template, sonar_search_weak,
                return_losses=False
            )

        sonar_quality_strong = out_strong['sonar_quality_physical']
        sonar_quality_weak = out_weak['sonar_quality_physical']

        # Strong signal should have higher quality
        assert sonar_quality_strong.mean() > sonar_quality_weak.mean(), \
            "Sonar quality estimator not responding to SNR"

    def test_quality_estimators_in_state_dict(self):
        """
        Test that quality estimators are saved in checkpoint.

        Ensures model persistence and reproducibility.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'quality_config': {
                'adaptive_weighting': {'learnable_fusion': True}  # Enable learnable for state_dict test
            }
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        state_dict = tracker.state_dict()

        # Check quality estimator keys are in state_dict
        rgb_quality_keys = [k for k in state_dict.keys() if 'rgb_quality_estimator' in k]
        sonar_quality_keys = [k for k in state_dict.keys() if 'sonar_quality_estimator' in k]
        adaptive_keys = [k for k in state_dict.keys() if 'adaptive_weighting' in k]

        assert len(rgb_quality_keys) > 0, \
            f"RGB quality estimator not in state_dict. Keys: {list(state_dict.keys())[:10]}"
        assert len(sonar_quality_keys) > 0, \
            f"Sonar quality estimator not in state_dict. Keys: {list(state_dict.keys())[:10]}"
        assert len(adaptive_keys) > 0, \
            f"Adaptive weighting not in state_dict. Keys: {list(state_dict.keys())[:10]}"

    def test_quality_estimators_trainable(self):
        """
        Test that learnable quality estimators have trainable parameters.

        Verifies that quality networks can be jointly optimized.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'quality_config': {
                'rgb': {'learnable': True},
                'sonar': {'learnable': True},
                'adaptive_weighting': {'learnable_fusion': True}
            }
        }

        tracker = AquaSOT3DTracker(**tracker_config)

        # Check that quality networks have parameters
        rgb_params = list(tracker.rgb_quality_estimator.parameters())
        sonar_params = list(tracker.sonar_quality_estimator.parameters())

        assert len(rgb_params) > 0, \
            "RGB quality estimator has no parameters"
        assert len(sonar_params) > 0, \
            "Sonar quality estimator has no parameters"

        # Check parameters require gradients
        assert all(p.requires_grad for p in rgb_params), \
            "RGB quality estimator parameters not trainable"
        assert all(p.requires_grad for p in sonar_params), \
            "Sonar quality estimator parameters not trainable"

    def test_gradient_flow_through_quality_estimators(self):
        """
        Test that gradients flow through quality estimators during training.

        Ensures joint optimization of tracking and quality estimation.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'use_quality_gating': True,
            'use_temporal': False,
            'use_weak3d': False,
            'quality_config': {
                'rgb': {'learnable': True},
                'sonar': {'learnable': True}
            }
        }

        tracker = AquaSOT3DTracker(**tracker_config)
        tracker.train()

        # Create inputs
        B = 2
        rgb_template = torch.randn(B, 3, 128, 128)
        rgb_search = torch.randn(B, 3, 256, 256)
        sonar_template = torch.randn(B, 1, 128, 128)
        sonar_search = torch.randn(B, 1, 256, 256)

        # Forward pass
        outputs = tracker(
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            return_losses=False
        )

        # Create dummy loss using quality scores
        rgb_quality = outputs['rgb_quality_physical']
        sonar_quality = outputs['sonar_quality_physical']
        loss = (rgb_quality.mean() + sonar_quality.mean())

        # Backward pass
        loss.backward()

        # Check gradients exist in quality estimators
        rgb_estimator_params = list(tracker.rgb_quality_estimator.parameters())
        sonar_estimator_params = list(tracker.sonar_quality_estimator.parameters())

        rgb_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in rgb_estimator_params)
        sonar_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in sonar_estimator_params)

        assert rgb_has_grad, \
            "No gradients in RGB quality estimator"
        assert sonar_has_grad, \
            "No gradients in Sonar quality estimator"


    def test_quality_consistency_loss_computed(self):
        """
        Test that quality-consistency loss is computed during training.

        Verifies that FusionHead.compute_fusion_loss is called and loss aligns
        learned quality with physical quality (SNR/sharpness).
        Addresses audit requirement: "align loss to encourage consistency between
        predicted quality and IoU/visibility signals".
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'fusion_head_config': {},  # Enable fusion_head
            'use_quality_gating': True,
            'use_temporal': False,
            'use_weak3d': False,
        }

        tracker = AquaSOT3DTracker(**tracker_config)
        tracker.train()  # Set to training mode

        # Create dummy inputs
        B = 2
        rgb_template = torch.randn(B, 3, 128, 128)
        rgb_search = torch.randn(B, 3, 256, 256)
        sonar_template = torch.randn(B, 1, 128, 128)
        sonar_search = torch.randn(B, 1, 256, 256)

        # Forward pass with loss computation
        outputs = tracker(
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            return_losses=True  # Request losses
        )

        # Check that quality_fusion_loss is computed
        assert 'losses' in outputs, \
            "No losses in output"

        losses = outputs['losses']
        assert 'quality_fusion_loss' in losses, \
            "quality_fusion_loss not computed during training"

        quality_loss = losses['quality_fusion_loss']
        assert quality_loss is not None, \
            "quality_fusion_loss is None"

        # Loss should be a scalar tensor
        assert isinstance(quality_loss, torch.Tensor), \
            "quality_fusion_loss not a tensor"
        assert quality_loss.numel() == 1, \
            f"quality_fusion_loss should be scalar, got shape {quality_loss.shape}"

        # Loss should be finite
        assert torch.isfinite(quality_loss), \
            "quality_fusion_loss is not finite"

    def test_quality_loss_gradient_flow(self):
        """
        Test that gradients flow through quality-consistency loss.

        Ensures that quality estimators and fusion head can be jointly optimized
        via the quality-consistency loss.
        """
        tracker_config = {
            'rgb_backbone_config': {'type': 'null'},
            'sonar_backbone_config': {'type': 'null'},
            'fusion_config': {'type': 'null'},
            'fusion_head_config': {},
            'use_quality_gating': True,
            'use_temporal': False,
            'use_weak3d': False,
            'quality_config': {
                'rgb': {'learnable': True},
                'sonar': {'learnable': True}
            }
        }

        tracker = AquaSOT3DTracker(**tracker_config)
        tracker.train()

        # Create inputs
        B = 2
        rgb_template = torch.randn(B, 3, 128, 128)
        rgb_search = torch.randn(B, 3, 256, 256)
        sonar_template = torch.randn(B, 1, 128, 128)
        sonar_search = torch.randn(B, 1, 256, 256)

        # Forward pass
        outputs = tracker(
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            return_losses=True
        )

        # Get quality loss
        quality_loss = outputs['losses']['quality_fusion_loss']

        # Backward pass
        quality_loss.backward()

        # Check gradients in quality estimators
        rgb_estimator_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tracker.rgb_quality_estimator.parameters()
        )
        sonar_estimator_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tracker.sonar_quality_estimator.parameters()
        )

        assert rgb_estimator_has_grad or sonar_estimator_has_grad, \
            "No gradients in quality estimators from quality-consistency loss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
