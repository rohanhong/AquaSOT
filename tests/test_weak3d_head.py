"""
Unit tests for Weak3DHead with single-modality awareness.

This test suite verifies that Weak3DHead correctly routes samples to
appropriate 3D estimation branches based on modality availability.

Author: AquaSOT-3D Team
Date: 2025-11-21
"""

import unittest
import torch
import sys
sys.path.insert(0, '/home/rohan/Workspace/AquaSOT-3D/src')

from aquasot3d.models.heads import Weak3DHead, build_weak3d_head


class TestWeak3DHeadV2(unittest.TestCase):
    """Test suite for Weak3DHead."""

    def setUp(self):
        """Set up test fixtures."""
        self.rgb_feat_dim = 256
        self.sonar_feat_dim = 256
        self.num_radial_bins = 50
        self.batch_size = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Build head
        self.head = build_weak3d_head(
            rgb_feat_dim=self.rgb_feat_dim,
            sonar_feat_dim=self.sonar_feat_dim,
            num_radial_bins=self.num_radial_bins,
            radial_range=(0.5, 10.0),
            hidden_dim=512,
            dropout=0.1,
            default_depth=5.0
        ).to(self.device)
        self.head.eval()

    def test_dual_modality_routing(self):
        """
        Test 3.1.1: Dual-modality routing uses original Weak3D logic.
        """
        B = self.batch_size
        rgb_features = torch.randn(B, self.rgb_feat_dim, device=self.device)
        sonar_features = torch.randn(B, self.sonar_feat_dim, device=self.device)
        sonar_radial = torch.rand(B, self.num_radial_bins, device=self.device)
        sonar_radial = sonar_radial / sonar_radial.sum(dim=-1, keepdim=True)  # Normalize

        # All samples are dual-modality
        is_rgb_valid = torch.ones(B, dtype=torch.bool, device=self.device)
        is_sonar_valid = torch.ones(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            outputs = self.head(
                rgb_features=rgb_features,
                sonar_features=sonar_features,
                is_rgb_valid=is_rgb_valid,
                is_sonar_valid=is_sonar_valid,
                sonar_radial_activation=sonar_radial
            )

        # Check outputs
        self.assertIn('center_3d', outputs)
        self.assertIn('routing_info', outputs)

        center_3d = outputs['center_3d']
        routing_info = outputs['routing_info']

        # Verify shapes
        self.assertEqual(center_3d.shape, (B, 3))

        # Verify routing: all samples to dual-modality
        self.assertEqual(routing_info['num_dual'], B)
        self.assertEqual(routing_info['num_single_rgb'], 0)
        self.assertEqual(routing_info['num_single_sonar'], 0)
        self.assertEqual(routing_info['num_zero'], 0)

        # Verify Z is within radial range
        Z = outputs['Z']
        self.assertTrue(torch.all(Z >= 0.5))
        self.assertTrue(torch.all(Z <= 10.0))

        print(f"\nTest 3.1.1 PASSED: Dual-modality routing")
        print(f"  center_3d shape: {center_3d.shape}")
        print(f"  Z range: [{Z.min().item():.2f}, {Z.max().item():.2f}]")
        print(f"  Routing: {routing_info}")

    def test_single_rgb_routing(self):
        """
        Test 3.1.2: Single-RGB routing uses RGB depth estimator.
        """
        B = self.batch_size
        rgb_features = torch.randn(B, self.rgb_feat_dim, device=self.device)
        sonar_features = torch.randn(B, self.sonar_feat_dim, device=self.device)
        sonar_radial = torch.rand(B, self.num_radial_bins, device=self.device)

        # All samples are RGB-only
        is_rgb_valid = torch.ones(B, dtype=torch.bool, device=self.device)
        is_sonar_valid = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            outputs = self.head(
                rgb_features=rgb_features,
                sonar_features=sonar_features,
                is_rgb_valid=is_rgb_valid,
                is_sonar_valid=is_sonar_valid,
                sonar_radial_activation=sonar_radial
            )

        center_3d = outputs['center_3d']
        routing_info = outputs['routing_info']

        # Verify routing: all samples to single-RGB
        self.assertEqual(routing_info['num_dual'], 0)
        self.assertEqual(routing_info['num_single_rgb'], B)
        self.assertEqual(routing_info['num_single_sonar'], 0)
        self.assertEqual(routing_info['num_zero'], 0)

        # Verify Z is from RGB depth estimator (should be within radial range)
        Z = outputs['Z']
        self.assertTrue(torch.all(Z >= 0.5))
        self.assertTrue(torch.all(Z <= 10.0))

        print(f"\nTest 3.1.2 PASSED: Single-RGB routing")
        print(f"  center_3d shape: {center_3d.shape}")
        print(f"  Z range (RGB depth): [{Z.min().item():.2f}, {Z.max().item():.2f}]")
        print(f"  Routing: {routing_info}")

    def test_single_sonar_routing(self):
        """
        Test 3.1.3: Single-Sonar routing uses sonar Y-branch.
        """
        B = self.batch_size
        rgb_features = torch.randn(B, self.rgb_feat_dim, device=self.device)
        sonar_features = torch.randn(B, self.sonar_feat_dim, device=self.device)
        sonar_radial = torch.rand(B, self.num_radial_bins, device=self.device)
        sonar_radial = sonar_radial / sonar_radial.sum(dim=-1, keepdim=True)

        # All samples are Sonar-only
        is_rgb_valid = torch.zeros(B, dtype=torch.bool, device=self.device)
        is_sonar_valid = torch.ones(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            outputs = self.head(
                rgb_features=rgb_features,
                sonar_features=sonar_features,
                is_rgb_valid=is_rgb_valid,
                is_sonar_valid=is_sonar_valid,
                sonar_radial_activation=sonar_radial
            )

        center_3d = outputs['center_3d']
        routing_info = outputs['routing_info']

        # Verify routing: all samples to single-Sonar
        self.assertEqual(routing_info['num_dual'], 0)
        self.assertEqual(routing_info['num_single_rgb'], 0)
        self.assertEqual(routing_info['num_single_sonar'], B)
        self.assertEqual(routing_info['num_zero'], 0)

        # Verify Z is from sonar radial (should be within radial range)
        Z = outputs['Z']
        self.assertTrue(torch.all(Z >= 0.5))
        self.assertTrue(torch.all(Z <= 10.0))

        # Verify Y is from sonar Y-branch (not NaN)
        Y = outputs['Y']
        self.assertFalse(torch.any(torch.isnan(Y)))

        print(f"\nTest 3.1.3 PASSED: Single-Sonar routing")
        print(f"  center_3d shape: {center_3d.shape}")
        print(f"  Z range (Sonar radial): [{Z.min().item():.2f}, {Z.max().item():.2f}]")
        print(f"  Y range (Sonar Y-branch): [{Y.min().item():.2f}, {Y.max().item():.2f}]")
        print(f"  Routing: {routing_info}")

    def test_zero_modality_routing(self):
        """
        Test 3.1.4: Zero-modality routing returns default predictions.
        """
        B = self.batch_size
        rgb_features = torch.randn(B, self.rgb_feat_dim, device=self.device)
        sonar_features = torch.randn(B, self.sonar_feat_dim, device=self.device)
        sonar_radial = torch.rand(B, self.num_radial_bins, device=self.device)

        # All samples have no valid modalities
        is_rgb_valid = torch.zeros(B, dtype=torch.bool, device=self.device)
        is_sonar_valid = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            outputs = self.head(
                rgb_features=rgb_features,
                sonar_features=sonar_features,
                is_rgb_valid=is_rgb_valid,
                is_sonar_valid=is_sonar_valid,
                sonar_radial_activation=sonar_radial
            )

        center_3d = outputs['center_3d']
        routing_info = outputs['routing_info']

        # Verify routing: all samples to zero-modality
        self.assertEqual(routing_info['num_dual'], 0)
        self.assertEqual(routing_info['num_single_rgb'], 0)
        self.assertEqual(routing_info['num_single_sonar'], 0)
        self.assertEqual(routing_info['num_zero'], B)

        # Verify default predictions: (0, 0, default_depth=5.0)
        X = outputs['X']
        Y = outputs['Y']
        Z = outputs['Z']

        self.assertTrue(torch.allclose(X, torch.zeros_like(X), atol=1e-6))
        self.assertTrue(torch.allclose(Y, torch.zeros_like(Y), atol=1e-6))
        self.assertTrue(torch.allclose(Z, torch.full_like(Z, 5.0), atol=1e-6))

        print(f"\nTest 3.1.4 PASSED: Zero-modality routing")
        print(f"  center_3d (default): {center_3d[0].tolist()}")
        print(f"  Routing: {routing_info}")

    def test_mixed_modality_routing(self):
        """
        Test 3.1.5: Mixed modality batch correctly routes to different branches.
        """
        B = 8  # Larger batch for mixed scenarios
        rgb_features = torch.randn(B, self.rgb_feat_dim, device=self.device)
        sonar_features = torch.randn(B, self.sonar_feat_dim, device=self.device)
        sonar_radial = torch.rand(B, self.num_radial_bins, device=self.device)
        sonar_radial = sonar_radial / sonar_radial.sum(dim=-1, keepdim=True)

        # Mixed modality:
        # Samples 0-1: Dual-modality
        # Samples 2-3: Single-RGB
        # Samples 4-5: Single-Sonar
        # Samples 6-7: Zero-modality
        is_rgb_valid = torch.tensor(
            [True, True, True, True, False, False, False, False],
            dtype=torch.bool, device=self.device
        )
        is_sonar_valid = torch.tensor(
            [True, True, False, False, True, True, False, False],
            dtype=torch.bool, device=self.device
        )

        with torch.no_grad():
            outputs = self.head(
                rgb_features=rgb_features,
                sonar_features=sonar_features,
                is_rgb_valid=is_rgb_valid,
                is_sonar_valid=is_sonar_valid,
                sonar_radial_activation=sonar_radial
            )

        center_3d = outputs['center_3d']
        routing_info = outputs['routing_info']

        # Verify routing distribution
        self.assertEqual(routing_info['num_dual'], 2)
        self.assertEqual(routing_info['num_single_rgb'], 2)
        self.assertEqual(routing_info['num_single_sonar'], 2)
        self.assertEqual(routing_info['num_zero'], 2)

        # Verify shapes
        self.assertEqual(center_3d.shape, (B, 3))

        # Verify zero-modality samples have default predictions
        X = outputs['X']
        Y = outputs['Y']
        Z = outputs['Z']

        zero_idx = torch.tensor([6, 7], device=self.device)
        self.assertTrue(torch.allclose(X[zero_idx], torch.zeros_like(X[zero_idx]), atol=1e-6))
        self.assertTrue(torch.allclose(Y[zero_idx], torch.zeros_like(Y[zero_idx]), atol=1e-6))
        self.assertTrue(torch.allclose(Z[zero_idx], torch.full_like(Z[zero_idx], 5.0), atol=1e-6))

        # Verify non-zero-modality samples have non-default predictions
        # (at least Z should be different for dual/single-sonar samples)
        dual_sonar_idx = torch.tensor([0, 1, 4, 5], device=self.device)
        # These should have Z from sonar radial, which is unlikely to be exactly 5.0
        # We just check that they're within valid range
        self.assertTrue(torch.all(Z[dual_sonar_idx] >= 0.5))
        self.assertTrue(torch.all(Z[dual_sonar_idx] <= 10.0))

        print(f"\nTest 3.1.5 PASSED: Mixed modality routing")
        print(f"  Routing distribution: {routing_info}")
        print(f"  Zero-modality samples (6-7) center_3d: {center_3d[zero_idx].tolist()}")
        print(f"  Dual-modality samples (0-1) Z: {Z[:2].tolist()}")
        print(f"  Single-RGB samples (2-3) Z: {Z[2:4].tolist()}")
        print(f"  Single-Sonar samples (4-5) Z: {Z[4:6].tolist()}")

    def test_gradient_flow(self):
        """
        Test 3.1.6: Verify gradients flow through all branches.
        """
        B = 4
        rgb_features = torch.randn(B, self.rgb_feat_dim, device=self.device, requires_grad=True)
        sonar_features = torch.randn(B, self.sonar_feat_dim, device=self.device, requires_grad=True)
        sonar_radial = torch.rand(B, self.num_radial_bins, device=self.device, requires_grad=True)

        # Mixed modality
        is_rgb_valid = torch.tensor([True, True, False, False], dtype=torch.bool, device=self.device)
        is_sonar_valid = torch.tensor([True, False, True, False], dtype=torch.bool, device=self.device)

        # Enable gradient computation
        self.head.train()

        outputs = self.head(
            rgb_features=rgb_features,
            sonar_features=sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            sonar_radial_activation=sonar_radial
        )

        center_3d = outputs['center_3d']

        # Compute loss and backpropagate
        loss = center_3d.sum()
        loss.backward()

        # Check gradients for dual-modality sample (index 0)
        self.assertIsNotNone(rgb_features.grad)
        self.assertIsNotNone(sonar_features.grad)
        self.assertTrue(torch.any(rgb_features.grad[0] != 0.0))
        self.assertTrue(torch.any(sonar_features.grad[0] != 0.0))

        # Check gradients for single-RGB sample (index 1)
        self.assertTrue(torch.any(rgb_features.grad[1] != 0.0))

        # Check gradients for single-Sonar sample (index 2)
        self.assertTrue(torch.any(sonar_features.grad[2] != 0.0))

        # Zero-modality sample (index 3) should have zero gradients (no trainable path)
        # This is expected since zero-modality returns static defaults

        self.head.eval()

        print(f"\nTest 3.1.6 PASSED: Gradient flow")
        print(f"  RGB grad norm (dual): {rgb_features.grad[0].norm().item():.4f}")
        print(f"  Sonar grad norm (dual): {sonar_features.grad[0].norm().item():.4f}")
        print(f"  RGB grad norm (single-RGB): {rgb_features.grad[1].norm().item():.4f}")
        print(f"  Sonar grad norm (single-Sonar): {sonar_features.grad[2].norm().item():.4f}")

    def test_consistency_across_batch_sizes(self):
        """
        Test 3.1.7: Verify output consistency across different batch sizes.
        """
        # Create identical samples
        rgb_features_single = torch.randn(1, self.rgb_feat_dim, device=self.device)
        sonar_features_single = torch.randn(1, self.sonar_feat_dim, device=self.device)
        sonar_radial_single = torch.rand(1, self.num_radial_bins, device=self.device)
        sonar_radial_single = sonar_radial_single / sonar_radial_single.sum(dim=-1, keepdim=True)

        is_rgb_valid = torch.ones(1, dtype=torch.bool, device=self.device)
        is_sonar_valid = torch.ones(1, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # Process as batch of 1
            outputs_single = self.head(
                rgb_features=rgb_features_single,
                sonar_features=sonar_features_single,
                is_rgb_valid=is_rgb_valid,
                is_sonar_valid=is_sonar_valid,
                sonar_radial_activation=sonar_radial_single
            )
            center_3d_single = outputs_single['center_3d']

            # Process same sample in batch of 4 (replicated)
            rgb_features_batch = rgb_features_single.repeat(4, 1)
            sonar_features_batch = sonar_features_single.repeat(4, 1)
            sonar_radial_batch = sonar_radial_single.repeat(4, 1)
            is_rgb_valid_batch = torch.ones(4, dtype=torch.bool, device=self.device)
            is_sonar_valid_batch = torch.ones(4, dtype=torch.bool, device=self.device)

            outputs_batch = self.head(
                rgb_features=rgb_features_batch,
                sonar_features=sonar_features_batch,
                is_rgb_valid=is_rgb_valid_batch,
                is_sonar_valid=is_sonar_valid_batch,
                sonar_radial_activation=sonar_radial_batch
            )
            center_3d_batch = outputs_batch['center_3d']

            # All batch samples should be identical to single sample
            for i in range(4):
                self.assertTrue(
                    torch.allclose(center_3d_batch[i], center_3d_single[0], atol=1e-5),
                    f"Batch sample {i} differs from single sample"
                )

        print(f"\nTest 3.1.7 PASSED: Consistency across batch sizes")
        print(f"  Single sample: {center_3d_single[0].tolist()}")
        print(f"  Batch sample 0: {center_3d_batch[0].tolist()}")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
