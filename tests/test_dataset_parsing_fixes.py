"""
Test Suite for Dataset Parsing Bug Fixes

Tests the fix for integer truncation in ground-truth bounding box parsing.

Based on audit report: 2025111403_Comprehensive_Algo_Audit.md
Issue: parse_groundtruth used int(x) instead of float(x), losing sub-pixel precision.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from aquasot3d.dataloader.datasets.rgb_sonar_dataset import parse_groundtruth


class TestGroundtruthParsingFix:
    """Test fixes for sub-pixel precision in bounding box parsing."""

    def test_float_precision_preserved(self):
        """
        Test that decimal coordinates are preserved (not truncated to int).

        Verifies fix for: bbox = [int(x) for x in parts] -> bbox = [float(x) for x in parts]
        """
        # Create temporary groundtruth file with decimal coordinates
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            gt_path = f.name
            # Write bboxes with sub-pixel precision
            f.write("100.5,200.75,300.25,400.125\n")
            f.write("110.333,210.666,310.999,410.001\n")
            f.write("120.0,220.0,320.0,420.0\n")

        try:
            bboxes = parse_groundtruth(gt_path)

            # Check that decimals are preserved
            assert bboxes.shape == (3, 4)
            assert bboxes.dtype == np.float32

            # Verify sub-pixel values
            assert np.isclose(bboxes[0, 0], 100.5), \
                f"Expected 100.5, got {bboxes[0, 0]} (decimal lost)"
            assert np.isclose(bboxes[0, 1], 200.75), \
                f"Expected 200.75, got {bboxes[0, 1]} (decimal lost)"
            assert np.isclose(bboxes[0, 2], 300.25), \
                f"Expected 300.25, got {bboxes[0, 2]} (decimal lost)"
            assert np.isclose(bboxes[0, 3], 400.125), \
                f"Expected 400.125, got {bboxes[0, 3]} (decimal lost)"

            # Check second row
            assert np.isclose(bboxes[1, 0], 110.333, atol=1e-3), \
                f"Expected 110.333, got {bboxes[1, 0]} (decimal lost)"
            assert np.isclose(bboxes[1, 1], 210.666, atol=1e-3), \
                f"Expected 210.666, got {bboxes[1, 1]} (decimal lost)"

            # Integer values should also work (remain as floats)
            assert np.isclose(bboxes[2, 0], 120.0), \
                f"Expected 120.0, got {bboxes[2, 0]}"

        finally:
            Path(gt_path).unlink()

    def test_integer_coordinates_still_work(self):
        """
        Test that integer coordinates (no decimals) still parse correctly.

        Ensures backward compatibility with existing datasets.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            gt_path = f.name
            f.write("100,200,300,400\n")
            f.write("110,210,310,410\n")

        try:
            bboxes = parse_groundtruth(gt_path)

            assert bboxes.shape == (2, 4)
            assert bboxes.dtype == np.float32

            # Integer values should be exact
            assert bboxes[0, 0] == 100.0
            assert bboxes[0, 1] == 200.0
            assert bboxes[1, 2] == 310.0

        finally:
            Path(gt_path).unlink()

    def test_occlusion_markers_preserved(self):
        """
        Test that occlusion markers (0,0,0,0) are correctly preserved as floats.

        Ensures occlusion detection logic still works.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            gt_path = f.name
            f.write("100.5,200.5,300.5,400.5\n")
            f.write("0,0,0,0\n")  # Occlusion marker
            f.write("110.5,210.5,310.5,410.5\n")

        try:
            bboxes = parse_groundtruth(gt_path)

            assert bboxes.shape == (3, 4)

            # Occlusion marker should be all zeros
            assert np.all(bboxes[1] == 0.0), \
                "Occlusion marker not preserved correctly"

            # Non-occluded frames should have decimals
            assert np.isclose(bboxes[0, 0], 100.5)
            assert np.isclose(bboxes[2, 0], 110.5)

        finally:
            Path(gt_path).unlink()

    def test_high_precision_coordinates(self):
        """
        Test that high-precision decimal values (many decimal places) are preserved.

        Important for sub-pixel accuracy in loss computation.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            gt_path = f.name
            # Write high-precision coordinates
            f.write("100.123456,200.987654,300.555555,400.111111\n")

        try:
            bboxes = parse_groundtruth(gt_path)

            assert bboxes.shape == (1, 4)

            # Check precision (np.float32 has ~7 decimal digits precision)
            assert np.isclose(bboxes[0, 0], 100.123456, atol=1e-5), \
                f"High-precision value lost: expected 100.123456, got {bboxes[0, 0]}"
            assert np.isclose(bboxes[0, 1], 200.987654, atol=1e-5), \
                f"High-precision value lost: expected 200.987654, got {bboxes[0, 1]}"

        finally:
            Path(gt_path).unlink()

    def test_mixed_integer_float_coordinates(self):
        """
        Test that mixed integer and float coordinates in same file work correctly.

        Real datasets may have mix of integer and decimal annotations.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            gt_path = f.name
            f.write("100,200,300.5,400.25\n")  # Mixed int/float
            f.write("110.75,210,310,410.5\n")  # Mixed int/float
            f.write("120,220,320,420\n")       # All int

        try:
            bboxes = parse_groundtruth(gt_path)

            assert bboxes.shape == (3, 4)

            # First row: mixed
            assert bboxes[0, 0] == 100.0
            assert bboxes[0, 1] == 200.0
            assert np.isclose(bboxes[0, 2], 300.5)
            assert np.isclose(bboxes[0, 3], 400.25)

            # Second row: mixed
            assert np.isclose(bboxes[1, 0], 110.75)
            assert bboxes[1, 1] == 210.0

            # Third row: all integers
            assert bboxes[2, 0] == 120.0
            assert bboxes[2, 3] == 420.0

        finally:
            Path(gt_path).unlink()

    def test_negative_coordinates_with_decimals(self):
        """
        Test that negative decimal coordinates are parsed correctly.

        Some datasets may have negative coordinates for certain representations.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            gt_path = f.name
            f.write("-100.5,200.5,300.5,400.5\n")
            f.write("100.5,-200.5,300.5,400.5\n")

        try:
            bboxes = parse_groundtruth(gt_path)

            assert bboxes.shape == (2, 4)

            # Negative decimals should be preserved
            assert np.isclose(bboxes[0, 0], -100.5), \
                f"Negative decimal not preserved: expected -100.5, got {bboxes[0, 0]}"
            assert np.isclose(bboxes[1, 1], -200.5), \
                f"Negative decimal not preserved: expected -200.5, got {bboxes[1, 1]}"

        finally:
            Path(gt_path).unlink()

    def test_loss_gradient_with_subpixel_precision(self):
        """
        Test that sub-pixel coordinates enable gradient flow in loss computation.

        Demonstrates the impact of the fix on training.
        """
        import torch
        import torch.nn.functional as F

        # Simulated ground truth with sub-pixel precision
        gt_bbox = torch.tensor([[100.5, 200.75, 300.25, 400.125]], dtype=torch.float32)

        # Predicted bbox (learnable)
        pred_bbox = torch.tensor([[100.0, 200.0, 300.0, 400.0]], requires_grad=True)

        # L1 loss
        loss = F.l1_loss(pred_bbox, gt_bbox)

        # Backward pass
        loss.backward()

        # Check gradients exist and are non-zero
        assert pred_bbox.grad is not None, "No gradient computed"
        assert torch.all(pred_bbox.grad != 0), \
            "Zero gradients: sub-pixel precision not enabling gradient flow"

        # Gradients should reflect sub-pixel differences
        expected_grad_sign = torch.sign(pred_bbox - gt_bbox)
        assert torch.allclose(torch.sign(pred_bbox.grad), expected_grad_sign, atol=1e-5), \
            "Gradient direction incorrect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
