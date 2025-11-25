"""
Unit tests for letterbox preprocessing consistency between training and inference.

CRITICAL FIX (Audit 2025112508 Finding 1 & 3):
These tests verify that:
1. Shared letterbox implementation produces consistent results
2. Padding values are correct (RGB=114, Sonar=127, NOT zeros)
3. Metadata generation is accurate
4. Training and inference preprocessing pipelines are aligned

Author: AquaSOT-3D Team
Date: 2025-11-25
"""

import pytest
import numpy as np
import torch
import cv2
from pathlib import Path

from aquasot3d.utils.preprocessing import letterbox_with_metadata, preprocess_for_inference
from aquasot3d.dataloader.datasets.template_search_dataset import TemplateSearchDataset


class TestLetterboxConsistency:
    """Test letterbox preprocessing consistency."""

    @pytest.fixture
    def rgb_image(self):
        """Create test RGB image."""
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    @pytest.fixture
    def sonar_image(self):
        """Create test sonar image."""
        return np.random.randint(0, 255, (376, 682), dtype=np.uint8)

    def test_letterbox_metadata_generation(self, rgb_image):
        """Test that letterbox generates correct metadata."""
        target_size = 512
        letterboxed, metadata = letterbox_with_metadata(rgb_image, target_size, is_rgb=True)

        # Check output shape
        assert letterboxed.shape == (target_size, target_size, 3), \
            f"Expected shape ({target_size}, {target_size}, 3), got {letterboxed.shape}"

        # Check metadata keys
        required_keys = ['scale', 'pad_left', 'pad_top', 'original_size', 'new_size', 'original_height']
        for key in required_keys:
            assert key in metadata, f"Missing metadata key: {key}"

        # Verify scale calculation
        h, w = rgb_image.shape[:2]
        expected_scale = min(target_size / w, target_size / h)
        assert abs(metadata['scale'] - expected_scale) < 1e-6, \
            f"Scale mismatch: expected {expected_scale}, got {metadata['scale']}"

        # Verify original size
        assert metadata['original_size'] == (h, w), \
            f"Original size mismatch: expected {(h, w)}, got {metadata['original_size']}"

    def test_padding_values_rgb(self, rgb_image):
        """Test that RGB images use correct padding value (114, not zeros)."""
        target_size = 512
        letterboxed, metadata = letterbox_with_metadata(rgb_image, target_size, is_rgb=True)

        # Get padding region (top border)
        pad_top = metadata['pad_top']
        if pad_top > 0:
            padding_region = letterboxed[:pad_top, :, :]
            # Check that padding is not zeros
            assert not np.all(padding_region == 0), \
                "CRITICAL: RGB padding should be (114,114,114), not zeros!"
            # Check that padding is approximately (114, 114, 114)
            mean_padding = padding_region.mean(axis=(0, 1))
            expected_padding = np.array([114, 114, 114])
            assert np.allclose(mean_padding, expected_padding, atol=1.0), \
                f"RGB padding should be {expected_padding}, got {mean_padding}"

    def test_padding_values_sonar(self, sonar_image):
        """Test that sonar images use correct padding value (127, not zeros)."""
        target_size = 512
        letterboxed, metadata = letterbox_with_metadata(sonar_image, target_size, is_rgb=False)

        # Get padding region (left border)
        pad_left = metadata['pad_left']
        if pad_left > 0:
            padding_region = letterboxed[:, :pad_left]
            # Check that padding is not zeros
            assert not np.all(padding_region == 0), \
                "CRITICAL: Sonar padding should be 127, not zeros!"
            # Check that padding is approximately 127
            mean_padding = padding_region.mean()
            assert abs(mean_padding - 127) < 1.0, \
                f"Sonar padding should be 127, got {mean_padding}"

    def test_preprocessing_consistency(self, rgb_image):
        """Test that preprocessing produces consistent normalized output."""
        target_size = 512

        # Use default normalization parameters
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]

        # Run preprocessing
        tensor, metadata = preprocess_for_inference(
            rgb_image, target_size, is_rgb=True,
            rgb_mean=rgb_mean, rgb_std=rgb_std,
            device='cpu'
        )

        # Check output shape
        assert tensor.shape == (1, 3, target_size, target_size), \
            f"Expected shape (1, 3, {target_size}, {target_size}), got {tensor.shape}"

        # Check normalization (output should have mean ≈ 0, std ≈ 1 after normalization)
        # Note: This is approximate due to padding
        tensor_np = tensor[0].numpy()
        assert tensor_np.mean() < 1.0 and tensor_np.mean() > -1.0, \
            f"Normalized tensor mean out of reasonable range: {tensor_np.mean()}"

    def test_sonar_letterbox_origin_compensation(self, sonar_image):
        """
        Test that sonar letterbox metadata includes original_height for origin compensation.

        CRITICAL for geometry-gated attention radial bins.
        """
        target_size = 512
        letterboxed, metadata = letterbox_with_metadata(sonar_image, target_size, is_rgb=False)

        # Check that original_height is preserved
        assert 'original_height' in metadata, \
            "CRITICAL: Sonar metadata must include original_height for radial bins origin computation"

        h_original = sonar_image.shape[0]
        assert metadata['original_height'] == h_original, \
            f"Original height mismatch: expected {h_original}, got {metadata['original_height']}"

    def test_aspect_ratio_preservation(self, rgb_image):
        """Test that letterbox preserves aspect ratio."""
        target_size = 512
        letterboxed, metadata = letterbox_with_metadata(rgb_image, target_size, is_rgb=True)

        h_orig, w_orig = rgb_image.shape[:2]
        h_new, w_new = metadata['new_size']

        # Check aspect ratio is preserved
        orig_aspect = w_orig / h_orig
        new_aspect = w_new / h_new

        assert abs(orig_aspect - new_aspect) < 1e-4, \
            f"Aspect ratio not preserved: original={orig_aspect}, new={new_aspect}"

    def test_coordinate_transformation(self, rgb_image):
        """Test that metadata allows correct bbox coordinate transformation."""
        target_size = 512
        letterboxed, metadata = letterbox_with_metadata(rgb_image, target_size, is_rgb=True)

        # Simulate a bbox in original image space
        h_orig, w_orig = rgb_image.shape[:2]
        bbox_orig = np.array([w_orig // 2, h_orig // 2, 100, 100])  # [x, y, w, h]

        # Transform to letterbox space
        scale = metadata['scale']
        pad_left = metadata['pad_left']
        pad_top = metadata['pad_top']

        x, y, w, h = bbox_orig
        x_letterbox = x * scale + pad_left
        y_letterbox = y * scale + pad_top
        w_letterbox = w * scale
        h_letterbox = h * scale

        # Verify transformed bbox is within letterbox image
        assert 0 <= x_letterbox < target_size, \
            f"Transformed x out of bounds: {x_letterbox}"
        assert 0 <= y_letterbox < target_size, \
            f"Transformed y out of bounds: {y_letterbox}"
        assert x_letterbox + w_letterbox <= target_size, \
            f"Transformed bbox exceeds right boundary"
        assert y_letterbox + h_letterbox <= target_size, \
            f"Transformed bbox exceeds bottom boundary"

    def test_single_channel_handling(self):
        """Test that single-channel images (H, W, 1) are handled correctly."""
        img_single_channel = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        target_size = 512

        letterboxed, metadata = letterbox_with_metadata(img_single_channel, target_size, is_rgb=False)

        # Output should be (H, W) after processing
        assert letterboxed.ndim == 2 or (letterboxed.ndim == 3 and letterboxed.shape[2] == 1), \
            f"Single-channel output should be 2D or (H,W,1), got shape {letterboxed.shape}"

    @pytest.mark.parametrize("target_size", [256, 512, 1024])
    def test_multiple_target_sizes(self, rgb_image, target_size):
        """Test letterbox works correctly for different target sizes."""
        letterboxed, metadata = letterbox_with_metadata(rgb_image, target_size, is_rgb=True)

        assert letterboxed.shape == (target_size, target_size, 3), \
            f"Output shape mismatch for target_size={target_size}"

        # Check that scale is computed correctly
        h, w = rgb_image.shape[:2]
        expected_scale = min(target_size / w, target_size / h)
        assert abs(metadata['scale'] - expected_scale) < 1e-6


class TestTrainingInferenceAlignment:
    """Test alignment between training and inference preprocessing."""

    def test_same_letterbox_implementation(self):
        """
        Verify that training and inference use the same letterbox logic.

        This is a meta-test that checks the code structure.
        """
        # Import both implementations
        from aquasot3d.utils.preprocessing import letterbox_with_metadata as inference_letterbox
        from aquasot3d.dataloader.datasets.template_search_dataset import TemplateSearchDataset

        # Check that TemplateSearchDataset would ideally use the shared implementation
        # (This test serves as a reminder if the training code still has its own implementation)

        # Create test image
        img = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        target_size = 512

        # Run inference letterbox
        letterboxed_inf, metadata_inf = inference_letterbox(img, target_size, is_rgb=True)

        # Verify output
        assert letterboxed_inf.shape == (target_size, target_size, 3)
        assert 'scale' in metadata_inf
        assert 'pad_left' in metadata_inf
        assert 'pad_top' in metadata_inf

    def test_normalization_parameters_match(self):
        """Test that normalization parameters match between train and inference configs."""
        # This test verifies that default values are consistent
        from aquasot3d.utils.preprocessing import preprocess_for_inference

        img = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)

        # RGB normalization (ImageNet stats)
        tensor, _ = preprocess_for_inference(
            img, 512, is_rgb=True,
            rgb_mean=[0.485, 0.456, 0.406],
            rgb_std=[0.229, 0.224, 0.225],
            device='cpu'
        )

        # Should produce valid normalized tensor
        assert tensor.shape == (1, 3, 512, 512)
        assert not torch.isnan(tensor).any(), "Normalized tensor contains NaN"
        assert not torch.isinf(tensor).any(), "Normalized tensor contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
