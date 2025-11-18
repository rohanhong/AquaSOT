"""
Unit Tests for Bbox-Aware Augmentation Pipeline

Tests the integrated augmentation pipeline with all bbox-aware transforms
from Audit 2025111801 Phase 5.

Author: AquaSOT-3D Team
Date: 2025-11-18
"""

import pytest
import numpy as np
import torch

from aquasot3d.dataloader.datasets.augmentation import (
    TrackingAugmentationBboxAware,
    NoAugmentationBboxAware,
    build_augmentation_bbox_aware,
)


class TestTrackingAugmentationBboxAware:
    """Test bbox-aware tracking augmentation pipeline."""

    def test_basic_augmentation_without_logpolar(self):
        """Test basic augmentation without log-polar transform."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            rgb_config={'enable_color_jitter': True},
            sonar_config={'enable_speckle': True},
            use_logpolar=False
        )

        data = {
            'rgb_img': np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 200, 50, 80]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
        }

        result = aug(data)

        # Check outputs
        assert isinstance(result['rgb_img'], torch.Tensor)
        assert isinstance(result['sonar_img'], torch.Tensor)
        assert isinstance(result['rgb_bbox'], torch.Tensor)
        assert isinstance(result['sonar_bbox'], torch.Tensor)

        # Check shapes
        assert result['rgb_img'].shape == (3, 376, 682)
        assert result['sonar_img'].shape == (1, 376, 682)

    def test_augmentation_with_logpolar(self):
        """
        Test augmentation with log-polar transform enabled.

        CRITICAL: This validates the fix for Audit Finding 1.
        """
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            sonar_config={'enable_speckle': True},
            use_logpolar=True,
            logpolar_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256,
                'fov_angle': 130.0
            }
        )

        data = {
            'rgb_img': np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 200, 50, 80]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
        }

        result = aug(data)

        # RGB unchanged (376×682)
        assert result['rgb_img'].shape == (3, 376, 682)

        # Sonar transformed to log-polar (256×256)
        assert result['sonar_img'].shape == (1, 256, 256)

        # Sonar bbox transformed to log-polar coordinates
        assert 0 <= result['sonar_bbox'][0].item() < 256
        assert 0 <= result['sonar_bbox'][1].item() < 256

    def test_radial_jitter_bbox_sync(self):
        """
        Test that RadialScalingJitter maintains bbox-image sync.

        CRITICAL: This validates the fix for Audit Finding 2.
        """
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            sonar_config={
                'enable_speckle': False,
                'enable_radial_jitter': True,
                'radial_scale_range': (1.05, 1.05),  # Fixed scale for testing
                'radial_jitter_prob': 1.0
            },
            use_logpolar=False
        )

        # Create image with target at bbox location
        sonar_img = np.zeros((376, 682, 1), dtype=np.uint8)
        bbox = np.array([250, 120, 80, 60])
        x, y, w, h = bbox.astype(int)
        sonar_img[y:y+h, x:x+w, :] = 255

        data = {
            'rgb_img': None,
            'sonar_img': sonar_img,
            'rgb_bbox': np.array([0, 0, 1, 1]),
            'sonar_bbox': bbox,
        }

        result = aug(data)

        # Extract transformed bbox region
        sonar_result = (result['sonar_img'].cpu().numpy() * 255).astype(np.uint8)
        sonar_result = sonar_result.transpose(1, 2, 0)  # (H, W, C)

        bbox_result = result['sonar_bbox'].cpu().numpy()
        x_new, y_new, w_new, h_new = bbox_result.astype(int)

        # Clamp to bounds
        x_new = max(0, min(x_new, 681))
        y_new = max(0, min(y_new, 375))
        w_new = max(1, min(w_new, 682 - x_new))
        h_new = max(1, min(h_new, 376 - y_new))

        bbox_region = sonar_result[y_new:y_new+h_new, x_new:x_new+w_new, :]

        # Bbox should contain the scaled target
        if bbox_region.size > 0:
            mean_intensity = np.mean(bbox_region)
            assert mean_intensity > 100, (
                f"Bbox desynchronized: intensity {mean_intensity} < 100"
            )

    def test_metadata_passthrough(self):
        """Test that metadata is preserved through pipeline."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            use_logpolar=False
        )

        data = {
            'rgb_img': np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 200, 50, 80]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
            'frame_id': 42,
            'sequence_name': 'uuv1',
        }

        result = aug(data)

        # Metadata should be preserved
        assert result['frame_id'] == 42
        assert result['sequence_name'] == 'uuv1'

    def test_none_rgb_handling(self):
        """Test handling None RGB images (sonar-only training)."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            sonar_config={'enable_speckle': True},
            use_logpolar=False
        )

        data = {
            'rgb_img': None,
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([0, 0, 1, 1]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
        }

        result = aug(data)

        assert result['rgb_img'] is None
        assert result['sonar_img'] is not None


class TestNoAugmentationBboxAware:
    """Test bbox-aware no-augmentation pipeline."""

    def test_evaluation_mode_without_logpolar(self):
        """Test evaluation mode without log-polar."""
        aug = NoAugmentationBboxAware(use_logpolar=False)

        data = {
            'rgb_img': np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 200, 50, 80]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
        }

        result = aug(data)

        # Should only normalize, not augment
        assert result['rgb_img'].shape == (3, 376, 682)
        assert result['sonar_img'].shape == (1, 376, 682)

    def test_evaluation_mode_with_logpolar(self):
        """Test evaluation mode with log-polar enabled."""
        aug = NoAugmentationBboxAware(
            use_logpolar=True,
            logpolar_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256
            }
        )

        data = {
            'rgb_img': np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 200, 50, 80]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
        }

        result = aug(data)

        # RGB unchanged
        assert result['rgb_img'].shape == (3, 376, 682)

        # Sonar transformed to log-polar
        assert result['sonar_img'].shape == (1, 256, 256)

        # Sonar bbox transformed
        assert 0 <= result['sonar_bbox'][0].item() < 256
        assert 0 <= result['sonar_bbox'][1].item() < 256


class TestBuildAugmentationBboxAware:
    """Test augmentation pipeline builder."""

    def test_build_training_pipeline(self):
        """Test building training pipeline from config."""
        cfg = {
            'type': 'TrackingAugmentationBboxAware',
            'rgb_config': {'enable_color_jitter': True},
            'sonar_config': {'enable_speckle': True},
            'logpolar_config': {
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256
            }
        }

        aug = build_augmentation_bbox_aware(cfg, is_train=True, use_logpolar=True)

        assert isinstance(aug, TrackingAugmentationBboxAware)
        assert aug.use_logpolar is True

    def test_build_evaluation_pipeline(self):
        """Test building evaluation pipeline from config."""
        cfg = {
            'type': 'NoAugmentationBboxAware',
            'logpolar_config': {
                'sonar_height': 376,
                'sonar_width': 682
            }
        }

        aug = build_augmentation_bbox_aware(cfg, is_train=False, use_logpolar=True)

        assert isinstance(aug, NoAugmentationBboxAware)

    def test_unknown_type_raises_error(self):
        """Test that unknown augmentation type raises error."""
        cfg = {'type': 'UnknownAugmentation'}

        with pytest.raises(ValueError, match="Unknown augmentation type"):
            build_augmentation_bbox_aware(cfg, is_train=True)


class TestIntegrationScenarios:
    """Integration tests for complete augmentation pipeline."""

    def test_full_sonar_pretraining_pipeline(self):
        """
        Test complete sonar pretraining pipeline with log-polar.

        This is the critical test for sonar convergence (val_loss 2.0 → < 0.5).
        """
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            sonar_config={
                'enable_speckle': True,
                'speckle_level': 0.1,
                'enable_radial_jitter': False,  # Disabled until tested
            },
            use_logpolar=True,
            logpolar_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256,
                'fov_angle': 130.0,
                'alpha': 1.0,
                'beta': 1.0
            }
        )

        # Simulate dataloader output
        data = {
            'rgb_img': None,  # Sonar-only pretraining
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([0, 0, 1, 1]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
        }

        result = aug(data)

        # Verify outputs ready for training
        assert result['sonar_img'].shape == (1, 256, 256)  # Log-polar space
        assert result['sonar_bbox'].shape == (4,)  # Transformed bbox
        assert 0 <= result['sonar_bbox'][0].item() < 256  # Valid coordinates
        assert 0 <= result['sonar_bbox'][1].item() < 256

    def test_multi_modal_fusion_pipeline(self):
        """Test multi-modal fusion training pipeline."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            rgb_config={'enable_color_jitter': True},
            sonar_config={'enable_speckle': True},
            use_logpolar=True,
            logpolar_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256
            }
        )

        data = {
            'rgb_img': np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 200, 50, 80]),
            'sonar_bbox': np.array([250, 120, 80, 60]),
        }

        result = aug(data)

        # Both modalities processed
        assert result['rgb_img'] is not None
        assert result['sonar_img'] is not None
        assert result['rgb_img'].shape == (3, 376, 682)
        assert result['sonar_img'].shape == (1, 256, 256)

    def test_bbox_image_sync_maintained_end_to_end(self):
        """
        End-to-end test: verify bbox-image sync is maintained through
        entire augmentation pipeline.

        This is the ultimate test for Audit 2025111801 fixes.
        """
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            sonar_config={
                'enable_speckle': True,
                'enable_radial_jitter': False,
            },
            use_logpolar=True,
            logpolar_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256,
                'fov_angle': 130.0
            }
        )

        # Create synthetic image with target at bbox location
        sonar_img = np.zeros((376, 682, 1), dtype=np.uint8)
        bbox = np.array([250, 120, 80, 60])
        x, y, w, h = bbox.astype(int)
        sonar_img[y:y+h, x:x+w, :] = 255

        data = {
            'rgb_img': None,
            'sonar_img': sonar_img,
            'rgb_bbox': np.array([0, 0, 1, 1]),
            'sonar_bbox': bbox,
        }

        result = aug(data)

        # Extract transformed bbox region in log-polar space
        sonar_result = (result['sonar_img'].cpu().numpy() * 255).astype(np.uint8)
        sonar_result = sonar_result.transpose(1, 2, 0)  # (H, W, C)

        bbox_result = result['sonar_bbox'].cpu().numpy()
        x_lp, y_lp, w_lp, h_lp = bbox_result.astype(int)

        # Clamp to log-polar output bounds
        x_lp = max(0, min(x_lp, 255))
        y_lp = max(0, min(y_lp, 255))
        w_lp = max(1, min(w_lp, 256 - x_lp))
        h_lp = max(1, min(h_lp, 256 - y_lp))

        bbox_region = sonar_result[y_lp:y_lp+h_lp, x_lp:x_lp+w_lp, :]

        # Bbox should contain the transformed target
        if bbox_region.size > 0:
            mean_intensity = np.mean(bbox_region)
            assert mean_intensity > 50, (
                f"End-to-end bbox-image desynchronization: "
                f"intensity {mean_intensity} < 50. "
                f"Target should be visible in transformed bbox region."
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
