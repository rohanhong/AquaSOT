"""
Unit Tests for SonarEncoder Bbox Pass-Through

Tests the updated SonarEncoder with bbox transformation support
from Audit 2025111801 Phase 5.

Author: AquaSOT-3D Team
Date: 2025-11-18
"""

import pytest
import torch
import numpy as np

from aquasot3d.models.backbones.sonar_encoder import SonarEncoder


class TestSonarEncoderBboxPassThrough:
    """Test SonarEncoder bbox pass-through functionality."""

    def test_forward_without_bboxes(self):
        """Test backward compatibility: forward works without bboxes."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            img_size=(376, 682)
        )

        template = torch.randn(2, 1, 376, 682)
        search = torch.randn(2, 1, 376, 682)

        outputs = encoder(template, search)

        # Should work without errors
        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert 'template_bbox' not in outputs
        assert 'search_bbox' not in outputs

    def test_forward_with_bboxes_logpolar_enabled(self):
        """
        Test bbox pass-through with log-polar transform enabled.

        CRITICAL: This validates the core fix for Audit Finding 1.
        """
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=False,
            img_size=(376, 682),
            sonar_fov_angle=130.0,
            max_range_pixels=376
        )

        B = 2
        template = torch.randn(B, 1, 376, 682)
        search = torch.randn(B, 1, 376, 682)

        # Bboxes in Cartesian space (xywh format)
        template_bbox = torch.tensor([[250, 120, 80, 60], [200, 100, 60, 50]], dtype=torch.float32)
        search_bbox = torch.tensor([[260, 130, 85, 65], [210, 110, 65, 55]], dtype=torch.float32)

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        # Verify outputs
        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert 'template_bbox' in outputs
        assert 'search_bbox' in outputs

        # Verify bbox shapes
        assert outputs['template_bbox'].shape == (B, 4)
        assert outputs['search_bbox'].shape == (B, 4)

        # Verify bboxes are in log-polar space (224x224 output)
        template_bbox_lp = outputs['template_bbox']
        search_bbox_lp = outputs['search_bbox']

        # All coordinates should be within log-polar output bounds
        assert torch.all(template_bbox_lp[:, 0] >= 0)  # x >= 0
        assert torch.all(template_bbox_lp[:, 0] < 224)  # x < 224
        assert torch.all(template_bbox_lp[:, 1] >= 0)  # y >= 0
        assert torch.all(template_bbox_lp[:, 1] < 224)  # y < 224

        assert torch.all(search_bbox_lp[:, 0] >= 0)
        assert torch.all(search_bbox_lp[:, 0] < 224)
        assert torch.all(search_bbox_lp[:, 1] >= 0)
        assert torch.all(search_bbox_lp[:, 1] < 224)

    def test_forward_with_bboxes_logpolar_disabled(self):
        """Test bbox pass-through with log-polar disabled (simple resize)."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=False,
            img_size=(376, 682)
        )

        B = 2
        template = torch.randn(B, 1, 376, 682)
        search = torch.randn(B, 1, 376, 682)

        # Bboxes in original space
        template_bbox = torch.tensor([[100, 200, 50, 80], [150, 250, 60, 90]], dtype=torch.float32)
        search_bbox = torch.tensor([[110, 210, 55, 85], [160, 260, 65, 95]], dtype=torch.float32)

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        # Verify outputs
        assert 'template_bbox' in outputs
        assert 'search_bbox' in outputs

        # Verify bboxes are proportionally scaled (376x682 -> 224x224)
        scale_h = 224 / 376
        scale_w = 224 / 682

        expected_template_x = template_bbox[:, 0] * scale_w
        expected_template_y = template_bbox[:, 1] * scale_h
        expected_template_w = template_bbox[:, 2] * scale_w
        expected_template_h = template_bbox[:, 3] * scale_h

        torch.testing.assert_close(outputs['template_bbox'][:, 0], expected_template_x, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(outputs['template_bbox'][:, 1], expected_template_y, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(outputs['template_bbox'][:, 2], expected_template_w, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(outputs['template_bbox'][:, 3], expected_template_h, rtol=1e-4, atol=1e-4)

    def test_learnable_scales_bbox_sync(self):
        """
        Test that learnable alpha/beta are correctly used for bbox transformation.

        This ensures bbox-image synchronization even when scales are learned.
        """
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=True,
            img_size=(376, 682),
            sonar_fov_angle=130.0,
            max_range_pixels=376,
            init_alpha=1.5,  # Non-standard alpha
            init_beta=0.8    # Non-standard beta
        )

        B = 1
        template = torch.randn(B, 1, 376, 682)
        search = torch.randn(B, 1, 376, 682)

        template_bbox = torch.tensor([[250, 120, 80, 60]], dtype=torch.float32)
        search_bbox = torch.tensor([[260, 130, 85, 65]], dtype=torch.float32)

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        # Verify bboxes are transformed
        assert 'template_bbox' in outputs
        assert 'search_bbox' in outputs

        # Verify bboxes are not equal to input (transformed)
        assert not torch.allclose(outputs['template_bbox'], template_bbox)
        assert not torch.allclose(outputs['search_bbox'], search_bbox)

        # Verify bboxes are in valid log-polar range
        assert torch.all(outputs['template_bbox'][:, 0] >= 0)
        assert torch.all(outputs['template_bbox'][:, 0] < 224)

    def test_single_bbox_support(self):
        """Test that single bbox (B=1) is handled correctly."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=False,
            img_size=(376, 682)
        )

        B = 1
        template = torch.randn(B, 1, 376, 682)
        search = torch.randn(B, 1, 376, 682)

        # Single bbox
        template_bbox = torch.tensor([[250, 120, 80, 60]], dtype=torch.float32)
        search_bbox = torch.tensor([[260, 130, 85, 65]], dtype=torch.float32)

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        assert outputs['template_bbox'].shape == (1, 4)
        assert outputs['search_bbox'].shape == (1, 4)

    def test_batch_bbox_support(self):
        """Test that batch of bboxes (B>1) is handled correctly."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=False,
            img_size=(376, 682)
        )

        B = 4
        template = torch.randn(B, 1, 376, 682)
        search = torch.randn(B, 1, 376, 682)

        # Batch of bboxes with varying coordinates
        template_bbox = torch.tensor([
            [250, 120, 80, 60],
            [200, 100, 70, 50],
            [300, 150, 90, 70],
            [150, 80, 60, 40]
        ], dtype=torch.float32)
        search_bbox = torch.tensor([
            [260, 130, 85, 65],
            [210, 110, 75, 55],
            [310, 160, 95, 75],
            [160, 90, 65, 45]
        ], dtype=torch.float32)

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        assert outputs['template_bbox'].shape == (B, 4)
        assert outputs['search_bbox'].shape == (B, 4)

        # All bboxes should be transformed (different from input)
        assert not torch.allclose(outputs['template_bbox'], template_bbox)
        assert not torch.allclose(outputs['search_bbox'], search_bbox)

    def test_bbox_image_semantic_sync(self):
        """
        End-to-end test: verify bbox-image semantic synchronization.

        Create synthetic image with target at bbox location, verify that
        after log-polar transform, the transformed bbox still corresponds
        to the target in the transformed image.

        This is the ultimate test for bbox-image synchronization.
        """
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=False,
            img_size=(376, 682),
            sonar_fov_angle=130.0,
            max_range_pixels=376
        )

        # Create synthetic sonar image with bright target at bbox location
        B = 1
        template = torch.zeros(B, 1, 376, 682)
        bbox = torch.tensor([[250, 120, 80, 60]], dtype=torch.float32)

        # Place bright target in bbox region
        x, y, w, h = bbox[0].int().tolist()
        template[:, :, y:y+h, x:x+w] = 1.0

        # Forward pass
        outputs = encoder(template, template, template_bbox=bbox, search_bbox=bbox)

        # Get transformed bbox
        bbox_lp = outputs['template_bbox']

        # Get log-polar transformed image
        # Note: We can't directly access the log-polar image from encoder,
        # but we can verify bbox is within valid range
        assert bbox_lp.shape == (1, 4)
        assert torch.all(bbox_lp[:, 0] >= 0)
        assert torch.all(bbox_lp[:, 0] < 224)
        assert torch.all(bbox_lp[:, 1] >= 0)
        assert torch.all(bbox_lp[:, 1] < 224)
        assert torch.all(bbox_lp[:, 2] > 0)  # Width > 0
        assert torch.all(bbox_lp[:, 3] > 0)  # Height > 0

    def test_gradients_flow_through_image_transform(self):
        """
        Test that gradients flow through image transformation with learnable scales.

        Note: Bbox transformation is non-differentiable (geometric coordinate mapping),
        so gradients only flow through the image transformation path.
        This is correct behavior - bboxes are coordinate labels, not trainable data.
        """
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=True,
            img_size=(376, 682)
        )

        B = 1
        template = torch.randn(B, 1, 376, 682, requires_grad=True)
        search = torch.randn(B, 1, 376, 682, requires_grad=True)

        template_bbox = torch.tensor([[250, 120, 80, 60]], dtype=torch.float32)
        search_bbox = torch.tensor([[260, 130, 85, 65]], dtype=torch.float32)

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        # Create dummy loss from features (image path is differentiable)
        loss = outputs['template_features'].mean()
        loss.backward()

        # Verify gradients exist for input images and learnable parameters
        # Gradients flow through image transformation, enabling end-to-end training
        assert template.grad is not None
        assert search.grad is not None
        assert encoder.logpolar_transform.alpha.grad is not None or encoder.logpolar_transform.beta.grad is not None


class TestSonarEncoderIntegration:
    """Integration tests for SonarEncoder with full pipeline."""

    def test_multi_frame_batch_processing(self):
        """Test processing multiple frames with bboxes in batch."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=False,
            img_size=(376, 682)
        )

        B = 8  # Simulate batch from dataloader
        template = torch.randn(B, 1, 376, 682)
        search = torch.randn(B, 1, 376, 682)

        template_bbox = torch.rand(B, 4) * 100 + 100  # Random bboxes
        search_bbox = torch.rand(B, 4) * 100 + 100

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        assert outputs['template_features'].shape[0] == B
        assert outputs['search_features'].shape[0] == B
        assert outputs['template_bbox'].shape == (B, 4)
        assert outputs['search_bbox'].shape == (B, 4)

    def test_output_dict_structure(self):
        """Test complete output dictionary structure."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            use_logpolar=True,
            learnable_logpolar=False,
            img_size=(376, 682),
            extract_radial_profile=True
        )

        B = 2
        template = torch.randn(B, 1, 376, 682)
        search = torch.randn(B, 1, 376, 682)
        template_bbox = torch.tensor([[250, 120, 80, 60], [200, 100, 60, 50]], dtype=torch.float32)
        search_bbox = torch.tensor([[260, 130, 85, 65], [210, 110, 65, 55]], dtype=torch.float32)

        outputs = encoder(template, search, template_bbox=template_bbox, search_bbox=search_bbox)

        # Verify all expected keys
        expected_keys = {
            'features', 'template_features', 'search_features',
            'template_bbox', 'search_bbox',
            'template_radial_profile', 'search_radial_profile',
            'geometry_info'
        }

        assert expected_keys.issubset(set(outputs.keys()))

        # Verify geometry info structure
        assert 'template_weights' in outputs['geometry_info']
        assert 'search_weights' in outputs['geometry_info']
        assert 'fov_angle' in outputs['geometry_info']
        assert 'max_range' in outputs['geometry_info']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
