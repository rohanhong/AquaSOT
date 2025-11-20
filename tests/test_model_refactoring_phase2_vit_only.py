"""
Unit Tests for Phase 2 Model Layer Refactoring (ViT Only)

Tests the ViT Backbone refactoring from template-search paradigm to
frame-level temporal tracking.

Author: AquaSOT-3D Team
Date: 2025-11-20
Audit: 2025112006
"""

import pytest
import torch

from aquasot3d.models.backbones.vit import ViTBackbone


class TestViTBackboneSingleFrame:
    """Test ViT Backbone single-frame forward pass."""

    def test_single_frame_forward(self):
        """Test single frame [B, C, H, W] forward pass."""
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,  # Don't download weights for test
            img_size=224,
            in_channels=3
        )
        backbone.eval()

        # Single frame input
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            outputs = backbone(x)

        # Check output structure
        assert 'features' in outputs
        assert 'is_sequence' in outputs
        assert 'batch_size' in outputs
        assert 'sequence_length' in outputs

        # Check shapes
        assert outputs['features'].shape[0] == 2  # Batch size
        assert outputs['features'].shape[2] == 768  # ViT-Base embedding dim

        # Check metadata
        assert outputs['is_sequence'] == False
        assert outputs['batch_size'] == 2
        assert outputs['sequence_length'] is None

    def test_sequence_forward(self):
        """Test frame sequence [B, T, C, H, W] forward pass."""
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            img_size=224,
            in_channels=3
        )
        backbone.eval()

        # Frame sequence input
        x = torch.randn(2, 5, 3, 224, 224)  # B=2, T=5

        with torch.no_grad():
            outputs = backbone(x)

        # Check output structure
        assert 'features' in outputs
        assert 'is_sequence' in outputs

        # Check shapes: should be reshaped to [B*T, N, C]
        assert outputs['features'].shape[0] == 10  # B * T = 2 * 5
        assert outputs['features'].shape[2] == 768

        # Check metadata
        assert outputs['is_sequence'] == True
        assert outputs['batch_size'] == 2
        assert outputs['sequence_length'] == 5

    def test_rgb_sonar_channel_adapter(self):
        """Test single-channel (Sonar) input with channel adapter."""
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            img_size=224,
            in_channels=1  # Single-channel Sonar
        )
        backbone.eval()

        # Single-channel input
        x = torch.randn(2, 1, 224, 224)

        with torch.no_grad():
            outputs = backbone(x)

        # Should work with channel adapter (1 -> 3)
        assert outputs['features'].shape[0] == 2
        assert outputs['features'].shape[2] == 768

    def test_variable_input_size(self):
        """Test variable input size (dynamic_img_size=True)."""
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            img_size=224,
            in_channels=3
        )
        backbone.eval()

        # Different input size
        x = torch.randn(2, 3, 384, 384)

        with torch.no_grad():
            outputs = backbone(x)

        # Should handle different input size
        assert outputs['features'].shape[0] == 2
        assert outputs['features'].shape[2] == 768
        # Number of patches will be different: (384/16)^2 = 576
        # Plus CLS token = 577
        assert outputs['features'].shape[1] == 577

    def test_sequence_with_different_sizes(self):
        """Test sequence processing with unified frame size."""
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            img_size=384,  # Larger image size
            in_channels=3
        )
        backbone.eval()

        # Frame sequence with larger size
        x = torch.randn(4, 5, 3, 384, 384)  # B=4, T=5

        with torch.no_grad():
            outputs = backbone(x)

        # Check shapes
        assert outputs['features'].shape[0] == 20  # B * T = 4 * 5
        assert outputs['features'].shape[1] == 577  # (384/16)^2 + 1 CLS
        assert outputs['features'].shape[2] == 768

        # Check metadata
        assert outputs['is_sequence'] == True
        assert outputs['batch_size'] == 4
        assert outputs['sequence_length'] == 5

    def test_batch_processing_consistency(self):
        """Test that sequence and batch processing are equivalent."""
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            img_size=224,
            in_channels=3
        )
        backbone.eval()

        # Create same frames in different formats
        B, T = 2, 3
        frames = torch.randn(B, T, 3, 224, 224)

        with torch.no_grad():
            # Process as sequence
            seq_out = backbone(frames)

            # Process individually and stack
            individual_outs = []
            for b in range(B):
                for t in range(T):
                    frame = frames[b:b+1, t]  # [1, 3, H, W]
                    out = backbone(frame)
                    individual_outs.append(out['features'])

            individual_stacked = torch.cat(individual_outs, dim=0)

        # Results should be close (minor numerical differences due to computation order)
        torch.testing.assert_close(
            seq_out['features'],
            individual_stacked,
            rtol=1e-4,  # Relaxed tolerance for numerical stability
            atol=5e-6
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
