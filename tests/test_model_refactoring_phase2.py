"""
Unit Tests for Phase 2 Model Layer Refactoring

Tests the model layer refactoring from template-search paradigm to
frame-level temporal tracking including:
- ViT Encoder single-frame forward
- Vision-Mamba dual-mode operation (training/inference)
- Sequence handling and shape validation

Author: AquaSOT-3D Team
Date: 2025-11-20
Audit: 2025112006
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

from aquasot3d.models.backbones.vit import ViTBackbone
from aquasot3d.models.temporal.mamba_head import VisionMambaHead, MAMBA_AVAILABLE


@pytest.fixture
def temp_checkpoint():
    """Create temporary checkpoint file for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_path = temp_dir / "test_checkpoint.pth"

    # Create dummy checkpoint
    checkpoint = {
        'model_state_dict': {
            'pos_embed': torch.randn(1, 197, 768),  # Standard ViT-B position embedding
        }
    }
    torch.save(checkpoint, checkpoint_path)

    yield checkpoint_path

    # Cleanup
    os.remove(checkpoint_path)
    os.rmdir(temp_dir)


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



@pytest.mark.skipif(not MAMBA_AVAILABLE or not torch.cuda.is_available(),
                    reason="mamba_ssm requires CUDA")
class TestVisionMambaDualMode:
    """Test Vision-Mamba dual-mode operation (training/inference)."""

    def test_training_mode_output_shape(self):
        """Test training mode returns all temporal features [B, T, C, H, W]."""
        head = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            bidirectional=False,
            use_occlusion_gating=False
        )
        head.train()  # Training mode
        head = head.cuda()  # Move to CUDA

        # Input sequence
        features = torch.randn(2, 5, 256, 8, 8).cuda()  # B=2, T=5, C=256, H=8, W=8

        outputs, temporal_state = head(features)

        # Training mode should return all frames [B, T, C, H, W]
        assert outputs.shape == (2, 5, 256, 8, 8)

    def test_inference_mode_output_shape(self):
        """Test inference mode returns single frame [B, C, H, W]."""
        head = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            bidirectional=False,
            use_occlusion_gating=False
        )
        head.eval()  # Inference mode

        # Input sequence
        features = torch.randn(2, 5, 256, 8, 8)

        with torch.no_grad():
            outputs, temporal_state = head(features)

        # Inference mode should return last frame [B, C, H, W]
        assert outputs.shape == (2, 256, 8, 8)

    def test_temporal_state_persistence(self):
        """Test temporal state persistence in inference mode."""
        head = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            bidirectional=False
        )
        head.eval()

        # First frame
        frame1 = torch.randn(1, 1, 256, 8, 8)  # B=1, T=1

        with torch.no_grad():
            out1, state1 = head(frame1, temporal_state=None)

        # Check temporal state
        assert state1 is not None
        assert len(state1) == 1  # One frame in buffer

        # Second frame (with temporal state)
        frame2 = torch.randn(1, 1, 256, 8, 8)

        with torch.no_grad():
            out2, state2 = head(frame2, temporal_state=state1)

        # Check temporal state updated
        assert len(state2) == 2  # Two frames in buffer

    def test_temporal_window_size(self):
        """Test temporal window size constraint (FIFO buffer)."""
        head = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            bidirectional=False
        )
        head.eval()

        temporal_state = None
        max_window = head.temporal_window_size  # Should be 5

        # Add frames beyond window size
        for i in range(max_window + 3):  # Add 8 frames (window=5)
            frame = torch.randn(1, 1, 256, 8, 8)
            with torch.no_grad():
                _, temporal_state = head(frame, temporal_state=temporal_state)

        # Buffer should be capped at window size
        assert len(temporal_state) == max_window

    def test_single_frame_auto_unsqueeze(self):
        """Test single frame input auto-unsqueezes to [B, 1, C, H, W]."""
        head = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            bidirectional=False,
            use_occlusion_gating=False
        )
        head.eval()

        # Single frame (4D input)
        frame = torch.randn(2, 256, 8, 8)  # B=2, C=256, H=8, W=8

        with torch.no_grad():
            outputs, temporal_state = head(frame)

        # Should handle single frame and return [B, C, H, W] in inference
        assert outputs.shape == (2, 256, 8, 8)

    def test_quality_scores_optional(self):
        """Test that quality scores are optional."""
        head = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            use_occlusion_gating=True
        )
        head.eval()

        features = torch.randn(2, 5, 256, 8, 8)

        # Without quality scores
        with torch.no_grad():
            out1, _ = head(features, quality_scores=None)

        assert out1.shape == (2, 256, 8, 8)  # Inference mode

        # With quality scores
        quality_scores = torch.rand(2, 5)  # [B, T]
        with torch.no_grad():
            out2, _ = head(features, quality_scores=quality_scores)

        assert out2.shape == (2, 256, 8, 8)

    def test_bidirectional_processing(self):
        """Test bidirectional processing."""
        head_uni = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            bidirectional=False
        )

        head_bi = VisionMambaHead(
            feature_dim=256,
            num_layers=2,
            bidirectional=True
        )

        head_uni.eval()
        head_bi.eval()

        features = torch.randn(2, 5, 256, 8, 8)

        with torch.no_grad():
            out_uni, _ = head_uni(features)
            out_bi, _ = head_bi(features)

        # Both should have same output shape
        assert out_uni.shape == out_bi.shape == (2, 256, 8, 8)


class TestIntegration:
    """Test integration between ViT and Mamba."""

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba_ssm not installed")
    def test_vit_to_mamba_pipeline(self):
        """Test full pipeline: ViT Encoder -> Mamba Head."""
        # Create models
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            img_size=224,
            in_channels=3
        )

        head = VisionMambaHead(
            feature_dim=768,  # ViT-Base output
            num_layers=2,
            bidirectional=False,
            use_occlusion_gating=False
        )

        backbone.eval()
        head.train()  # Training mode for Mamba

        # Input sequence
        frames = torch.randn(2, 5, 3, 224, 224)  # B=2, T=5

        with torch.no_grad():
            # ViT forward
            vit_outputs = backbone(frames)
            features = vit_outputs['features']  # [B*T, N, C]

        # Reshape for Mamba: [B*T, N, C] -> [B, T, C, H, W]
        B, T = 2, 5
        N, C = features.shape[1], features.shape[2]
        H = W = int(N ** 0.5)  # Assume square spatial dimension (approximate)

        # For ViT with CLS token: N = H*W + 1
        # Remove CLS token
        features_no_cls = features[:, 1:, :]  # Remove first token (CLS)
        N_spatial = features_no_cls.shape[1]
        H = W = int(N_spatial ** 0.5)

        features_reshaped = features_no_cls.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # [B*T, C, H, W]
        features_seq = features_reshaped.reshape(B, T, C, H, W)  # [B, T, C, H, W]

        # Mamba forward
        outputs, _ = head(features_seq)

        # Training mode should return all frames
        assert outputs.shape == (B, T, C, H, W)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
