"""
Tests for OSTrack-style model architecture.

Tests the new simplified model implementation.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'lib'))

from lib.config.aquasot import cfg
from lib.models.aquasot import (
    AquaSOT, build_aquasot,
    VisionTransformer, vit_base_patch16_224,
    build_vit_rgb, build_vit_sonar,
    DCMA, IdentityFusion, build_fusion,
    combine_tokens, recover_tokens,
)
from lib.models.layers import CenterPredictor, PatchEmbed


class TestPatchEmbed:
    """Test PatchEmbed module."""

    def test_rgb_input(self):
        """Test 3-channel RGB input."""
        patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        x = torch.randn(2, 3, 224, 224)
        out = patch_embed(x)
        assert out.shape == (2, 196, 768)  # 14x14 patches

    def test_sonar_input(self):
        """Test 1-channel Sonar input."""
        patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=1, embed_dim=768)
        x = torch.randn(2, 1, 224, 224)
        out = patch_embed(x)
        assert out.shape == (2, 196, 768)

    def test_custom_sizes(self):
        """Test with template/search sizes."""
        # Template: 128x128 -> 8x8 patches
        patch_embed_t = PatchEmbed(img_size=128, patch_size=16, in_chans=3, embed_dim=768)
        x_t = torch.randn(2, 3, 128, 128)
        out_t = patch_embed_t(x_t)
        assert out_t.shape == (2, 64, 768)

        # Search: 256x256 -> 16x16 patches
        patch_embed_s = PatchEmbed(img_size=256, patch_size=16, in_chans=3, embed_dim=768)
        x_s = torch.randn(2, 3, 256, 256)
        out_s = patch_embed_s(x_s)
        assert out_s.shape == (2, 256, 768)


class TestCenterPredictor:
    """Test CenterPredictor head."""

    def test_forward(self):
        """Test forward pass."""
        head = CenterPredictor(inplanes=768, channel=256, feat_sz=16, stride=16)
        x = torch.randn(2, 768, 16, 16)
        score_map, bbox, size_map, offset_map = head(x)

        assert score_map.shape == (2, 1, 16, 16)
        assert bbox.shape == (2, 4)  # cx, cy, w, h
        assert size_map.shape == (2, 2, 16, 16)
        assert offset_map.shape == (2, 2, 16, 16)

    def test_bbox_normalized(self):
        """Test bbox is normalized to [0, 1]."""
        head = CenterPredictor(inplanes=768, channel=256, feat_sz=16, stride=16)
        x = torch.randn(2, 768, 16, 16)
        _, bbox, _, _ = head(x)

        # All values should be in [0, 1] range (approximately)
        assert bbox[:, :2].min() >= 0  # cx, cy
        assert bbox[:, :2].max() <= 1


class TestCombineTokens:
    """Test token combination utilities."""

    def test_direct_mode(self):
        """Test direct concatenation."""
        z = torch.randn(2, 64, 768)  # Template
        x = torch.randn(2, 256, 768)  # Search
        combined = combine_tokens(z, x, mode='direct')
        assert combined.shape == (2, 320, 768)

    def test_recover_direct(self):
        """Test token recovery."""
        combined = torch.randn(2, 320, 768)
        recovered = recover_tokens(combined, 64, 256, mode='direct')
        assert recovered.shape == (2, 320, 768)


class TestDCMA:
    """Test DCMA fusion module."""

    def test_forward(self):
        """Test forward pass."""
        dcma = DCMA(embed_dim=768, num_heads=8, num_layers=2)
        rgb = torch.randn(2, 320, 768)
        sonar = torch.randn(2, 320, 768)
        rgb_out, sonar_out = dcma(rgb, sonar)

        assert rgb_out.shape == (2, 320, 768)
        assert sonar_out.shape == (2, 320, 768)

    def test_identity_fusion(self):
        """Test identity fusion (no cross-modal interaction)."""
        fusion = IdentityFusion(embed_dim=768)
        rgb = torch.randn(2, 320, 768)
        sonar = torch.randn(2, 320, 768)
        rgb_out, sonar_out = fusion(rgb, sonar)

        # Should be identical to input
        assert torch.allclose(rgb_out, rgb)
        assert torch.allclose(sonar_out, sonar)


class TestViT:
    """Test ViT backbone."""

    def test_rgb_vit(self):
        """Test ViT with RGB input."""
        model = vit_base_patch16_224(pretrained=False, in_chans=3)
        model.finetune_track(cfg)

        z = torch.randn(2, 3, 128, 128)
        x = torch.randn(2, 3, 256, 256)
        out, aux = model(z, x)

        # Template: 64 tokens, Search: 256 tokens
        assert out.shape == (2, 320, 768)

    def test_sonar_vit(self):
        """Test ViT with Sonar input (1-channel)."""
        model = vit_base_patch16_224(pretrained=False, in_chans=1)
        model.finetune_track(cfg)

        z = torch.randn(2, 1, 128, 128)
        x = torch.randn(2, 1, 256, 256)
        out, aux = model(z, x)

        assert out.shape == (2, 320, 768)


class TestAquaSOT:
    """Test full AquaSOT model."""

    @pytest.fixture
    def model(self):
        """Create model without pretrained weights."""
        # Temporarily disable pretrained loading
        original_pretrain = cfg.MODEL.PRETRAIN_FILE
        cfg.MODEL.PRETRAIN_FILE = ""

        model = build_aquasot(cfg, training=False)

        cfg.MODEL.PRETRAIN_FILE = original_pretrain
        return model

    def test_model_creation(self, model):
        """Test model can be created."""
        assert isinstance(model, AquaSOT)
        assert hasattr(model, 'rgb_backbone')
        assert hasattr(model, 'sonar_backbone')
        assert hasattr(model, 'fusion')
        assert hasattr(model, 'box_head')

    def test_forward(self, model):
        """Test forward pass."""
        rgb_t = torch.randn(2, 3, 128, 128)
        rgb_s = torch.randn(2, 3, 256, 256)
        sonar_t = torch.randn(2, 1, 128, 128)
        sonar_s = torch.randn(2, 1, 256, 256)

        out = model(rgb_t, rgb_s, sonar_t, sonar_s)

        assert 'rgb_boxes' in out
        assert 'sonar_boxes' in out
        assert out['rgb_boxes'].shape == (2, 4)
        assert out['sonar_boxes'].shape == (2, 4)

    def test_online_tracking(self, model):
        """Test online tracking interface."""
        model.eval()

        # Initialize with template
        rgb_t = torch.randn(1, 3, 128, 128)
        sonar_t = torch.randn(1, 1, 128, 128)
        model.set_online(rgb_t, sonar_t)

        # Track
        rgb_s = torch.randn(1, 3, 256, 256)
        sonar_s = torch.randn(1, 1, 256, 256)
        out = model.track(rgb_s, sonar_s)

        assert 'rgb_boxes' in out
        assert 'sonar_boxes' in out

    def test_output_range(self, model):
        """Test output bbox values are in reasonable range."""
        rgb_t = torch.randn(2, 3, 128, 128)
        rgb_s = torch.randn(2, 3, 256, 256)
        sonar_t = torch.randn(2, 1, 128, 128)
        sonar_s = torch.randn(2, 1, 256, 256)

        out = model(rgb_t, rgb_s, sonar_t, sonar_s)

        # cx, cy should be in [0, 1]
        assert out['rgb_boxes'][:, :2].min() >= 0
        assert out['rgb_boxes'][:, :2].max() <= 1

    def test_model_parameter_count(self, model):
        """Test model is not overly complex."""
        total_params = sum(p.numel() for p in model.parameters())
        # Should be around 206M parameters (2x ViT-Base ~86M each + fusion + head)
        # Less than 250M is reasonable
        assert total_params < 250_000_000


class TestBuildFunctions:
    """Test model building functions."""

    def test_build_fusion_dcma(self):
        """Test building DCMA fusion."""
        fusion = build_fusion(cfg)
        assert isinstance(fusion, DCMA)

    def test_build_fusion_identity(self):
        """Test building identity fusion."""
        original = cfg.MODEL.FUSION.TYPE
        cfg.MODEL.FUSION.TYPE = "identity"
        fusion = build_fusion(cfg)
        assert isinstance(fusion, IdentityFusion)
        cfg.MODEL.FUSION.TYPE = original


class TestModelSize:
    """Test model complexity is within bounds."""

    def test_aquasot_lines(self):
        """Verify model file is simplified."""
        model_file = project_root / 'lib' / 'models' / 'aquasot' / 'aquasot.py'
        if model_file.exists():
            with open(model_file) as f:
                lines = len(f.readlines())
            # Should be under 200 lines (vs. 1349 in old implementation)
            assert lines < 200, f"Model file has {lines} lines, should be < 200"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
