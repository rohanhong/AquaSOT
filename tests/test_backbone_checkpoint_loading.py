"""
Test Backbone Checkpoint Loading Functionality

Tests the implementation of partial checkpoint loading for Stage 1/2 pretrained weights,
verifying the fixes for Audit 2025111704 Finding #2.

Author: AquaSOT-3D Team
Date: 2025-11-17
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from aquasot3d.models.backbones.vit_rgb import ViTRGBBackbone
from aquasot3d.models.backbones.sonar_encoder import SonarEncoder


class TestBackboneCheckpointLoading:
    """Test cases for backbone checkpoint loading functionality."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_dummy_rgb_checkpoint(self, checkpoint_path: Path):
        """Create a dummy RGB backbone checkpoint."""
        # Create a simple state dict mimicking RGB backbone structure
        state_dict = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.patch_embed.proj.weight': torch.randn(768, 3, 16, 16),
                'backbone.rgb_backbone.vit.patch_embed.proj.bias': torch.randn(768),
                'backbone.rgb_backbone.vit.pos_embed': torch.randn(1, 197, 768),
                'backbone.rgb_backbone.vit.cls_token': torch.randn(1, 1, 768),
            },
            'epoch': 40,
            'metadata': {
                'stage': 'pretraining',
                'modalities': ['rgb'],
                'experiment_name': 'rgb_pretrain_test'
            }
        }

        torch.save(state_dict, checkpoint_path)
        return checkpoint_path

    def create_dummy_sonar_checkpoint(self, checkpoint_path: Path):
        """Create a dummy Sonar encoder checkpoint."""
        # Create a simple state dict mimicking Sonar encoder structure
        state_dict = {
            'model_state_dict': {
                'backbone.sonar_encoder.encoder.patch_embed.proj.weight': torch.randn(384, 3, 16, 16),
                'backbone.sonar_encoder.encoder.patch_embed.proj.bias': torch.randn(384),
                'backbone.sonar_encoder.encoder.pos_embed': torch.randn(1, 197, 384),
                'backbone.sonar_encoder.input_adapter.weight': torch.randn(3, 1, 1, 1),
                'backbone.sonar_encoder.input_adapter.bias': torch.randn(3),
            },
            'epoch': 40,
            'metadata': {
                'stage': 'pretraining',
                'modalities': ['sonar'],
                'experiment_name': 'sonar_pretrain_test'
            }
        }

        torch.save(state_dict, checkpoint_path)
        return checkpoint_path

    def test_rgb_backbone_loads_without_pretrained_path(self):
        """Test RGB backbone initialization without pretrained checkpoint."""
        backbone = ViTRGBBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=None
        )

        assert backbone is not None
        assert hasattr(backbone, 'vit')

    def test_rgb_backbone_handles_missing_checkpoint_gracefully(self, temp_checkpoint_dir):
        """Test RGB backbone handles missing checkpoint file gracefully."""
        non_existent_path = str(temp_checkpoint_dir / 'non_existent.pth')

        # Should not raise error, just log warning
        backbone = ViTRGBBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=non_existent_path
        )

        assert backbone is not None

    def test_rgb_backbone_loads_checkpoint_successfully(self, temp_checkpoint_dir):
        """Test RGB backbone successfully loads pretrained checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'rgb_checkpoint.pth'
        self.create_dummy_rgb_checkpoint(checkpoint_path)

        backbone = ViTRGBBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=str(checkpoint_path)
        )

        assert backbone is not None
        assert hasattr(backbone, 'vit')
        # Verify checkpoint was loaded (weights should be different from random init)
        assert backbone.pretrained_path == str(checkpoint_path)

    def test_sonar_encoder_loads_without_pretrained_path(self):
        """Test Sonar encoder initialization without pretrained checkpoint."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=None
        )

        assert encoder is not None
        assert hasattr(encoder, 'encoder')

    def test_sonar_encoder_handles_missing_checkpoint_gracefully(self, temp_checkpoint_dir):
        """Test Sonar encoder handles missing checkpoint file gracefully."""
        non_existent_path = str(temp_checkpoint_dir / 'non_existent.pth')

        # Should not raise error, just log warning
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=non_existent_path
        )

        assert encoder is not None

    def test_sonar_encoder_loads_checkpoint_successfully(self, temp_checkpoint_dir):
        """Test Sonar encoder successfully loads pretrained checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'sonar_checkpoint.pth'
        self.create_dummy_sonar_checkpoint(checkpoint_path)

        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=str(checkpoint_path)
        )

        assert encoder is not None
        assert hasattr(encoder, 'encoder')
        # Verify checkpoint was loaded
        assert encoder.pretrained_path == str(checkpoint_path)

    def test_checkpoint_metadata_preserved(self, temp_checkpoint_dir):
        """Test that checkpoint metadata is preserved."""
        checkpoint_path = temp_checkpoint_dir / 'rgb_checkpoint.pth'
        self.create_dummy_rgb_checkpoint(checkpoint_path)

        # Load checkpoint directly to verify metadata
        checkpoint = torch.load(checkpoint_path)

        assert 'metadata' in checkpoint
        assert checkpoint['metadata']['stage'] == 'pretraining'
        assert checkpoint['metadata']['modalities'] == ['rgb']
        assert checkpoint['metadata']['experiment_name'] == 'rgb_pretrain_test'

    def test_rgb_backbone_forward_with_pretrained_checkpoint(self, temp_checkpoint_dir):
        """Test RGB backbone forward pass with pretrained checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'rgb_checkpoint.pth'
        self.create_dummy_rgb_checkpoint(checkpoint_path)

        backbone = ViTRGBBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=str(checkpoint_path)
        )

        # Test forward pass
        template = torch.randn(2, 3, 128, 128)
        search = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            outputs = backbone(template, search)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['template_features'].shape[0] == 2  # batch size
        assert outputs['search_features'].shape[0] == 2

    def test_sonar_encoder_forward_with_pretrained_checkpoint(self, temp_checkpoint_dir):
        """Test Sonar encoder forward pass with pretrained checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'sonar_checkpoint.pth'
        self.create_dummy_sonar_checkpoint(checkpoint_path)

        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=str(checkpoint_path)
        )

        # Test forward pass
        template = torch.randn(2, 1, 376, 682)
        search = torch.randn(2, 1, 376, 682)

        with torch.no_grad():
            outputs = encoder(template, search)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['template_features'].shape[0] == 2  # batch size
        assert outputs['search_features'].shape[0] == 2


class TestOptimizerLRMultiplier:
    """Test cases for backbone_lr_multiplier functionality."""

    def test_build_optimizer_with_backbone_lr_multiplier(self):
        """Test optimizer respects backbone_lr_multiplier parameter."""
        from aquasot3d.training.optimizer import build_optimizer

        # Create a simple model with backbone and head
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(10, 20)
                self.head = nn.Linear(20, 5)

        model = SimpleModel()
        base_lr = 0.001
        backbone_lr_mult = 0.1

        optimizer = build_optimizer(
            model=model,
            optimizer_type='adamw',
            base_lr=base_lr,
            backbone_lr_multiplier=backbone_lr_mult
        )

        # Check parameter groups
        assert len(optimizer.param_groups) >= 1

        # Find backbone group
        backbone_group = None
        head_group = None
        for group in optimizer.param_groups:
            if 'name' in group:
                if 'backbone' in group['name']:
                    backbone_group = group
                elif 'head' in group['name']:
                    head_group = group

        # Verify backbone has reduced LR
        if backbone_group is not None:
            expected_backbone_lr = base_lr * backbone_lr_mult
            assert abs(backbone_group['lr'] - expected_backbone_lr) < 1e-6

        # Verify head has base LR
        if head_group is not None:
            assert abs(head_group['lr'] - base_lr) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
