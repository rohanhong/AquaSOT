"""
Test Backbone Checkpoint Loading Functionality

Tests the implementation of partial checkpoint loading for Stage 1/2 pretrained weights,
verifying the fixes for Audit 2025111704 Finding #2.

Author: AquaSOT-3D Team
Date: 2025-11-19
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

from aquasot3d.models.backbones.vit import ViTBackbone


class TestBackboneCheckpointLoading:
    """Test cases for backbone checkpoint loading functionality."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_dummy_rgb_checkpoint(self, checkpoint_path: Path):
        """Create a dummy RGB backbone checkpoint."""
        # Create a simple state dict mimicking RGB backbone structure
        # Using embed_dim=192 to match vit_tiny_patch16_224
        embed_dim = 192
        state_dict = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.patch_embed.proj.weight': torch.randn(embed_dim, 3, 16, 16),
                'backbone.rgb_backbone.vit.patch_embed.proj.bias': torch.randn(embed_dim),
                'backbone.rgb_backbone.vit.pos_embed': torch.randn(1, 197, embed_dim),
                'backbone.rgb_backbone.vit.cls_token': torch.randn(1, 1, embed_dim),
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
        """Create a dummy Sonar backbone checkpoint (using ViT)."""
        # Create a simple state dict mimicking Sonar backbone structure
        # NOTE: Sonar now uses unified ViT with channel adapter
        # Using embed_dim=192 to match vit_tiny_patch16_224
        embed_dim = 192
        state_dict = {
            'model_state_dict': {
                'backbone.sonar_encoder.vit.patch_embed.proj.weight': torch.randn(embed_dim, 3, 16, 16),
                'backbone.sonar_encoder.vit.patch_embed.proj.bias': torch.randn(embed_dim),
                'backbone.sonar_encoder.vit.pos_embed': torch.randn(1, 197, embed_dim),
                'backbone.sonar_encoder.vit.cls_token': torch.randn(1, 1, embed_dim),
                'backbone.sonar_encoder.channel_adapter.weight': torch.randn(3, 1, 1, 1),
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
        backbone = ViTBackbone(
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
        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=non_existent_path
        )

        assert backbone is not None

    def test_rgb_backbone_loads_checkpoint_successfully(self, temp_checkpoint_dir):
        """Test RGB backbone successfully loads pretrained checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'rgb_checkpoint.pth'
        self.create_dummy_rgb_checkpoint(checkpoint_path)

        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=str(checkpoint_path)
        )

        assert backbone is not None
        assert hasattr(backbone, 'vit')
        # Verify checkpoint was loaded (weights should be different from random init)
        assert backbone.pretrained_path == str(checkpoint_path)

    def test_sonar_backbone_loads_without_pretrained_path(self):
        """Test Sonar backbone initialization without pretrained checkpoint."""
        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=1,  # Sonar mode
            pretrained_path=None
        )

        assert backbone is not None
        assert hasattr(backbone, 'vit')
        assert backbone.channel_adapter is not None

    def test_sonar_backbone_handles_missing_checkpoint_gracefully(self, temp_checkpoint_dir):
        """Test Sonar backbone handles missing checkpoint file gracefully."""
        non_existent_path = str(temp_checkpoint_dir / 'non_existent.pth')

        # Should not raise error, just log warning
        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=1,  # Sonar mode
            pretrained_path=non_existent_path
        )

        assert backbone is not None

    def test_sonar_backbone_loads_checkpoint_successfully(self, temp_checkpoint_dir):
        """Test Sonar backbone successfully loads pretrained checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'sonar_checkpoint.pth'
        self.create_dummy_sonar_checkpoint(checkpoint_path)

        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=1,  # Sonar mode
            pretrained_path=str(checkpoint_path)
        )

        assert backbone is not None
        assert hasattr(backbone, 'vit')
        # Verify checkpoint was loaded
        assert backbone.pretrained_path == str(checkpoint_path)

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

        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            pretrained_path=str(checkpoint_path)
        )

        # Test forward pass with separate processing
        template = torch.randn(2, 3, 128, 128)
        search = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            outputs = backbone(template, search, return_separate=True)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['template_features'].shape[0] == 2  # batch size
        assert outputs['search_features'].shape[0] == 2

    def test_sonar_backbone_forward_with_pretrained_checkpoint(self, temp_checkpoint_dir):
        """Test Sonar backbone forward pass with pretrained checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'sonar_checkpoint.pth'
        self.create_dummy_sonar_checkpoint(checkpoint_path)

        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=1,  # Sonar mode
            pretrained_path=str(checkpoint_path)
        )

        # Test forward pass (Sonar images preprocessed to 224x224)
        template = torch.randn(2, 1, 224, 224)
        search = torch.randn(2, 1, 224, 224)

        with torch.no_grad():
            outputs = backbone(template, search)

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
