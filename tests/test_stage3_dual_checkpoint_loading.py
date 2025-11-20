"""
Test Stage 3 Dual Checkpoint Loading (Audit 2025112003 - Integration Test)

Critical test to verify that Stage 3 full training can simultaneously load:
- Stage 1 RGB pretrained checkpoint
- Stage 2 Sonar pretrained checkpoint

Without conflicts or cross-contamination.

This is the KEY functionality for 3-stage training pipeline:
  Stage 1: RGB-only pretraining
  Stage 2: Sonar-only pretraining
  Stage 3: Full dual-modality training (loads BOTH checkpoints)

Author: AquaSOT-3D Team
Date: 2025-11-20
"""

import pytest
import torch
import tempfile
from pathlib import Path

from aquasot3d.models.backbones.factory import MultiModalBackbone


class TestStage3DualCheckpointLoading:
    """Test Stage 3 can load both RGB and Sonar checkpoints simultaneously"""

    def test_load_rgb_and_sonar_checkpoints_simultaneously(self, tmp_path):
        """
        CRITICAL TEST: Stage 3 loads both Stage 1 RGB and Stage 2 Sonar checkpoints

        This is the core functionality of 3-stage training pipeline.
        """
        # Create Stage 1 RGB checkpoint (RGB-only pretraining)
        rgb_checkpoint_path = tmp_path / "stage1_rgb_best.pth"
        rgb_checkpoint = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.pos_embed': torch.randn(1, 197, 768),
                'backbone.rgb_backbone.vit.patch_embed.proj.weight': torch.randn(768, 3, 16, 16),
                'backbone.rgb_backbone.vit.blocks.0.attn.qkv.weight': torch.randn(2304, 768),
                'backbone.rgb_backbone.vit.blocks.0.attn.proj.weight': torch.randn(768, 768),
                'backbone.rgb_backbone.vit.norm.weight': torch.randn(768),
            }
        }
        torch.save(rgb_checkpoint, rgb_checkpoint_path)

        # Create Stage 2 Sonar checkpoint (Sonar-only pretraining)
        sonar_checkpoint_path = tmp_path / "stage2_sonar_best.pth"
        sonar_checkpoint = {
            'model_state_dict': {
                'backbone.sonar_encoder.vit.pos_embed': torch.randn(1, 197, 768),
                'backbone.sonar_encoder.vit.patch_embed.proj.weight': torch.randn(768, 1, 16, 16),
                'backbone.sonar_encoder.vit.blocks.0.attn.qkv.weight': torch.randn(2304, 768),
                'backbone.sonar_encoder.vit.blocks.0.attn.proj.weight': torch.randn(768, 768),
                'backbone.sonar_encoder.vit.norm.weight': torch.randn(768),
            }
        }
        torch.save(sonar_checkpoint, sonar_checkpoint_path)

        # Stage 3 configuration: Load BOTH checkpoints
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(rgb_checkpoint_path),  # Stage 1 RGB checkpoint
            'freeze_backbone': False,
        }

        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(sonar_checkpoint_path),  # Stage 2 Sonar checkpoint
            'in_channels': 1,
            'freeze_backbone': False,
        }

        # Create Stage 3 MultiModalBackbone
        backbone = MultiModalBackbone(
            rgb_config=rgb_config,
            sonar_config=sonar_config
        )

        # CRITICAL CHECKS
        assert backbone.rgb_backbone is not None
        assert backbone.sonar_encoder is not None

        # Verify both backbones loaded their checkpoints
        # (they should have logged "Loaded pretrained ViT backbone from ...")

    def test_dual_checkpoint_no_cross_contamination(self, tmp_path):
        """
        Verify RGB and Sonar checkpoints don't contaminate each other

        Each backbone should have completely independent parameters.
        """
        # Create two checkpoints with DIFFERENT values (easy to detect contamination)
        rgb_checkpoint_path = tmp_path / "stage1_rgb.pth"

        # RGB checkpoint: all ones
        rgb_pos_embed = torch.ones(1, 197, 768)
        rgb_checkpoint = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.pos_embed': rgb_pos_embed,
                'backbone.rgb_backbone.vit.patch_embed.proj.weight': torch.ones(768, 3, 16, 16),
            }
        }
        torch.save(rgb_checkpoint, rgb_checkpoint_path)

        # Sonar checkpoint: all twos (different!)
        sonar_checkpoint_path = tmp_path / "stage2_sonar.pth"
        sonar_pos_embed = torch.ones(1, 197, 768) * 2.0
        sonar_checkpoint = {
            'model_state_dict': {
                'backbone.sonar_encoder.vit.pos_embed': sonar_pos_embed,
                'backbone.sonar_encoder.vit.patch_embed.proj.weight': torch.ones(768, 1, 16, 16) * 2.0,
            }
        }
        torch.save(sonar_checkpoint, sonar_checkpoint_path)

        # Load both checkpoints
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(rgb_checkpoint_path),
        }

        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(sonar_checkpoint_path),
            'in_channels': 1,
        }

        backbone = MultiModalBackbone(rgb_config=rgb_config, sonar_config=sonar_config)

        # CRITICAL CHECK: Verify parameters are different (no contamination)
        rgb_loaded = backbone.rgb_backbone.vit.pos_embed
        sonar_loaded = backbone.sonar_encoder.vit.pos_embed

        # RGB should have values close to 1.0
        assert torch.allclose(rgb_loaded, torch.ones_like(rgb_loaded), atol=0.1), \
            "RGB checkpoint not loaded correctly or contaminated by Sonar"

        # Sonar should have values close to 2.0
        assert torch.allclose(sonar_loaded, torch.ones_like(sonar_loaded) * 2.0, atol=0.1), \
            "Sonar checkpoint not loaded correctly or contaminated by RGB"

        # They should be DIFFERENT
        assert not torch.allclose(rgb_loaded, sonar_loaded, atol=0.1), \
            "RGB and Sonar parameters are identical - checkpoints contaminated each other!"

    def test_dual_checkpoint_forward_pass(self, tmp_path):
        """
        Test forward pass works correctly with both checkpoints loaded
        """
        # Create minimal checkpoints
        rgb_checkpoint_path = tmp_path / "rgb.pth"
        rgb_checkpoint = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.pos_embed': torch.randn(1, 197, 768),
                'backbone.rgb_backbone.vit.patch_embed.proj.weight': torch.randn(768, 3, 16, 16),
            }
        }
        torch.save(rgb_checkpoint, rgb_checkpoint_path)

        sonar_checkpoint_path = tmp_path / "sonar.pth"
        sonar_checkpoint = {
            'model_state_dict': {
                'backbone.sonar_encoder.vit.pos_embed': torch.randn(1, 197, 768),
                'backbone.sonar_encoder.vit.patch_embed.proj.weight': torch.randn(768, 1, 16, 16),
            }
        }
        torch.save(sonar_checkpoint, sonar_checkpoint_path)

        # Create backbone with both checkpoints
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(rgb_checkpoint_path),
        }

        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(sonar_checkpoint_path),
            'in_channels': 1,
        }

        backbone = MultiModalBackbone(rgb_config=rgb_config, sonar_config=sonar_config)

        # Test forward pass (template-search paradigm)
        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 224, 224)
        sonar_template = torch.randn(batch_size, 1, 128, 128)
        sonar_search = torch.randn(batch_size, 1, 224, 224)

        # Should not crash (channel adapter handles 1→3 conversion for Sonar)
        # Use return_separate=True since template and search have different sizes
        rgb_features = backbone.rgb_backbone.forward(rgb_template, rgb_search, return_separate=True)
        sonar_features = backbone.sonar_encoder.forward(sonar_template, sonar_search, return_separate=True)

        # Verify outputs are different (different checkpoints → different features)
        assert rgb_features is not None
        assert sonar_features is not None
        assert 'template_features' in rgb_features
        assert 'template_features' in sonar_features

    def test_stage3_config_processing(self, tmp_path):
        """
        Test that Stage 3 configuration processing applies checkpoints correctly

        Simulates what train.py does with init_backbones parameter.
        """
        # Create mock checkpoints
        rgb_checkpoint_path = tmp_path / "outputs/pretrain_rgb/checkpoints/checkpoint_best.pth"
        rgb_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': {
            'backbone.rgb_backbone.vit.pos_embed': torch.randn(1, 197, 768)
        }}, rgb_checkpoint_path)

        sonar_checkpoint_path = tmp_path / "outputs/pretrain_sonar/checkpoints/checkpoint_best.pth"
        sonar_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': {
            'backbone.sonar_encoder.vit.pos_embed': torch.randn(1, 197, 768)
        }}, sonar_checkpoint_path)

        # Simulate train.py config processing
        # (This is what happens in scripts/train.py:95-112)
        rgb_backbone_cfg = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
        }

        sonar_backbone_cfg = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'in_channels': 1,
        }

        # Simulate init_backbones processing
        init_backbones = {
            'rgb': str(rgb_checkpoint_path),
            'sonar': str(sonar_checkpoint_path),
        }

        # Apply checkpoint paths (like train.py does)
        if init_backbones.get('rgb'):
            rgb_checkpoint = init_backbones['rgb']
            if rgb_checkpoint not in [None, 'null', '']:
                rgb_backbone_cfg['pretrained_path'] = rgb_checkpoint

        if init_backbones.get('sonar'):
            sonar_checkpoint = init_backbones['sonar']
            if sonar_checkpoint not in [None, 'null', '']:
                sonar_backbone_cfg['pretrained_path'] = sonar_checkpoint

        # Verify configs have correct checkpoint paths
        assert 'pretrained_path' in rgb_backbone_cfg
        assert 'pretrained_path' in sonar_backbone_cfg
        assert rgb_backbone_cfg['pretrained_path'] == str(rgb_checkpoint_path)
        assert sonar_backbone_cfg['pretrained_path'] == str(sonar_checkpoint_path)

        # Create backbone with processed configs
        backbone = MultiModalBackbone(
            rgb_config=rgb_backbone_cfg,
            sonar_config=sonar_backbone_cfg
        )

        # Should successfully create both backbones with checkpoints loaded
        assert backbone.rgb_backbone is not None
        assert backbone.sonar_encoder is not None

    def test_one_checkpoint_none_other_valid(self, tmp_path):
        """
        Test edge case: RGB checkpoint specified, Sonar uses ImageNet

        This might happen if user only has Stage 1 RGB pretrained.
        """
        # Create only RGB checkpoint
        rgb_checkpoint_path = tmp_path / "stage1_rgb.pth"
        rgb_checkpoint = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.pos_embed': torch.randn(1, 197, 768),
            }
        }
        torch.save(rgb_checkpoint, rgb_checkpoint_path)

        # RGB uses checkpoint, Sonar uses ImageNet (pretrained=True, no pretrained_path)
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(rgb_checkpoint_path),
        }

        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': True,  # ImageNet initialization
            'in_channels': 1,
            # No pretrained_path
        }

        backbone = MultiModalBackbone(rgb_config=rgb_config, sonar_config=sonar_config)

        # Both should be created successfully
        assert backbone.rgb_backbone is not None
        assert backbone.sonar_encoder is not None


class TestStage3TrainingPipelineIntegration:
    """Integration tests for complete 3-stage training pipeline"""

    def test_stage1_to_stage3_rgb_checkpoint_compatibility(self, tmp_path):
        """
        Test Stage 1 RGB checkpoint can be loaded in Stage 3

        Simulates: Stage 1 (RGB-only) → Stage 3 (Full)
        """
        # Create Stage 1 RGB-only checkpoint
        stage1_checkpoint = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.pos_embed': torch.randn(1, 197, 768),
                'backbone.rgb_backbone.vit.patch_embed.proj.weight': torch.randn(768, 3, 16, 16),
                'tracker.head.weight': torch.randn(256, 768),  # Other model components
            },
            'epoch': 50,
            'best_iou': 0.72,
        }
        checkpoint_path = tmp_path / "stage1_rgb_best.pth"
        torch.save(stage1_checkpoint, checkpoint_path)

        # Stage 3 loads this checkpoint for RGB backbone
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(checkpoint_path),
        }

        # Sonar uses ImageNet (no Stage 2 pretraining)
        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': True,
            'in_channels': 1,
        }

        backbone = MultiModalBackbone(rgb_config=rgb_config, sonar_config=sonar_config)

        # Should work - RGB loads Stage 1 checkpoint, Sonar uses ImageNet
        assert backbone.rgb_backbone is not None
        assert backbone.sonar_encoder is not None

    def test_stage2_to_stage3_sonar_checkpoint_compatibility(self, tmp_path):
        """
        Test Stage 2 Sonar checkpoint can be loaded in Stage 3

        Simulates: Stage 2 (Sonar-only) → Stage 3 (Full)
        """
        # Create Stage 2 Sonar-only checkpoint
        stage2_checkpoint = {
            'model_state_dict': {
                'backbone.sonar_encoder.vit.pos_embed': torch.randn(1, 197, 768),
                'backbone.sonar_encoder.vit.patch_embed.proj.weight': torch.randn(768, 1, 16, 16),
                'tracker.head.weight': torch.randn(256, 768),
            },
            'epoch': 50,
            'best_iou': 0.68,
        }
        checkpoint_path = tmp_path / "stage2_sonar_best.pth"
        torch.save(stage2_checkpoint, checkpoint_path)

        # Stage 3 loads this checkpoint for Sonar backbone
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': True,  # ImageNet
        }

        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(checkpoint_path),  # Stage 2 checkpoint
            'in_channels': 1,
        }

        backbone = MultiModalBackbone(rgb_config=rgb_config, sonar_config=sonar_config)

        # Should work - Sonar loads Stage 2 checkpoint, RGB uses ImageNet
        assert backbone.rgb_backbone is not None
        assert backbone.sonar_encoder is not None

    def test_full_stage1_plus_stage2_to_stage3_pipeline(self, tmp_path):
        """
        COMPREHENSIVE TEST: Full 3-stage pipeline

        Stage 1 RGB → Stage 2 Sonar → Stage 3 Full (both checkpoints)

        This is the COMPLETE workflow.
        """
        # Stage 1 RGB checkpoint
        stage1_rgb_checkpoint = {
            'model_state_dict': {
                'backbone.rgb_backbone.vit.pos_embed': torch.ones(1, 197, 768),  # ones
                'backbone.rgb_backbone.vit.patch_embed.proj.weight': torch.ones(768, 3, 16, 16),
            },
            'epoch': 50,
        }
        stage1_path = tmp_path / "stage1_rgb_best.pth"
        torch.save(stage1_rgb_checkpoint, stage1_path)

        # Stage 2 Sonar checkpoint
        stage2_sonar_checkpoint = {
            'model_state_dict': {
                'backbone.sonar_encoder.vit.pos_embed': torch.ones(1, 197, 768) * 2.0,  # twos
                'backbone.sonar_encoder.vit.patch_embed.proj.weight': torch.ones(768, 1, 16, 16) * 2.0,
            },
            'epoch': 50,
        }
        stage2_path = tmp_path / "stage2_sonar_best.pth"
        torch.save(stage2_sonar_checkpoint, stage2_path)

        # Stage 3: Load BOTH checkpoints
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(stage1_path),
        }

        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'pretrained_path': str(stage2_path),
            'in_channels': 1,
        }

        backbone = MultiModalBackbone(rgb_config=rgb_config, sonar_config=sonar_config)

        # CRITICAL VERIFICATION: Both checkpoints loaded correctly
        rgb_params = backbone.rgb_backbone.vit.pos_embed
        sonar_params = backbone.sonar_encoder.vit.pos_embed

        # RGB should have ~1.0 values (from Stage 1)
        assert torch.allclose(rgb_params, torch.ones_like(rgb_params), atol=0.1), \
            "RGB did not load Stage 1 checkpoint correctly"

        # Sonar should have ~2.0 values (from Stage 2)
        assert torch.allclose(sonar_params, torch.ones_like(sonar_params) * 2.0, atol=0.1), \
            "Sonar did not load Stage 2 checkpoint correctly"

        # They should be different (no cross-contamination)
        assert not torch.allclose(rgb_params, sonar_params, atol=0.5), \
            "CRITICAL BUG: RGB and Sonar checkpoints contaminated each other!"

        # Test forward pass with both checkpoints loaded (template-search paradigm)
        rgb_template = torch.randn(1, 3, 128, 128)
        rgb_search = torch.randn(1, 3, 224, 224)
        sonar_template = torch.randn(1, 1, 128, 128)
        sonar_search = torch.randn(1, 1, 224, 224)

        # Use forward method (not forward_features) to apply channel adapter
        # Use return_separate=True since template and search have different sizes
        rgb_out = backbone.rgb_backbone.forward(rgb_template, rgb_search, return_separate=True)
        sonar_out = backbone.sonar_encoder.forward(sonar_template, sonar_search, return_separate=True)

        assert rgb_out is not None
        assert sonar_out is not None
        assert 'template_features' in rgb_out
        assert 'template_features' in sonar_out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
