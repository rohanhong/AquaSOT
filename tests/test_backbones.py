"""
Unit Tests for Backbone Networks

Tests for unified ViT backbone (RGB and Sonar) and Backbone Factory.

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# Import backbone modules
from aquasot3d.models.backbones import (
    ViTBackbone,
    BackboneFactory,
    MultiModalBackbone,
    build_backbone,
    build_multimodal_backbone)


class TestViTBackbone:
    """Tests for unified ViT Backbone (RGB and Sonar)."""

    @pytest.fixture
    def vit_rgb_backbone(self):
        """Create ViT backbone for RGB testing."""
        return ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=3,
            img_size=224,
            embed_dim=192,
            depth=12,
            num_heads=3)

    @pytest.fixture
    def vit_sonar_backbone(self):
        """Create ViT backbone for Sonar testing."""
        return ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=1,  # Single-channel sonar
            img_size=224,
            embed_dim=192,
            depth=12,
            num_heads=3)

    def test_vit_rgb_initialization(self, vit_rgb_backbone):
        """Test ViT RGB backbone initialization."""
        assert isinstance(vit_rgb_backbone, ViTBackbone)
        assert vit_rgb_backbone.embed_dim > 0
        assert vit_rgb_backbone.patch_size > 0
        assert vit_rgb_backbone.in_channels == 3
        assert vit_rgb_backbone.channel_adapter is None

    def test_vit_sonar_initialization(self, vit_sonar_backbone):
        """Test ViT Sonar backbone initialization."""
        assert isinstance(vit_sonar_backbone, ViTBackbone)
        assert vit_sonar_backbone.in_channels == 1
        assert vit_sonar_backbone.channel_adapter is not None
        # Channel adapter should convert 1ch -> 3ch
        test_input = torch.randn(1, 1, 224, 224)
        adapted = vit_sonar_backbone.channel_adapter(test_input)
        assert adapted.shape[1] == 3

    def test_vit_forward_separate_rgb(self, vit_rgb_backbone):
        """Test ViT forward pass with separate processing (RGB)."""
        template = torch.randn(2, 3, 128, 128)
        search = torch.randn(2, 3, 256, 256)

        outputs = vit_rgb_backbone(template, search, return_separate=True)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['template_features'].shape[0] == 2
        assert outputs['search_features'].shape[0] == 2

    def test_vit_forward_separate_sonar(self, vit_sonar_backbone):
        """Test ViT forward pass with separate processing (Sonar)."""
        template = torch.randn(2, 1, 128, 128)
        search = torch.randn(2, 1, 256, 256)

        outputs = vit_sonar_backbone(template, search, return_separate=True)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['template_features'].shape[0] == 2
        assert outputs['search_features'].shape[0] == 2

    def test_vit_forward_combined(self, vit_rgb_backbone):
        """Test ViT forward pass with combined processing."""
        template = torch.randn(2, 3, 224, 224)
        search = torch.randn(2, 3, 224, 224)

        outputs = vit_rgb_backbone(template, search, return_separate=False)

        assert 'features' in outputs
        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['features'].shape[0] == 4  # 2*B

    def test_vit_feature_dim(self, vit_rgb_backbone):
        """Test feature dimension getter."""
        dim = vit_rgb_backbone.get_feature_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_vit_freeze_backbone(self):
        """Test freezing backbone parameters."""
        backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            freeze_backbone=True)

        # Check if parameters are frozen
        for param in backbone.vit.parameters():
            assert not param.requires_grad

    def test_channel_adapter_applied(self, vit_sonar_backbone):
        """Test that channel adapter is applied in forward pass."""
        template = torch.randn(2, 1, 224, 224)
        search = torch.randn(2, 1, 224, 224)

        # Verify channel adapter exists and has correct shape
        assert vit_sonar_backbone.channel_adapter is not None
        assert isinstance(vit_sonar_backbone.channel_adapter, nn.Conv2d)

        # Test channel adapter converts 1ch to 3ch
        adapted = vit_sonar_backbone.channel_adapter(template)
        assert adapted.shape == (2, 3, 224, 224)

        # Verify forward pass works with 1ch input
        outputs = vit_sonar_backbone(template, search)
        assert 'template_features' in outputs
        assert 'search_features' in outputs


class TestBackboneFactory:
    """Tests for Backbone Factory."""

    def test_factory_create_vit_rgb(self):
        """Test creating ViT RGB backbone via factory."""
        backbone = BackboneFactory.create('vit_rgb',
                                         model_name='vit_tiny_patch16_224',
                                         pretrained=False)

        assert isinstance(backbone, ViTBackbone)
        assert backbone.in_channels == 3

    def test_factory_create_vit_sonar(self):
        """Test creating ViT Sonar backbone via factory."""
        backbone = BackboneFactory.create('vit_sonar',
                                         model_name='vit_tiny_patch16_224',
                                         pretrained=False,
                                         in_channels=1)

        assert isinstance(backbone, ViTBackbone)
        assert backbone.in_channels == 1
        assert backbone.channel_adapter is not None

    def test_factory_invalid_type(self):
        """Test factory with invalid backbone type."""
        with pytest.raises(ValueError):
            BackboneFactory.create('invalid_type')

    def test_factory_from_config_vit_rgb(self):
        """Test creating backbone from config dict."""
        config = {
            'type': 'vit_rgb',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False
        }
        backbone = BackboneFactory.create_from_config(config)
        assert isinstance(backbone, ViTBackbone)

    def test_factory_from_config_vit_sonar(self):
        """Test creating sonar backbone from config dict."""
        config = {
            'type': 'vit_sonar',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False,
            'in_channels': 1
        }
        backbone = BackboneFactory.create_from_config(config)
        assert isinstance(backbone, ViTBackbone)
        assert backbone.in_channels == 1

    def test_factory_list_available_models(self):
        """Test listing available models."""
        models_rgb = BackboneFactory.list_available_models('vit_rgb')
        models_sonar = BackboneFactory.list_available_models('vit_sonar')

        assert isinstance(models_rgb, list)
        assert isinstance(models_sonar, list)
        assert len(models_rgb) > 0
        assert len(models_sonar) > 0
        assert 'vit_base_patch16_224' in models_rgb
        assert 'vit_base_patch16_224' in models_sonar


class TestMultiModalBackbone:
    """Tests for Multi-Modal Backbone."""

    @pytest.fixture
    def multimodal_backbone(self):
        """Create multi-modal backbone for testing."""
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False
        }
        sonar_config = {
            'type': 'vit_sonar',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False,
            'in_channels': 1
        }
        return MultiModalBackbone(rgb_config, sonar_config)

    def test_multimodal_initialization(self, multimodal_backbone):
        """Test multi-modal backbone initialization."""
        assert multimodal_backbone.rgb_backbone is not None
        assert multimodal_backbone.sonar_encoder is not None
        assert isinstance(multimodal_backbone.rgb_backbone, ViTBackbone)
        assert isinstance(multimodal_backbone.sonar_encoder, ViTBackbone)

    def test_multimodal_forward(self, multimodal_backbone):
        """Test multi-modal forward pass."""
        rgb_template = torch.randn(2, 3, 224, 224)
        rgb_search = torch.randn(2, 3, 224, 224)
        sonar_template = torch.randn(2, 1, 224, 224)
        sonar_search = torch.randn(2, 1, 224, 224)

        outputs = multimodal_backbone(rgb_template, rgb_search,
                                     sonar_template, sonar_search)

        assert 'rgb' in outputs
        assert 'sonar' in outputs
        assert outputs['rgb'] is not None
        assert outputs['sonar'] is not None

    def test_multimodal_feature_dims(self, multimodal_backbone):
        """Test getting feature dimensions."""
        dims = multimodal_backbone.get_feature_dims()

        assert 'rgb' in dims
        assert 'sonar' in dims
        assert isinstance(dims['rgb'], int)
        assert isinstance(dims['sonar'], int)


class TestBackboneBuildFunctions:
    """Tests for convenience build functions."""

    def test_build_backbone_vit_rgb(self):
        """Test build_backbone for ViT RGB."""
        backbone = build_backbone('vit_rgb', pretrained=False)
        assert isinstance(backbone, ViTBackbone)
        assert backbone.in_channels == 3

    def test_build_backbone_vit_sonar(self):
        """Test build_backbone for ViT Sonar."""
        backbone = build_backbone('vit_sonar', pretrained=False, in_channels=1)
        assert isinstance(backbone, ViTBackbone)
        assert backbone.in_channels == 1

    def test_build_multimodal_backbone(self):
        """Test building multi-modal backbone."""
        backbone = build_multimodal_backbone(
            rgb_type='vit_rgb',
            rgb_model='vit_tiny_patch16_224',
            sonar_type='vit_sonar',
            sonar_model='vit_tiny_patch16_224',
            rgb_pretrained=False,
            sonar_pretrained=False
        )

        assert isinstance(backbone, MultiModalBackbone)
        assert backbone.rgb_backbone is not None
        assert backbone.sonar_encoder is not None


class TestBackboneIntegration:
    """Integration tests for backbone compatibility."""

    def test_rgb_sonar_feature_compatibility(self):
        """Test that RGB and Sonar backbones produce compatible features."""
        rgb_backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=3)

        sonar_backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=1)

        # Test with same spatial size
        template = torch.randn(2, 3, 224, 224)
        search = torch.randn(2, 3, 224, 224)

        sonar_template = torch.randn(2, 1, 224, 224)
        sonar_search = torch.randn(2, 1, 224, 224)

        rgb_out = rgb_backbone(template, search)
        sonar_out = sonar_backbone(sonar_template, sonar_search)

        # Features should have same dimension (except batch concat)
        assert rgb_out['template_features'].shape[1:] == sonar_out['template_features'].shape[1:]
        assert rgb_out['search_features'].shape[1:] == sonar_out['search_features'].shape[1:]

    def test_shared_architecture_different_inputs(self):
        """Test that same ViT architecture works for different input channels."""
        # Create two backbones with same architecture but different input channels
        rgb_backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=3,
            embed_dim=192)

        sonar_backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            in_channels=1,
            embed_dim=192)

        # Both should have same embed_dim despite different input channels
        assert rgb_backbone.embed_dim == sonar_backbone.embed_dim
        assert rgb_backbone.get_feature_dim() == sonar_backbone.get_feature_dim()

    def test_pretrained_weight_sharing(self):
        """Test that RGB and Sonar can both use pretrained weights."""
        # This test verifies the architecture supports loading same pretrained weights
        rgb_backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,  # Set to False for testing
            in_channels=3)

        sonar_backbone = ViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,  # Set to False for testing
            in_channels=1)

        # Both should use the same underlying ViT architecture
        assert rgb_backbone.model_name == sonar_backbone.model_name
        # Sonar should have channel adapter
        assert sonar_backbone.channel_adapter is not None
        assert rgb_backbone.channel_adapter is None
