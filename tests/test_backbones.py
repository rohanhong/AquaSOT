"""
Unit Tests for Backbone Networks

Tests for ViT RGB, ConvNeXt RGB, Sonar Encoder, and Backbone Factory.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# Import backbone modules
from aquasot3d.models.backbones import (
    ViTRGBBackbone,
    ConvNeXtRGBBackbone,
    SonarEncoder,
    BackboneFactory,
    MultiModalBackbone,
    build_backbone,
    build_multimodal_backbone,
)


class TestViTRGBBackbone:
    """Tests for ViT RGB Backbone."""

    @pytest.fixture
    def vit_backbone(self):
        """Create ViT backbone for testing."""
        return ViTRGBBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            img_size=224,
            embed_dim=192,
            depth=12,
            num_heads=3,
        )

    def test_vit_initialization(self, vit_backbone):
        """Test ViT backbone initialization."""
        assert isinstance(vit_backbone, ViTRGBBackbone)
        assert vit_backbone.embed_dim > 0
        assert vit_backbone.patch_size > 0

    def test_vit_forward_separate(self, vit_backbone):
        """Test ViT forward pass with separate processing."""
        template = torch.randn(2, 3, 128, 128)
        search = torch.randn(2, 3, 256, 256)

        outputs = vit_backbone(template, search, return_separate=True)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['template_features'].shape[0] == 2
        assert outputs['search_features'].shape[0] == 2

    def test_vit_forward_combined(self, vit_backbone):
        """Test ViT forward pass with combined processing."""
        template = torch.randn(2, 3, 224, 224)
        search = torch.randn(2, 3, 224, 224)

        outputs = vit_backbone(template, search, return_separate=False)

        assert 'features' in outputs
        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert outputs['features'].shape[0] == 4  # 2*B

    def test_vit_feature_dim(self, vit_backbone):
        """Test feature dimension getter."""
        dim = vit_backbone.get_feature_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_vit_freeze_backbone(self):
        """Test freezing backbone parameters."""
        backbone = ViTRGBBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            freeze_backbone=True,
        )

        # Check if parameters are frozen
        for param in backbone.vit.parameters():
            assert not param.requires_grad


class TestConvNeXtRGBBackbone:
    """Tests for ConvNeXt RGB Backbone."""

    @pytest.fixture
    def convnext_backbone(self):
        """Create ConvNeXt backbone for testing."""
        return ConvNeXtRGBBackbone(
            model_name='convnext_tiny',
            pretrained=False,
            out_indices=(0, 1, 2, 3),
        )

    def test_convnext_initialization(self, convnext_backbone):
        """Test ConvNeXt backbone initialization."""
        assert isinstance(convnext_backbone, ConvNeXtRGBBackbone)
        assert len(convnext_backbone.feature_dims) > 0
        assert len(convnext_backbone.feature_strides) > 0

    def test_convnext_forward_separate(self, convnext_backbone):
        """Test ConvNeXt forward pass with separate processing."""
        template = torch.randn(2, 3, 128, 128)
        search = torch.randn(2, 3, 256, 256)

        outputs = convnext_backbone(template, search, return_separate=True)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert 'template_multi_scale' in outputs
        assert 'search_multi_scale' in outputs
        assert isinstance(outputs['template_multi_scale'], list)

    def test_convnext_forward_combined(self, convnext_backbone):
        """Test ConvNeXt forward pass with combined processing."""
        template = torch.randn(2, 3, 224, 224)
        search = torch.randn(2, 3, 224, 224)

        outputs = convnext_backbone(template, search, return_separate=False)

        assert 'features' in outputs
        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert 'multi_scale_features' in outputs
        assert 'feature_info' in outputs

    def test_convnext_feature_info(self, convnext_backbone):
        """Test feature dimension and stride getters."""
        dims = convnext_backbone.get_feature_dims()
        strides = convnext_backbone.get_feature_strides()
        final_dim = convnext_backbone.get_final_feature_dim()

        assert isinstance(dims, list)
        assert isinstance(strides, list)
        assert len(dims) == len(strides)
        assert final_dim == dims[-1]

    def test_convnext_freeze_stages(self):
        """Test freezing early stages."""
        backbone = ConvNeXtRGBBackbone(
            model_name='convnext_tiny',
            pretrained=False,
            freeze_stages=2,
        )

        assert backbone.freeze_stages == 2


class TestSonarEncoder:
    """Tests for Sonar Encoder."""

    @pytest.fixture
    def sonar_encoder_vit(self):
        """Create ViT-based Sonar encoder for testing."""
        return SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            pretrained=False,
            use_logpolar=True,
            learnable_logpolar=True,
            img_size=(682, 376),
        )

    @pytest.fixture
    def sonar_encoder_conv(self):
        """Create Conv-based Sonar encoder for testing."""
        return SonarEncoder(
            encoder_type='conv',
            model_name='resnet18',
            pretrained=False,
            use_logpolar=True,
            img_size=(682, 376),
        )

    def test_sonar_initialization_vit(self, sonar_encoder_vit):
        """Test Sonar encoder initialization with ViT."""
        assert isinstance(sonar_encoder_vit, SonarEncoder)
        assert sonar_encoder_vit.encoder_type == 'vit'
        assert sonar_encoder_vit.use_logpolar
        assert hasattr(sonar_encoder_vit, 'logpolar_transform')
        assert hasattr(sonar_encoder_vit, 'geometry_handler')

    def test_sonar_initialization_conv(self, sonar_encoder_conv):
        """Test Sonar encoder initialization with Conv."""
        assert isinstance(sonar_encoder_conv, SonarEncoder)
        assert sonar_encoder_conv.encoder_type == 'conv'

    def test_sonar_forward_separate(self, sonar_encoder_vit):
        """Test Sonar forward pass with separate processing."""
        template = torch.randn(2, 1, 682, 376)  # Grayscale sonar
        search = torch.randn(2, 1, 682, 376)

        outputs = sonar_encoder_vit(template, search, return_separate=True)

        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert 'geometry_info' in outputs

    def test_sonar_forward_combined(self, sonar_encoder_vit):
        """Test Sonar forward pass with combined processing."""
        template = torch.randn(2, 1, 682, 376)
        search = torch.randn(2, 1, 682, 376)

        outputs = sonar_encoder_vit(template, search, return_separate=False)

        assert 'features' in outputs
        assert 'template_features' in outputs
        assert 'search_features' in outputs
        assert 'geometry_info' in outputs

    def test_sonar_radial_profile_extraction(self, sonar_encoder_vit):
        """Test radial profile extraction for depth proxy."""
        template = torch.randn(2, 1, 682, 376)
        search = torch.randn(2, 1, 682, 376)

        outputs = sonar_encoder_vit(template, search)

        if sonar_encoder_vit.extract_radial_profile:
            assert 'template_radial_profile' in outputs
            assert 'search_radial_profile' in outputs

    def test_sonar_logpolar_disabled(self):
        """Test Sonar encoder without log-polar transformation."""
        encoder = SonarEncoder(
            encoder_type='vit',
            pretrained=False,
            use_logpolar=False,
        )

        template = torch.randn(2, 1, 682, 376)
        search = torch.randn(2, 1, 682, 376)

        outputs = encoder(template, search)
        assert 'features' in outputs


class TestBackboneFactory:
    """Tests for Backbone Factory."""

    def test_factory_create_vit_rgb(self):
        """Test factory creating ViT RGB backbone."""
        backbone = BackboneFactory.create(
            'vit_rgb',
            model_name='vit_tiny_patch16_224',
            pretrained=False,
        )

        assert isinstance(backbone, ViTRGBBackbone)

    def test_factory_create_convnext_rgb(self):
        """Test factory creating ConvNeXt RGB backbone."""
        backbone = BackboneFactory.create(
            'convnext_rgb',
            model_name='convnext_tiny',
            pretrained=False,
        )

        assert isinstance(backbone, ConvNeXtRGBBackbone)

    def test_factory_create_sonar(self):
        """Test factory creating Sonar encoder."""
        encoder = BackboneFactory.create(
            'sonar',
            encoder_type='vit',
            pretrained=False,
        )

        assert isinstance(encoder, SonarEncoder)

    def test_factory_invalid_type(self):
        """Test factory with invalid backbone type."""
        with pytest.raises(ValueError):
            BackboneFactory.create('invalid_type')

    def test_factory_from_config(self):
        """Test creating backbone from config dict."""
        config = {
            'type': 'vit_rgb',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False,
        }

        backbone = BackboneFactory.create_from_config(config)
        assert isinstance(backbone, ViTRGBBackbone)

    def test_factory_list_available_models(self):
        """Test listing available models."""
        vit_models = BackboneFactory.list_available_models('vit_rgb')
        convnext_models = BackboneFactory.list_available_models('convnext_rgb')
        sonar_models = BackboneFactory.list_available_models('sonar')

        assert isinstance(vit_models, list)
        assert isinstance(convnext_models, list)
        assert isinstance(sonar_models, list)
        assert len(vit_models) > 0
        assert len(convnext_models) > 0
        assert len(sonar_models) > 0


class TestMultiModalBackbone:
    """Tests for Multi-Modal Backbone."""

    @pytest.fixture
    def multimodal_backbone(self):
        """Create multi-modal backbone for testing."""
        rgb_config = {
            'type': 'vit_rgb',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False,
        }
        sonar_config = {
            'type': 'sonar',
            'encoder_type': 'vit',
            'pretrained': False,
        }

        return MultiModalBackbone(rgb_config, sonar_config)

    def test_multimodal_initialization(self, multimodal_backbone):
        """Test multi-modal backbone initialization."""
        assert isinstance(multimodal_backbone, MultiModalBackbone)
        assert hasattr(multimodal_backbone, 'rgb_backbone')
        assert hasattr(multimodal_backbone, 'sonar_encoder')

    def test_multimodal_forward(self, multimodal_backbone):
        """Test multi-modal forward pass."""
        rgb_template = torch.randn(2, 3, 224, 224)
        rgb_search = torch.randn(2, 3, 224, 224)
        sonar_template = torch.randn(2, 1, 682, 376)
        sonar_search = torch.randn(2, 1, 682, 376)

        outputs = multimodal_backbone(
            rgb_template, rgb_search,
            sonar_template, sonar_search
        )

        assert 'rgb' in outputs
        assert 'sonar' in outputs
        assert isinstance(outputs['rgb'], dict)
        assert isinstance(outputs['sonar'], dict)

    def test_multimodal_feature_dims(self, multimodal_backbone):
        """Test getting feature dimensions."""
        dims = multimodal_backbone.get_feature_dims()

        assert 'rgb' in dims
        assert 'sonar' in dims
        assert isinstance(dims['rgb'], int)
        assert isinstance(dims['sonar'], int)


class TestBackboneBuildFunctions:
    """Tests for convenience build functions."""

    def test_build_backbone_vit(self):
        """Test build_backbone with ViT."""
        backbone = build_backbone('vit_rgb', pretrained=False)
        assert isinstance(backbone, ViTRGBBackbone)

    def test_build_backbone_convnext(self):
        """Test build_backbone with ConvNeXt."""
        backbone = build_backbone('convnext_rgb', pretrained=False)
        assert isinstance(backbone, ConvNeXtRGBBackbone)

    def test_build_backbone_sonar(self):
        """Test build_backbone with Sonar."""
        backbone = build_backbone('sonar', pretrained=False)
        assert isinstance(backbone, SonarEncoder)

    def test_build_multimodal_backbone(self):
        """Test build_multimodal_backbone."""
        backbone = build_multimodal_backbone(
            rgb_type='vit_rgb',
            rgb_model='vit_tiny_patch16_224',
            sonar_type='vit',
            sonar_model='vit_tiny_patch16_224',
            rgb_pretrained=False,
            sonar_pretrained=False,
        )

        assert isinstance(backbone, MultiModalBackbone)


# Integration tests
class TestBackboneIntegration:
    """Integration tests for backbone modules."""

    def test_vit_to_convnext_compatibility(self):
        """Test that ViT and ConvNeXt outputs are compatible."""
        vit = ViTRGBBackbone(model_name='vit_tiny_patch16_224', pretrained=False)
        convnext = ConvNeXtRGBBackbone(model_name='convnext_tiny', pretrained=False)

        template = torch.randn(2, 3, 224, 224)
        search = torch.randn(2, 3, 224, 224)

        vit_out = vit(template, search)
        convnext_out = convnext(template, search)

        # Both should have 'features' key
        assert 'features' in vit_out
        assert 'features' in convnext_out

    def test_rgb_sonar_feature_compatibility(self):
        """Test that RGB and Sonar features are compatible."""
        rgb_backbone = ViTRGBBackbone(model_name='vit_tiny_patch16_224', pretrained=False)
        sonar_encoder = SonarEncoder(encoder_type='vit', pretrained=False)

        rgb_template = torch.randn(2, 3, 224, 224)
        rgb_search = torch.randn(2, 3, 224, 224)
        sonar_template = torch.randn(2, 1, 682, 376)
        sonar_search = torch.randn(2, 1, 682, 376)

        rgb_out = rgb_backbone(rgb_template, rgb_search)
        sonar_out = sonar_encoder(sonar_template, sonar_search)

        # Both should have template and search features
        assert 'template_features' in rgb_out
        assert 'template_features' in sonar_out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
