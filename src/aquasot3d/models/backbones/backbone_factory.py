"""
Backbone Factory for Creating RGB and Sonar Encoders

This module provides a unified factory interface for creating various backbone
networks (ViT, ConvNeXt, Sonar encoders) with consistent configuration.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
import logging

from aquasot3d.core import BACKBONE_REGISTRY
from aquasot3d.models.backbones.vit_rgb import ViTRGBBackbone
from aquasot3d.models.backbones.convnext_rgb import ConvNeXtRGBBackbone
from aquasot3d.models.backbones.sonar_encoder import SonarEncoder


logger = logging.getLogger(__name__)


class DummyBackbone(nn.Module):
    """
    Dummy backbone for unit testing without loading full models.

    Returns fixed-size random features for fast testing of downstream components
    without requiring actual pretrained weights or heavy computation.

    Matches the interface of real backbones (ViTRGBBackbone, SonarEncoder):
    - forward(template, search, return_separate=False)
    - Returns {'template_features': ..., 'search_features': ...} or {'features': ...}

    Args:
        output_dim: Feature dimension to output
        spatial_size: Spatial resolution of output features (H=W)
    """

    def __init__(self, output_dim: int = 256, spatial_size: int = 14):
        super().__init__()
        self.output_dim = output_dim
        self.spatial_size = spatial_size
        # Minimal parameters to make it a valid nn.Module
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        return_separate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Return dummy features with correct shape.

        Args:
            template: Template frame (B, C, H_t, W_t)
            search: Search frame (B, C, H_s, W_s)
            return_separate: Whether to return separate template/search features

        Returns:
            Dictionary with dummy features matching real backbone interface
        """
        B = template.shape[0]
        device = template.device

        # Generate deterministic random features
        # Use fixed seed for reproducibility in tests
        gen_template = torch.Generator(device='cpu').manual_seed(42)
        gen_search = torch.Generator(device='cpu').manual_seed(43)

        template_features = torch.randn(
            B, self.output_dim, self.spatial_size, self.spatial_size,
            generator=gen_template
        ).to(device)

        search_features = torch.randn(
            B, self.output_dim, self.spatial_size, self.spatial_size,
            generator=gen_search
        ).to(device)

        if return_separate:
            # Return separate template and search features
            return {
                'template_features': template_features,
                'search_features': search_features,
            }
        else:
            # Return concatenated features (typical for tracking)
            return {
                'features': search_features,  # Use search features as default
                'template_features': template_features,  # Also include for compatibility
                'search_features': search_features,
            }

    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.output_dim


class BackboneFactory:
    """
    Factory class for creating backbone networks.

    Supports creating various backbone types:
    - RGB backbones: ViT, ConvNeXt
    - Sonar encoders: ViT-based, CNN-based

    Example:
        >>> factory = BackboneFactory()
        >>> rgb_backbone = factory.create('vit_rgb', model_name='vit_base_patch16_224')
        >>> sonar_backbone = factory.create('sonar', encoder_type='vit')
    """

    @staticmethod
    def create(
        backbone_type: Literal['vit_rgb', 'convnext_rgb', 'sonar'],
        **kwargs
    ) -> nn.Module:
        """
        Create backbone network based on type.

        Args:
            backbone_type: Type of backbone to create
            **kwargs: Additional arguments for specific backbone

        Returns:
            Initialized backbone module

        Raises:
            ValueError: If backbone_type is not supported

        Example:
            >>> backbone = BackboneFactory.create('vit_rgb', pretrained=True)
        """
        if backbone_type == 'vit_rgb':
            return BackboneFactory.create_vit_rgb(**kwargs)
        elif backbone_type == 'convnext_rgb':
            return BackboneFactory.create_convnext_rgb(**kwargs)
        elif backbone_type == 'sonar':
            return BackboneFactory.create_sonar_encoder(**kwargs)
        else:
            raise ValueError(
                f"Unsupported backbone_type: {backbone_type}. "
                f"Supported types: ['vit_rgb', 'convnext_rgb', 'sonar']"
            )

    @staticmethod
    def create_vit_rgb(
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        **kwargs
    ) -> ViTRGBBackbone:
        """
        Create ViT RGB backbone.

        Args:
            model_name: timm model name for ViT
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments for ViTRGBBackbone

        Returns:
            ViTRGBBackbone instance

        Example:
            >>> backbone = BackboneFactory.create_vit_rgb('vit_base_patch16_224')
        """
        logger.info(f"Creating ViT RGB backbone: {model_name}")
        return ViTRGBBackbone(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def create_convnext_rgb(
        model_name: str = 'convnext_base',
        pretrained: bool = True,
        **kwargs
    ) -> ConvNeXtRGBBackbone:
        """
        Create ConvNeXt RGB backbone.

        Args:
            model_name: timm model name for ConvNeXt
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments for ConvNeXtRGBBackbone

        Returns:
            ConvNeXtRGBBackbone instance

        Example:
            >>> backbone = BackboneFactory.create_convnext_rgb('convnext_base')
        """
        logger.info(f"Creating ConvNeXt RGB backbone: {model_name}")
        return ConvNeXtRGBBackbone(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def create_sonar_encoder(
        encoder_type: Literal['vit', 'conv'] = 'vit',
        model_name: str = 'vit_tiny_patch16_224',
        pretrained: bool = False,
        **kwargs
    ) -> SonarEncoder:
        """
        Create Sonar encoder.

        Args:
            encoder_type: Type of encoder architecture ('vit' or 'conv')
            model_name: timm model name
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments for SonarEncoder

        Returns:
            SonarEncoder instance

        Example:
            >>> encoder = BackboneFactory.create_sonar_encoder('vit')
        """
        logger.info(f"Creating Sonar encoder: type={encoder_type}, model={model_name}")
        return SonarEncoder(
            encoder_type=encoder_type,
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> nn.Module:
        """
        Create backbone from configuration dictionary.

        Args:
            config: Configuration dictionary with 'type' and parameters

        Returns:
            Initialized backbone module

        Example:
            >>> config = {
            ...     'type': 'vit_rgb',
            ...     'model_name': 'vit_base_patch16_224',
            ...     'pretrained': True
            ... }
            >>> backbone = BackboneFactory.create_from_config(config)
        """
        # Handle disabled backbone (type=null or config is None/empty)
        if config is None or not config:
            return None

        config = config.copy()
        backbone_type = config.pop('type', None)

        # Handle null type: return DummyBackbone for testing
        if backbone_type is None or backbone_type == 'null':
            # Create dummy backbone with configurable dimensions
            output_dim = config.get('output_dim', 256)
            spatial_size = config.get('spatial_size', 14)
            logger.info(f"Creating DummyBackbone for testing: output_dim={output_dim}, spatial_size={spatial_size}")
            return DummyBackbone(output_dim=output_dim, spatial_size=spatial_size)

        return BackboneFactory.create(backbone_type, **config)

    @staticmethod
    def list_available_models(backbone_type: str) -> list:
        """
        List available pretrained models for given backbone type.

        Args:
            backbone_type: Type of backbone

        Returns:
            List of available model names

        Example:
            >>> models = BackboneFactory.list_available_models('vit_rgb')
        """
        if backbone_type == 'vit_rgb':
            return [
                'vit_tiny_patch16_224',
                'vit_small_patch16_224',
                'vit_base_patch16_224',
                'vit_base_patch16_384',
                'vit_large_patch16_224',
                'vit_large_patch16_384',
            ]
        elif backbone_type == 'convnext_rgb':
            return [
                'convnext_tiny',
                'convnext_small',
                'convnext_base',
                'convnext_large',
                'convnext_xlarge',
            ]
        elif backbone_type == 'sonar':
            return [
                'vit_tiny_patch16_224',
                'vit_small_patch16_224',
                'resnet18',
                'resnet34',
                'resnet50',
                'efficientnet_b0',
                'efficientnet_b1',
            ]
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")


class MultiModalBackbone(nn.Module):
    """
    Multi-modal backbone that combines RGB and Sonar encoders.

    This module manages separate encoders for RGB and Sonar modalities
    and provides a unified interface for multi-modal feature extraction.

    Args:
        rgb_config: Configuration dict for RGB backbone
        sonar_config: Configuration dict for Sonar encoder

    Example:
        >>> rgb_config = {'type': 'vit_rgb', 'model_name': 'vit_base_patch16_224'}
        >>> sonar_config = {'type': 'sonar', 'encoder_type': 'vit'}
        >>> backbone = MultiModalBackbone(rgb_config, sonar_config)
        >>> rgb_t, rgb_s = torch.randn(4, 3, 128, 128), torch.randn(4, 3, 256, 256)
        >>> son_t, son_s = torch.randn(4, 1, 682, 376), torch.randn(4, 1, 682, 376)
        >>> outputs = backbone(rgb_t, rgb_s, son_t, son_s)
    """

    def __init__(
        self,
        rgb_config: Optional[Dict[str, Any]],
        sonar_config: Optional[Dict[str, Any]]
    ):
        super().__init__()

        # Create backbones (can be None for single-modal training)
        self.rgb_backbone = BackboneFactory.create_from_config(rgb_config) if rgb_config else None
        self.sonar_encoder = BackboneFactory.create_from_config(sonar_config) if sonar_config else None

        # Log initialization status
        if self.rgb_backbone and self.sonar_encoder:
            logger.info("Initialized MultiModalBackbone with RGB and Sonar encoders")
        elif self.rgb_backbone:
            logger.info("Initialized MultiModalBackbone with RGB encoder only (sonar disabled)")
        elif self.sonar_encoder:
            logger.info("Initialized MultiModalBackbone with Sonar encoder only (RGB disabled)")
        else:
            raise ValueError("At least one modality (RGB or Sonar) must be enabled")

    def forward(
        self,
        rgb_template: torch.Tensor,
        rgb_search: torch.Tensor,
        sonar_template: torch.Tensor,
        sonar_search: torch.Tensor,
        return_separate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both modalities.

        Args:
            rgb_template: RGB template frame (B, 3, H_t, W_t)
            rgb_search: RGB search frame (B, 3, H_s, W_s)
            sonar_template: Sonar template frame (B, 1, H_t, W_t)
            sonar_search: Sonar search frame (B, 1, H_s, W_s)
            return_separate: Whether to process modalities separately

        Returns:
            Dictionary containing features from both modalities
        """
        # Extract RGB features (if RGB encoder is available)
        rgb_outputs = None
        if self.rgb_backbone is not None:
            rgb_outputs = self.rgb_backbone(
                rgb_template, rgb_search, return_separate=return_separate
            )

        # Extract Sonar features (if Sonar encoder is available)
        sonar_outputs = None
        if self.sonar_encoder is not None:
            sonar_outputs = self.sonar_encoder(
                sonar_template, sonar_search, return_separate=return_separate
            )

        # Combine outputs
        outputs = {
            'rgb': rgb_outputs,
            'sonar': sonar_outputs,
        }

        return outputs

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for both modalities."""
        return {
            'rgb': self.rgb_backbone.get_feature_dim(),
            'sonar': self.sonar_encoder.get_feature_dim(),
        }


def build_backbone(
    backbone_type: str,
    model_name: Optional[str] = None,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Convenience function to build backbone.

    Args:
        backbone_type: Type of backbone ('vit_rgb', 'convnext_rgb', 'sonar')
        model_name: Model name (optional, uses default if None)
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments

    Returns:
        Backbone module

    Example:
        >>> backbone = build_backbone('vit_rgb', pretrained=True)
    """
    # Set default model names if not provided
    if model_name is None:
        defaults = {
            'vit_rgb': 'vit_base_patch16_224',
            'convnext_rgb': 'convnext_base',
            'sonar': 'vit_tiny_patch16_224',
        }
        model_name = defaults.get(backbone_type, 'vit_base_patch16_224')

    return BackboneFactory.create(
        backbone_type,
        model_name=model_name,
        pretrained=pretrained,
        **kwargs
    )


def build_multimodal_backbone(
    rgb_type: str = 'vit_rgb',
    rgb_model: str = 'vit_base_patch16_224',
    sonar_type: str = 'vit',
    sonar_model: str = 'vit_tiny_patch16_224',
    rgb_pretrained: bool = True,
    sonar_pretrained: bool = False,
    **kwargs
) -> MultiModalBackbone:
    """
    Build multi-modal backbone with RGB and Sonar encoders.

    Args:
        rgb_type: Type of RGB backbone
        rgb_model: RGB model name
        sonar_type: Type of Sonar encoder
        sonar_model: Sonar model name
        rgb_pretrained: Whether RGB backbone is pretrained
        sonar_pretrained: Whether Sonar encoder is pretrained
        **kwargs: Additional arguments

    Returns:
        MultiModalBackbone instance

    Example:
        >>> backbone = build_multimodal_backbone(
        ...     rgb_type='vit_rgb',
        ...     sonar_type='vit'
        ... )
    """
    rgb_config = {
        'type': rgb_type,
        'model_name': rgb_model,
        'pretrained': rgb_pretrained,
    }

    sonar_config = {
        'type': 'sonar',
        'encoder_type': sonar_type,
        'model_name': sonar_model,
        'pretrained': sonar_pretrained,
    }

    return MultiModalBackbone(rgb_config, sonar_config)
