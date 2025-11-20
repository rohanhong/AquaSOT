"""
Backbone Factory for Creating RGB and Sonar Encoders

This module provides a unified factory interface for creating ViT backbone
networks with consistent configuration.

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
import logging

from aquasot3d.core import BACKBONE_REGISTRY
from aquasot3d.models.backbones.vit import ViTBackbone


logger = logging.getLogger(__name__)


class DummyBackbone(nn.Module):
    """
    Dummy backbone for unit testing without loading full models.

    Returns fixed-size random features for fast testing of downstream components
    without requiring actual pretrained weights or heavy computation.

    ARCHITECTURE CHANGE (Audit 2025112104):
    Updated to match template-search ViTBackbone interface.

    Matches the interface of real backbones (ViTBackbone):
    - forward(template, search) -> (feat_template, feat_search)
    - Returns Tuple[Tensor, Tensor] with shapes [B, N, D]

    Args:
        output_dim: Feature dimension to output (D)
        num_patches_template: Number of template patches (N_z, default: 256)
        num_patches_search: Number of search patches (N_x, default: 1024)
    """

    def __init__(
        self,
        output_dim: int = 384,
        num_patches_template: int = 256,
        num_patches_search: int = 1024
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches_template = num_patches_template
        self.num_patches_search = num_patches_search
        # Minimal parameters to make it a valid nn.Module
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor
    ) -> tuple:
        """
        Return dummy features with correct shape.

        Args:
            template: Template frame (B, C, template_size, template_size)
            search: Search frame (B, C, search_size, search_size)

        Returns:
            Tuple of (feat_template, feat_search):
                - feat_template: [B, N_z, D]
                - feat_search: [B, N_x, D]
        """
        B = template.shape[0]
        device = template.device

        # Generate deterministic random features
        # Use fixed seed for reproducibility in tests
        gen_template = torch.Generator(device='cpu').manual_seed(42)
        gen_search = torch.Generator(device='cpu').manual_seed(43)

        feat_template = torch.randn(
            B, self.num_patches_template, self.output_dim,
            generator=gen_template
        ).to(device)

        feat_search = torch.randn(
            B, self.num_patches_search, self.output_dim,
            generator=gen_search
        ).to(device)

        return feat_template, feat_search

    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.output_dim


class BackboneFactory:
    """
    Factory class for creating backbone networks.

    Supports creating ViT backbones for RGB and Sonar:
    - RGB backbones: vit_rgb
    - Sonar backbones: vit_sonar (unified ViT with 1-channel adapter)

    Example:
        >>> factory = BackboneFactory()
        >>> rgb_backbone = factory.create('vit_rgb', model_name='vit_base_patch16_224')
        >>> sonar_backbone = factory.create('vit_sonar', model_name='vit_base_patch16_224', in_channels=1)
    """

    @staticmethod
    def create(
        backbone_type: Literal['vit_rgb', 'vit_sonar'],
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
            >>> sonar_backbone = BackboneFactory.create('vit_sonar', in_channels=1, pretrained=True)
        """
        if backbone_type == 'vit_rgb':
            return BackboneFactory.create_vit_rgb(**kwargs)
        elif backbone_type == 'vit_sonar':
            return BackboneFactory.create_vit_sonar(**kwargs)
        else:
            raise ValueError(
                f"Unsupported backbone_type: {backbone_type}. "
                f"Supported types: ['vit_rgb', 'vit_sonar']"
            )

    @staticmethod
    def create_vit_rgb(
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        **kwargs
    ) -> ViTBackbone:
        """
        Create ViT RGB backbone.

        Args:
            model_name: timm model name for ViT
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments for ViTBackbone

        Returns:
            ViTBackbone instance

        Example:
            >>> backbone = BackboneFactory.create_vit_rgb('vit_base_patch16_224')
        """
        logger.info(f"Creating ViT RGB backbone: {model_name}")
        return ViTBackbone(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def create_vit_sonar(
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        in_channels: int = 1,
        **kwargs
    ) -> ViTBackbone:
        """
        Create ViT Sonar backbone (unified ViT with 1-channel adapter).

        Args:
            model_name: timm model name for ViT
            pretrained: Whether to load pretrained weights
            in_channels: Input channels (1 for sonar)
            **kwargs: Additional arguments for ViTBackbone

        Returns:
            ViTBackbone instance configured for sonar

        Example:
            >>> sonar_backbone = BackboneFactory.create_vit_sonar('vit_base_patch16_224')
        """
        logger.info(f"Creating ViT Sonar backbone: {model_name} (in_channels={in_channels})")
        return ViTBackbone(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
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
            output_dim = config.get('output_dim', 384)
            num_patches_template = config.get('num_patches_template', 256)
            num_patches_search = config.get('num_patches_search', 1024)
            logger.info(
                f"Creating DummyBackbone for testing: output_dim={output_dim}, "
                f"num_patches_template={num_patches_template}, num_patches_search={num_patches_search}"
            )
            return DummyBackbone(
                output_dim=output_dim,
                num_patches_template=num_patches_template,
                num_patches_search=num_patches_search
            )

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
        if backbone_type in ['vit_rgb', 'vit_sonar']:
            return [
                'vit_tiny_patch16_224',
                'vit_small_patch16_224',
                'vit_base_patch16_224',
                'vit_base_patch16_384',
                'vit_large_patch16_224',
                'vit_large_patch16_384',
            ]
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")


class MultiModalBackbone(nn.Module):
    """
    Multi-modal backbone that combines RGB and Sonar encoders for template-search tracking.

    This module manages separate encoders for RGB and Sonar modalities
    and provides a unified interface for multi-modal feature extraction using
    template-search paradigm (OSTrack-style).

    ARCHITECTURE CHANGE (Audit 2025112104):
    Migrated from frame-sequence to template-search paradigm. Now accepts
    template and search inputs separately for each modality.

    IMPORTANT: Both RGB and Sonar backbones are required for training.
    Single-modality training is not supported.

    Args:
        rgb_config: Configuration dict for RGB backbone (required)
        sonar_config: Configuration dict for Sonar encoder (required)

    Example:
        >>> rgb_config = {'type': 'vit_rgb', 'model_name': 'vit_base_patch16_224', 'template_size': 256, 'search_size': 512}
        >>> sonar_config = {'type': 'vit_sonar', 'model_name': 'vit_base_patch16_224', 'template_size': 256, 'search_size': 512}
        >>> backbone = MultiModalBackbone(rgb_config, sonar_config)
        >>> rgb_template = torch.randn(4, 3, 256, 256)
        >>> rgb_search = torch.randn(4, 3, 512, 512)
        >>> sonar_template = torch.randn(4, 1, 256, 256)
        >>> sonar_search = torch.randn(4, 1, 512, 512)
        >>> outputs = backbone(rgb_template, rgb_search, sonar_template, sonar_search)
    """

    def __init__(
        self,
        rgb_config: Dict[str, Any],
        sonar_config: Dict[str, Any]
    ):
        super().__init__()

        # Validate that both configs are provided
        if not rgb_config or not sonar_config:
            raise ValueError("Both RGB and Sonar configurations are required for MultiModalBackbone")

        # Create backbones
        self.rgb_backbone = BackboneFactory.create_from_config(rgb_config)
        self.sonar_encoder = BackboneFactory.create_from_config(sonar_config)

        logger.info("Initialized MultiModalBackbone with RGB and Sonar encoders")

    def forward(
        self,
        rgb_template: torch.Tensor,
        rgb_search: torch.Tensor,
        sonar_template: torch.Tensor,
        sonar_search: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass through both modalities using template-search paradigm.

        ARCHITECTURE CHANGE (Audit 2025112104):
        Migrated from frame-sequence to template-search processing.

        Args:
            rgb_template: RGB template (B, 3, template_size, template_size)
            rgb_search: RGB search (B, 3, search_size, search_size)
            sonar_template: Sonar template (B, 1, template_size, template_size)
            sonar_search: Sonar search (B, 1, search_size, search_size)

        Returns:
            Dictionary containing features from both modalities:
            {
                'rgb': {
                    'template_features': [B, N_z, D],
                    'search_features': [B, N_x, D]
                },
                'sonar': {
                    'template_features': [B, N_z, D],
                    'search_features': [B, N_x, D]
                }
            }
            where N_z = (template_size // patch_size)^2, N_x = (search_size // patch_size)^2
        """
        # Extract RGB template-search features
        rgb_feat_template, rgb_feat_search = self.rgb_backbone(rgb_template, rgb_search)

        # Extract Sonar template-search features
        sonar_feat_template, sonar_feat_search = self.sonar_encoder(sonar_template, sonar_search)

        # Package outputs
        outputs = {
            'rgb': {
                'template_features': rgb_feat_template,
                'search_features': rgb_feat_search,
            },
            'sonar': {
                'template_features': sonar_feat_template,
                'search_features': sonar_feat_search,
            },
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
        backbone_type: Type of backbone ('vit_rgb', 'vit_sonar')
        model_name: Model name (optional, uses default if None)
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments

    Returns:
        Backbone module

    Example:
        >>> backbone = build_backbone('vit_rgb', pretrained=True)
        >>> sonar_backbone = build_backbone('vit_sonar', in_channels=1, pretrained=True)
    """
    # Set default model names if not provided
    if model_name is None:
        defaults = {
            'vit_rgb': 'vit_base_patch16_224',
            'vit_sonar': 'vit_base_patch16_224',
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
    sonar_type: str = 'vit_sonar',
    sonar_model: str = 'vit_base_patch16_224',
    rgb_pretrained: bool = True,
    sonar_pretrained: bool = True,
    **kwargs
) -> MultiModalBackbone:
    """
    Build multi-modal backbone with RGB and Sonar encoders.

    Args:
        rgb_type: Type of RGB backbone (default: 'vit_rgb')
        rgb_model: RGB model name (default: 'vit_base_patch16_224')
        sonar_type: Type of Sonar backbone (default: 'vit_sonar')
        sonar_model: Sonar model name (default: 'vit_base_patch16_224')
        rgb_pretrained: Whether RGB backbone is pretrained (default: True)
        sonar_pretrained: Whether Sonar backbone is pretrained (default: True)
        **kwargs: Additional arguments

    Returns:
        MultiModalBackbone instance

    Example:
        >>> backbone = build_multimodal_backbone(
        ...     rgb_type='vit_rgb',
        ...     sonar_type='vit_sonar'
        ... )
    """
    rgb_config = {
        'type': rgb_type,
        'model_name': rgb_model,
        'pretrained': rgb_pretrained,
    }

    sonar_config = {
        'type': sonar_type,
        'model_name': sonar_model,
        'pretrained': sonar_pretrained,
        'in_channels': 1,  # Sonar is single-channel
    }

    return MultiModalBackbone(rgb_config, sonar_config)
