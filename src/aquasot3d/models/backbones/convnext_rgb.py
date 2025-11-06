"""
ConvNeXt Backbone for RGB Images

This module implements a ConvNeXt-based backbone as an alternative to ViT
for RGB image encoding. ConvNeXt provides hierarchical features and is
compatible with the OSTrack template-search paradigm.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import timm

from aquasot3d.core import BACKBONE_REGISTRY, BaseModule


@BACKBONE_REGISTRY.register("ConvNeXt_RGB")
class ConvNeXtRGBBackbone(BaseModule):
    """
    ConvNeXt backbone for RGB image encoding.

    Provides hierarchical multi-scale features through a pure convolutional
    architecture. Supports OSTrack-style template-search paradigm.

    Args:
        model_name: timm model name (e.g., 'convnext_base', 'convnext_tiny')
        pretrained: Whether to load ImageNet pretrained weights
        output_stride: Output stride for feature maps (default: 16)
        out_indices: Which stages to output features from (default: (0, 1, 2, 3))
        freeze_stages: Number of stages to freeze (default: -1, no freezing)
        norm_eval: Whether to set norm layers to eval mode (default: False)
        return_interm_layers: Whether to return intermediate layer features

    Returns:
        Dictionary containing:
            - 'features': Final stage features
            - 'multi_scale_features': List of features from selected stages
            - 'feature_info': Information about feature dimensions and strides

    Example:
        >>> backbone = ConvNeXtRGBBackbone(model_name='convnext_base', pretrained=True)
        >>> template = torch.randn(4, 3, 128, 128)
        >>> search = torch.randn(4, 3, 256, 256)
        >>> outputs = backbone(template, search)
        >>> print(outputs['features'].shape)  # (4, C, H/16, W/16)
    """

    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained: bool = True,
        output_stride: int = 16,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        freeze_stages: int = -1,
        norm_eval: bool = False,
        return_interm_layers: bool = True,
        **kwargs
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.output_stride = output_stride
        self.out_indices = out_indices
        self.freeze_stages = freeze_stages
        self.norm_eval = norm_eval
        self.return_interm_layers = return_interm_layers

        # Load pretrained ConvNeXt from timm
        try:
            self.convnext = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,  # Return feature maps from multiple stages
                out_indices=out_indices,
                num_classes=0,
            )
        except Exception as e:
            raise ValueError(f"Failed to load ConvNeXt model '{model_name}' from timm: {e}")

        # Get feature information
        self.feature_info = self.convnext.feature_info.info
        self.num_stages = len(self.feature_info)

        # Extract feature dimensions and strides
        self.feature_dims = [info['num_chs'] for info in self.feature_info]
        self.feature_strides = [info['reduction'] for info in self.feature_info]

        # Freeze early stages if requested
        if freeze_stages >= 0:
            self._freeze_stages()

        self.logger.info(
            f"Initialized ConvNeXtRGBBackbone: model={model_name}, "
            f"feature_dims={self.feature_dims}, strides={self.feature_strides}, "
            f"pretrained={pretrained}"
        )

    def _freeze_stages(self):
        """Freeze parameters of early stages."""
        # ConvNeXt has 4 stages typically
        # Freeze stem and early stages
        if hasattr(self.convnext, 'stem'):
            for param in self.convnext.stem.parameters():
                param.requires_grad = False

        if hasattr(self.convnext, 'stages'):
            for i in range(min(self.freeze_stages, len(self.convnext.stages))):
                for param in self.convnext.stages[i].parameters():
                    param.requires_grad = False

        self.logger.info(f"Froze {self.freeze_stages} stages of ConvNeXt")

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features from input image.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary with multi-scale features
        """
        # Extract features from all selected stages
        features = self.convnext(x)  # Returns list of feature maps

        # Organize output
        outputs = {
            'features': features[-1],  # Final stage features
            'multi_scale_features': features,
        }

        # Add feature dimension information
        outputs['feature_info'] = {
            'dims': self.feature_dims,
            'strides': self.feature_strides,
        }

        return outputs

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        return_separate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for template-search paradigm.

        Args:
            template: Template frame (B, C, H_t, W_t)
            search: Search region frame (B, C, H_s, W_s)
            return_separate: If True, return template and search features separately

        Returns:
            Dictionary containing encoded features
        """
        B = template.shape[0]

        if return_separate:
            # Process template and search separately
            template_feat = self.forward_features(template)
            search_feat = self.forward_features(search)

            return {
                'template_features': template_feat['features'],
                'search_features': search_feat['features'],
                'template_multi_scale': template_feat['multi_scale_features'],
                'search_multi_scale': search_feat['multi_scale_features'],
                'feature_info': template_feat['feature_info'],
            }
        else:
            # Process together (batch concatenation)
            combined = torch.cat([template, search], dim=0)  # (2*B, C, H, W)
            combined_feat = self.forward_features(combined)

            # Split back into template and search
            template_feat = combined_feat['features'][:B]
            search_feat = combined_feat['features'][B:]

            # Split multi-scale features
            template_multi_scale = [f[:B] for f in combined_feat['multi_scale_features']]
            search_multi_scale = [f[B:] for f in combined_feat['multi_scale_features']]

            return {
                'features': combined_feat['features'],
                'template_features': template_feat,
                'search_features': search_feat,
                'template_multi_scale': template_multi_scale,
                'search_multi_scale': search_multi_scale,
                'multi_scale_features': combined_feat['multi_scale_features'],
                'feature_info': combined_feat['feature_info'],
            }

    def get_feature_dims(self) -> List[int]:
        """Get output feature dimensions for all stages."""
        return self.feature_dims

    def get_feature_strides(self) -> List[int]:
        """Get output feature strides for all stages."""
        return self.feature_strides

    def get_final_feature_dim(self) -> int:
        """Get final stage feature dimension."""
        return self.feature_dims[-1]

    def train(self, mode: bool = True):
        """
        Override train method to handle norm_eval.

        Args:
            mode: Training mode flag
        """
        super().train(mode)

        if mode and self.norm_eval:
            # Set all norm layers to eval mode
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    m.eval()


class ConvNeXtFeaturePyramid(nn.Module):
    """
    Feature Pyramid Network adapter for ConvNeXt.

    Combines multi-scale features from ConvNeXt backbone into a unified
    feature representation suitable for tracking.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_outs: int = 1
    ):
        """
        Initialize Feature Pyramid.

        Args:
            in_channels_list: List of input channel dimensions for each stage
            out_channels: Output channel dimension
            num_outs: Number of output feature levels
        """
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_outs = num_outs

        # Lateral convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through feature pyramid.

        Args:
            features: List of feature maps from backbone stages

        Returns:
            List of pyramid feature maps
        """
        # Build lateral features
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher-level feature
            upsampled = torch.nn.functional.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )
            # Add to lateral feature
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convolutions
        outs = [
            self.output_convs[i](laterals[i])
            for i in range(len(laterals))
        ]

        return outs[:self.num_outs] if self.num_outs > 0 else outs


def create_convnext_rgb_backbone(
    model_name: str = "convnext_base",
    pretrained: bool = True,
    **kwargs
) -> ConvNeXtRGBBackbone:
    """
    Factory function to create ConvNeXt RGB backbone.

    Args:
        model_name: timm model name
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments for ConvNeXtRGBBackbone

    Returns:
        ConvNeXtRGBBackbone instance

    Example:
        >>> backbone = create_convnext_rgb_backbone('convnext_base', pretrained=True)
    """
    return ConvNeXtRGBBackbone(model_name=model_name, pretrained=pretrained, **kwargs)
