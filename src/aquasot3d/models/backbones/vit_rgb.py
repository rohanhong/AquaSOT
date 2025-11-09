"""
Vision Transformer (ViT) Backbone for RGB Images

This module implements a ViT-based backbone compatible with OSTrack template-search paradigm.
Loads pretrained ViT from timm and adapts it for tracking tasks.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import timm

from aquasot3d.core import BACKBONE_REGISTRY, BaseModule


@BACKBONE_REGISTRY.register("ViT_RGB")
class ViTRGBBackbone(BaseModule):
    """
    Vision Transformer backbone for RGB image encoding.

    Supports OSTrack-style template-search paradigm by processing concatenated
    template and search region patches through a shared ViT encoder.

    Args:
        model_name: timm model name (e.g., 'vit_base_patch16_224')
        pretrained: Whether to load ImageNet pretrained weights
        img_size: Input image size (default: 224)
        patch_size: Patch size for ViT (default: 16)
        embed_dim: Embedding dimension (default: 768 for ViT-Base)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        drop_rate: Dropout rate (default: 0.0)
        drop_path_rate: Stochastic depth rate (default: 0.1)
        return_interm_layers: Whether to return intermediate layer features
        freeze_backbone: Whether to freeze backbone parameters

    Returns:
        Dictionary containing:
            - 'features': Final layer features (B, N, C)
            - 'interm_features': Intermediate layer features (optional)
            - 'attention_weights': Self-attention weights (optional)

    Example:
        >>> backbone = ViTRGBBackbone(model_name='vit_base_patch16_224', pretrained=True)
        >>> template = torch.randn(4, 3, 128, 128)  # Template frames
        >>> search = torch.randn(4, 3, 256, 256)    # Search region frames
        >>> outputs = backbone(template, search)
        >>> print(outputs['features'].shape)  # (4, num_patches, 768)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        return_interm_layers: bool = False,
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.return_interm_layers = return_interm_layers
        self.freeze_backbone = freeze_backbone

        # Load pretrained ViT from timm
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                img_size=img_size,
                num_classes=0,  # Remove classification head
                global_pool='',  # Remove global pooling
                dynamic_img_size=True,  # Allow variable input sizes
            )
        except Exception as e:
            raise ValueError(f"Failed to load ViT model '{model_name}' from timm: {e}")

        # Extract configuration from loaded model
        if hasattr(self.vit, 'embed_dim'):
            self.embed_dim = self.vit.embed_dim
        if hasattr(self.vit, 'patch_embed'):
            self.patch_size = self.vit.patch_embed.patch_size[0]

        # Calculate number of patches
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_parameters()

        # Position encoding adaptation for different input sizes
        self.pos_embed_adapter = PositionEmbeddingAdapter(
            embed_dim=self.embed_dim,
            default_size=self.num_patches_per_side
        )

        self.logger.info(
            f"Initialized ViTRGBBackbone: model={model_name}, "
            f"embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"img_size={img_size}, pretrained={pretrained}"
        )

    def _freeze_parameters(self):
        """Freeze all backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.logger.info("Froze ViT backbone parameters")

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input image.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary with extracted features
        """
        B = x.shape[0]

        # Patch embedding
        if hasattr(self.vit, 'patch_embed'):
            x = self.vit.patch_embed(x)  # (B, N, C) or (B, H, W, C) with dynamic_img_size
            # If dynamic_img_size is enabled, flatten spatial dimensions
            if x.dim() == 4:  # (B, H, W, C)
                B, H, W, C = x.shape
                x = x.reshape(B, H * W, C)  # (B, N, C)
        else:
            raise AttributeError("ViT model does not have patch_embed attribute")

        # Add class token if exists
        if hasattr(self.vit, 'cls_token'):
            cls_tokens = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding with adaptation
        if hasattr(self.vit, 'pos_embed'):
            x = x + self.pos_embed_adapter(self.vit.pos_embed, x.shape[1])

        # Apply position dropout
        if hasattr(self.vit, 'pos_drop'):
            x = self.vit.pos_drop(x)

        # Transformer blocks
        interm_features = []
        if hasattr(self.vit, 'blocks'):
            for i, blk in enumerate(self.vit.blocks):
                x = blk(x)
                if self.return_interm_layers and i in [self.depth // 4, self.depth // 2, 3 * self.depth // 4]:
                    interm_features.append(x)

        # Final layer norm
        if hasattr(self.vit, 'norm'):
            x = self.vit.norm(x)

        outputs = {'features': x}
        if self.return_interm_layers:
            outputs['interm_features'] = interm_features

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
            }
        else:
            # Concatenate and process together (OSTrack style)
            # This allows cross-attention between template and search
            combined = torch.cat([template, search], dim=0)  # (2*B, C, H, W)
            combined_feat = self.forward_features(combined)

            # Split back into template and search
            feat_dim = combined_feat['features'].shape[1]
            template_feat = combined_feat['features'][:B]
            search_feat = combined_feat['features'][B:]

            return {
                'features': combined_feat['features'],
                'template_features': template_feat,
                'search_features': search_feat,
            }

    def get_num_patches(self, img_size: int) -> int:
        """Calculate number of patches for given image size."""
        return (img_size // self.patch_size) ** 2

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.embed_dim


class PositionEmbeddingAdapter(nn.Module):
    """
    Adapts position embeddings for different input resolutions.

    Uses bilinear interpolation to resize position embeddings when input
    resolution differs from the default resolution.
    """

    def __init__(self, embed_dim: int, default_size: int = 14):
        super().__init__()
        self.embed_dim = embed_dim
        self.default_size = default_size

    def forward(self, pos_embed: torch.Tensor, target_num_patches: int) -> torch.Tensor:
        """
        Adapt position embeddings to target number of patches.

        Args:
            pos_embed: Original position embeddings (1, N+1, C) or (1, N, C)
            target_num_patches: Target number of patches (including cls token if present)

        Returns:
            Adapted position embeddings
        """
        if pos_embed.shape[1] == target_num_patches:
            return pos_embed

        # Check if cls token is present
        has_cls_token = False
        if pos_embed.shape[1] == self.default_size ** 2 + 1:
            has_cls_token = True
            cls_pos_embed = pos_embed[:, :1, :]
            pos_embed = pos_embed[:, 1:, :]
            target_num_patches -= 1

        # Calculate target grid size
        target_size = int(target_num_patches ** 0.5)

        # Reshape to 2D
        pos_embed = pos_embed.reshape(1, self.default_size, self.default_size, self.embed_dim)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # (1, C, H, W)

        # Interpolate
        pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, target_num_patches, self.embed_dim)

        # Add cls token back if present
        if has_cls_token:
            pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)

        return pos_embed


def create_vit_rgb_backbone(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    **kwargs
) -> ViTRGBBackbone:
    """
    Factory function to create ViT RGB backbone.

    Args:
        model_name: timm model name
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments for ViTRGBBackbone

    Returns:
        ViTRGBBackbone instance

    Example:
        >>> backbone = create_vit_rgb_backbone('vit_base_patch16_224', pretrained=True)
    """
    return ViTRGBBackbone(model_name=model_name, pretrained=pretrained, **kwargs)
