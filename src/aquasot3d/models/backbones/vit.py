"""
Vision Transformer (ViT) Backbone for RGB Images

This module implements a ViT-based backbone for frame-level temporal tracking.
Loads pretrained ViT from timm and adapts it for tracking tasks.

ARCHITECTURE CHANGE (Audit 2025112006):
Refactored from template-search paradigm to single-frame processing for
Vision-Mamba temporal modeling.

Author: AquaSOT-3D Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import timm

from aquasot3d.core import BACKBONE_REGISTRY, BaseModule


@BACKBONE_REGISTRY.register("vit_rgb")
@BACKBONE_REGISTRY.register("vit_sonar")
@BACKBONE_REGISTRY.register("vit")
class ViTBackbone(BaseModule):
    """
    Unified Vision Transformer backbone for RGB and Sonar image encoding.

    Supports both RGB (3-channel) and Sonar (1-channel) inputs through optional
    channel adaptation. Processes frames independently for temporal modeling.

    ARCHITECTURE CHANGE (Audit 2025112006):
    Refactored from template-search paradigm to single-frame processing.
    Now supports both single-frame [B, C, H, W] and frame-sequence [B, T, C, H, W] inputs.

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
        in_channels: Input channels (3 for RGB, 1 for Sonar, default: 3)

    Returns:
        Dictionary containing:
            - 'features': Final layer features (B, N, C) or (B*T, N, C) for sequences
            - 'interm_features': Intermediate layer features (optional)
            - 'attention_weights': Self-attention weights (optional)

    Example:
        >>> # RGB backbone
        >>> rgb_backbone = ViTBackbone(model_name='vit_base_patch16_224', in_channels=3)
        >>> # Single frame
        >>> frame = torch.randn(4, 3, 384, 384)
        >>> outputs = rgb_backbone(frame)
        >>> # Frame sequence
        >>> frames = torch.randn(4, 5, 3, 384, 384)  # [B, T, C, H, W]
        >>> outputs = rgb_backbone(frames)  # Processes as [B*T, C, H, W]
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        return_interm_layers: bool = False,
        freeze_backbone: bool = False,
        in_channels: int = 3,  # Support both RGB (3) and Sonar (1)
        **kwargs
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.return_interm_layers = return_interm_layers
        self.freeze_backbone = freeze_backbone
        self.in_channels = in_channels

        # Channel adapter for single-channel inputs (e.g., Sonar)
        if in_channels == 1:
            self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            self.logger.info("Added 1->3 channel adapter for single-channel input")
        else:
            self.channel_adapter = None

        # Load pretrained ViT from timm (ImageNet weights)
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

        # Load pretrained checkpoint if specified (Stage 1 RGB-only weights)
        if pretrained_path is not None:
            self._load_pretrained_checkpoint(pretrained_path)

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
            f"img_size={img_size}, pretrained={pretrained}, "
            f"pretrained_path={pretrained_path}"
        )

    def _freeze_parameters(self):
        """Freeze all backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.logger.info("Froze ViT backbone parameters")

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        """
        Load pretrained backbone weights from Stage 1 checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pth)
        """
        import os
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter backbone keys (supports both RGB and Sonar)
            # Expected formats:
            #   - RGB: 'backbone.rgb_backbone.vit.*'
            #   - Sonar: 'backbone.sonar_encoder.vit.*'
            #   - Direct: 'vit.*'
            backbone_keys = {}
            rgb_key_count = 0
            sonar_key_count = 0
            direct_key_count = 0

            for key, value in state_dict.items():
                if 'rgb_backbone.vit.' in key:
                    # Remove prefix to match local vit module
                    new_key = key.split('rgb_backbone.vit.')[-1]
                    backbone_keys[new_key] = value
                    rgb_key_count += 1
                elif 'sonar_encoder.vit.' in key:
                    # Support Sonar single-modality checkpoint (Critical Fix: Audit 2025112003)
                    new_key = key.split('sonar_encoder.vit.')[-1]
                    backbone_keys[new_key] = value
                    sonar_key_count += 1
                elif key.startswith('vit.'):
                    # Direct vit key (if saved without model wrapper)
                    new_key = key[4:]  # Remove 'vit.' prefix
                    backbone_keys[new_key] = value
                    direct_key_count += 1

            # CRITICAL CHECK (Audit 2025112003): Detect mixed RGB+Sonar keys
            if rgb_key_count > 0 and sonar_key_count > 0:
                self.logger.warning(
                    f"Checkpoint contains BOTH RGB ({rgb_key_count}) and Sonar ({sonar_key_count}) keys. "
                    f"Sonar keys will override RGB keys with same names. This may indicate an unexpected checkpoint."
                )

            if not backbone_keys:
                self.logger.warning(
                    f"No compatible backbone keys found in checkpoint {checkpoint_path}. "
                    f"Available keys: {list(state_dict.keys())[:5]}..."
                )
                return

            # CRITICAL CHECK (Audit 2025112003): Validate critical parameter shapes
            # before loading to catch incompatible checkpoints early
            current_state = self.vit.state_dict()
            # Use actual timm ViT key names
            # Note: For Sonar (in_channels=1), patch_embed.proj.weight shape mismatch is EXPECTED
            # because Sonar checkpoints have [768, 1, 16, 16] but current ViT is [768, 3, 16, 16]
            # (Sonar uses channel adapter to convert 1→3 channels before ViT)
            critical_keys = ['pos_embed']  # Only check pos_embed (always should match)
            if self.in_channels != 1:  # RGB: also check patch_embed
                critical_keys.append('patch_embed.proj.weight')

            shape_mismatches = []

            for key in critical_keys:
                if key in backbone_keys and key in current_state:
                    checkpoint_shape = backbone_keys[key].shape
                    current_shape = current_state[key].shape
                    if checkpoint_shape != current_shape:
                        shape_mismatches.append(
                            f"{key}: checkpoint {checkpoint_shape} vs current {current_shape}"
                        )

            if shape_mismatches:
                self.logger.error(
                    f"Critical parameter shape mismatch detected in checkpoint {checkpoint_path}:\n" +
                    "\n".join(f"  - {mismatch}" for mismatch in shape_mismatches) +
                    f"\n  This usually indicates incompatible model configurations (e.g., different img_size or patch_size)."
                )
                raise ValueError(
                    f"Checkpoint {checkpoint_path} has incompatible shapes for critical parameters. "
                    f"Cannot load checkpoint safely."
                )

            # CRITICAL FIX (Audit 2025112003): For Sonar (in_channels=1), remove patch_embed.proj.weight
            # because its shape [768, 1, 16, 16] doesn't match current ViT [768, 3, 16, 16]
            # (Sonar uses channel adapter, doesn't need this parameter from checkpoint)
            if self.in_channels == 1 and 'patch_embed.proj.weight' in backbone_keys:
                self.logger.info(
                    "Sonar mode: Skipping patch_embed.proj.weight from checkpoint "
                    "(shape mismatch expected, using channel adapter instead)"
                )
                del backbone_keys['patch_embed.proj.weight']

            # Load state dict with strict=False to allow missing keys
            missing_keys, unexpected_keys = self.vit.load_state_dict(
                backbone_keys, strict=False
            )

            self.logger.info(
                f"Loaded pretrained ViT backbone from {checkpoint_path} "
                f"({len(backbone_keys)} parameters)"
            )

            # CRITICAL CHECK (Audit 2025112003): Warn if too many missing keys
            if missing_keys:
                if len(missing_keys) > len(backbone_keys) * 0.1:  # >10% missing
                    self.logger.warning(
                        f"Large number of missing keys ({len(missing_keys)}/{len(current_state)}): "
                        f"{missing_keys[:5]}... "
                        f"Checkpoint may be incompatible or incomplete."
                    )
                else:
                    self.logger.debug(f"Missing keys: {missing_keys[:5]}...")

            if unexpected_keys:
                self.logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")

        except Exception as e:
            self.logger.error(
                f"Failed to load pretrained checkpoint {checkpoint_path}: {e}"
            )
            raise

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-frame or frame-sequence processing.

        ARCHITECTURE CHANGE (Audit 2025112006):
        Replaced template-search paradigm with single-frame processing.

        Args:
            x: Input frames
                - Single frame: (B, C, H, W)
                - Frame sequence: (B, T, C, H, W) -> reshaped to (B*T, C, H, W)

        Returns:
            Dictionary containing:
                - 'features': Encoded features (B, N, C) or (B*T, N, C)
                - 'interm_features': Intermediate features (optional)
                - 'is_sequence': Boolean indicating if input was a sequence
                - 'batch_size': Original batch size B (for sequence reshaping)
                - 'sequence_length': T (for sequence reshaping, None for single frame)
        """
        # Detect input shape
        is_sequence = x.ndim == 5  # [B, T, C, H, W]

        if is_sequence:
            B, T, C, H, W = x.shape
            # Reshape to process all frames independently
            x = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        else:
            B = x.shape[0]
            T = None

        # Apply channel adapter for single-channel inputs (e.g., Sonar)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        # Extract features
        outputs = self.forward_features(x)

        # Add metadata for sequence handling
        outputs['is_sequence'] = is_sequence
        outputs['batch_size'] = B
        outputs['sequence_length'] = T

        return outputs


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


def create_vit_backbone(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    **kwargs
) -> ViTBackbone:
    """
    Factory function to create ViT backbone (RGB or Sonar).

    Args:
        model_name: timm model name
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments for ViTBackbone

    Returns:
        ViTBackbone instance

    Example:
        >>> rgb_backbone = create_vit_backbone('vit_base_patch16_224', pretrained=True)
        >>> sonar_backbone = create_vit_backbone('vit_base_patch16_224', in_channels=1, pretrained=True)
    """
    return ViTBackbone(model_name=model_name, pretrained=pretrained, **kwargs)
