"""
Vision Transformer (ViT) Backbone for RGB Images

This module implements a ViT-based backbone for template-search tracking.
Loads pretrained ViT from timm and adapts it for tracking tasks.

ARCHITECTURE CHANGE (Audit 2025112104):
Migrated to OSTrack one-stream template-search paradigm. Processes template
and search regions jointly through self-attention for better feature interaction.

Author: AquaSOT-3D Team
Date: 2025-11-21
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
    Unified Vision Transformer backbone for RGB and Sonar template-search tracking.

    Implements OSTrack one-stream architecture: concatenates template and search tokens
    before transformer processing, enabling self-attention across both regions.

    ARCHITECTURE CHANGE (Audit 2025112104):
    Migrated from frame-sequence processing to template-search paradigm.
    Separate positional embeddings for template (128x128) and search (256x256).

    Args:
        model_name: timm model name (e.g., 'vit_base_patch16_224')
        pretrained: Whether to load ImageNet pretrained weights
        template_size: Template input size (default: 128, OSTrack-256 standard)
        search_size: Search input size (default: 256, OSTrack-256 standard)
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
        Tuple of (feat_template, feat_search):
            - feat_template: [B, N_z, C] where N_z = (template_size // patch_size)^2
            - feat_search: [B, N_x, C] where N_x = (search_size // patch_size)^2

    Example:
        >>> # RGB backbone for template-search (OSTrack-256 standard)
        >>> rgb_backbone = ViTBackbone(
        ...     model_name='vit_base_patch16_224',
        ...     template_size=128,
        ...     search_size=256,
        ...     in_channels=3
        ... )
        >>> template = torch.randn(4, 3, 128, 128)
        >>> search = torch.randn(4, 3, 256, 256)
        >>> feat_z, feat_x = rgb_backbone(template, search)
        >>> print(feat_z.shape)  # [4, 64, 768]  (128/16)^2 = 64
        >>> print(feat_x.shape)  # [4, 256, 768] (256/16)^2 = 256
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        template_size: int = 128,  # OSTrack-256 standard: template 128x128
        search_size: int = 256,    # OSTrack-256 standard: search 256x256
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
        self.template_size = template_size  # OSTrack-256: 128
        self.search_size = search_size      # OSTrack-256: 256
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
            # CRITICAL FIX: Initialize to replicate input channel to all 3 outputs
            # Random initialization causes gradient explosion with pretrained ViT
            nn.init.constant_(self.channel_adapter.weight, 1.0)
            self.logger.info("Added 1->3 channel adapter (initialized to replicate)")
        else:
            self.channel_adapter = None

        # Load pretrained ViT from timm (ImageNet weights)
        # Note: Use template_size as base img_size for pretrained loading
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                img_size=template_size,  # Use template size for initialization
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

        # Calculate number of patches for template and search
        self.num_patches_template = (template_size // self.patch_size) ** 2  # 128/16 = 8, 8*8 = 64
        self.num_patches_search = (search_size // self.patch_size) ** 2      # 256/16 = 16, 16*16 = 256

        # CRITICAL: Create SEPARATE positional embeddings for template and search (OSTrack style)
        # Remove original pos_embed from ViT (will use our custom embeddings)
        if hasattr(self.vit, 'pos_embed'):
            # Copy pretrained pos_embed for initialization
            pretrained_pos_embed = self.vit.pos_embed.data.clone()
            # Delete ViT's pos_embed to avoid confusion
            delattr(self.vit, 'pos_embed')

        # Initialize learnable positional embeddings
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, self.embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, self.embed_dim))

        # Initialize from pretrained pos_embed via interpolation
        if 'pretrained_pos_embed' in locals():
            # Remove cls token if exists
            if pretrained_pos_embed.shape[1] == (template_size // self.patch_size) ** 2 + 1:
                pretrained_pos_embed = pretrained_pos_embed[:, 1:, :]  # Remove cls token

            # Interpolate to template size (may not need if already correct size)
            self.pos_embed_z.data = self._interpolate_pos_embed(
                pretrained_pos_embed, self.num_patches_template
            )

            # Interpolate to search size
            self.pos_embed_x.data = self._interpolate_pos_embed(
                pretrained_pos_embed, self.num_patches_search
            )

        # Load pretrained checkpoint if specified (Stage 1 RGB-only weights)
        if pretrained_path is not None:
            self._load_pretrained_checkpoint(pretrained_path)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_parameters()

        self.logger.info(
            f"Initialized ViTBackbone (OSTrack one-stream): model={model_name}, "
            f"embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"template_size={template_size}, search_size={search_size}, "
            f"num_patches_template={self.num_patches_template}, "
            f"num_patches_search={self.num_patches_search}, "
            f"pretrained={pretrained}, pretrained_path={pretrained_path}"
        )

    def _freeze_parameters(self):
        """Freeze all backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.logger.info("Froze ViT backbone parameters")

    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        target_num_patches: int
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings to target number of patches.

        Args:
            pos_embed: Source position embeddings [1, N_src, C]
            target_num_patches: Target number of patches

        Returns:
            Interpolated position embeddings [1, target_num_patches, C]
        """
        if pos_embed.shape[1] == target_num_patches:
            return pos_embed

        # Get source and target grid sizes
        src_num_patches = pos_embed.shape[1]
        src_size = int(src_num_patches ** 0.5)
        target_size = int(target_num_patches ** 0.5)

        # Reshape to 2D grid [1, C, H, W]
        C = pos_embed.shape[2]
        pos_embed_2d = pos_embed.reshape(1, src_size, src_size, C).permute(0, 3, 1, 2)

        # Interpolate using bilinear
        pos_embed_interp = torch.nn.functional.interpolate(
            pos_embed_2d,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back to [1, N, C]
        pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).reshape(1, target_num_patches, C)

        return pos_embed_interp

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

    def forward(
        self,
        x_template: torch.Tensor,
        x_search: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        OSTrack one-stream forward pass: process template and search jointly.

        ARCHITECTURE CHANGE (Audit 2025112104):
        Migrated to template-search paradigm. Concatenates template and search tokens
        before transformer, enabling cross-attention between regions.

        Args:
            x_template: Template image [B, C, template_size, template_size] (e.g., [B, 3, 128, 128])
            x_search: Search image [B, C, search_size, search_size] (e.g., [B, 3, 256, 256])

        Returns:
            Tuple of (feat_template, feat_search):
                - feat_template: [B, N_z, D] where N_z = (template_size // patch_size)^2
                - feat_search: [B, N_x, D] where N_x = (search_size // patch_size)^2

        Example:
            >>> template = torch.randn(4, 3, 128, 128)
            >>> search = torch.randn(4, 3, 256, 256)
            >>> feat_z, feat_x = backbone(template, search)
            >>> print(feat_z.shape)  # [4, 64, 768]
            >>> print(feat_x.shape)  # [4, 256, 768]
        """
        B = x_template.shape[0]

        # Apply channel adapter if needed (for Sonar)
        if self.channel_adapter is not None:
            x_template = self.channel_adapter(x_template)
            x_search = self.channel_adapter(x_search)

        # 1. Patch embedding
        tokens_z = self.vit.patch_embed(x_template)  # [B, N_z, D]
        tokens_x = self.vit.patch_embed(x_search)    # [B, N_x, D]

        # Handle dynamic_img_size mode (flatten if needed)
        if tokens_z.dim() == 4:  # [B, H, W, C]
            B, H_z, W_z, D = tokens_z.shape
            tokens_z = tokens_z.reshape(B, H_z * W_z, D)
        if tokens_x.dim() == 4:
            B, H_x, W_x, D = tokens_x.shape
            tokens_x = tokens_x.reshape(B, H_x * W_x, D)

        # 2. Add SEPARATE positional embeddings (OSTrack key innovation)
        tokens_z = tokens_z + self.pos_embed_z  # [B, N_z, D]
        tokens_x = tokens_x + self.pos_embed_x  # [B, N_x, D]

        # 3. Concatenate template and search tokens (ONE-STREAM)
        tokens = torch.cat([tokens_z, tokens_x], dim=1)  # [B, N_z + N_x, D]

        # Apply position dropout
        if hasattr(self.vit, 'pos_drop'):
            tokens = self.vit.pos_drop(tokens)

        # 4. Transformer blocks (self-attention across BOTH template and search)
        if hasattr(self.vit, 'blocks'):
            for blk in self.vit.blocks:
                tokens = blk(tokens)

        # 5. Final layer norm
        if hasattr(self.vit, 'norm'):
            tokens = self.vit.norm(tokens)

        # 6. Split back into template and search features
        N_z = self.num_patches_template
        feat_template = tokens[:, :N_z, :]      # [B, N_z, D]
        feat_search = tokens[:, N_z:, :]        # [B, N_x, D]

        return feat_template, feat_search


    def get_num_patches(self, img_size: int) -> int:
        """Calculate number of patches for given image size."""
        return (img_size // self.patch_size) ** 2

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.embed_dim


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
