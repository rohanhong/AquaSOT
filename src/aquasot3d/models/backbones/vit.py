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

    def _convert_spatial_mask_to_token_mask(
        self,
        spatial_mask: torch.Tensor,
        target_size: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Convert spatial attention mask [B, H, W] to token-level mask [B, N].

        Args:
            spatial_mask: Spatial mask [B, H, W], 1=valid, 0=padded
            target_size: Target image size (template_size or search_size)
            batch_size: Batch size

        Returns:
            Token mask [B, N] where N = (target_size // patch_size)^2
        """
        patch_h = target_size // self.patch_size
        patch_w = target_size // self.patch_size

        # Use average pooling - patch is valid if >50% pixels are valid
        mask_pooled = torch.nn.functional.adaptive_avg_pool2d(
            spatial_mask.unsqueeze(1).float(),
            (patch_h, patch_w)
        ).squeeze(1)  # [B, patch_h, patch_w]

        token_mask = (mask_pooled > 0.5).float().reshape(batch_size, -1)  # [B, N]
        return token_mask

    def forward(
        self,
        x_template: torch.Tensor,
        x_search: torch.Tensor,
        template_att_mask: Optional[torch.Tensor] = None,
        search_att_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        OSTrack one-stream forward pass: process template and search jointly.

        ARCHITECTURE CHANGE (Audit 2025112104):
        Migrated to template-search paradigm. Concatenates template and search tokens
        before transformer, enabling cross-attention between regions.

        CRITICAL FIX (Audit 2025120902 Issue 15, 17 - COMPLETE REVISION):
        Proper attention masking implementation:
        1. Convert spatial masks to token-level masks
        2. Add pos_embed FIRST, then apply token mask
        3. Create additive attention bias (padded positions get -inf)
        4. Apply mask inside transformer via custom attention wrapper
        5. Re-apply mask after transformer for safety

        Args:
            x_template: Template image [B, C, template_size, template_size]
            x_search: Search image [B, C, search_size, search_size]
            template_att_mask: Optional attention mask for template [B, H, W]
                               (1 = valid, 0 = padded)
            search_att_mask: Optional attention mask for search [B, H, W]
                             (1 = valid, 0 = padded)

        Returns:
            Tuple of (feat_template, feat_search):
                - feat_template: [B, N_z, D]
                - feat_search: [B, N_x, D]
        """
        B = x_template.shape[0]
        device = x_template.device

        # Apply channel adapter if needed (for Sonar)
        if self.channel_adapter is not None:
            x_template = self.channel_adapter(x_template)
            x_search = self.channel_adapter(x_search)

        # === Step 1: Patch embedding ===
        tokens_z = self.vit.patch_embed(x_template)  # [B, N_z, D]
        tokens_x = self.vit.patch_embed(x_search)    # [B, N_x, D]

        # Handle dynamic_img_size mode (flatten if needed)
        if tokens_z.dim() == 4:  # [B, H, W, C]
            B, H_z, W_z, D = tokens_z.shape
            tokens_z = tokens_z.reshape(B, H_z * W_z, D)
        if tokens_x.dim() == 4:
            B, H_x, W_x, D = tokens_x.shape
            tokens_x = tokens_x.reshape(B, H_x * W_x, D)

        N_z = tokens_z.shape[1]  # num_patches_template
        N_x = tokens_x.shape[1]  # num_patches_search

        # === Step 2: Add positional embeddings FIRST ===
        # This is critical - pos_embed must be added before masking
        tokens_z = tokens_z + self.pos_embed_z  # [B, N_z, D]
        tokens_x = tokens_x + self.pos_embed_x  # [B, N_x, D]

        # === Step 3: Process attention masks ===
        # Convert spatial masks to token-level masks
        mask_z = None
        mask_x = None

        if template_att_mask is not None:
            if template_att_mask.dim() == 3:  # [B, H, W]
                mask_z = self._convert_spatial_mask_to_token_mask(
                    template_att_mask, self.template_size, B
                )
            else:
                mask_z = template_att_mask.float()
        else:
            # Default: all template tokens valid
            mask_z = torch.ones(B, N_z, device=device)

        if search_att_mask is not None:
            if search_att_mask.dim() == 3:  # [B, H, W]
                mask_x = self._convert_spatial_mask_to_token_mask(
                    search_att_mask, self.search_size, B
                )
            else:
                mask_x = search_att_mask.float()
        else:
            # Default: all search tokens valid
            mask_x = torch.ones(B, N_x, device=device)

        # === Step 4: Apply token mask AFTER pos_embed ===
        # Zero out padded tokens so they don't contribute values
        tokens_z = tokens_z * mask_z.unsqueeze(-1)  # [B, N_z, D]
        tokens_x = tokens_x * mask_x.unsqueeze(-1)  # [B, N_x, D]

        # === Step 5: Concatenate for one-stream processing ===
        tokens = torch.cat([tokens_z, tokens_x], dim=1)  # [B, N_z + N_x, D]
        combined_mask = torch.cat([mask_z, mask_x], dim=1)  # [B, N_z + N_x]

        # === Step 6: Create additive attention bias ===
        # For proper masking, we need to set attention scores to -inf for padded positions
        # attention_bias: [B, 1, N, N] or [B, num_heads, N, N]
        # Where padded KEY positions get -inf (so softmax gives them 0 weight)
        N_total = N_z + N_x

        # Key mask: which positions should be ignored as keys
        # combined_mask: 1 = valid, 0 = padded
        # We want attention scores to padded keys to be -inf
        key_padding_mask = (combined_mask < 0.5)  # [B, N], True = padded

        # Create attention bias: [B, 1, 1, N] - broadcast across queries and heads
        # Set to -inf for padded key positions
        attn_bias = torch.zeros(B, 1, 1, N_total, device=device, dtype=tokens.dtype)
        attn_bias.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Apply position dropout
        if hasattr(self.vit, 'pos_drop'):
            tokens = self.vit.pos_drop(tokens)

        # === Step 7: Transformer blocks with attention masking ===
        if hasattr(self.vit, 'blocks'):
            for blk in self.vit.blocks:
                # Try to pass attention bias if block supports it
                # Different timm versions have different signatures
                try:
                    # Newer timm versions may support attn_mask parameter
                    if hasattr(blk, 'attn') and hasattr(blk.attn, 'forward'):
                        # Custom forward with attention bias
                        tokens = self._block_forward_with_mask(blk, tokens, attn_bias)
                    else:
                        tokens = blk(tokens)
                except TypeError:
                    # Fallback: standard forward without mask
                    tokens = blk(tokens)

        # === Step 8: Re-apply mask after transformer ===
        # Ensure padded positions remain zeroed (attention may leak small values)
        tokens = tokens * combined_mask.unsqueeze(-1)

        # === Step 9: Final layer norm ===
        if hasattr(self.vit, 'norm'):
            tokens = self.vit.norm(tokens)

        # Re-apply mask after norm (norm may introduce non-zero values for padded tokens)
        tokens = tokens * combined_mask.unsqueeze(-1)

        # === Step 10: Split back into template and search ===
        feat_template = tokens[:, :N_z, :]  # [B, N_z, D]
        feat_search = tokens[:, N_z:, :]    # [B, N_x, D]

        return feat_template, feat_search

    def _block_forward_with_mask(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        attn_bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through a transformer block with attention masking.

        This manually applies the block components to inject attention bias.

        Args:
            blk: Transformer block (timm Block)
            x: Input tokens [B, N, D]
            attn_bias: Attention bias [B, 1, 1, N], -inf for padded positions

        Returns:
            Output tokens [B, N, D]
        """
        # Standard transformer block: x = x + attn(norm1(x)), x = x + mlp(norm2(x))

        # Pre-norm for attention
        if hasattr(blk, 'norm1'):
            x_norm = blk.norm1(x)
        else:
            x_norm = x

        # Attention with masking
        if hasattr(blk, 'attn'):
            attn_out = self._attention_with_mask(blk.attn, x_norm, attn_bias)
        else:
            attn_out = x_norm

        # Residual connection
        if hasattr(blk, 'drop_path1'):
            x = x + blk.drop_path1(attn_out)
        elif hasattr(blk, 'drop_path'):
            x = x + blk.drop_path(attn_out)
        else:
            x = x + attn_out

        # Pre-norm for MLP
        if hasattr(blk, 'norm2'):
            x_norm = blk.norm2(x)
        else:
            x_norm = x

        # MLP
        if hasattr(blk, 'mlp'):
            mlp_out = blk.mlp(x_norm)
        else:
            mlp_out = x_norm

        # Residual connection
        if hasattr(blk, 'drop_path2'):
            x = x + blk.drop_path2(mlp_out)
        elif hasattr(blk, 'drop_path'):
            x = x + blk.drop_path(mlp_out)
        else:
            x = x + mlp_out

        return x

    def _attention_with_mask(
        self,
        attn_module: nn.Module,
        x: torch.Tensor,
        attn_bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention with additive attention bias for masking.

        Args:
            attn_module: timm Attention module
            x: Input [B, N, D]
            attn_bias: Attention bias [B, 1, 1, N], -inf for padded positions

        Returns:
            Attention output [B, N, D]
        """
        B, N, C = x.shape

        # Get QKV projection
        if hasattr(attn_module, 'qkv'):
            qkv = attn_module.qkv(x)  # [B, N, 3*D]
            qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # Fallback for different attention implementations
            return attn_module(x)

        # Compute attention scores
        scale = (C // attn_module.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]

        # Apply attention bias (masking padded keys with -inf)
        # attn_bias is [B, 1, 1, N], broadcast to [B, heads, N, N]
        attn = attn + attn_bias

        # Softmax (padded positions get 0 weight due to -inf)
        attn = attn.softmax(dim=-1)

        # Apply attention dropout if exists
        if hasattr(attn_module, 'attn_drop'):
            attn = attn_module.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, D]

        # Output projection
        if hasattr(attn_module, 'proj'):
            x = attn_module.proj(x)
        if hasattr(attn_module, 'proj_drop'):
            x = attn_module.proj_drop(x)

        return x


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
