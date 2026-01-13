"""
Vision Transformer backbone for AquaSOT-3D.

Based on OSTrack ViT with support for:
- 3-channel RGB input
- 1-channel Sonar input (with channel adaptation)

Reference:
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  https://arxiv.org/abs/2010.11929
"""

import os
import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.patch_embed import PatchEmbed
from .base_backbone import BaseBackbone

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformer(BaseBackbone):
    """Vision Transformer for tracking.

    Supports both RGB (3-channel) and Sonar (1-channel) inputs.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 weight_init=''):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels (3 for RGB, 1 for Sonar)
            num_classes: Number of classes (not used for tracking)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Enable QKV bias
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate (stochastic depth)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        self.in_chans = in_chans
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ViT weight initialization."""
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def named_apply(fn, module, name='', depth_first=True, include_root=False):
    """Apply function to module with name."""
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)


def _create_vision_transformer(pretrained=False, in_chans=3, **kwargs):
    """Create ViT model with optional pretrained weights."""
    model = VisionTransformer(in_chans=in_chans, **kwargs)

    if pretrained:
        if isinstance(pretrained, str):
            checkpoint_path = pretrained
            if 'npz' in checkpoint_path:
                model.load_pretrained(checkpoint_path, prefix='')
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'net' in checkpoint:
                    state_dict = checkpoint['net']
                else:
                    state_dict = checkpoint

                # Handle channel mismatch for sonar
                if in_chans != 3 and 'patch_embed.proj.weight' in state_dict:
                    pretrained_weight = state_dict['patch_embed.proj.weight']
                    if pretrained_weight.shape[1] == 3:
                        # Average RGB channels for 1-channel input
                        state_dict['patch_embed.proj.weight'] = pretrained_weight.mean(dim=1, keepdim=True)
                        _logger.info(f"Adapted patch_embed from 3-channel to {in_chans}-channel")

                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                _logger.info(f'Loaded pretrained: {checkpoint_path}')
                if missing_keys:
                    _logger.info(f'Missing keys: {missing_keys}')
                if unexpected_keys:
                    _logger.info(f'Unexpected keys: {unexpected_keys}')

    return model


def vit_base_patch16_224(pretrained=False, in_chans=3, **kwargs):
    """ViT-Base model (ViT-B/16).

    Args:
        pretrained: Path to pretrained weights or False
        in_chans: Number of input channels (3 for RGB, 1 for Sonar)
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(pretrained=pretrained, in_chans=in_chans, **model_kwargs)
    return model


def vit_large_patch16_224(pretrained=False, in_chans=3, **kwargs):
    """ViT-Large model (ViT-L/16)."""
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(pretrained=pretrained, in_chans=in_chans, **model_kwargs)
    return model


# Factory functions for AquaSOT-3D

def _resolve_pretrained_path(cfg, pretrained=None):
    """Resolve pretrained checkpoint path.

    Args:
        cfg: Config object
        pretrained: Explicit pretrained path (overrides config)

    Returns:
        Full path to pretrained checkpoint, or None if not specified
    """
    if pretrained:
        checkpoint = pretrained
    else:
        checkpoint = cfg.MODEL.PRETRAIN_FILE

    if not checkpoint:
        return None

    # If already absolute path or exists, use as-is
    if os.path.isabs(checkpoint) or os.path.exists(checkpoint):
        return checkpoint

    # Try with PRETRAINED_DIR
    pretrained_dir = getattr(cfg.PATHS, 'PRETRAINED_DIR', 'pretrained')
    full_path = os.path.join(pretrained_dir, checkpoint)
    if os.path.exists(full_path):
        return full_path

    # Return original (will fail with clear error if not found)
    return checkpoint


def build_vit_rgb(cfg, pretrained=None):
    """Build ViT backbone for RGB input."""
    backbone_cfg = cfg.MODEL.BACKBONE.RGB
    drop_path = cfg.TRAIN.DROP_PATH_RATE if hasattr(cfg.TRAIN, 'DROP_PATH_RATE') else 0.1

    pretrained_path = _resolve_pretrained_path(cfg, pretrained)

    model = vit_base_patch16_224(
        pretrained=pretrained_path,
        in_chans=backbone_cfg.IN_CHANNELS,
        drop_path_rate=drop_path
    )
    model.finetune_track(cfg)
    return model


def build_vit_sonar(cfg, pretrained=None):
    """Build ViT backbone for Sonar input (1-channel)."""
    backbone_cfg = cfg.MODEL.BACKBONE.SONAR
    drop_path = cfg.TRAIN.DROP_PATH_RATE if hasattr(cfg.TRAIN, 'DROP_PATH_RATE') else 0.1

    pretrained_path = _resolve_pretrained_path(cfg, pretrained)

    model = vit_base_patch16_224(
        pretrained=pretrained_path,
        in_chans=backbone_cfg.IN_CHANNELS,
        drop_path_rate=drop_path
    )
    model.finetune_track(cfg)
    return model
