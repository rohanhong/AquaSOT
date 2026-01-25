"""
Base Backbone class for AquaSOT.

Adapted from OSTrack with support for variable input channels.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    """Base class for tracking backbones.

    Implements template-search concatenation and position embedding handling.
    """

    def __init__(self):
        super().__init__()

        # Default ViT settings
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384
        self.in_chans = 3  # Can be 1 for sonar

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):
        """Setup for tracking by resizing position embeddings.

        Args:
            cfg: Configuration object
            patch_start_index: Index where patch embeddings start (after CLS token)
        """
        # Get sizes from config (handle both RGB and SONAR configs)
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)

        # Determine which backbone config to use
        if hasattr(cfg.MODEL.BACKBONE, 'RGB'):
            new_patch_size = cfg.MODEL.BACKBONE.RGB.STRIDE
            cat_mode = cfg.MODEL.BACKBONE.RGB.CAT_MODE
        else:
            new_patch_size = cfg.MODEL.BACKBONE.STRIDE
            cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        self.cat_mode = cat_mode

        # Resize patch embedding if needed
        if new_patch_size != self.patch_size:
            print(f'Interpolating patch embedding: {self.patch_size} -> {new_patch_size}')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = F.interpolate(
                        param, size=(new_patch_size, new_patch_size),
                        mode='bicubic', align_corners=False
                    )
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=new_patch_size,
                in_chans=self.in_chans, embed_dim=self.embed_dim
            )
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # Resize position embeddings
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # For search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False
        )
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # For template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False
        )
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # CLS token position
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # Segment embeddings
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # Intermediate norm layers
        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, z_online=None):
        """Extract features from template and search.

        Args:
            z: Template image [B, C, H_z, W_z]
            x: Search image [B, C, H_x, W_x]
            z_online: Online templates [B, N, C, H_z, W_z] or [B*N, C, H_z, W_z] or None

        Returns:
            x: Combined features [B, L_z + L_x, C] (or [B, L_z*(1+N) + L_x, C] if z_online)
            aux_dict: Auxiliary outputs
        """
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        # Process online templates if provided
        if z_online is not None:
            # Handle different input formats
            if z_online.dim() == 5:
                # [B, N, C, H, W] -> [B*N, C, H, W]
                N = z_online.shape[1]
                z_online = z_online.view(-1, *z_online.shape[2:])
            else:
                # [B*N, C, H, W] - infer N
                N = z_online.shape[0] // B

            # Patch embed online templates
            z_online = self.patch_embed(z_online)  # [B*N, L_z, C]
            z_online += self.pos_embed_z

            if self.add_sep_seg:
                z_online += self.template_segment_pos_embed

            # Reshape to [B, N*L_z, C]
            L_z = z_online.shape[1]
            z_online = z_online.reshape(B, N * L_z, -1)

            # Concatenate first-frame template with online templates
            z = torch.cat([z, z_online], dim=1)  # [B, L_z*(1+N), C]

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        # Calculate template length (including online templates)
        lens_z_base = self.pos_embed_z.shape[1]
        if z_online is not None:
            lens_z = lens_z_base * (1 + N)
        else:
            lens_z = lens_z_base
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None, "lens_z": lens_z, "lens_x": lens_x}
        return self.norm(x), aux_dict

    def forward(self, z, x, z_online=None, **kwargs):
        """Joint feature extraction for template and search.

        Args:
            z: Template [B, C, H_z, W_z]
            x: Search [B, C, H_x, W_x]
            z_online: Online templates [B, N, C, H_z, W_z] or [B*N, C, H_z, W_z] or None

        Returns:
            x: Merged features [B, L_z + L_x, C]
            aux_dict: Auxiliary outputs
        """
        x, aux_dict = self.forward_features(z, x, z_online)
        return x, aux_dict
