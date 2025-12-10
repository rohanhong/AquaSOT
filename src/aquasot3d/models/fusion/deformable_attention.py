"""
Deformable Cross-Modal Attention for DCMA Fusion

This module implements deformable attention with learnable sampling offsets
for cross-modal alignment between RGB and Sonar. Unlike standard attention that
operates on regular grids, deformable attention predicts sparse sampling locations
to handle spatial misalignment between modalities.

Components:
- DeformableOffsetNetwork: Predict sampling offsets from query features
- BilinearSampler: Sample features at offset locations using bilinear interpolation
- DeformableCrossAttention: Cross-modal attention with deformable sampling
- MultiScaleDeformableAttention: Multi-scale variant with pyramid features

Reference:
    Deformable DETR (ICLR 2021): https://arxiv.org/abs/2010.04159
    AquaSOT-3D Technical Specification Section 3(c): Deformable Offsets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from aquasot3d.core.base_module import BaseModule
from aquasot3d.core.registry import FUSION_REGISTRY


class DeformableOffsetNetwork(BaseModule):
    """
    Predict sparse sampling offsets for deformable attention.

    Given query features (RGB), this network predicts 2D offsets (Δx, Δy)
    for each sampling point. These offsets allow the attention mechanism
    to adaptively sample from key features (Sonar) at non-grid locations.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_points: Number of sampling points per head
        offset_scale: Scale factor for offsets (default 1.0)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        offset_scale: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.offset_scale = offset_scale

        # Offset prediction network
        # Output: [num_heads, num_points, 2] offsets per query location
        self.offset_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_heads * num_points * 2)
        )

        # Initialize offsets to zero (identity initialization)
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Predict sampling offsets from query features.

        Args:
            query_features: Query features [B, N_query, C]

        Returns:
            offsets: Predicted offsets [B, N_query, num_heads, num_points, 2]
                    Values in normalized coordinates [-1, 1]
        """
        B, N_query, C = query_features.shape

        # Predict offsets
        offsets = self.offset_predictor(query_features)  # [B, N_query, num_heads*num_points*2]
        offsets = offsets.view(B, N_query, self.num_heads, self.num_points, 2)

        # Scale offsets
        offsets = torch.tanh(offsets) * self.offset_scale

        return offsets


class BilinearSampler(BaseModule):
    """
    Sample features at arbitrary locations using bilinear interpolation.

    Given a feature map and sampling coordinates, this module performs
    differentiable bilinear sampling, allowing gradient flow for learning
    optimal sampling locations.

    Args:
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection')
    """

    def __init__(self, padding_mode: str = 'zeros'):
        super().__init__()
        self.padding_mode = padding_mode

    def forward(
        self,
        feature_map: torch.Tensor,
        sampling_locations: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample features at specified locations.

        Args:
            feature_map: Feature map [B, C, H, W]
            sampling_locations: Normalized coordinates [B, N_samples, 2]
                               Values in [-1, 1] where (-1,-1) is top-left

        Returns:
            sampled_features: Features at sampling locations [B, N_samples, C]
        """
        B, C, H, W = feature_map.shape
        N_samples = sampling_locations.shape[1]

        # Reshape sampling_locations for grid_sample
        # grid_sample expects [B, H_out, W_out, 2]
        grid = sampling_locations.unsqueeze(1)  # [B, 1, N_samples, 2]

        # Sample features using bilinear interpolation
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=False
        )  # [B, C, 1, N_samples]

        # Reshape to [B, N_samples, C]
        sampled = sampled.squeeze(2).permute(0, 2, 1)

        return sampled


@FUSION_REGISTRY.register("deformable_cross_attention")
class DeformableCrossAttention(BaseModule):
    """
    Deformable Cross-Modal Attention between RGB and Sonar.

    This module implements cross-attention with learnable sampling offsets.
    Instead of attending to all key locations, it predicts sparse sampling
    points and performs attention only at those locations, enabling flexible
    spatial alignment between misaligned modalities.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_points: Number of sampling points per head
        dropout: Dropout probability
        offset_scale: Scale factor for offsets
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
        offset_scale: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Offset prediction
        self.offset_network = DeformableOffsetNetwork(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_points=num_points,
            offset_scale=offset_scale
        )

        # Bilinear sampler
        self.sampler = BilinearSampler(padding_mode='zeros')

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Attention weights for each sampling point
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def compute_reference_points(
        self,
        H: int,
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute reference points (grid centers) in normalized coordinates.

        Args:
            H: Height of feature map
            W: Width of feature map
            device: Device for tensor allocation

        Returns:
            reference_points: Grid coordinates [H*W, 2] in [-1, 1]
        """
        # Create normalized grid
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack and flatten
        reference_points = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        reference_points = reference_points.view(-1, 2)  # [H*W, 2]

        return reference_points

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        value_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply deformable cross-attention.

        Args:
            query_features: Query features (RGB) [B, C, H_q, W_q]
            key_features: Key features (Sonar) [B, C, H_k, W_k]
            value_features: Value features (default: same as key) [B, C, H_k, W_k]

        Returns:
            dict containing:
                - fused_features: Attended features [B, C, H_q, W_q]
                - offsets: Predicted offsets [B, N_q, num_heads, num_points, 2]
                - sampling_locations: Absolute sampling locations [B, N_q, num_heads, num_points, 2]
        """
        if value_features is None:
            value_features = key_features

        B, C_q, H_q, W_q = query_features.shape
        _, C_k, H_k, W_k = key_features.shape
        _, C_v, H_v, W_v = value_features.shape
        N_q = H_q * W_q
        N_k = H_k * W_k
        N_v = H_v * W_v

        # Flatten spatial dimensions
        query_flat = query_features.flatten(2).permute(0, 2, 1)  # [B, N_q, C_q]
        key_flat = key_features.flatten(2).permute(0, 2, 1)  # [B, N_k, C_k]
        value_flat = value_features.flatten(2).permute(0, 2, 1)  # [B, N_v, C_v]

        # Check if feature dimensions match embed_dim, if not, project them
        if C_q != self.embed_dim:
            if not hasattr(self, 'query_input_proj'):
                self.query_input_proj = nn.Linear(C_q, self.embed_dim).to(query_flat.device)
            query_flat = self.query_input_proj(query_flat)  # [B, N_q, embed_dim]

        if C_k != self.embed_dim:
            if not hasattr(self, 'key_input_proj'):
                self.key_input_proj = nn.Linear(C_k, self.embed_dim).to(key_flat.device)
            key_flat = self.key_input_proj(key_flat)  # [B, N_k, embed_dim]

        if C_v != self.embed_dim:
            if not hasattr(self, 'value_input_proj'):
                self.value_input_proj = nn.Linear(C_v, self.embed_dim).to(value_flat.device)
            value_flat = self.value_input_proj(value_flat)  # [B, N_k, embed_dim]

        # Project to Q, K, V
        q = self.q_proj(query_flat)  # [B, N_q, embed_dim]
        k = self.k_proj(key_flat)  # [B, N_k, embed_dim]
        v = self.v_proj(value_flat)  # [B, N_k, embed_dim]

        # Reshape query to multi-head
        q = q.view(B, N_q, self.num_heads, self.head_dim)  # [B, N_q, num_heads, head_dim]

        # Reshape projected k and v back to spatial for sampling
        # Note: k and v may have different spatial dimensions if value_features != key_features
        k_spatial = k.view(B, H_k, W_k, -1).permute(0, 3, 1, 2)  # [B, embed_dim, H_k, W_k]
        v_spatial = v.view(B, H_v, W_v, -1).permute(0, 3, 1, 2)  # [B, embed_dim, H_v, W_v]

        # Predict sampling offsets
        offsets = self.offset_network(query_flat)  # [B, N_q, num_heads, num_points, 2]

        # Compute reference points (query grid centers)
        reference_points = self.compute_reference_points(H_q, W_q, query_features.device)  # [N_q, 2]
        reference_points = reference_points.unsqueeze(0).expand(B, -1, -1)  # [B, N_q, 2]

        # Compute absolute sampling locations
        # reference_points: [B, N_q, 2] -> [B, N_q, 1, 1, 2]
        # offsets: [B, N_q, num_heads, num_points, 2]
        reference_expanded = reference_points.unsqueeze(2).unsqueeze(3)  # [B, N_q, 1, 1, 2]
        sampling_locations = reference_expanded + offsets  # [B, N_q, num_heads, num_points, 2]

        # Clamp to valid range
        sampling_locations = torch.clamp(sampling_locations, -1.0, 1.0)

        # For each head, sample key and value features at offset locations
        # We need to sample separately for each head or reshape properly after sampling
        k_sampled_list = []
        v_sampled_list = []

        for head_idx in range(self.num_heads):
            # Get sampling locations for this head: [B, N_q, num_points, 2]
            head_sampling_locs = sampling_locations[:, :, head_idx, :, :]  # [B, N_q, num_points, 2]
            head_sampling_flat = head_sampling_locs.reshape(B, -1, 2)  # [B, N_q*num_points, 2]

            # Sample from the head-specific channel slice
            # Extract features for this head from k_spatial and v_spatial
            head_start = head_idx * self.head_dim
            head_end = (head_idx + 1) * self.head_dim

            k_head_spatial = k_spatial[:, head_start:head_end, :, :]  # [B, head_dim, H_k, W_k]
            v_head_spatial = v_spatial[:, head_start:head_end, :, :]  # [B, head_dim, H_v, W_v]

            # Sample features
            k_head_sampled = self.sampler(k_head_spatial, head_sampling_flat)  # [B, N_q*num_points, head_dim]
            v_head_sampled = self.sampler(v_head_spatial, head_sampling_flat)  # [B, N_q*num_points, head_dim]

            # Reshape to [B, N_q, num_points, head_dim]
            k_head_sampled = k_head_sampled.view(B, N_q, self.num_points, self.head_dim)
            v_head_sampled = v_head_sampled.view(B, N_q, self.num_points, self.head_dim)

            k_sampled_list.append(k_head_sampled)
            v_sampled_list.append(v_head_sampled)

        # Stack heads: [B, N_q, num_heads, num_points, head_dim]
        k_sampled = torch.stack(k_sampled_list, dim=2)  # [B, N_q, num_heads, num_points, head_dim]
        v_sampled = torch.stack(v_sampled_list, dim=2)  # [B, N_q, num_heads, num_points, head_dim]

        # Compute attention scores for each sampling point
        # q: [B, N_q, num_heads, head_dim]
        # k_sampled: [B, N_q, num_heads, num_points, head_dim]
        attn_scores = torch.einsum(
            'bnhd,bnhpd->bnhp',
            q,
            k_sampled
        ) * self.scale  # [B, N_q, num_heads, num_points]

        # Apply softmax over sampling points
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N_q, num_heads, num_points]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to sampled values
        # attn_weights: [B, N_q, num_heads, num_points]
        # v_sampled: [B, N_q, num_heads, num_points, head_dim]
        attn_output = torch.einsum(
            'bnhp,bnhpd->bnhd',
            attn_weights,
            v_sampled
        )  # [B, N_q, num_heads, head_dim]

        # Reshape and project output
        attn_output = attn_output.reshape(B, N_q, self.embed_dim)  # [B, N_q, embed_dim]
        attn_output = self.out_proj(attn_output)  # [B, N_q, embed_dim]

        # If input query dimension was different, project back to original dimension
        if C_q != self.embed_dim:
            if not hasattr(self, 'query_output_proj'):
                self.query_output_proj = nn.Linear(self.embed_dim, C_q).to(attn_output.device)
            attn_output = self.query_output_proj(attn_output)  # [B, N_q, C_q]

        # Reshape back to spatial dimensions
        output_dim = C_q if C_q != self.embed_dim else self.embed_dim
        fused_features = attn_output.permute(0, 2, 1).view(B, output_dim, H_q, W_q)  # [B, C_q, H_q, W_q]

        return {
            'fused_features': fused_features,
            'offsets': offsets,
            'sampling_locations': sampling_locations,
            'attention_weights': attn_weights
        }


@FUSION_REGISTRY.register("multi_scale_deformable_attention")
class MultiScaleDeformableAttention(BaseModule):
    """
    Multi-scale variant of deformable cross-attention.

    Performs deformable attention across multiple feature pyramid levels,
    enabling the network to attend to features at different scales.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_levels: Number of pyramid levels
        num_points: Number of sampling points per head per level
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Per-level deformable attention
        self.level_attentions = nn.ModuleList([
            DeformableCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_points=num_points,
                dropout=dropout
            )
            for _ in range(num_levels)
        ])

        # Level fusion weights
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

    def forward(
        self,
        query_pyramid: List[torch.Tensor],
        key_pyramid: List[torch.Tensor],
        value_pyramid: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply multi-scale deformable attention.

        Args:
            query_pyramid: List of query features at different scales
            key_pyramid: List of key features at different scales
            value_pyramid: List of value features (default: same as key)

        Returns:
            dict containing:
                - fused_pyramid: List of attended features per level
                - level_weights: Normalized fusion weights [num_levels]
        """
        if value_pyramid is None:
            value_pyramid = key_pyramid

        assert len(query_pyramid) == self.num_levels
        assert len(key_pyramid) == self.num_levels
        assert len(value_pyramid) == self.num_levels

        # Apply deformable attention at each level
        level_outputs = []
        for i, (attn, q, k, v) in enumerate(zip(
            self.level_attentions,
            query_pyramid,
            key_pyramid,
            value_pyramid
        )):
            output = attn(q, k, v)
            level_outputs.append(output['fused_features'])

        # Normalize level weights
        level_weights = F.softmax(self.level_weights, dim=0)

        # Weighted fusion of levels (resize to match first level)
        B, C, H, W = query_pyramid[0].shape
        fused = torch.zeros(B, C, H, W, device=query_pyramid[0].device)

        for i, (feat, weight) in enumerate(zip(level_outputs, level_weights)):
            # Resize to target size
            feat_resized = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            fused = fused + weight * feat_resized

        return {
            'fused_pyramid': level_outputs,
            'fused_features': fused,
            'level_weights': level_weights
        }


def build_deformable_cross_attention(config: dict) -> DeformableCrossAttention:
    """
    Factory function to build DeformableCrossAttention from config.

    Args:
        config: Configuration dictionary

    Returns:
        DeformableCrossAttention instance
    """
    return DeformableCrossAttention(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_points=config.get('num_points', 4),
        dropout=config.get('dropout', 0.1),
        offset_scale=config.get('offset_scale', 1.0)
    )


def build_multi_scale_deformable_attention(config: dict) -> MultiScaleDeformableAttention:
    """
    Factory function to build MultiScaleDeformableAttention from config.

    Args:
        config: Configuration dictionary

    Returns:
        MultiScaleDeformableAttention instance
    """
    return MultiScaleDeformableAttention(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_levels=config.get('num_levels', 3),
        num_points=config.get('num_points', 4),
        dropout=config.get('dropout', 0.1)
    )
