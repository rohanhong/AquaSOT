"""
Geometric Gating Module for G2-DCMA Fusion

This module implements geometry-aware gating for cross-modal attention between
RGB and Sonar modalities. It uses radial activation from sonar and scale information
from RGB to modulate attention weights, amplifying matches and suppressing mismatches.

Components:
- RadialProfileExtractor: Extract radial activation distribution from sonar features
- ScalePyramidExtractor: Extract scale/depth information from RGB feature pyramid
- GeometricGatingFunction: Learnable gating function g(r, s)
- GeometryGatedAttention: Apply geometric gating to cross-modal attention

Reference:
    AquaSOT-3D Technical Specification Section 3(b): Geometry-Gated Cross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from aquasot3d.core.base_module import BaseModule
from aquasot3d.core.registry import FUSION_REGISTRY


class RadialProfileExtractor(BaseModule):
    """
    Extract radial activation profile from sonar feature maps.

    Sonar sensors have fan-shaped field of view with radial structure.
    This module extracts a 1D radial activation profile that encodes
    distance information from the sensor origin.

    Args:
        num_bins: Number of radial bins for discretization
        feature_dim: Dimensionality of input feature maps
        learnable_weights: Whether to use learnable aggregation weights
    """

    def __init__(
        self,
        num_bins: int = 50,
        feature_dim: int = 256,
        learnable_weights: bool = True
    ):
        super().__init__()
        self.num_bins = num_bins
        self.feature_dim = feature_dim
        self.learnable_weights = learnable_weights

        # Learnable channel-wise weights for aggregation
        if learnable_weights:
            self.channel_weights = nn.Parameter(torch.ones(feature_dim))
            self.channel_weights.data.fill_(1.0 / feature_dim)

        # Projection layer to encode radial profile
        self.profile_encoder = nn.Sequential(
            nn.Linear(num_bins, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )

    def extract_radial_bins(
        self,
        feature_map: torch.Tensor,
        sensor_position: str = "bottom_center"
    ) -> torch.Tensor:
        """
        Extract radial binned activations from sonar feature map.

        Args:
            feature_map: Sonar features [B, C, H, W]
            sensor_position: Position of sonar sensor origin
                           ("bottom_center" for upward-facing FLS)

        Returns:
            radial_profile: Radial activation distribution [B, num_bins]
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # Sensor position in feature map coordinates
        if sensor_position == "bottom_center":
            cx, cy = W // 2, H - 1
        elif sensor_position == "center":
            cx, cy = W // 2, H // 2
        else:
            raise ValueError(f"Unknown sensor_position: {sensor_position}")

        # Create coordinate grid
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Calculate radial distance from sensor origin
        r = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)  # [H, W]

        # Normalize to bin indices [0, num_bins)
        r_max = r.max()
        r_normalized = (r / (r_max + 1e-6)) * (self.num_bins - 1)
        r_bins = r_normalized.long().clamp(0, self.num_bins - 1)  # [H, W]

        # Check if channel weights need adjustment
        if self.learnable_weights and self.channel_weights.shape[0] != C:
            # Dynamically adjust channel weights to match input dimension
            # This handles cases where input dimension differs from initialization
            if C > self.feature_dim:
                # Pad with zeros
                padding = torch.zeros(C - self.feature_dim, device=device)
                adjusted_weights = torch.cat([self.channel_weights, padding], dim=0)
            else:
                # Truncate
                adjusted_weights = self.channel_weights[:C]
            # Renormalize
            adjusted_weights = adjusted_weights / (adjusted_weights.sum() + 1e-6)
        else:
            adjusted_weights = self.channel_weights if self.learnable_weights else None

        # Aggregate feature activations per bin using vectorized operations
        radial_profiles = []
        for b in range(B):
            # Weighted channel aggregation
            if self.learnable_weights:
                feat = (feature_map[b] * adjusted_weights.view(C, 1, 1)).sum(dim=0)  # [H, W]
            else:
                feat = feature_map[b].mean(dim=0)  # [H, W]

            # Vectorized bin-wise aggregation using scatter_add
            feat_flat = feat.flatten()  # [H*W]
            r_bins_flat = r_bins.flatten()  # [H*W]

            # Accumulate values and counts for each bin
            profile = torch.zeros(self.num_bins, device=device)
            counts = torch.zeros(self.num_bins, device=device)

            profile.scatter_add_(0, r_bins_flat, feat_flat)
            counts.scatter_add_(0, r_bins_flat, torch.ones_like(feat_flat))

            # Compute mean: divide accumulated values by counts (single normalization)
            profile = profile / (counts + 1e-6)
            radial_profiles.append(profile)

        radial_profile = torch.stack(radial_profiles, dim=0)  # [B, num_bins]
        return radial_profile

    def forward(
        self,
        sonar_features: torch.Tensor,
        sensor_position: str = "bottom_center"
    ) -> torch.Tensor:
        """
        Extract and encode radial profile.

        Args:
            sonar_features: Sonar feature maps [B, C, H, W]
            sensor_position: Sensor origin position

        Returns:
            radial_encoding: Encoded radial profile [B, 64]
        """
        radial_profile = self.extract_radial_bins(sonar_features, sensor_position)
        radial_encoding = self.profile_encoder(radial_profile)
        return radial_encoding


class ScalePyramidExtractor(BaseModule):
    """
    Extract scale/depth information from RGB feature pyramid.

    RGB images have scale-dependent information that correlates with depth.
    Larger bounding boxes typically indicate closer objects, smaller boxes
    indicate farther objects. This module extracts scale features from
    multi-scale RGB feature pyramid.

    Args:
        feature_dim: Dimensionality of input features
        num_scales: Number of pyramid levels to extract
        use_attention: Whether to use attention-based aggregation
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_scales: int = 3,
        use_attention: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.use_attention = use_attention

        # Channel dimension adapters (registered as ModuleDict for state dict persistence)
        self.channel_adapters = nn.ModuleDict()

        # Scale-specific processing layers
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
            for _ in range(num_scales)
        ])

        # Attention-based scale aggregation
        if use_attention:
            self.scale_attention = nn.Sequential(
                nn.Linear(num_scales * (feature_dim // 2), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_scales),
                nn.Softmax(dim=-1)
            )
            # Encoder for attention-aggregated features
            encoder_input_dim = feature_dim // 2
        else:
            encoder_input_dim = num_scales * (feature_dim // 2)

        # Projection to scale encoding
        self.scale_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        pyramid_levels: Optional[list] = None
    ) -> torch.Tensor:
        """
        Extract scale encoding from RGB features.

        Args:
            rgb_features: RGB feature maps [B, C, H, W] or list of pyramid levels
            pyramid_levels: Optional list of feature maps at different scales

        Returns:
            scale_encoding: Encoded scale information [B, 64]
        """
        if pyramid_levels is None:
            # Generate pyramid from single feature map
            pyramid_levels = [rgb_features]
            for i in range(1, self.num_scales):
                downsampled = F.avg_pool2d(rgb_features, kernel_size=2**i)
                pyramid_levels.append(downsampled)

        B = rgb_features.shape[0] if not isinstance(rgb_features, list) else pyramid_levels[0].shape[0]

        # Check if input dimension matches expected dimension
        actual_dim = pyramid_levels[0].shape[1]
        if actual_dim != self.feature_dim:
            # Get or create channel adapter for this dimension
            adapter_key = str(actual_dim)
            if adapter_key not in self.channel_adapters:
                # Create and register new adapter for this input dimension
                self.channel_adapters[adapter_key] = nn.Conv2d(
                    actual_dim, self.feature_dim, kernel_size=1
                )
            # Use registered adapter
            projection = self.channel_adapters[adapter_key]
            # Project all pyramid levels to expected dimension
            pyramid_levels = [projection(level) for level in pyramid_levels[:self.num_scales]]

        # Extract scale-specific features
        scale_features = []
        for i, (level, conv) in enumerate(zip(pyramid_levels[:self.num_scales], self.scale_convs)):
            feat = conv(level).view(B, -1)  # [B, feature_dim//2]
            scale_features.append(feat)

        scale_features = torch.cat(scale_features, dim=-1)  # [B, num_scales * (feature_dim//2)]

        # Attention-based aggregation (optional)
        if self.use_attention:
            attn_weights = self.scale_attention(scale_features)  # [B, num_scales]
            # Reshape for weighted aggregation
            scale_features_reshaped = scale_features.view(B, self.num_scales, -1)
            scale_features = (scale_features_reshaped * attn_weights.unsqueeze(-1)).sum(dim=1)
            scale_features = scale_features.view(B, -1)

        # Encode scale information
        scale_encoding = self.scale_encoder(scale_features)
        return scale_encoding


class GeometricGatingFunction(BaseModule):
    """
    Learnable gating function g(r, s) that computes geometry-aware weights.

    This function learns to amplify attention when radial distance (r) from sonar
    and scale (s) from RGB are geometrically consistent (e.g., large scale + close range
    or small scale + far range), and suppress when inconsistent.

    Args:
        radial_dim: Dimensionality of radial encoding
        scale_dim: Dimensionality of scale encoding
        hidden_dim: Hidden layer dimension
        num_heads: Number of attention heads (for per-head gating)
    """

    def __init__(
        self,
        radial_dim: int = 64,
        scale_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 8
    ):
        super().__init__()
        self.radial_dim = radial_dim
        self.scale_dim = scale_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Geometric consistency network
        self.gating_net = nn.Sequential(
            nn.Linear(radial_dim + scale_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Sigmoid()  # Output in [0, 1] for gating
        )

    def forward(
        self,
        radial_encoding: torch.Tensor,
        scale_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geometry-aware gating weights.

        Args:
            radial_encoding: Radial profile encoding from sonar [B, radial_dim]
            scale_encoding: Scale pyramid encoding from RGB [B, scale_dim]

        Returns:
            gating_weights: Per-head gating weights [B, num_heads]
        """
        # Concatenate geometric features
        geo_features = torch.cat([radial_encoding, scale_encoding], dim=-1)  # [B, radial_dim + scale_dim]

        # Compute gating weights
        gating_weights = self.gating_net(geo_features)  # [B, num_heads]

        return gating_weights


@FUSION_REGISTRY.register("geometry_gated_attention")
class GeometryGatedAttention(BaseModule):
    """
    Geometry-Gated Cross-Modal Attention.

    Applies geometric gating to cross-attention between RGB and Sonar modalities.
    The gating weights modulate attention scores based on geometric consistency
    between radial distance (sonar) and scale (RGB).

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_radial_bins: Number of radial bins for sonar
        num_scales: Number of pyramid scales for RGB
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_radial_bins: int = 50,
        num_scales: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Geometric feature extractors
        self.radial_extractor = RadialProfileExtractor(
            num_bins=num_radial_bins,
            feature_dim=embed_dim
        )
        self.scale_extractor = ScalePyramidExtractor(
            feature_dim=embed_dim,
            num_scales=num_scales
        )

        # Gating function
        self.gating_function = GeometricGatingFunction(
            radial_dim=64,
            scale_dim=64,
            num_heads=num_heads
        )

        # Cross-attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dynamic dimension adapters (registered for state dict persistence)
        self.rgb_adapters = nn.ModuleDict()
        self.sonar_adapters = nn.ModuleDict()
        self.rgb_out_adapters = nn.ModuleDict()

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_pyramid: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply geometry-gated cross-attention.

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional multi-scale RGB features

        Returns:
            dict containing:
                - fused_features: Gated cross-modal features [B, C, H, W]
                - gating_weights: Geometric gating weights [B, num_heads]
                - radial_encoding: Radial profile encoding [B, 64]
                - scale_encoding: Scale pyramid encoding [B, 64]
        """
        B, C_rgb, H_rgb, W_rgb = rgb_features.shape
        _, C_sonar, H_sonar, W_sonar = sonar_features.shape

        # Check if feature dimensions match embed_dim, if not, project them
        if C_rgb != self.embed_dim:
            # Get or create RGB adapter for this dimension
            adapter_key = str(C_rgb)
            if adapter_key not in self.rgb_adapters:
                self.rgb_adapters[adapter_key] = nn.Linear(C_rgb, self.embed_dim)
            rgb_proj = self.rgb_adapters[adapter_key]
            rgb_flat = rgb_features.flatten(2).permute(0, 2, 1)  # [B, H_rgb*W_rgb, C_rgb]
            rgb_flat = rgb_proj(rgb_flat)  # [B, H_rgb*W_rgb, embed_dim]
            rgb_features_proj = rgb_flat.permute(0, 2, 1).view(B, self.embed_dim, H_rgb, W_rgb)
        else:
            rgb_features_proj = rgb_features
            rgb_flat = rgb_features.flatten(2).permute(0, 2, 1)

        if C_sonar != self.embed_dim:
            # Get or create Sonar adapter for this dimension
            adapter_key = str(C_sonar)
            if adapter_key not in self.sonar_adapters:
                self.sonar_adapters[adapter_key] = nn.Linear(C_sonar, self.embed_dim)
            sonar_proj = self.sonar_adapters[adapter_key]
            sonar_flat = sonar_features.flatten(2).permute(0, 2, 1)  # [B, H_sonar*W_sonar, C_sonar]
            sonar_flat = sonar_proj(sonar_flat)  # [B, H_sonar*W_sonar, embed_dim]
        else:
            sonar_flat = sonar_features.flatten(2).permute(0, 2, 1)

        # Extract geometric encodings (using original features)
        radial_encoding = self.radial_extractor(sonar_features)  # [B, 64]
        scale_encoding = self.scale_extractor(rgb_features, rgb_pyramid)  # [B, 64]

        # Compute gating weights
        gating_weights = self.gating_function(radial_encoding, scale_encoding)  # [B, num_heads]

        N_rgb = H_rgb * W_rgb
        N_sonar = H_sonar * W_sonar

        # Project to Q, K, V
        q = self.q_proj(rgb_flat).view(B, N_rgb, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_rgb, head_dim]
        k = self.k_proj(sonar_flat).view(B, N_sonar, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_sonar, head_dim]
        v = self.v_proj(sonar_flat).view(B, N_sonar, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_sonar, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_rgb, N_sonar]

        # Apply geometric gating
        # Gating weights shape: [B, num_heads] -> [B, num_heads, 1, 1]
        gating_weights_expanded = gating_weights.unsqueeze(-1).unsqueeze(-1)
        attn_scores = attn_scores * gating_weights_expanded  # Modulate by gating

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N_rgb, N_sonar]
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, N_rgb, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_rgb, self.embed_dim)  # [B, N_rgb, embed_dim]
        attn_output = self.out_proj(attn_output)  # [B, N_rgb, embed_dim]

        # If input RGB dimension was different, project back to original dimension
        if C_rgb != self.embed_dim:
            # Get or create RGB output adapter for this dimension
            adapter_key = str(C_rgb)
            if adapter_key not in self.rgb_out_adapters:
                self.rgb_out_adapters[adapter_key] = nn.Linear(self.embed_dim, C_rgb)
            rgb_out_proj = self.rgb_out_adapters[adapter_key]
            attn_output = rgb_out_proj(attn_output)  # [B, N_rgb, C_rgb]

        # Reshape back to spatial dimensions
        output_dim = C_rgb if C_rgb != self.embed_dim else self.embed_dim
        fused_features = attn_output.permute(0, 2, 1).view(B, output_dim, H_rgb, W_rgb)  # [B, C_rgb, H_rgb, W_rgb]

        return {
            'fused_features': fused_features,
            'gating_weights': gating_weights,
            'radial_encoding': radial_encoding,
            'scale_encoding': scale_encoding,
            'attention_probs': attn_probs.mean(dim=1)  # Average over heads for visualization
        }


def build_geometry_gated_attention(config: dict) -> GeometryGatedAttention:
    """
    Factory function to build GeometryGatedAttention from config.

    Args:
        config: Configuration dictionary

    Returns:
        GeometryGatedAttention instance
    """
    return GeometryGatedAttention(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_radial_bins=config.get('num_radial_bins', 50),
        num_scales=config.get('num_scales', 3),
        dropout=config.get('dropout', 0.1)
    )