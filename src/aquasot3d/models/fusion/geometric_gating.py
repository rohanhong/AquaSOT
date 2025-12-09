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
from typing import Tuple, Optional, Dict, Union, List

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
        default_target_size: Default letterbox target size for stride estimation
                            when metadata is incomplete (default: 256 for search, OSTrack-256)
    """

    def __init__(
        self,
        num_bins: int = 50,
        feature_dim: int = 256,
        learnable_weights: bool = True,
        default_target_size: int = 256  # OSTrack-256 standard
    ):
        super().__init__()
        self.num_bins = num_bins
        self.feature_dim = feature_dim
        self.learnable_weights = learnable_weights
        # CRITICAL FIX (Audit 2025112509 Bug B2): Add explicit default_target_size parameter
        # This provides a reliable fallback when metadata is incomplete, avoiding fragile inference
        self.default_target_size = default_target_size

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
        sensor_position: str = "bottom_center",
        letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> torch.Tensor:
        """
        Extract radial binned activations from sonar feature map.

        CRITICAL FIX (Audit 2025112403 Finding 3): Letterbox-aware origin computation
        Letterboxing moves the physical sensor origin. Without compensation, radial bins
        treat padding as "near range", corrupting geometry-based gating.

        ENHANCED FIX (Audit 2025112403 Follow-up): Per-sample metadata support
        Now accepts list of dicts for heterogeneous batches with different letterbox params.

        Args:
            feature_map: Sonar features [B, C, H, W]
            sensor_position: Position of sonar sensor origin
                           ("bottom_center" for upward-facing FLS)
            letterbox_metadata: Either:
                              - Single dict: used for all samples (backward compatible)
                              - List of dicts: per-sample metadata for heterogeneous batches
                              Each dict contains 'scale', 'pad_top', 'original_height'

        Returns:
            radial_profile: Radial activation distribution [B, num_bins]
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # Normalize metadata to list form for unified processing
        if letterbox_metadata is None:
            metadata_list = [None] * B
        elif isinstance(letterbox_metadata, dict):
            # Single dict: use for all samples (backward compatible)
            metadata_list = [letterbox_metadata] * B
        elif isinstance(letterbox_metadata, list):
            # Per-sample metadata
            if len(letterbox_metadata) != B:
                raise ValueError(f"letterbox_metadata list length {len(letterbox_metadata)} must match batch size {B}")
            metadata_list = letterbox_metadata
        else:
            raise TypeError(f"letterbox_metadata must be dict, list, or None, got {type(letterbox_metadata)}")

        # Compute per-sample sensor origins
        sensor_origins = []  # List of (cx, cy) tuples
        for meta in metadata_list:
            # CRITICAL FIX (Audit 2025120909): Use direct sensor_origin_x/y if available
            # These are pre-computed in crop_metadata_to_letterbox considering:
            # - Source image dimensions (where sensor actually is)
            # - Crop position relative to source
            # - Resize factor and padding
            # The old approach assumed sensor at bottom of CROP, which was WRONG for
            # target-centered crops that don't include the sensor origin.
            if meta is not None and 'sensor_origin_x' in meta and 'sensor_origin_y' in meta:
                # Best case: use pre-computed sensor origin in output coordinates
                # Note: sensor_origin is in PIXEL (letterbox output) coordinates
                # Keep in pixel coordinates - later code (y_grid_pixels) also uses pixel space
                cx = meta['sensor_origin_x']
                cy = meta['sensor_origin_y']
            elif meta is not None and 'scale' in meta:
                # Fallback: old computation (for backward compatibility with non-crop metadata)
                scale = meta.get('scale', 1.0)
                pad_top = meta.get('pad_top', 0)
                original_height = meta.get('original_height', H)

                if sensor_position == "bottom_center":
                    # Physical origin at bottom-center: (W/2, original_height-1)
                    # After letterbox: (W/2, pad_top + (original_height-1) * scale)
                    cx = W // 2
                    cy = pad_top + (original_height - 1) * scale
                elif sensor_position == "center":
                    cx = W // 2
                    cy = pad_top + (original_height // 2) * scale
                else:
                    raise ValueError(f"Unknown sensor_position: {sensor_position}")
            else:
                # Fallback: no letterbox compensation (backward compatibility)
                if sensor_position == "bottom_center":
                    cx, cy = W // 2, H - 1
                elif sensor_position == "center":
                    cx, cy = W // 2, H // 2
                else:
                    raise ValueError(f"Unknown sensor_position: {sensor_position}")

            sensor_origins.append((cx, cy))

        # Create coordinate grid (shared across batch)
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

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

        # CRITICAL FIX (Audit 2025112502 Finding 3a): Stride scaling to unify coordinate systems
        # Problem: Feature map grid (e.g., 32x32) and letterbox pixel coordinates (e.g., 512x512)
        #          are in different scales. Without scaling, distance computation is incorrect.
        # Solution: Compute stride (pixels_per_feature_cell) and scale grid to pixel space
        #
        # REVISION (Audit 2025112502): Per-sample stride computation for heterogeneous batches
        # Previous approach used only first sample's metadata and incorrect padding formula.
        # New approach: For each sample, infer letterbox target size from metadata or use default.

        # Aggregate feature activations per bin using per-sample origins
        radial_profiles = []
        for b in range(B):
            # Get per-sample sensor origin (in pixel coordinates)
            cx, cy = sensor_origins[b]

            # CRITICAL FIX: Compute per-sample stride
            # Letterbox target size is typically fixed (e.g., 512x512 for training)
            # stride = target_size / feature_map_size
            meta = metadata_list[b]

            # CRITICAL FIX (Audit 2025112502 Final): Support evaluation metadata keys
            # Training uses: scale, pad_top, original_height, new_size
            # Evaluation uses: scale_x, scale_y, pad_x, pad_y, original_height, original_width
            # Map evaluation keys to training keys for unified processing
            if meta is not None:
                # Create unified metadata with training key names
                unified_meta = {}
                # Map scale (use scale_y for vertical, which matters for stride)
                if 'scale' in meta:
                    unified_meta['scale'] = meta['scale']
                elif 'scale_y' in meta:
                    unified_meta['scale'] = meta['scale_y']  # Use vertical scale

                # Map pad_top (evaluation uses pad_y for top padding)
                if 'pad_top' in meta:
                    unified_meta['pad_top'] = meta['pad_top']
                elif 'pad_y' in meta:
                    unified_meta['pad_top'] = meta['pad_y']  # Evaluation top padding

                # Map original_height (should already be consistent)
                if 'original_height' in meta:
                    unified_meta['original_height'] = meta['original_height']

                # Preserve new_size if exists (training-specific)
                if 'new_size' in meta:
                    unified_meta['new_size'] = meta['new_size']

                meta = unified_meta if unified_meta else meta

            # CRITICAL FIX (Audit 2025120902 Issue 11): Use explicit output_size from metadata
            # Previous approach used heuristic estimation which was fragile
            # Now metadata contains explicit output_size from preprocessing, use it directly
            if meta is not None and 'output_size' in meta:
                # Best case: output_size is explicitly stored in metadata
                # This is the most reliable method, set during preprocessing
                target_size = meta['output_size']
                if isinstance(target_size, (tuple, list)):
                    target_size = target_size[0]  # Use height (assume square)
                stride = target_size / H  # pixels per feature cell
            elif meta is not None and 'new_size' in meta:
                # Fallback: Use new_size (after resize, before padding) + padding
                new_h, new_w = meta['new_size']
                pad_top = meta.get('pad_top', 0)
                pad_bottom = meta.get('pad_bottom', pad_top)  # Assume symmetric if not specified
                # More accurate: use both top and bottom padding
                target_size = new_h + pad_top + pad_bottom
                stride = target_size / H  # pixels per feature cell
            elif meta is not None and 'scale' in meta:
                # Fallback: estimate from scale
                # If original_height * scale = new_height, and padding added to reach target_size
                original_h = meta.get('original_height', H)
                scale = meta['scale']
                pad_top = meta.get('pad_top', 0)
                # Estimate target_size assuming letterbox pads to square
                target_size = original_h * scale + 2 * pad_top  # Rough estimate
                stride = target_size / H
            else:
                # No metadata: use default_target_size for stride estimation
                # CRITICAL FIX (Audit 2025112509 Bug B2): Use configurable default instead of fragile stride=1.0
                # This provides a reliable fallback when metadata is incomplete
                stride = self.default_target_size / H

            # CRITICAL FIX: Sonar rectangular coordinates are (bearing_angle, range)
            # - x-axis: X-direction angle (bearing, horizontal FOV)
            # - y-axis: Z-direction distance (range from sensor)
            # Cannot mix angle and distance with Euclidean formula!
            # Correct radial distance is simply the y-axis coordinate (range axis)

            # Scale feature map grid to pixel coordinates
            y_grid_pixels = y_grid * stride  # [H, W], in pixel space

            # Now both cy and y_grid_pixels are in pixel space
            r = cy - y_grid_pixels  # [H, W], range from sensor (bottom=0, top=max_range)

            # CRITICAL FIX (Audit 2025112507 Finding 2): Mask out letterbox padding regions
            # Problem: Padding pixels produce negative ranges that dominate bin 0, breaking geometry gating
            # Solution: Create valid mask to exclude padding, only bin active sonar regions
            if meta is not None and 'pad_top' in meta and 'scale' in meta:
                # Compute valid region bounds (exclude top/bottom padding)
                # Note: metadata values are in letterbox pixel space (e.g., 512x512)
                # y_grid is in feature space (e.g., 32x32), scaled to pixel space via stride
                pad_top_letterbox = meta.get('pad_top', 0)  # Top padding in letterbox pixels
                original_h = meta.get('original_height', H)
                scale_meta = meta.get('scale', 1.0)  # Letterbox scale factor

                # Active region in letterbox pixels: [pad_top, pad_top + original_h * scale_meta)
                active_start_pixels = pad_top_letterbox
                active_end_pixels = pad_top_letterbox + original_h * scale_meta

                # Valid mask: feature pixels that correspond to active sonar region
                # y_grid_pixels is already in pixel space (via y_grid * stride)
                valid_mask = (y_grid_pixels >= active_start_pixels) & (y_grid_pixels < active_end_pixels)
            else:
                # No metadata: assume all pixels are valid (fallback for non-letterboxed inputs)
                valid_mask = torch.ones_like(r, dtype=torch.bool)

            # Clamp negative ranges to zero (safety measure, should not occur with proper metadata)
            r = torch.clamp(r, min=0.0)

            # Normalize to bin indices [0, num_bins)
            r_max = r[valid_mask].max() if valid_mask.any() else r.max()  # Only consider valid pixels
            r_normalized = (r / (r_max + 1e-6)) * (self.num_bins - 1)
            r_bins = r_normalized.long().clamp(0, self.num_bins - 1)  # [H, W]

            # Weighted channel aggregation
            if self.learnable_weights:
                feat = (feature_map[b] * adjusted_weights.view(C, 1, 1)).sum(dim=0)  # [H, W]
            else:
                feat = feature_map[b].mean(dim=0)  # [H, W]

            # CRITICAL FIX (Audit 2025112507 Finding 2): Apply valid mask before binning
            # Only bin pixels from active sonar region, ignore padding
            feat_masked = feat * valid_mask.float()  # Zero out padding features
            valid_mask_flat = valid_mask.flatten()  # [H*W]

            # Vectorized bin-wise aggregation using scatter_add (only valid pixels)
            feat_flat = feat_masked.flatten()  # [H*W]
            r_bins_flat = r_bins.flatten()  # [H*W]

            # Accumulate values and counts for each bin (only valid pixels contribute)
            profile = torch.zeros(self.num_bins, device=device)
            counts = torch.zeros(self.num_bins, device=device)

            # Use valid mask to filter: only scatter valid pixels
            profile.scatter_add_(0, r_bins_flat[valid_mask_flat], feat_flat[valid_mask_flat])
            counts.scatter_add_(0, r_bins_flat[valid_mask_flat], torch.ones(valid_mask_flat.sum(), device=device))

            # Compute mean: divide accumulated values by counts (single normalization)
            profile = profile / (counts + 1e-6)
            radial_profiles.append(profile)

        radial_profile = torch.stack(radial_profiles, dim=0)  # [B, num_bins]
        return radial_profile

    def forward(
        self,
        sonar_features: torch.Tensor,
        sensor_position: str = "bottom_center",
        letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> torch.Tensor:
        """
        Extract and encode radial profile.

        CRITICAL FIX (Audit 2025112403 Finding 3): Added letterbox_metadata parameter
        ENHANCED FIX (Audit 2025112403 Follow-up): Per-sample metadata support

        Args:
            sonar_features: Sonar feature maps [B, C, H, W]
            sensor_position: Sensor origin position
            letterbox_metadata: Either single dict or list of dicts for per-sample metadata

        Returns:
            radial_encoding: Encoded radial profile [B, 64]
        """
        radial_profile = self.extract_radial_bins(
            sonar_features,
            sensor_position,
            letterbox_metadata
        )
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
        # CRITICAL FIX (Audit 2025112509 Bug B3): Pre-initialize common dimension adapters
        # to ensure checkpoint completeness. Dynamically created adapters during forward
        # may be missing if model is saved before all dimensions are encountered.
        common_input_dims = [768, 384, 512, 1024]  # ViT-Base, ViT-Small, common, ViT-Large
        self.channel_adapters = nn.ModuleDict()
        for dim in common_input_dims:
            if dim != feature_dim:
                self.channel_adapters[str(dim)] = nn.Conv2d(
                    dim, feature_dim, kernel_size=1
                )

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
                ).to(pyramid_levels[0].device)
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
        # CRITICAL FIX (Audit 2025112509 Bug B3): Pre-initialize common dimension adapters
        # to ensure checkpoint completeness for cross-modal attention
        common_input_dims = [768, 384, 512, 1024]  # ViT-Base, ViT-Small, common, ViT-Large
        self.rgb_adapters = nn.ModuleDict()
        self.sonar_adapters = nn.ModuleDict()
        self.rgb_out_adapters = nn.ModuleDict()
        for dim in common_input_dims:
            if dim != embed_dim:
                self.rgb_adapters[str(dim)] = nn.Linear(dim, embed_dim)
                self.sonar_adapters[str(dim)] = nn.Linear(dim, embed_dim)
                self.rgb_out_adapters[str(dim)] = nn.Linear(embed_dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_pyramid: Optional[list] = None,
        sonar_letterbox_metadata: Union[dict, List[dict], None] = None,
        rgb_letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply geometry-gated cross-attention.

        CRITICAL FIX (Audit 2025112403 Finding 3): Added sonar_letterbox_metadata parameter
        to enable letterbox-aware radial binning that respects physical sensor geometry.

        ENHANCED FIX (Audit 2025112403 Follow-up): Per-sample metadata support
        Now accepts list of dicts for heterogeneous batches with different letterbox params.

        CRITICAL FIX (Audit 2025120401 Finding 9): Added rgb_letterbox_metadata parameter
        for symmetric cross-modal metadata passing. Currently stored for future scale
        consistency verification when dynamic search expansion is used.

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional multi-scale RGB features
            sonar_letterbox_metadata: Either:
                                     - Single dict: used for all samples
                                     - List of dicts: per-sample metadata for heterogeneous batches
            rgb_letterbox_metadata: RGB letterbox metadata (symmetric to sonar, for future use)

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
                self.rgb_adapters[adapter_key] = nn.Linear(C_rgb, self.embed_dim).to(rgb_features.device)
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
                self.sonar_adapters[adapter_key] = nn.Linear(C_sonar, self.embed_dim).to(sonar_features.device)
            sonar_proj = self.sonar_adapters[adapter_key]
            sonar_flat = sonar_features.flatten(2).permute(0, 2, 1)  # [B, H_sonar*W_sonar, C_sonar]
            sonar_flat = sonar_proj(sonar_flat)  # [B, H_sonar*W_sonar, embed_dim]
        else:
            sonar_flat = sonar_features.flatten(2).permute(0, 2, 1)

        # Extract geometric encodings (using original features)
        # CRITICAL FIX (Audit 2025112403 Finding 3): Pass letterbox metadata for radial binning
        radial_encoding = self.radial_extractor(
            sonar_features,
            sensor_position="bottom_center",
            letterbox_metadata=sonar_letterbox_metadata
        )  # [B, 64]
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
                self.rgb_out_adapters[adapter_key] = nn.Linear(self.embed_dim, C_rgb).to(attn_output.device)
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