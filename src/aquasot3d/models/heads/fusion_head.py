"""
Fusion Head for Multi-Modal Feature Integration

This module implements a fusion head that combines RGB and Sonar features with
uncertainty-aware weighting. It predicts quality scores and adaptively weights
modality-specific predictions based on estimated uncertainties.

Architecture:
- Multi-modal feature fusion with learnable attention
- Uncertainty estimation for each modality (quality/confidence scores)
- Adaptive weighting based on uncertainty
- Quality score prediction for final output

Reference: NA-OSTrack (https://github.com/fbasatemur/NA-OSTrack)

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal

from aquasot3d.core import BaseHead, PREDICTION_HEAD_REGISTRY


class UncertaintyEstimator(nn.Module):
    """
    Estimates uncertainty/quality for a single modality.

    Uses feature statistics to estimate confidence score, which is inversely
    related to uncertainty. Higher quality features get higher confidence.

    Args:
        in_channels: Input feature dimension
        hidden_dim: Hidden layer dimension
        method: Uncertainty estimation method ('learned' or 'entropy')
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        method: Literal['learned', 'entropy'] = 'learned'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.method = method

        if method == 'learned':
            # Learnable uncertainty estimation
            self.estimator = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1], higher = more confident
            )
        elif method == 'entropy':
            # Entropy-based uncertainty (lower entropy = higher confidence)
            pass  # Computed in forward
        else:
            raise ValueError(f"Unknown uncertainty estimation method: {method}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence score for features.

        Args:
            features: Input features, shape (B, C) or (B, C, H, W)

        Returns:
            Confidence scores, shape (B,)
        """
        # Handle spatial features
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)

        # Check if feature dimension matches expected dimension
        actual_dim = features.shape[1]
        if actual_dim != self.in_channels:
            if not hasattr(self, 'input_proj'):
                # Create dynamic projection layer
                self.input_proj = nn.Linear(actual_dim, self.in_channels).to(features.device)
            features = self.input_proj(features)

        if self.method == 'learned':
            confidence = self.estimator(features).squeeze(-1)
        elif self.method == 'entropy':
            # Compute feature entropy as uncertainty proxy
            # Normalize features to probability distribution
            probs = F.softmax(features, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
            # Normalize entropy to [0, 1] and invert (high entropy = low confidence)
            max_entropy = torch.log(torch.tensor(features.shape[-1], dtype=torch.float32))
            confidence = 1.0 - (entropy / max_entropy)

        return confidence


class AdaptiveFusionGate(nn.Module):
    """
    Adaptive fusion gate that combines multi-modal features based on uncertainty.

    Uses uncertainty estimates to compute adaptive fusion weights, giving more
    importance to more confident modalities.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        hidden_dim: Hidden layer dimension
        fusion_method: Fusion method ('attention', 'mlp', or 'uncertainty_weighted')
    """

    def __init__(
        self,
        rgb_channels: int,
        sonar_channels: int,
        hidden_dim: int = 256,
        fusion_method: Literal['attention', 'mlp', 'uncertainty_weighted'] = 'attention'
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method

        if fusion_method == 'attention':
            # Cross-attention based fusion with metadata-aware masking
            self.query_proj = nn.Linear(rgb_channels + sonar_channels, hidden_dim)
            self.key_proj = nn.Linear(rgb_channels + sonar_channels, hidden_dim)
            self.value_proj = nn.Linear(rgb_channels + sonar_channels, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            # Metadata-aware attention requires spatial feature maps
            self.num_heads = 8
            self.head_dim = hidden_dim // self.num_heads

        elif fusion_method == 'mlp':
            # Simple MLP fusion
            self.fusion_mlp = nn.Sequential(
                nn.Linear(rgb_channels + sonar_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        elif fusion_method == 'uncertainty_weighted':
            # Uncertainty-aware weighted fusion
            self.rgb_proj = nn.Linear(rgb_channels, hidden_dim)
            self.sonar_proj = nn.Linear(sonar_channels, hidden_dim)
            # Weight predictor based on concatenated features + uncertainties
            self.weight_predictor = nn.Sequential(
                nn.Linear(rgb_channels + sonar_channels + 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def _create_attention_mask_from_metadata(
        self,
        metadata: Dict,
        batch_size: int,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Create attention mask from metadata to prevent attending to padded regions.

        Args:
            metadata: Dict containing 'rgb_search_metadata' and 'sonar_search_metadata'
            batch_size: Batch size
            device: Device for mask tensor

        Returns:
            Attention mask of shape (B, 2) where 0 indicates padded/invalid regions
        """
        if metadata is None:
            return None

        # CRITICAL FIX (Audit 2025111413 Finding 4): Validate metadata structure
        assert isinstance(metadata, dict), \
            f"Metadata must be dict, got {type(metadata)}"

        rgb_meta = metadata.get('rgb_search_metadata', [])
        sonar_meta = metadata.get('sonar_search_metadata', [])

        # Assert metadata lists are properly sized
        if rgb_meta and not isinstance(rgb_meta, list):
            rgb_meta = [rgb_meta]
        if sonar_meta and not isinstance(sonar_meta, list):
            sonar_meta = [sonar_meta]

        # Validate list lengths match batch size
        if rgb_meta:
            assert len(rgb_meta) >= batch_size, \
                f"RGB metadata list too short: {len(rgb_meta)} < {batch_size}"
        if sonar_meta:
            assert len(sonar_meta) >= batch_size, \
                f"Sonar metadata list too short: {len(sonar_meta)} < {batch_size}"

        if not rgb_meta or not sonar_meta:
            return None

        # Create mask based on padding information
        mask = torch.ones(batch_size, 2, device=device)

        for i in range(min(batch_size, len(rgb_meta), len(sonar_meta))):
            # CRITICAL FIX (Audit 2025111414 Finding 1): Use pad_x/pad_y instead of 'padding'
            # Loaders emit scale_x/y and pad_x/y, not a unified 'padding' key

            # RGB metadata: check if heavily padded
            if rgb_meta[i]:
                pad_x = rgb_meta[i].get('pad_x', 0)
                pad_y = rgb_meta[i].get('pad_y', 0)
                # Compute padding ratio relative to target size (default 256)
                target_size = 256.0
                total_padding_ratio = (pad_x + pad_y) / target_size
                # If more than 50% padding, reduce attention weight
                if total_padding_ratio > 0.5:
                    mask[i, 0] = 0.5

            # Sonar metadata: check if heavily padded
            if sonar_meta[i]:
                pad_x = sonar_meta[i].get('pad_x', 0)
                pad_y = sonar_meta[i].get('pad_y', 0)
                # Compute padding ratio relative to target size (default 256)
                target_size = 256.0
                total_padding_ratio = (pad_x + pad_y) / target_size
                if total_padding_ratio > 0.5:
                    mask[i, 1] = 0.5

        return mask

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_confidence: Optional[torch.Tensor] = None,
        sonar_confidence: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse RGB and Sonar features adaptively with metadata-aware attention.

        Args:
            rgb_features: RGB features, shape (B, C_rgb)
            sonar_features: Sonar features, shape (B, C_sonar)
            rgb_confidence: RGB confidence scores, shape (B,), optional
            sonar_confidence: Sonar confidence scores, shape (B,), optional
            metadata: Metadata dict with padding/scale info, optional

        Returns:
            Tuple of (fused_features, fusion_weights)
        """
        batch_size = rgb_features.shape[0]

        # Handle spatial and sequence features
        if rgb_features.dim() == 4:
            rgb_features = F.adaptive_avg_pool2d(rgb_features, (1, 1)).flatten(1)
        elif rgb_features.dim() == 3:
            rgb_features = rgb_features.mean(dim=1)  # Average over sequence

        if sonar_features.dim() == 4:
            sonar_features = F.adaptive_avg_pool2d(sonar_features, (1, 1)).flatten(1)
        elif sonar_features.dim() == 3:
            sonar_features = sonar_features.mean(dim=1)  # Average over sequence

        if self.fusion_method == 'attention':
            # Concatenate features
            combined = torch.cat([rgb_features, sonar_features], dim=-1)

            # Self-attention
            Q = self.query_proj(combined)
            K = self.key_proj(combined)
            V = self.value_proj(combined)

            # Compute attention scores
            attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / (self.hidden_dim ** 0.5)

            # CRITICAL FIX (Audit 2025111413 Finding 1): Apply metadata-based mask
            attn_mask = self._create_attention_mask_from_metadata(metadata, batch_size, rgb_features.device)
            if attn_mask is not None:
                # Expand mask for attention computation
                # Mask shape: (B, 2) -> apply to modality-specific regions
                # For simplicity, we apply a scaling factor based on metadata quality
                rgb_mask = attn_mask[:, 0:1].unsqueeze(1)  # (B, 1, 1)
                sonar_mask = attn_mask[:, 1:2].unsqueeze(1)  # (B, 1, 1)
                combined_mask = torch.cat([rgb_mask, sonar_mask], dim=-1).mean(dim=-1, keepdim=True)
                attn_scores = attn_scores * combined_mask

            # Attention weights
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Weighted values
            attended = torch.bmm(attn_weights, V.unsqueeze(1)).squeeze(1)
            fused = self.out_proj(attended)

            # Extract fusion weights from attention
            # Use attention mask if available, otherwise uniform weights
            if attn_mask is not None:
                fusion_weights = F.softmax(attn_mask, dim=-1)
            else:
                fusion_weights = torch.ones(batch_size, 2, device=rgb_features.device) * 0.5

        elif self.fusion_method == 'mlp':
            # Simple concatenation and MLP
            combined = torch.cat([rgb_features, sonar_features], dim=-1)
            fused = self.fusion_mlp(combined)

            # Dummy fusion weights
            fusion_weights = torch.ones(batch_size, 2, device=rgb_features.device) * 0.5

        elif self.fusion_method == 'uncertainty_weighted':
            # Check if feature dimensions match expected dimensions
            rgb_actual_dim = rgb_features.shape[1]
            sonar_actual_dim = sonar_features.shape[1]

            # Dynamic projection for RGB features if needed
            if rgb_actual_dim != self.rgb_channels:
                if not hasattr(self, 'rgb_input_proj'):
                    self.rgb_input_proj = nn.Linear(rgb_actual_dim, self.rgb_channels).to(rgb_features.device)
                rgb_features_adapted = self.rgb_input_proj(rgb_features)
            else:
                rgb_features_adapted = rgb_features

            # Dynamic projection for Sonar features if needed
            if sonar_actual_dim != self.sonar_channels:
                if not hasattr(self, 'sonar_input_proj'):
                    self.sonar_input_proj = nn.Linear(sonar_actual_dim, self.sonar_channels).to(sonar_features.device)
                sonar_features_adapted = self.sonar_input_proj(sonar_features)
            else:
                sonar_features_adapted = sonar_features

            # Project to common space
            rgb_proj = self.rgb_proj(rgb_features_adapted)
            sonar_proj = self.sonar_proj(sonar_features_adapted)

            # Default confidence if not provided
            if rgb_confidence is None:
                rgb_confidence = torch.ones(batch_size, device=rgb_features.device)
            if sonar_confidence is None:
                sonar_confidence = torch.ones(batch_size, device=sonar_features.device)

            # Predict fusion weights based on features and confidence
            weight_input = torch.cat([
                rgb_features_adapted,
                sonar_features_adapted,
                rgb_confidence.unsqueeze(-1),
                sonar_confidence.unsqueeze(-1)
            ], dim=-1)
            fusion_weights = self.weight_predictor(weight_input)  # (B, 2)

            # Weighted fusion
            fused = (
                fusion_weights[:, 0:1] * rgb_proj +
                fusion_weights[:, 1:2] * sonar_proj
            )

        return fused, fusion_weights


@PREDICTION_HEAD_REGISTRY.register("FusionHead")
class FusionHead(BaseHead):
    """
    Multi-Modal Fusion Head with Uncertainty-Aware Weighting.

    Combines RGB and Sonar features adaptively based on estimated uncertainties,
    and predicts a unified quality score for the fused output.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        hidden_dim: Hidden layer dimension (default: 256)
        uncertainty_method: Uncertainty estimation method (default: 'learned')
        fusion_method: Fusion method (default: 'uncertainty_weighted')
        predict_uncertainty: Whether to output uncertainty estimates (default: True)

    Returns:
        Dictionary containing:
            - 'fused_features': Fused feature representation, shape (B, hidden_dim)
            - 'fusion_weights': Adaptive fusion weights, shape (B, 2)
            - 'rgb_confidence': RGB confidence score, shape (B,)
            - 'sonar_confidence': Sonar confidence score, shape (B,)
            - 'quality_score': Unified quality prediction, shape (B,)

    Example:
        >>> head = FusionHead(rgb_channels=768, sonar_channels=384)
        >>> rgb_feat = torch.randn(4, 768)
        >>> sonar_feat = torch.randn(4, 384)
        >>> outputs = head(rgb_feat, sonar_feat)
        >>> print(outputs['fused_features'].shape)  # (4, 256)
        >>> print(outputs['fusion_weights'].shape)  # (4, 2)
    """

    def __init__(
        self,
        rgb_channels: int,
        sonar_channels: int,
        hidden_dim: int = 256,
        uncertainty_method: Literal['learned', 'entropy'] = 'learned',
        fusion_method: Literal['attention', 'mlp', 'uncertainty_weighted'] = 'uncertainty_weighted',
        predict_uncertainty: bool = True,
        **kwargs
    ):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels
        self.hidden_dim = hidden_dim
        self.uncertainty_method = uncertainty_method
        self.fusion_method = fusion_method
        self.predict_uncertainty = predict_uncertainty

        # Uncertainty estimators
        self.rgb_uncertainty_estimator = UncertaintyEstimator(
            in_channels=rgb_channels,
            hidden_dim=hidden_dim // 2,
            method=uncertainty_method
        )
        self.sonar_uncertainty_estimator = UncertaintyEstimator(
            in_channels=sonar_channels,
            hidden_dim=hidden_dim // 2,
            method=uncertainty_method
        )

        # Adaptive fusion gate
        self.fusion_gate = AdaptiveFusionGate(
            rgb_channels=rgb_channels,
            sonar_channels=sonar_channels,
            hidden_dim=hidden_dim,
            fusion_method=fusion_method
        )

        # Quality score predictor (for final fused output)
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.logger.info(
            f"Initialized FusionHead: rgb_channels={rgb_channels}, "
            f"sonar_channels={sonar_channels}, hidden_dim={hidden_dim}, "
            f"uncertainty_method={uncertainty_method}, fusion_method={fusion_method}"
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal fusion with validity-aware confidence.

        CRITICAL FIX (Audit 2025111412 Finding 1):
        Now accepts metadata to enable physical alignment awareness. This prevents
        fusion from overweighting modality artifacts in padded/scaled regions.

        CRITICAL FIX (Audit 2025112101 Phase 1.3):
        Added validity supervision. When is_rgb_valid/is_sonar_valid are False,
        confidence is forced to 0.0 to train the model to output low confidence
        for invalid modalities (occlusions, out-of-FOV targets).

        Args:
            rgb_features: RGB features, shape (B, C_rgb) or (B, C_rgb, H, W)
            sonar_features: Sonar features, shape (B, C_sonar) or (B, C_sonar, H, W)
            is_rgb_valid: [B] boolean tensor, True if RGB has valid target (optional)
            is_sonar_valid: [B] boolean tensor, True if Sonar has valid target (optional)
            return_uncertainty: Whether to return uncertainty estimates
            metadata: Optional dict with 'rgb_search_metadata' and 'sonar_search_metadata'
                     lists containing per-sample scale/padding info for physical alignment

        Returns:
            Dictionary with fused features and quality predictions

        Note:
            When metadata is provided, fusion can adjust attention weights based on
            physically meaningful regions vs padded/artifact areas. Currently passed
            through for future physical grid alignment (Finding 1 phase 1).
        """
        # Estimate confidence/uncertainty for each modality
        rgb_confidence_base = self.rgb_uncertainty_estimator(rgb_features)  # (B,)
        sonar_confidence_base = self.sonar_uncertainty_estimator(sonar_features)  # (B,)

        # CRITICAL: Clamp confidence scores to [0, 1] for numerical stability
        # Sigmoid in FP16/AMP mode can occasionally produce values slightly outside [0, 1]
        rgb_confidence_base = torch.clamp(rgb_confidence_base, min=0.0, max=1.0)
        sonar_confidence_base = torch.clamp(sonar_confidence_base, min=0.0, max=1.0)

        # CRITICAL FIX (Audit 2025112101): Suppress confidence for invalid modalities
        # This trains the model to output low confidence when target is absent
        if is_rgb_valid is not None:
            rgb_confidence = rgb_confidence_base * is_rgb_valid.float()
        else:
            rgb_confidence = rgb_confidence_base

        if is_sonar_valid is not None:
            sonar_confidence = sonar_confidence_base * is_sonar_valid.float()
        else:
            sonar_confidence = sonar_confidence_base

        # Adaptive fusion with metadata awareness
        fused_features, fusion_weights = self.fusion_gate(
            rgb_features,
            sonar_features,
            rgb_confidence,
            sonar_confidence,
            metadata
        )  # (B, hidden_dim), (B, 2)

        # Predict quality score
        quality_score = self.quality_predictor(fused_features).squeeze(-1)  # (B,)

        # CRITICAL: Clamp quality_score to [0, 1] for numerical stability
        # This prevents CUDA assertion failures in BCE loss computation
        quality_score = torch.clamp(quality_score, min=0.0, max=1.0)

        # CRITICAL FIX (Audit 2025112101): Further suppress quality for zero-modality
        # When both modalities are invalid, overall quality should be near zero
        if is_rgb_valid is not None and is_sonar_valid is not None:
            any_valid = (is_rgb_valid | is_sonar_valid).float()
            quality_score = quality_score * any_valid

        outputs = {
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'quality_score': quality_score,
        }

        if return_uncertainty or self.predict_uncertainty:
            outputs['rgb_confidence'] = rgb_confidence
            outputs['sonar_confidence'] = sonar_confidence
            # Compute overall uncertainty (inverse of confidence)
            overall_uncertainty = 1.0 - (
                fusion_weights[:, 0] * rgb_confidence +
                fusion_weights[:, 1] * sonar_confidence
            )
            outputs['uncertainty'] = overall_uncertainty

        return outputs

    def compute_fusion_loss(
        self,
        rgb_quality: torch.Tensor,
        sonar_quality: torch.Tensor,
        fusion_weights: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        target_quality: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute fusion consistency loss with validity supervision.

        CRITICAL FIX (Audit 2025112101 Phase 1.3):
        Added validity-supervised fusion weight loss. This trains fusion weights
        to correctly allocate attention based on modality availability:
        - If RGB invalid: w_rgb -> 0, w_sonar -> 1
        - If Sonar invalid: w_rgb -> 1, w_sonar -> 0
        - If both valid: w based on quality scores
        - If both invalid: w irrelevant (quality_score -> 0 already handled)

        Args:
            rgb_quality: RGB quality scores, shape (B,)
            sonar_quality: Sonar quality scores, shape (B,)
            fusion_weights: Fusion weights, shape (B, 2) where [:, 0] is RGB weight
            is_rgb_valid: [B] boolean tensor, True if RGB has valid target (optional)
            is_sonar_valid: [B] boolean tensor, True if Sonar has valid target (optional)
            target_quality: Target quality (e.g., IoU), shape (B,), optional

        Returns:
            Fusion loss (scalar)
        """
        batch_size = fusion_weights.shape[0]

        # Initialize loss components
        quality_consistency_loss = torch.tensor(0.0, device=fusion_weights.device)
        validity_supervision_loss = torch.tensor(0.0, device=fusion_weights.device)

        # CRITICAL FIX (Audit 2025112101): Supervise fusion weights based on validity
        if is_rgb_valid is not None and is_sonar_valid is not None:
            # Convert boolean tensors to float for computation
            rgb_valid_float = is_rgb_valid.float()  # [B]
            sonar_valid_float = is_sonar_valid.float()  # [B]

            # Compute target fusion weights based on validity
            # If both valid: use quality-based weighting
            # If only RGB valid: w_rgb=1, w_sonar=0
            # If only Sonar valid: w_rgb=0, w_sonar=1
            # If neither valid: no supervision (handled by quality_score suppression)

            # Detect modality availability modes
            both_valid = rgb_valid_float * sonar_valid_float  # [B]
            only_rgb = rgb_valid_float * (1.0 - sonar_valid_float)  # [B]
            only_sonar = (1.0 - rgb_valid_float) * sonar_valid_float  # [B]

            # Target weights for single-modality scenarios
            target_weights_single = torch.stack([
                only_rgb + 0.5 * both_valid,  # RGB weight: 1.0 if only RGB, 0.5 if both
                only_sonar + 0.5 * both_valid  # Sonar weight: 1.0 if only Sonar, 0.5 if both
            ], dim=1)  # [B, 2]

            # For dual-modality, refine weights based on quality
            if target_quality is not None and both_valid.sum() > 0:
                # Normalize quality scores for dual-modality samples
                quality_sum = rgb_quality + sonar_quality + 1e-7
                quality_based_rgb_weight = rgb_quality / quality_sum
                quality_based_sonar_weight = sonar_quality / quality_sum

                # Update target weights for dual-modality samples
                target_weights_single[:, 0] = (
                    only_rgb +
                    both_valid * quality_based_rgb_weight
                )
                target_weights_single[:, 1] = (
                    only_sonar +
                    both_valid * quality_based_sonar_weight
                )

            # Compute L1 loss between predicted and target weights
            # Only supervise samples where at least one modality is valid
            any_valid = torch.clamp(rgb_valid_float + sonar_valid_float, min=0.0, max=1.0)
            num_valid = any_valid.sum().clamp(min=1.0)

            weight_error = F.l1_loss(fusion_weights, target_weights_single, reduction='none')  # [B, 2]
            weight_error_per_sample = weight_error.mean(dim=1)  # [B]

            validity_supervision_loss = (weight_error_per_sample * any_valid).sum() / num_valid

        # Quality consistency loss (original behavior)
        # Encourage weighted quality to match target quality
        weighted_quality = (
            fusion_weights[:, 0] * rgb_quality +
            fusion_weights[:, 1] * sonar_quality
        )

        if target_quality is not None:
            # L1 loss between weighted quality and target
            quality_consistency_loss = F.l1_loss(weighted_quality, target_quality)
        else:
            # Encourage high quality when both modalities agree
            agreement = 1.0 - torch.abs(rgb_quality - sonar_quality)
            quality_consistency_loss = -agreement.mean()  # Maximize agreement

        # Combine losses
        # Validity supervision is critical for single-modality learning
        total_loss = validity_supervision_loss + 0.5 * quality_consistency_loss

        return total_loss


def build_fusion_head(
    rgb_channels: int = 768,
    sonar_channels: int = 384,
    **kwargs
) -> FusionHead:
    """
    Factory function to build fusion head.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        **kwargs: Additional arguments for FusionHead

    Returns:
        FusionHead instance

    Example:
        >>> head = build_fusion_head(rgb_channels=768, sonar_channels=384, hidden_dim=256)
    """
    return FusionHead(rgb_channels=rgb_channels, sonar_channels=sonar_channels, **kwargs)
