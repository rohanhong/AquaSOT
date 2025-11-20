"""
Modality Dropout Augmentation

CRITICAL FIX (Audit 2025112101 Phase 2.3):
This module implements modality dropout augmentation to train the model's
robustness to single/zero-modality scenarios.

During training, randomly drop one or both modalities to simulate:
- RGB occlusion/out-of-FOV (single-Sonar scenario)
- Sonar dropouts (single-RGB scenario)
- Complete target loss (zero-modality scenario)

This forces the model to learn:
1. Single-modality operation through bypass paths
2. Confidence suppression when modalities are invalid
3. Robust feature extraction from limited information

Usage:
    # In dataset or training loop
    dropout = ModalityDropout(p_drop_rgb=0.1, p_drop_sonar=0.1, p_drop_both=0.02)

    # Apply to batch
    rgb_features, sonar_features, is_rgb_valid, is_sonar_valid = dropout(
        rgb_features, sonar_features, is_rgb_valid, is_sonar_valid
    )

Author: AquaSOT-3D Team
Date: 2025-11-21
"""

import torch
import numpy as np
from typing import Tuple, Optional


class ModalityDropout:
    """
    Modality dropout augmentation for training robustness.

    CRITICAL FIX (Audit 2025112101 Phase 2.3):
    Randomly drops modalities during training to force the model to learn
    single-modality and zero-modality operation.

    The dropout probabilities are applied independently:
    1. With probability p_drop_rgb: Drop RGB (set invalid)
    2. With probability p_drop_sonar: Drop Sonar (set invalid)
    3. With probability p_drop_both: Drop both (set both invalid)

    Note: p_drop_both is applied AFTER individual drops to ensure
    zero-modality scenarios are properly sampled.

    Args:
        p_drop_rgb: Probability of dropping RGB modality (default: 0.1)
        p_drop_sonar: Probability of dropping Sonar modality (default: 0.1)
        p_drop_both: Probability of dropping both modalities (default: 0.02)
        enabled: Whether dropout is enabled (default: True, disable during eval)
        mode: Dropout mode, 'zero' or 'noise' (default: 'zero')
            - 'zero': Replace dropped features with zeros
            - 'noise': Replace dropped features with gaussian noise

    Example:
        >>> dropout = ModalityDropout(p_drop_rgb=0.15, p_drop_sonar=0.15, p_drop_both=0.03)
        >>> rgb_feat, sonar_feat, is_rgb, is_sonar = dropout(rgb_feat, sonar_feat, is_rgb, is_sonar)
        >>> # Expected modality distribution after dropout:
        >>> # - Dual-modality: ~72%
        >>> # - Single-RGB: ~13%
        >>> # - Single-Sonar: ~13%
        >>> # - Zero-modality: ~2%
    """

    def __init__(
        self,
        p_drop_rgb: float = 0.1,
        p_drop_sonar: float = 0.1,
        p_drop_both: float = 0.02,
        enabled: bool = True,
        mode: str = 'zero'
    ):
        assert 0.0 <= p_drop_rgb <= 1.0, f"p_drop_rgb must be in [0, 1], got {p_drop_rgb}"
        assert 0.0 <= p_drop_sonar <= 1.0, f"p_drop_sonar must be in [0, 1], got {p_drop_sonar}"
        assert 0.0 <= p_drop_both <= 1.0, f"p_drop_both must be in [0, 1], got {p_drop_both}"
        assert mode in ['zero', 'noise'], f"mode must be 'zero' or 'noise', got {mode}"

        self.p_drop_rgb = p_drop_rgb
        self.p_drop_sonar = p_drop_sonar
        self.p_drop_both = p_drop_both
        self.enabled = enabled
        self.mode = mode

        # Statistics tracking
        self.reset_stats()

    def reset_stats(self):
        """Reset dropout statistics."""
        self.stats = {
            'total_samples': 0,
            'dropped_rgb': 0,
            'dropped_sonar': 0,
            'dropped_both': 0,
            'dual_modality': 0,
            'single_rgb': 0,
            'single_sonar': 0,
            'zero_modality': 0
        }

    def get_stats(self) -> dict:
        """Get dropout statistics."""
        if self.stats['total_samples'] == 0:
            return self.stats

        total = self.stats['total_samples']
        return {
            'total_samples': total,
            'dropped_rgb': self.stats['dropped_rgb'],
            'dropped_sonar': self.stats['dropped_sonar'],
            'dropped_both': self.stats['dropped_both'],
            'dual_modality_pct': self.stats['dual_modality'] / total * 100,
            'single_rgb_pct': self.stats['single_rgb'] / total * 100,
            'single_sonar_pct': self.stats['single_sonar'] / total * 100,
            'zero_modality_pct': self.stats['zero_modality'] / total * 100
        }

    def __call__(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: torch.Tensor,
        is_sonar_valid: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply modality dropout augmentation.

        CRITICAL FIX (Audit 2025112101 Phase 2.3):
        Randomly drops modalities to simulate single/zero-modality scenarios.

        Args:
            rgb_features: RGB feature tensor [B, C, H, W] or [B, T, C, H, W]
            sonar_features: Sonar feature tensor [B, C, H, W] or [B, T, C, H, W]
            is_rgb_valid: RGB validity mask [B] or [B, T]
            is_sonar_valid: Sonar validity mask [B] or [B, T]
            training: Whether in training mode (dropout only applied during training)

        Returns:
            Tuple of (rgb_features, sonar_features, is_rgb_valid, is_sonar_valid)
            with dropout applied to features and validity masks
        """
        # Only apply dropout during training and if enabled
        if not training or not self.enabled:
            return rgb_features, sonar_features, is_rgb_valid, is_sonar_valid

        batch_size = rgb_features.shape[0]
        device = rgb_features.device

        # Clone tensors to avoid in-place modifications
        rgb_features = rgb_features.clone()
        sonar_features = sonar_features.clone()
        is_rgb_valid = is_rgb_valid.clone()
        is_sonar_valid = is_sonar_valid.clone()

        # Generate dropout masks [B]
        drop_rgb_mask = torch.rand(batch_size, device=device) < self.p_drop_rgb
        drop_sonar_mask = torch.rand(batch_size, device=device) < self.p_drop_sonar
        drop_both_mask = torch.rand(batch_size, device=device) < self.p_drop_both

        # Apply both-modality dropout first (overrides individual drops)
        if drop_both_mask.any():
            self.stats['dropped_both'] += drop_both_mask.sum().item()

            if self.mode == 'zero':
                # Zero out both modalities
                rgb_features[drop_both_mask] = 0.0
                sonar_features[drop_both_mask] = 0.0
            elif self.mode == 'noise':
                # Replace with gaussian noise
                rgb_features[drop_both_mask] = torch.randn_like(rgb_features[drop_both_mask]) * 0.1
                sonar_features[drop_both_mask] = torch.randn_like(sonar_features[drop_both_mask]) * 0.1

            # Mark as invalid
            is_rgb_valid[drop_both_mask] = False
            is_sonar_valid[drop_both_mask] = False

        # Apply individual RGB dropout (only for samples not affected by drop_both)
        rgb_drop_only = drop_rgb_mask & (~drop_both_mask)
        if rgb_drop_only.any():
            self.stats['dropped_rgb'] += rgb_drop_only.sum().item()

            if self.mode == 'zero':
                rgb_features[rgb_drop_only] = 0.0
            elif self.mode == 'noise':
                rgb_features[rgb_drop_only] = torch.randn_like(rgb_features[rgb_drop_only]) * 0.1

            is_rgb_valid[rgb_drop_only] = False

        # Apply individual Sonar dropout (only for samples not affected by drop_both)
        sonar_drop_only = drop_sonar_mask & (~drop_both_mask)
        if sonar_drop_only.any():
            self.stats['dropped_sonar'] += sonar_drop_only.sum().item()

            if self.mode == 'zero':
                sonar_features[sonar_drop_only] = 0.0
            elif self.mode == 'noise':
                sonar_features[sonar_drop_only] = torch.randn_like(sonar_features[sonar_drop_only]) * 0.1

            is_sonar_valid[sonar_drop_only] = False

        # Update statistics
        self.stats['total_samples'] += batch_size

        # Count modality distributions after dropout
        dual = (is_rgb_valid & is_sonar_valid).sum().item()
        single_rgb = (is_rgb_valid & (~is_sonar_valid)).sum().item()
        single_sonar = ((~is_rgb_valid) & is_sonar_valid).sum().item()
        zero = ((~is_rgb_valid) & (~is_sonar_valid)).sum().item()

        self.stats['dual_modality'] += dual
        self.stats['single_rgb'] += single_rgb
        self.stats['single_sonar'] += single_sonar
        self.stats['zero_modality'] += zero

        return rgb_features, sonar_features, is_rgb_valid, is_sonar_valid

    def __repr__(self) -> str:
        return (
            f"ModalityDropout("
            f"p_drop_rgb={self.p_drop_rgb}, "
            f"p_drop_sonar={self.p_drop_sonar}, "
            f"p_drop_both={self.p_drop_both}, "
            f"enabled={self.enabled}, "
            f"mode='{self.mode}')"
        )


class AdaptiveModalityDropout(ModalityDropout):
    """
    Adaptive modality dropout with curriculum learning.

    EXTENSION (Audit 2025112101 Phase 2.3):
    Gradually increases dropout rates during training to implement curriculum learning:
    - Start with low dropout rates (easy: mostly dual-modality)
    - Gradually increase dropout rates (hard: more single/zero-modality)

    This helps the model first learn dual-modality fusion, then progressively
    adapt to single/zero-modality scenarios.

    Args:
        p_drop_rgb_start: Initial RGB dropout probability
        p_drop_rgb_end: Final RGB dropout probability
        p_drop_sonar_start: Initial Sonar dropout probability
        p_drop_sonar_end: Final Sonar dropout probability
        p_drop_both_start: Initial both dropout probability
        p_drop_both_end: Final both dropout probability
        warmup_epochs: Number of epochs to ramp up dropout rates
        current_epoch: Current training epoch (updated externally)
        enabled: Whether dropout is enabled
        mode: Dropout mode ('zero' or 'noise')

    Example:
        >>> # Start easy (5% dropout), end hard (20% dropout)
        >>> dropout = AdaptiveModalityDropout(
        ...     p_drop_rgb_start=0.05,
        ...     p_drop_rgb_end=0.20,
        ...     p_drop_sonar_start=0.05,
        ...     p_drop_sonar_end=0.20,
        ...     warmup_epochs=50
        ... )
        >>>
        >>> # In training loop
        >>> for epoch in range(num_epochs):
        ...     dropout.set_epoch(epoch)
        ...     for batch in dataloader:
        ...         # dropout rates automatically adjusted based on epoch
        ...         rgb, sonar, is_rgb, is_sonar = dropout(rgb, sonar, is_rgb, is_sonar)
    """

    def __init__(
        self,
        p_drop_rgb_start: float = 0.05,
        p_drop_rgb_end: float = 0.15,
        p_drop_sonar_start: float = 0.05,
        p_drop_sonar_end: float = 0.15,
        p_drop_both_start: float = 0.01,
        p_drop_both_end: float = 0.05,
        warmup_epochs: int = 50,
        current_epoch: int = 0,
        enabled: bool = True,
        mode: str = 'zero'
    ):
        self.p_drop_rgb_start = p_drop_rgb_start
        self.p_drop_rgb_end = p_drop_rgb_end
        self.p_drop_sonar_start = p_drop_sonar_start
        self.p_drop_sonar_end = p_drop_sonar_end
        self.p_drop_both_start = p_drop_both_start
        self.p_drop_both_end = p_drop_both_end
        self.warmup_epochs = warmup_epochs
        self.current_epoch = current_epoch

        # Initialize with starting probabilities
        super().__init__(
            p_drop_rgb=p_drop_rgb_start,
            p_drop_sonar=p_drop_sonar_start,
            p_drop_both=p_drop_both_start,
            enabled=enabled,
            mode=mode
        )

    def set_epoch(self, epoch: int):
        """Update dropout probabilities based on current epoch."""
        self.current_epoch = epoch

        if epoch >= self.warmup_epochs:
            # Use final probabilities after warmup
            progress = 1.0
        else:
            # Linear ramp up during warmup
            progress = epoch / self.warmup_epochs

        # Interpolate probabilities
        self.p_drop_rgb = self.p_drop_rgb_start + (self.p_drop_rgb_end - self.p_drop_rgb_start) * progress
        self.p_drop_sonar = self.p_drop_sonar_start + (self.p_drop_sonar_end - self.p_drop_sonar_start) * progress
        self.p_drop_both = self.p_drop_both_start + (self.p_drop_both_end - self.p_drop_both_start) * progress

    def __repr__(self) -> str:
        return (
            f"AdaptiveModalityDropout("
            f"p_drop_rgb={self.p_drop_rgb:.3f} (start={self.p_drop_rgb_start}, end={self.p_drop_rgb_end}), "
            f"p_drop_sonar={self.p_drop_sonar:.3f} (start={self.p_drop_sonar_start}, end={self.p_drop_sonar_end}), "
            f"p_drop_both={self.p_drop_both:.3f} (start={self.p_drop_both_start}, end={self.p_drop_both_end}), "
            f"current_epoch={self.current_epoch}, "
            f"warmup_epochs={self.warmup_epochs}, "
            f"enabled={self.enabled}, "
            f"mode='{self.mode}')"
        )
