"""
Modality dropout utility for ablation experiments (testing-time only).

Provides deterministic dropout behavior for testing-time ablation studies.
Used to simulate single-modal operation (RGB-only or Sonar-only) while
preserving the dual-modal architecture.

When a modality is dropped:
1. Input tensor is replaced with zeros
2. Model produces low confidence score (~0) for that modality
3. Target Loss Handler detects dropped modality as LOST
4. Primary output uses the active modality exclusively

Configuration (in config.py or YAML):
    DATA:
      MODALITY_DROPOUT:
        ENABLED: true/false
        P_DROP_RGB: 0.0 or 1.0
        P_DROP_SONAR: 0.0 or 1.0

Note: This is testing-time only ablation. Training always uses both modalities
to ensure the model learns robust dual-modal representations.
"""

import torch


def apply_modality_dropout(rgb_tensor, sonar_tensor, cfg):
    """Apply deterministic modality dropout for testing-time ablation.

    This function is designed for ablation experiments where we want to
    evaluate single-modal performance using a model trained with both modalities.

    Args:
        rgb_tensor: RGB input tensor [B, C, H, W] or [1, C, H, W]
        sonar_tensor: Sonar input tensor [B, 1, H, W] or [1, 1, H, W]
        cfg: Configuration object with DATA.MODALITY_DROPOUT settings

    Returns:
        Tuple of (rgb_tensor, sonar_tensor, dropout_info)
        dropout_info: dict with keys:
            - 'rgb_dropped': bool, whether RGB was dropped
            - 'sonar_dropped': bool, whether Sonar was dropped

    Example:
        # Ablation mode: drop sonar entirely (P_DROP_SONAR=1.0)
        rgb, sonar, info = apply_modality_dropout(rgb, sonar, cfg)
        # info = {'rgb_dropped': False, 'sonar_dropped': True}
        # sonar is now all zeros

    Note:
        - When ENABLED=False, returns tensors unchanged (backward compatible)
        - When P_DROP_*=1.0, that modality is always dropped (deterministic)
        - Dropout is applied consistently to template and search in the caller
    """
    dropout_cfg = cfg.DATA.MODALITY_DROPOUT

    # Fast path: disabled dropout (default configuration)
    if not dropout_cfg.ENABLED:
        return rgb_tensor, sonar_tensor, {'rgb_dropped': False, 'sonar_dropped': False}

    p_rgb = dropout_cfg.P_DROP_RGB
    p_sonar = dropout_cfg.P_DROP_SONAR

    # Deterministic dropout for testing-time ablation
    # P_DROP >= 1.0 means always drop that modality
    drop_rgb = p_rgb >= 1.0
    drop_sonar = p_sonar >= 1.0

    # Apply dropout by replacing with zero tensor
    # Using zeros_like preserves device, dtype, and shape
    if drop_rgb:
        rgb_tensor = torch.zeros_like(rgb_tensor)
    if drop_sonar:
        sonar_tensor = torch.zeros_like(sonar_tensor)

    return rgb_tensor, sonar_tensor, {'rgb_dropped': drop_rgb, 'sonar_dropped': drop_sonar}
