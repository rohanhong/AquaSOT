"""
BatchNorm adaptation for single-modality inference.

When a dual-modal model is trained with both RGB and Sonar inputs, the BatchNorm
layers in prediction heads learn running statistics from fused features:
    fused_sonar = sonar_feat + 0.3 * cross_attn(sonar, rgb)

At single-modality inference (e.g., Sonar-only), the feature distribution changes
because cross-attention output is different (self-attention or zero). This causes
a distribution mismatch that degrades prediction quality.

This module provides functions to adapt BatchNorm statistics for single-modality
inference by running a few forward passes with the target modality configuration.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Iterator, Tuple


def adapt_batchnorm_for_modality(
    model: nn.Module,
    dataloader: Iterator,
    modality: str = 'sonar',
    num_batches: int = 10,
    device: str = 'cuda'
) -> nn.Module:
    """
    Adapt BatchNorm statistics for single-modality or no-fusion inference.

    The key insight: during dual-modal training, BN layers learned statistics
    from fused features (sonar_feat + 0.3 * cross_attn(sonar, rgb)). At
    single-modal or no-fusion inference, the feature distribution is different.
    This function re-calculates BN running_mean/running_var.

    Args:
        model: AquaSOT model
        dataloader: Iterator yielding (rgb_template, rgb_search,
                    sonar_template, sonar_search) tuples or dicts
        modality: 'sonar', 'rgb', or 'dual'
                  - 'sonar': Keep sonar, zero RGB (sonar-only mode)
                  - 'rgb': Keep RGB, zero sonar (rgb-only mode)
                  - 'dual': Keep both modalities (no-fusion mode)
        num_batches: Number of batches for adaptation (30-50 recommended)
        device: Device to run adaptation on

    Returns:
        model: Model with adapted BN statistics (in eval mode)

    Example:
        >>> model = build_aquasot(cfg)
        >>> model.load_state_dict(checkpoint['net'])
        >>> adapt_loader = create_adaptation_loader(cfg)
        >>> model = adapt_batchnorm_for_modality(model, adapt_loader, 'sonar')
        >>> # Now model's BN layers have Sonar-only statistics
    """
    model = model.to(device)

    # 1. Reset BN running stats and set to training mode
    bn_count = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.reset_running_stats()
            module.momentum = 0.2  # Higher momentum for faster adaptation convergence
            bn_count += 1

    if bn_count == 0:
        print('[BN Adaptation] Warning: No BatchNorm2d layers found in model')
        model.eval()
        return model

    mode_name = 'no-fusion' if modality == 'dual' else f'{modality}-only'
    print(f'[BN Adaptation] Adapting {bn_count} BatchNorm layers for {mode_name} mode...')

    # 2. Forward pass with dropped modality to update running stats
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch_count >= num_batches:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                rgb_t = batch.get('rgb_template', batch.get('template_rgb'))
                rgb_s = batch.get('rgb_search', batch.get('search_rgb'))
                sonar_t = batch.get('sonar_template', batch.get('template_sonar'))
                sonar_s = batch.get('sonar_search', batch.get('search_sonar'))
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 4:
                    rgb_t, rgb_s, sonar_t, sonar_s = batch[:4]
                else:
                    print(f'[BN Adaptation] Warning: Unexpected batch format, skipping')
                    continue
            else:
                print(f'[BN Adaptation] Warning: Unknown batch type {type(batch)}, skipping')
                continue

            # Move to device
            rgb_t = rgb_t.to(device)
            rgb_s = rgb_s.to(device)
            sonar_t = sonar_t.to(device)
            sonar_s = sonar_s.to(device)

            # Apply modality dropout based on mode
            if modality == 'sonar':
                rgb_t = torch.zeros_like(rgb_t)
                rgb_s = torch.zeros_like(rgb_s)
            elif modality == 'rgb':
                sonar_t = torch.zeros_like(sonar_t)
                sonar_s = torch.zeros_like(sonar_s)
            elif modality == 'dual':
                # No-fusion mode: keep both modalities, no dropout
                pass
            else:
                raise ValueError(f"modality must be 'sonar', 'rgb', or 'dual', got '{modality}'")

            # Forward updates running_mean/running_var
            try:
                _ = model(rgb_t, rgb_s, sonar_t, sonar_s)
                batch_count += 1
            except Exception as e:
                print(f'[BN Adaptation] Warning: Forward pass failed: {e}')
                continue

    print(f'[BN Adaptation] Completed adaptation with {batch_count} batches')

    # 3. Restore eval mode (but keep adapted running stats)
    model.eval()

    return model


def get_bn_stats(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Get current BN statistics for debugging/analysis.

    Args:
        model: Model to extract BN statistics from

    Returns:
        Dictionary mapping layer names to their running_mean and running_var
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            stats[name] = {
                'running_mean': module.running_mean.clone().cpu(),
                'running_var': module.running_var.clone().cpu(),
                'num_batches_tracked': module.num_batches_tracked.clone().cpu(),
            }
    return stats


def compare_bn_stats(
    stats_before: Dict[str, Dict[str, torch.Tensor]],
    stats_after: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    Compare BN statistics before and after adaptation.

    Args:
        stats_before: BN statistics before adaptation (from get_bn_stats)
        stats_after: BN statistics after adaptation (from get_bn_stats)

    Returns:
        Dictionary with comparison metrics (mean_diff, var_diff for each layer)
    """
    comparison = {}
    for name in stats_before:
        if name not in stats_after:
            continue

        mean_diff = (stats_after[name]['running_mean'] - stats_before[name]['running_mean']).abs().mean().item()
        var_diff = (stats_after[name]['running_var'] - stats_before[name]['running_var']).abs().mean().item()

        comparison[name] = {
            'mean_diff': mean_diff,
            'var_diff': var_diff,
        }

    # Aggregate statistics
    if comparison:
        all_mean_diffs = [v['mean_diff'] for v in comparison.values()]
        all_var_diffs = [v['var_diff'] for v in comparison.values()]
        comparison['_aggregate'] = {
            'avg_mean_diff': sum(all_mean_diffs) / len(all_mean_diffs),
            'avg_var_diff': sum(all_var_diffs) / len(all_var_diffs),
            'max_mean_diff': max(all_mean_diffs),
            'max_var_diff': max(all_var_diffs),
        }

    return comparison


def set_bn_mode(model: nn.Module, mode: str = 'eval') -> None:
    """
    Set BatchNorm mode without affecting other layers.

    Args:
        model: Model to modify
        mode: 'train' to use batch statistics, 'eval' to use running statistics
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if mode == 'train':
                module.train()
            else:
                module.eval()
