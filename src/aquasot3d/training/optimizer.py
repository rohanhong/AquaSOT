"""
Optimizer Configuration for AquaSOT-3D

This module provides utilities for configuring optimizers with parameter grouping,
learning rate scheduling, and gradient clipping for multi-modal tracking training.

Key Features:
- Parameter group separation (backbone, fusion, heads)
- Layer-wise learning rate decay (optional)
- Weight decay configuration
- Gradient clipping utilities

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Literal, Callable
import logging
import re

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    optimizer_type: Literal['sgd', 'adam', 'adamw'] = 'adamw',
    base_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple = (0.9, 0.999),
    parameter_groups: Optional[List[Dict[str, Any]]] = None,
    layer_decay: Optional[float] = None,
    backbone_lr_multiplier: float = 0.1,
    head_lr_multiplier: float = 1.0,
    filter_bias_and_bn: bool = True,
    **kwargs
) -> optim.Optimizer:
    """
    Build optimizer with flexible parameter grouping and layer-wise LR decay.

    Args:
        model: PyTorch model to optimize
        optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
        base_lr: Base learning rate
        weight_decay: Weight decay factor
        momentum: Momentum for SGD
        betas: Betas for Adam/AdamW
        parameter_groups: Custom parameter groups (overrides automatic grouping)
        layer_decay: Layer-wise learning rate decay factor (optional)
        backbone_lr_multiplier: Learning rate multiplier for backbone parameters (default: 0.1)
        head_lr_multiplier: Learning rate multiplier for head parameters (default: 1.0)
        filter_bias_and_bn: If True, no weight decay for bias and BatchNorm params
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer

    Example:
        >>> # Simple usage
        >>> optimizer = build_optimizer(model, 'adamw', base_lr=2e-4)
        >>>
        >>> # With custom backbone LR multiplier
        >>> optimizer = build_optimizer(model, 'adamw', base_lr=2e-4, backbone_lr_multiplier=0.05)
        >>>
        >>> # With parameter grouping
        >>> groups = [
        ...     {'params': model.backbone.parameters(), 'lr': 1e-4},
        ...     {'params': model.fusion.parameters(), 'lr': 2e-4},
        ...     {'params': model.heads.parameters(), 'lr': 2e-4}
        ... ]
        >>> optimizer = build_optimizer(model, parameter_groups=groups)
        >>>
        >>> # With layer-wise decay
        >>> optimizer = build_optimizer(model, layer_decay=0.95)
    """
    if parameter_groups is None:
        # Automatic parameter grouping
        if layer_decay is not None:
            parameter_groups = _get_layer_decay_groups(
                model, base_lr, weight_decay, layer_decay, filter_bias_and_bn
            )
        else:
            parameter_groups = _get_default_parameter_groups(
                model, base_lr, weight_decay, filter_bias_and_bn,
                backbone_lr_multiplier=backbone_lr_multiplier,
                head_lr_multiplier=head_lr_multiplier
            )

    # Build optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            parameter_groups,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            parameter_groups,
            lr=base_lr,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            parameter_groups,
            lr=base_lr,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger.info(f"Built {optimizer_type.upper()} optimizer with {len(parameter_groups)} parameter groups")
    logger.info(f"Base LR: {base_lr}, Weight Decay: {weight_decay}")

    return optimizer


def _get_default_parameter_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    filter_bias_and_bn: bool,
    backbone_lr_multiplier: float = 0.1,
    head_lr_multiplier: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Get default parameter groups with backbone/fusion/heads separation.

    CRITICAL FIX (Audit 2025120301 Finding 10):
    - Proper handling of bias, BatchNorm, LayerNorm parameters
    - These parameters should NOT have weight decay applied
    - Detection is now based on parameter names, not just dimensions

    Args:
        model: Model to extract parameters from
        base_lr: Base learning rate
        weight_decay: Weight decay factor
        filter_bias_and_bn: Filter bias and BatchNorm parameters
        backbone_lr_multiplier: Learning rate multiplier for backbone (default: 0.1)
        head_lr_multiplier: Learning rate multiplier for heads (default: 1.0)

    Returns:
        List of parameter groups
    """
    groups = []

    # CRITICAL FIX (Audit 2025120301 Finding 10):
    # Separate parameters by component AND by decay/no-decay
    # Store (name, param) tuples to enable name-based filtering
    backbone_decay = []
    backbone_no_decay = []
    fusion_decay = []
    fusion_no_decay = []
    head_decay = []
    head_no_decay = []
    other_decay = []
    other_no_decay = []

    def should_skip_weight_decay(name: str, param) -> bool:
        """
        Check if parameter should skip weight decay.

        CRITICAL FIX (Audit 2025120301 Finding 10):
        Bias, BatchNorm, LayerNorm parameters should NOT have weight decay.
        This prevents training instability and improves generalization.
        """
        # Check by name patterns (more reliable than dimension)
        name_lower = name.lower()
        skip_patterns = [
            'bias',           # All bias terms
            '.bn',            # BatchNorm (e.g., bn1.weight)
            'batchnorm',      # BatchNorm explicit
            'batch_norm',     # BatchNorm variant
            'layernorm',      # LayerNorm
            'layer_norm',     # LayerNorm variant
            '.norm',          # Generic norm layers
            'norm.',          # Norm prefix
            'gamma',          # Some norm implementations
            'beta',           # Some norm implementations
            'cls_token',      # ViT class token
            'pos_embed',      # Position embeddings
        ]
        for pattern in skip_patterns:
            if pattern in name_lower:
                return True

        # Fallback: 1D parameters are typically bias or norm parameters
        if filter_bias_and_bn and hasattr(param, 'ndim') and param.ndim == 1:
            return True

        return False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = should_skip_weight_decay(name, param) if filter_bias_and_bn else False

        # Categorize parameters by component
        if 'backbone' in name.lower() or 'encoder' in name.lower():
            if no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        elif 'fusion' in name.lower() or 'dcma' in name.lower() or 'g2dcma' in name.lower():
            if no_decay:
                fusion_no_decay.append(param)
            else:
                fusion_decay.append(param)
        elif 'head' in name.lower():
            if no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if no_decay:
                other_no_decay.append(param)
            else:
                other_decay.append(param)

    # Build parameter groups with different learning rates
    if backbone_decay:
        groups.append({
            'params': backbone_decay,
            'lr': base_lr * backbone_lr_multiplier,
            'weight_decay': weight_decay,
            'name': 'backbone'
        })
    if backbone_no_decay:
        groups.append({
            'params': backbone_no_decay,
            'lr': base_lr * backbone_lr_multiplier,
            'weight_decay': 0.0,  # CRITICAL FIX: no weight decay
            'name': 'backbone_no_decay'
        })

    if fusion_decay:
        groups.append({
            'params': fusion_decay,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'fusion'
        })
    if fusion_no_decay:
        groups.append({
            'params': fusion_no_decay,
            'lr': base_lr,
            'weight_decay': 0.0,  # CRITICAL FIX: no weight decay
            'name': 'fusion_no_decay'
        })

    if head_decay:
        groups.append({
            'params': head_decay,
            'lr': base_lr * head_lr_multiplier,
            'weight_decay': weight_decay,
            'name': 'heads'
        })
    if head_no_decay:
        groups.append({
            'params': head_no_decay,
            'lr': base_lr * head_lr_multiplier,
            'weight_decay': 0.0,  # CRITICAL FIX: no weight decay
            'name': 'heads_no_decay'
        })

    if other_decay:
        groups.append({
            'params': other_decay,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'other'
        })
    if other_no_decay:
        groups.append({
            'params': other_no_decay,
            'lr': base_lr,
            'weight_decay': 0.0,  # CRITICAL FIX: no weight decay
            'name': 'other_no_decay'
        })

    logger.info(f"Created {len(groups)} parameter groups (backbone_lr_mult={backbone_lr_multiplier}, head_lr_mult={head_lr_multiplier}):")
    for group in groups:
        logger.info(f"  - {group.get('name', 'unnamed')}: {len(group['params'])} params, lr={group['lr']:.6f}, wd={group['weight_decay']}")

    return groups


def _get_layer_decay_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
    filter_bias_and_bn: bool
) -> List[Dict[str, Any]]:
    """
    Get parameter groups with layer-wise learning rate decay.

    This is particularly useful for Vision Transformers where deeper layers
    (closer to output) should have higher learning rates.

    Args:
        model: Model to extract parameters from
        base_lr: Base learning rate
        weight_decay: Weight decay factor
        layer_decay: Decay factor for each layer (e.g., 0.95)
        filter_bias_and_bn: Filter bias and BatchNorm parameters

    Returns:
        List of parameter groups with layer-wise LR decay
    """
    # Get parameter names and depths
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Extract layer depth from name
        # Assume format like: backbone.encoder.layer.12.xxx or blocks.12.xxx
        layer_id = _extract_layer_id(name)

        if layer_id not in param_groups:
            param_groups[layer_id] = []

        param_groups[layer_id].append((name, param))

    # Sort by layer depth
    sorted_layers = sorted(param_groups.keys())
    num_layers = len(sorted_layers)

    groups = []
    for idx, layer_id in enumerate(sorted_layers):
        # Calculate layer-specific learning rate
        # Deeper layers (higher idx) get higher LR
        decay_factor = layer_decay ** (num_layers - idx - 1)
        layer_lr = base_lr * decay_factor

        layer_params = [p for _, p in param_groups[layer_id]]

        groups.append({
            'params': layer_params,
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'name': f'layer_{layer_id}'
        })

    if filter_bias_and_bn:
        groups = _split_weight_decay_groups(groups)

    logger.info(f"Created {len(groups)} layer-decay parameter groups with decay={layer_decay}")

    return groups


def _extract_layer_id(param_name: str) -> int:
    """
    Extract layer ID from parameter name.

    Args:
        param_name: Parameter name

    Returns:
        Layer ID (default 0 if not found)
    """
    # Try to match common patterns
    patterns = [
        r'layer\.(\d+)',
        r'layers\.(\d+)',
        r'block\.(\d+)',
        r'blocks\.(\d+)',
        r'encoder\.(\d+)',
        r'stage\.(\d+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, param_name)
        if match:
            return int(match.group(1))

    return 0  # Default layer ID


def _split_weight_decay_groups(
    groups: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Split parameter groups to separate bias and BatchNorm parameters
    (which typically don't use weight decay).

    Args:
        groups: Original parameter groups

    Returns:
        Split parameter groups
    """
    new_groups = []

    for group in groups:
        params = group['params']
        decay_params = []
        no_decay_params = []

        for param in params:
            # Check if parameter should have weight decay
            # Typically bias and BatchNorm parameters don't
            if hasattr(param, 'ndim') and param.ndim == 1:
                # 1D parameters are typically bias or BN parameters
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Create groups with/without weight decay
        if decay_params:
            decay_group = group.copy()
            decay_group['params'] = decay_params
            new_groups.append(decay_group)

        if no_decay_params:
            no_decay_group = group.copy()
            no_decay_group['params'] = no_decay_params
            no_decay_group['weight_decay'] = 0.0
            no_decay_group['name'] = group.get('name', 'unnamed') + '_no_decay'
            new_groups.append(no_decay_group)

    return new_groups


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> torch.Tensor:
    """
    Clip gradients by global norm.

    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use (2.0 for L2 norm)
        error_if_nonfinite: Raise error if gradient is inf or NaN

    Returns:
        Total gradient norm before clipping

    Example:
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> grad_norm = clip_gradients(model, max_norm=1.0)
        >>> optimizer.step()
    """
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite
    )

    return total_norm


def get_parameter_groups(
    model: nn.Module,
    base_lr: float = 2e-4,
    weight_decay: float = 1e-4
) -> List[Dict[str, Any]]:
    """
    Get parameter groups for AquaSOT-3D tracker.

    This is a convenience function that returns reasonable default parameter
    groups for the tracker architecture.

    Args:
        model: AquaSOT-3D tracker model
        base_lr: Base learning rate
        weight_decay: Weight decay factor

    Returns:
        List of parameter groups
    """
    return _get_default_parameter_groups(model, base_lr, weight_decay, filter_bias_and_bn=True)


def freeze_parameters(model: nn.Module, freeze_patterns: List[str]):
    """
    Freeze parameters matching specified patterns.

    Args:
        model: Model to freeze parameters in
        freeze_patterns: List of name patterns to freeze

    Example:
        >>> # Freeze backbone
        >>> freeze_parameters(model, ['backbone'])
        >>>
        >>> # Freeze multiple components
        >>> freeze_parameters(model, ['backbone', 'fusion.geometric'])
    """
    frozen_count = 0

    for name, param in model.named_parameters():
        for pattern in freeze_patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                break

    logger.info(f"Froze {frozen_count} parameters matching patterns: {freeze_patterns}")


def unfreeze_parameters(model: nn.Module, unfreeze_patterns: Optional[List[str]] = None):
    """
    Unfreeze parameters matching specified patterns.

    Args:
        model: Model to unfreeze parameters in
        unfreeze_patterns: List of name patterns to unfreeze (None = unfreeze all)

    Example:
        >>> # Unfreeze all
        >>> unfreeze_parameters(model)
        >>>
        >>> # Unfreeze specific components
        >>> unfreeze_parameters(model, ['fusion', 'heads'])
    """
    unfrozen_count = 0

    for name, param in model.named_parameters():
        if unfreeze_patterns is None:
            param.requires_grad = True
            unfrozen_count += 1
        else:
            for pattern in unfreeze_patterns:
                if pattern in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    break

    logger.info(f"Unfroze {unfrozen_count} parameters")
