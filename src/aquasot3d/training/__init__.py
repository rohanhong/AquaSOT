"""
Training Infrastructure for AquaSOT-3D

This package provides comprehensive training utilities including:
- Loss management with multi-task weighting
- Optimizer configuration with parameter grouping
- Learning rate scheduling with warmup
- Checkpoint management with best model tracking
- Complete training loop with AMP and DDP support

Example:
    >>> from aquasot3d.training import (
    ...     LossManager, build_optimizer, build_scheduler,
    ...     CheckpointManager, Trainer, build_trainer
    ... )
    >>>
    >>> # Build training components
    >>> loss_manager = LossManager()
    >>> optimizer = build_optimizer(model, 'adamw', base_lr=2e-4)
    >>> scheduler = build_scheduler(optimizer, 'cosine', max_epoch=60)
    >>> checkpoint_manager = CheckpointManager('outputs/checkpoints')
    >>>
    >>> # Build and run trainer
    >>> trainer = Trainer(
    ...     model, optimizer, scheduler, loss_manager,
    ...     train_loader, val_loader, checkpoint_manager
    ... )
    >>> trainer.train()

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

from .loss_manager import LossManager, build_loss_manager
from .optimizer import (
    build_optimizer,
    get_parameter_groups,
    clip_gradients,
    freeze_parameters,
    unfreeze_parameters
)
from .scheduler import (
    build_scheduler,
    CosineAnnealingWarmupRestarts,
    PolynomialLRDecay,
    WarmupScheduler,
    get_current_lr,
    adjust_learning_rate
)
from .checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint
)
from .trainer import Trainer, build_trainer

__all__ = [
    # Loss management
    'LossManager',
    'build_loss_manager',

    # Optimizer
    'build_optimizer',
    'get_parameter_groups',
    'clip_gradients',
    'freeze_parameters',
    'unfreeze_parameters',

    # Scheduler
    'build_scheduler',
    'CosineAnnealingWarmupRestarts',
    'PolynomialLRDecay',
    'WarmupScheduler',
    'get_current_lr',
    'adjust_learning_rate',

    # Checkpoint
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',

    # Trainer
    'Trainer',
    'build_trainer'
]
