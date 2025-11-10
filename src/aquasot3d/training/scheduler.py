"""
Learning Rate Scheduler for AquaSOT-3D

This module provides learning rate scheduling strategies optimized for
multi-modal tracking training, including cosine annealing with warmup.

Supported Schedulers:
- Cosine Annealing with Warmup (recommended)
- Step Decay
- Exponential Decay
- Multi-Step with Milestones
- Polynomial Decay

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    ExponentialLR
)
from typing import Optional, List, Literal, Callable
import math
import logging
import warnings

logger = logging.getLogger(__name__)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine Annealing with Warmup learning rate scheduler.

    This scheduler implements:
    1. Linear warmup from 0 to base_lr over warmup_epochs
    2. Cosine annealing from base_lr to eta_min over T_max epochs
    3. Optional restarts with period T_mult

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of epochs until first restart
        T_mult: Factor by which T_i increases after restart (default: 1)
        eta_min: Minimum learning rate
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting LR for warmup (default: 0)
        last_epoch: The index of last epoch

    Example:
        >>> scheduler = CosineAnnealingWarmupRestarts(
        ...     optimizer,
        ...     T_0=50,
        ...     T_mult=1,
        ...     eta_min=1e-6,
        ...     warmup_epochs=5,
        ...     warmup_start_lr=0
        ... )
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr

        self.T_cur = last_epoch
        self.T_i = T_0
        self.cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            epoch_after_warmup = self.last_epoch - self.warmup_epochs

            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult

            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

        self.last_epoch = math.floor(epoch)
        self.T_cur = epoch - self.warmup_epochs if epoch >= self.warmup_epochs else epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PolynomialLRDecay(_LRScheduler):
    """
    Polynomial learning rate decay.

    Args:
        optimizer: Wrapped optimizer
        max_epoch: Maximum number of epochs
        power: Polynomial power (default: 0.9)
        eta_min: Minimum learning rate
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_epoch: int,
        power: float = 0.9,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.max_epoch = max_epoch
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_epoch) ** self.power
        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: Literal['cosine', 'step', 'multistep', 'exponential', 'polynomial'] = 'cosine',
    max_epoch: int = 60,
    warmup_epochs: int = 5,
    warmup_start_lr: float = 0,
    eta_min: float = 1e-6,
    step_size: Optional[int] = None,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
    power: float = 0.9,
    T_mult: int = 1,
    **kwargs
) -> _LRScheduler:
    """
    Build learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        max_epoch: Maximum training epochs
        warmup_epochs: Number of warmup epochs (for cosine)
        warmup_start_lr: Starting LR for warmup
        eta_min: Minimum learning rate
        step_size: Step size for step scheduler
        milestones: Milestones for multistep scheduler
        gamma: Decay factor
        power: Power for polynomial decay
        T_mult: Restart multiplier for cosine annealing
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler

    Example:
        >>> # Cosine annealing with warmup (recommended)
        >>> scheduler = build_scheduler(
        ...     optimizer,
        ...     'cosine',
        ...     max_epoch=60,
        ...     warmup_epochs=5,
        ...     eta_min=1e-6
        ... )
        >>>
        >>> # Step decay
        >>> scheduler = build_scheduler(
        ...     optimizer,
        ...     'step',
        ...     step_size=20,
        ...     gamma=0.1
        ... )
        >>>
        >>> # Multi-step with milestones
        >>> scheduler = build_scheduler(
        ...     optimizer,
        ...     'multistep',
        ...     milestones=[20, 40],
        ...     gamma=0.1
        ... )
    """
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            T_0=max_epoch - warmup_epochs,
            T_mult=T_mult,
            eta_min=eta_min,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr
        )
        logger.info(
            f"Built Cosine Annealing scheduler: "
            f"max_epoch={max_epoch}, warmup={warmup_epochs}, "
            f"eta_min={eta_min}"
        )

    elif scheduler_type == 'step':
        if step_size is None:
            step_size = max_epoch // 3  # Default: decay 3 times

        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        logger.info(f"Built Step scheduler: step_size={step_size}, gamma={gamma}")

    elif scheduler_type == 'multistep':
        if milestones is None:
            milestones = [max_epoch * 2 // 3, max_epoch * 8 // 9]  # Default milestones

        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        logger.info(f"Built MultiStep scheduler: milestones={milestones}, gamma={gamma}")

    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
        logger.info(f"Built Exponential scheduler: gamma={gamma}")

    elif scheduler_type == 'polynomial':
        scheduler = PolynomialLRDecay(
            optimizer,
            max_epoch=max_epoch,
            power=power,
            eta_min=eta_min
        )
        logger.info(f"Built Polynomial scheduler: power={power}, eta_min={eta_min}")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class WarmupScheduler:
    """
    Wrapper for adding warmup to any scheduler.

    Args:
        scheduler: Base scheduler
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting learning rate for warmup
        warmup_mode: Warmup mode ('linear' or 'constant')

    Example:
        >>> base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        >>> scheduler = WarmupScheduler(
        ...     base_scheduler,
        ...     warmup_epochs=5,
        ...     warmup_start_lr=1e-6
        ... )
    """

    def __init__(
        self,
        scheduler: _LRScheduler,
        warmup_epochs: int,
        warmup_start_lr: float = 0,
        warmup_mode: Literal['linear', 'constant'] = 'linear'
    ):
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_mode = warmup_mode
        self.current_epoch = 0

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_mode == 'linear':
                alpha = self.current_epoch / self.warmup_epochs
                for param_group, base_lr in zip(self.scheduler.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
            else:  # constant
                for param_group in self.scheduler.optimizer.param_groups:
                    param_group['lr'] = self.warmup_start_lr
        else:
            # Normal scheduling
            self.scheduler.step(epoch=self.current_epoch - self.warmup_epochs)

    def state_dict(self):
        """Return scheduler state."""
        return {
            'base_scheduler': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'warmup_mode': self.warmup_mode,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict['base_scheduler'])
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.warmup_mode = state_dict['warmup_mode']
        self.base_lrs = state_dict['base_lrs']


def get_current_lr(optimizer: optim.Optimizer) -> List[float]:
    """
    Get current learning rates from optimizer.

    Args:
        optimizer: Optimizer to query

    Returns:
        List of current learning rates for each parameter group
    """
    return [group['lr'] for group in optimizer.param_groups]


def adjust_learning_rate(
    optimizer: optim.Optimizer,
    epoch: int,
    max_epoch: int,
    base_lr: float,
    schedule_type: Literal['cosine', 'step', 'linear'] = 'cosine',
    warmup_epochs: int = 0,
    eta_min: float = 0
) -> float:
    """
    Manually adjust learning rate without scheduler object.

    Args:
        optimizer: Optimizer to adjust
        epoch: Current epoch
        max_epoch: Maximum epochs
        base_lr: Base learning rate
        schedule_type: Schedule type
        warmup_epochs: Number of warmup epochs
        eta_min: Minimum learning rate

    Returns:
        New learning rate
    """
    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        epoch_after_warmup = epoch - warmup_epochs
        max_epoch_after_warmup = max_epoch - warmup_epochs

        if schedule_type == 'cosine':
            lr = eta_min + (base_lr - eta_min) * (
                1 + math.cos(math.pi * epoch_after_warmup / max_epoch_after_warmup)
            ) / 2
        elif schedule_type == 'linear':
            lr = base_lr * (1 - epoch_after_warmup / max_epoch_after_warmup)
        elif schedule_type == 'step':
            lr = base_lr * (0.1 ** (epoch_after_warmup // (max_epoch_after_warmup // 3)))
        else:
            lr = base_lr

    # Update all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
