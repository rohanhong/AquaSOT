"""
Training Loop and Logic for AquaSOT-3D

This module implements the complete training infrastructure including:
- Epoch-based training loop
- Validation loop
- Gradient accumulation
- Mixed precision training (AMP)
- Distributed training (DDP) support
- TensorBoard logging
- Progress tracking

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Optional, Any, Callable, Union
from pathlib import Path
from tqdm import tqdm
import logging
import time
from collections import defaultdict

from aquasot3d.training.loss_manager import LossManager
from aquasot3d.training.checkpoint import CheckpointManager
from aquasot3d.training.optimizer import clip_gradients

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


logger = logging.getLogger(__name__)


class Trainer:
    """
    Complete training pipeline for AquaSOT-3D tracker.

    This class orchestrates the entire training process including forward/backward passes,
    optimization, validation, checkpointing, and logging.

    Args:
        model: Tracking model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_manager: Loss computation and weighting
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        checkpoint_manager: Checkpoint management
        device: Device to train on
        max_epochs: Maximum training epochs
        gradient_accumulation_steps: Steps for gradient accumulation
        use_amp: Use automatic mixed precision
        grad_clip_norm: Maximum gradient norm (None = no clipping)
        log_interval: Logging interval in iterations
        eval_interval: Evaluation interval in epochs
        tensorboard_dir: TensorBoard log directory
        distributed: Whether using distributed training

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     loss_manager=loss_manager,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     checkpoint_manager=checkpoint_manager,
        ...     max_epochs=60
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: _LRScheduler,
        loss_manager: LossManager,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: Union[str, torch.device] = 'cuda',
        max_epochs: int = 60,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        log_interval: int = 10,
        eval_interval: int = 1,
        tensorboard_dir: Optional[Union[str, Path]] = None,
        distributed: bool = False,
        rank: int = 0,
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_manager = loss_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_manager = checkpoint_manager

        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        self.distributed = distributed
        self.rank = rank
        self.is_main_process = (rank == 0)

        # Move model to device
        self.model = self.model.to(self.device)

        # Gradient scaler for AMP
        self.scaler = GradScaler() if self.use_amp else None

        # TensorBoard writer
        self.writer = None
        if tensorboard_dir is not None and TENSORBOARD_AVAILABLE and self.is_main_process:
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            logger.info(f"TensorBoard logging to: {tensorboard_dir}")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        logger.info(f"Trainer initialized: epochs={max_epochs}, amp={use_amp}, distributed={distributed}")

    def train(self, start_epoch: int = 0):
        """
        Main training loop.

        Args:
            start_epoch: Epoch to start from (for resuming)

        Example:
            >>> # Fresh training
            >>> trainer.train()
            >>>
            >>> # Resume training
            >>> start_epoch = checkpoint_manager.resume(model, optimizer, scheduler)
            >>> trainer.train(start_epoch=start_epoch)
        """
        logger.info(f"Starting training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = None
            if self.val_loader is not None and (epoch + 1) % self.eval_interval == 0:
                val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            if self.checkpoint_manager is not None and self.is_main_process:
                metrics = {**train_metrics}
                if val_metrics is not None:
                    metrics.update(val_metrics)

                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics
                )

            # Log epoch summary
            if self.is_main_process:
                self._log_epoch_summary(epoch, train_metrics, val_metrics)

        logger.info("Training completed!")

        if self.writer is not None:
            self.writer.close()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of averaged training metrics
        """
        self.model.train()

        epoch_metrics = defaultdict(float)
        epoch_counts = defaultdict(int)

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.max_epochs}',
            disable=not self.is_main_process
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        batch['rgb_template'],
                        batch['rgb_search'],
                        batch['sonar_template'],
                        batch['sonar_search']
                    )
                    total_loss, loss_dict = self.loss_manager.compute_losses(
                        outputs,
                        batch,
                        epoch=epoch
                    )
            else:
                outputs = self.model(
                    batch['rgb_template'],
                    batch['rgb_search'],
                    batch['sonar_template'],
                    batch['sonar_search']
                )
                total_loss, loss_dict = self.loss_manager.compute_losses(
                    outputs,
                    batch,
                    epoch=epoch
                )

            # Scale loss for gradient accumulation
            loss = total_loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_norm is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = clip_gradients(self.model, self.grad_clip_norm)
                    else:
                        grad_norm = clip_gradients(self.model, self.grad_clip_norm)
                else:
                    grad_norm = 0.0

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Log to TensorBoard
                if self.is_main_process and self.global_step % self.log_interval == 0:
                    self._log_training_step(loss_dict, grad_norm)

                self.global_step += 1

            # Accumulate metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key] += value.item()
                else:
                    epoch_metrics[key] += value
                epoch_counts[key] += 1

            # Update progress bar
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

        # Average metrics
        averaged_metrics = {
            key: value / epoch_counts[key]
            for key, value in epoch_metrics.items()
        }

        return averaged_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_metrics = defaultdict(float)
        val_counts = defaultdict(int)

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc='Validation',
                disable=not self.is_main_process
            ):
                # Move batch to device
                batch = self._to_device(batch)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            batch['rgb_template'],
                            batch['rgb_search'],
                            batch['sonar_template'],
                            batch['sonar_search']
                        )
                        total_loss, loss_dict = self.loss_manager.compute_losses(
                            outputs,
                            batch,
                            epoch=epoch
                        )
                else:
                    outputs = self.model(
                        batch['rgb_template'],
                        batch['rgb_search'],
                        batch['sonar_template'],
                        batch['sonar_search']
                    )
                    total_loss, loss_dict = self.loss_manager.compute_losses(
                        outputs,
                        batch,
                        epoch=epoch
                    )

                # Accumulate metrics
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        val_metrics[key] += value.item()
                    else:
                        val_metrics[key] += value
                    val_counts[key] += 1

        # Average metrics
        averaged_metrics = {
            f'val_{key}': value / val_counts[key]
            for key, value in val_metrics.items()
        }

        # Log to TensorBoard
        if self.is_main_process and self.writer is not None:
            for key, value in averaged_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)

        return averaged_metrics

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch

    def _log_training_step(self, loss_dict: Dict[str, torch.Tensor], grad_norm: float):
        """Log training step to TensorBoard."""
        if self.writer is None:
            return

        # Log losses
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)

        # Log gradient norm
        if grad_norm > 0:
            self.writer.add_scalar('Train/grad_norm', grad_norm, self.global_step)

        # Log learning rates
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'Train/lr_group_{i}',
                param_group['lr'],
                self.global_step
            )

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ):
        """Log epoch summary."""
        summary = f"\nEpoch {epoch + 1}/{self.max_epochs} Summary:\n"
        summary += "=" * 60 + "\n"

        # Training metrics
        summary += "Training Metrics:\n"
        for key, value in train_metrics.items():
            summary += f"  {key}: {value:.4f}\n"

        # Validation metrics
        if val_metrics is not None:
            summary += "\nValidation Metrics:\n"
            for key, value in val_metrics.items():
                summary += f"  {key}: {value:.4f}\n"

        # Learning rate
        summary += f"\nLearning Rate: {self.optimizer.param_groups[0]['lr']:.2e}\n"
        summary += "=" * 60

        logger.info(summary)

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state."""
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.global_step = state_dict.get('global_step', 0)
        self.best_metric = state_dict.get('best_metric', float('-inf'))


def build_trainer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: _LRScheduler,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None
) -> Trainer:
    """
    Factory function to build trainer from configuration.

    Args:
        model: Tracking model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary

    Returns:
        Initialized Trainer

    Example:
        >>> config = {
        ...     'max_epochs': 60,
        ...     'use_amp': True,
        ...     'grad_clip_norm': 1.0,
        ...     'checkpoint_dir': 'outputs/checkpoints',
        ...     'tensorboard_dir': 'outputs/tensorboard'
        ... }
        >>> trainer = build_trainer(model, optimizer, scheduler, train_loader, val_loader, config)
    """
    config = config or {}

    # Build loss manager
    loss_manager = LossManager(
        loss_weights=config.get('loss_weights'),
        weight_schedule=config.get('weight_schedule'),
        schedule_config=config.get('schedule_config')
    )

    # Build checkpoint manager
    checkpoint_manager = None
    if 'checkpoint_dir' in config:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=config['checkpoint_dir'],
            max_to_keep=config.get('max_checkpoints_to_keep', 5),
            save_best_only=config.get('save_best_only', False),
            metric_name=config.get('best_metric_name', 'val_fusion_box'),
            mode=config.get('best_metric_mode', 'min')
        )

    # Build trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_manager=loss_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=checkpoint_manager,
        device=config.get('device', 'cuda'),
        max_epochs=config.get('max_epochs', 60),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        use_amp=config.get('use_amp', True),
        grad_clip_norm=config.get('grad_clip_norm', 1.0),
        log_interval=config.get('log_interval', 10),
        eval_interval=config.get('eval_interval', 1),
        tensorboard_dir=config.get('tensorboard_dir'),
        distributed=config.get('distributed', False),
        rank=config.get('rank', 0)
    )

    return trainer
