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
from typing import Dict, Optional, Any, Callable, Union, Tuple
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
        checkpoint_save_interval: Checkpoint save interval in epochs (default: 1)
        save_best: Whether to save best checkpoint when metric improves (default: True)
        save_last: Whether to save last checkpoint every epoch (default: True)
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
        checkpoint_save_interval: int = 1,
        save_best: bool = True,
        save_last: bool = True,
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
        self.checkpoint_save_interval = checkpoint_save_interval
        self.save_best = save_best
        self.save_last = save_last

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

        # Metric fallback tracking
        self._metric_fallback_applied = False

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

            # Always validate every epoch to determine best checkpoint
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Three-tier checkpoint saving
            if self.checkpoint_manager is not None and self.is_main_process:
                metrics = {**train_metrics}
                if val_metrics is not None:
                    metrics.update(val_metrics)

                # 1. Save "last" checkpoint every epoch (overwrite) - only if save_last is enabled
                if self.save_last:
                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=metrics,
                        filename='checkpoint_last.pth'
                    )

                # 2. Save "best" checkpoint if metric improves (overwrite) - only if save_best is enabled
                if self.save_best:
                    current_metric = metrics.get(self.checkpoint_manager.metric_name, None)

                    # Metric fallback logic: if monitor metric not found, try val_total
                    if current_metric is None and not self._metric_fallback_applied:
                        original_metric = self.checkpoint_manager.metric_name

                        # Try fallback to val_total
                        if 'val_total' in metrics and original_metric != 'val_total':
                            logger.warning(
                                f"Metric '{original_metric}' not found in metrics. "
                                f"Available metrics: {list(metrics.keys())}. "
                                f"Falling back to 'val_total' for best checkpoint tracking."
                            )
                            self.checkpoint_manager.metric_name = 'val_total'
                            self.checkpoint_manager.mode = 'min'
                            self.checkpoint_manager.best_metric = float('inf')  # Reset for min mode
                            self._metric_fallback_applied = True
                            current_metric = metrics.get('val_total')
                        else:
                            # Neither original metric nor val_total found - critical error
                            logger.error(
                                f"CRITICAL: Monitor metric '{original_metric}' not found and fallback 'val_total' "
                                f"is also unavailable. Available metrics: {list(metrics.keys())}. "
                                f"Best checkpoint WILL NOT be saved. Please fix metric configuration."
                            )

                    if current_metric is not None:
                        if self.checkpoint_manager._is_better(current_metric, self.checkpoint_manager.best_metric):
                            self.checkpoint_manager.best_metric = current_metric
                            self.checkpoint_manager.best_epoch = epoch
                            self.checkpoint_manager.save(
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                epoch=epoch,
                                metrics=metrics,
                                is_best=True,
                                filename='checkpoint_best.pth'
                            )
                            logger.info(f"Saved best checkpoint at epoch {epoch + 1} with {self.checkpoint_manager.metric_name}={current_metric:.4f}")

                # 3. Save interval checkpoint every N epochs (keep multiple)
                if (epoch + 1) % self.checkpoint_save_interval == 0:
                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=metrics,
                        filename=f'checkpoint_epoch_{epoch + 1}.pth'
                    )

            # Log epoch summary
            if self.is_main_process:
                self._log_epoch_summary(epoch, train_metrics, val_metrics)

        logger.info("Training completed!")

        if self.writer is not None:
            self.writer.close()

    def _compute_model_losses(self, model_losses: Dict[str, torch.Tensor], epoch: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted total loss from model-internal losses.

        Maps model loss names to config loss weight keys and applies weighting.

        Args:
            model_losses: Loss dictionary from model outputs['losses']
            epoch: Current epoch (for loss scheduling if needed)

        Returns:
            (total_loss, loss_dict) where loss_dict contains all component losses and total
        """
        loss_dict = {}
        total_loss = 0.0

        # Get loss weights from config
        loss_weights = self.loss_manager.loss_weights

        # Define loss name mapping from model internal names to config keys
        # Note: renamed for clarity (fusion geometric vs weak3d component consistency)
        loss_name_map = {
            'rgb_box_loss': 'rgb_box',
            'sonar_box_loss': 'sonar_box',
            'fusion_box_loss': 'fusion_box',
            'geometric_consistency': 'geometric_consistency',  # Fusion: scale vs radial encodings
            'component_consistency': 'component_consistency',  # Weak3D: X_rgb vs X_sonar agreement
            'representation_consistency': 'representation_consistency',
            'X_consistency_loss': 'component_consistency',  # Legacy: map old key to new name
        }

        # Process individual losses with name mapping
        for model_key, loss_value in model_losses.items():
            if loss_value is None or not isinstance(loss_value, torch.Tensor):
                continue

            # Map model loss name to config weight key
            config_key = loss_name_map.get(model_key, model_key)

            # Get weight for this loss component
            weight = loss_weights.get(config_key, 0.0)

            # Weighted loss
            weighted_loss = weight * loss_value
            total_loss = total_loss + weighted_loss

            # Store both raw and weighted loss for logging
            loss_dict[f'{config_key}_raw'] = loss_value.detach()
            loss_dict[config_key] = weighted_loss.detach()

        # Handle cyclic losses (combine rgb_cyclic_loss + sonar_cyclic_loss -> weak3d_cycle)
        cyclic_losses = []
        if 'rgb_cyclic_loss' in model_losses and model_losses['rgb_cyclic_loss'] is not None:
            cyclic_losses.append(model_losses['rgb_cyclic_loss'])
        if 'sonar_cyclic_loss' in model_losses and model_losses['sonar_cyclic_loss'] is not None:
            cyclic_losses.append(model_losses['sonar_cyclic_loss'])

        if cyclic_losses:
            avg_cyclic_loss = sum(cyclic_losses) / len(cyclic_losses)
            weight = loss_weights.get('weak3d_cycle', 0.0)
            weighted_cyclic = weight * avg_cyclic_loss
            total_loss = total_loss + weighted_cyclic
            loss_dict['weak3d_cycle_raw'] = avg_cyclic_loss.detach()
            loss_dict['weak3d_cycle'] = weighted_cyclic.detach()

        # Handle projection losses (combine rgb_proj_loss + sonar_proj_loss)
        proj_losses = []
        if 'rgb_proj_loss' in model_losses and model_losses['rgb_proj_loss'] is not None:
            proj_losses.append(model_losses['rgb_proj_loss'])
        if 'sonar_proj_loss' in model_losses and model_losses['sonar_proj_loss'] is not None:
            proj_losses.append(model_losses['sonar_proj_loss'])

        if proj_losses:
            avg_proj_loss = sum(proj_losses) / len(proj_losses)
            # Use same weight as cyclic or create a new one
            weight = loss_weights.get('weak3d_proj', loss_weights.get('weak3d_cycle', 0.0))
            weighted_proj = weight * avg_proj_loss
            total_loss = total_loss + weighted_proj
            loss_dict['weak3d_proj_raw'] = avg_proj_loss.detach()
            loss_dict['weak3d_proj'] = weighted_proj.detach()

        # Add total loss to dict
        loss_dict['total'] = total_loss.detach()

        return total_loss, loss_dict

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

            # Prepare ground truth boxes
            gt_boxes = {}
            if 'rgb_boxes' in batch:
                gt_boxes['rgb'] = batch['rgb_boxes']
            if 'sonar_boxes' in batch:
                gt_boxes['sonar'] = batch['sonar_boxes']

            # CRITICAL FIX (Audit 2025111408 Finding 3):
            # Add pixel coordinates for quality estimators
            if 'rgb_boxes_pixels' in batch:
                gt_boxes['rgb_pixels'] = batch['rgb_boxes_pixels']
            if 'sonar_boxes_pixels' in batch:
                gt_boxes['sonar_pixels'] = batch['sonar_boxes_pixels']

            # Priority 0.2: Add fusion GT fallback (use RGB GT as fusion target)
            if 'fusion' not in gt_boxes and 'rgb' in gt_boxes:
                gt_boxes['fusion'] = gt_boxes['rgb']

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        batch['rgb_template'],
                        batch['rgb_search'],
                        batch['sonar_template'],
                        batch['sonar_search'],
                        gt_boxes=gt_boxes,
                        return_losses=True
                    )
                    # Priority 0.1: Use model-provided losses if available
                    if 'losses' in outputs and len(outputs['losses']) > 0:
                        total_loss, loss_dict = self._compute_model_losses(outputs['losses'], epoch)
                    else:
                        # Fallback to LossManager
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
                    batch['sonar_search'],
                    gt_boxes=gt_boxes,
                    return_losses=True
                )
                # Priority 0.1: Use model-provided losses if available
                if 'losses' in outputs and len(outputs['losses']) > 0:
                    total_loss, loss_dict = self._compute_model_losses(outputs['losses'], epoch)
                else:
                    # Fallback to LossManager
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

        # Reset temporal state at the start of validation epoch
        # Prevents cross-sequence state leakage in adaptive weighting and temporal modules
        if hasattr(self.model, 'reset_temporal_state'):
            self.model.reset_temporal_state()

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

                # Prepare ground truth boxes (for validation loss computation)
                gt_boxes = {}
                if 'rgb_boxes' in batch:
                    gt_boxes['rgb'] = batch['rgb_boxes']
                if 'sonar_boxes' in batch:
                    gt_boxes['sonar'] = batch['sonar_boxes']

                # CRITICAL FIX (Audit 2025111408 Finding 3):
                # Add pixel coordinates for quality estimators
                if 'rgb_boxes_pixels' in batch:
                    gt_boxes['rgb_pixels'] = batch['rgb_boxes_pixels']
                if 'sonar_boxes_pixels' in batch:
                    gt_boxes['sonar_pixels'] = batch['sonar_boxes_pixels']

                # Add fusion GT fallback
                if 'fusion' not in gt_boxes and 'rgb' in gt_boxes:
                    gt_boxes['fusion'] = gt_boxes['rgb']

                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            batch['rgb_template'],
                            batch['rgb_search'],
                            batch['sonar_template'],
                            batch['sonar_search'],
                            gt_boxes=gt_boxes if gt_boxes else None,
                            return_losses=True if gt_boxes else False
                        )
                        # Use model-provided losses if available
                        if 'losses' in outputs and len(outputs['losses']) > 0:
                            total_loss, loss_dict = self._compute_model_losses(outputs['losses'], epoch)
                        else:
                            # Fallback to LossManager
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
                        batch['sonar_search'],
                        gt_boxes=gt_boxes if gt_boxes else None,
                        return_losses=True if gt_boxes else False
                    )
                    # Use model-provided losses if available
                    if 'losses' in outputs and len(outputs['losses']) > 0:
                        total_loss, loss_dict = self._compute_model_losses(outputs['losses'], epoch)
                    else:
                        # Fallback to LossManager
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

    def save_checkpoint(
        self,
        epoch: int = -1,
        is_best: bool = False,
        filename: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> Optional[Path]:
        """
        Save a checkpoint (typically called during exception handling).

        Args:
            epoch: Current epoch (-1 for emergency save)
            is_best: Whether this is the best checkpoint
            filename: Custom filename (optional)
            metrics: Optional metrics to save

        Returns:
            Path to saved checkpoint or None

        Example:
            >>> # Emergency save on exception
            >>> trainer.save_checkpoint(epoch=-1, is_best=False)
        """
        if self.checkpoint_manager is None:
            logger.warning("No checkpoint manager available, cannot save checkpoint")
            return None

        if not self.is_main_process:
            return None

        # Use empty metrics if not provided
        if metrics is None:
            metrics = {}

        # Generate default filename if not provided
        if filename is None:
            if epoch == -1:
                filename = 'checkpoint_emergency.pth'
            elif is_best:
                filename = 'checkpoint_best.pth'
            else:
                filename = f'checkpoint_epoch_{epoch + 1}.pth'

        logger.info(f"Saving checkpoint: {filename}")

        return self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best,
            filename=filename
        )

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict: bool = True
    ) -> int:
        """
        Load a checkpoint and return the starting epoch.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            strict: Strict mode for model loading

        Returns:
            Starting epoch for resuming training

        Example:
            >>> start_epoch = trainer.load_checkpoint('outputs/checkpoints/checkpoint_epoch_10.pth')
            >>> trainer.train(start_epoch=start_epoch)
        """
        if self.checkpoint_manager is None:
            raise RuntimeError("No checkpoint manager available, cannot load checkpoint")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = self.checkpoint_manager.load(
            filename=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer if load_optimizer else None,
            scheduler=self.scheduler if load_scheduler else None,
            strict=strict
        )

        # Update trainer state
        epoch = checkpoint.get('epoch', 0)
        self.current_epoch = epoch

        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']

        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']

        logger.info(f"Resumed from epoch {epoch + 1}")

        # Return next epoch to start from
        return epoch + 1

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
            save_best_only=False,  # Handle save_best/save_last in Trainer instead
            metric_name=config.get('best_metric_name', 'val_total'),
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
        checkpoint_save_interval=config.get('checkpoint_save_interval', 1),
        save_best=config.get('save_best', True),
        save_last=config.get('save_last', True),
        tensorboard_dir=config.get('tensorboard_dir'),
        distributed=config.get('distributed', False),
        rank=config.get('rank', 0)
    )

    return trainer
