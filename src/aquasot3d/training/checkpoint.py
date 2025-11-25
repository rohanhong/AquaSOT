"""
Checkpoint Management for AquaSOT-3D

This module provides utilities for saving and loading model checkpoints,
supporting resume training, best model tracking, and distributed training.

Features:
- Model and optimizer state persistence
- Automatic best model tracking
- Resume training from checkpoint
- Distributed training support
- Scheduler state saving
- Custom metadata storage

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
import logging
import os
import shutil
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with automatic best model tracking.

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep (None = keep all)
        save_best_only: If True, only save when metric improves
        metric_name: Name of metric for best model tracking
        mode: 'min' or 'max' for metric comparison
        prefix: Prefix for checkpoint filenames

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='outputs/checkpoints',
        ...     max_to_keep=5,
        ...     save_best_only=False,
        ...     metric_name='val_total',
        ...     mode='min'
        ... )
        >>> # Save checkpoint
        >>> manager.save(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     epoch=10,
        ...     metrics={'val_total': 1.25, 'total': 1.30}
        ... )
        >>> # Load checkpoint
        >>> checkpoint = manager.load('checkpoint_epoch_10.pth')
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_to_keep: Optional[int] = None,
        save_best_only: bool = False,
        metric_name: str = 'val_total',
        mode: str = 'min',
        prefix: str = 'checkpoint'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode
        self.prefix = prefix

        # Track best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1

        # Track saved checkpoints
        self.checkpoints = []

        logger.info(f"CheckpointManager initialized: dir={checkpoint_dir}, max_keep={max_to_keep}")

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        filename: Optional[str] = None,
        skip_best_check: bool = False
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            metrics: Dictionary of metrics
            metadata: Additional metadata to save
            is_best: Force save as best checkpoint
            filename: Custom filename (optional)
            skip_best_check: If True, skip internal best metric check/update.
                This prevents race conditions when caller manages best_metric externally.

        Returns:
            Path to saved checkpoint

        Example:
            >>> manager.save(
            ...     model=model,
            ...     optimizer=optimizer,
            ...     epoch=10,
            ...     metrics={'fusion_auc': 0.75, 'precision': 0.82}
            ... )
        """
        metrics = metrics or {}
        metadata = metadata or {}

        # Check if this is the best model (if not explicitly specified)
        # Skip this check when caller manages best_metric externally (skip_best_check=True)
        if not is_best and not skip_best_check:
            current_metric = metrics.get(self.metric_name, None)
            if current_metric is not None:
                if self._is_better(current_metric, self.best_metric):
                    is_best = True
                    self.best_metric = current_metric
                    self.best_epoch = epoch

        # Determine filename
        if filename is None:
            filename = f'{self.prefix}_epoch_{epoch}.pth'

        # Special handling for last/best checkpoints
        is_special_checkpoint = 'last' in filename or 'best' in filename

        # Skip saving if save_best_only and not best (except for special checkpoints)
        if self.save_best_only and not is_best and not is_special_checkpoint:
            return None

        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self._get_state_dict(model),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'metadata': metadata
        }

        # Add optimizer state
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Only track interval checkpoints for cleanup (not last/best)
        if not is_special_checkpoint:
            self.checkpoints.append(checkpoint_path)
            # Clean up old interval checkpoints only
            self._cleanup_checkpoints()

        return checkpoint_path

    def load(
        self,
        filename: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            filename: Checkpoint filename
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            strict: Strict mode for model loading
            map_location: Device to map tensors to

        Returns:
            Checkpoint dictionary

        Example:
            >>> # Load checkpoint (after training completes)
            >>> checkpoint = manager.load('checkpoint_best.pth', model=model, optimizer=optimizer)
            >>> start_epoch = checkpoint['epoch'] + 1
        """
        checkpoint_path = self.checkpoint_dir / filename if not Path(filename).is_absolute() else Path(filename)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

        # Load model state
        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.info(f"Loaded model state from epoch {checkpoint.get('epoch', 'unknown')}")

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state")

        # Update best metric tracking
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']

        return checkpoint

    def load_best(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            **kwargs: Additional arguments for load()

        Returns:
            Best checkpoint dictionary
        """
        return self.load(
            f'{self.prefix}_best.pth',
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs
        )

    def resume(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        resume_from: Optional[Union[str, Path]] = None
    ) -> int:
        """
        Resume training from a checkpoint.

        Args:
            model: Model to resume
            optimizer: Optimizer to resume
            scheduler: Scheduler to resume
            resume_from: Checkpoint path (if None, auto-detect latest)

        Returns:
            Epoch to resume from

        Example:
            >>> start_epoch = manager.resume(model, optimizer, scheduler)
            >>> for epoch in range(start_epoch, max_epochs):
            ...     train(...)
        """
        if resume_from is None:
            resume_from = self._find_latest_checkpoint()
            if resume_from is None:
                logger.warning("No checkpoint found to resume from, starting fresh")
                return 0

        checkpoint = self.load(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )

        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resuming training from epoch {start_epoch}")

        return start_epoch

    def _get_state_dict(self, model: nn.Module) -> OrderedDict:
        """
        Get model state dict, handling DataParallel/DistributedDataParallel.

        Args:
            model: Model to extract state dict from

        Returns:
            State dictionary
        """
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module.state_dict()
        return model.state_dict()

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best
        else:  # max
            return current > best

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in the directory."""
        checkpoints = list(self.checkpoint_dir.glob(f'{self.prefix}_epoch_*.pth'))

        if not checkpoints:
            return None

        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest

    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to maintain max_to_keep."""
        if self.max_to_keep is None or self.max_to_keep <= 0:
            return

        # Get all non-best checkpoints
        checkpoints = [
            p for p in self.checkpoint_dir.glob(f'{self.prefix}_epoch_*.pth')
        ]

        if len(checkpoints) <= self.max_to_keep:
            return

        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest checkpoints
        to_remove = checkpoints[:len(checkpoints) - self.max_to_keep]
        for checkpoint_path in to_remove:
            checkpoint_path.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint_path}")
            if checkpoint_path in self.checkpoints:
                self.checkpoints.remove(checkpoint_path)

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / f'{self.prefix}_best.pth'
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        return sorted(
            self.checkpoint_dir.glob(f'{self.prefix}_*.pth'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    **kwargs
):
    """
    Save a simple checkpoint (convenience function).

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        **kwargs: Additional data to save

    Example:
        >>> save_checkpoint(
        ...     'outputs/model.pth',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'auc': 0.75}
        ... )
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict()
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
        else model.state_dict(),
        **kwargs
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a simple checkpoint (convenience function).

    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        map_location: Device to map to
        strict: Strict mode for loading

    Returns:
        Checkpoint dictionary

    Example:
        >>> checkpoint = load_checkpoint(
        ...     'outputs/model.pth',
        ...     model=model,
        ...     optimizer=optimizer
        ... )
    """
    checkpoint = torch.load(filepath, map_location=map_location)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logger.info(f"Loaded checkpoint from {filepath}")
    return checkpoint
