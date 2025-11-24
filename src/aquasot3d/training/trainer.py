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
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_manager = loss_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_metadata = checkpoint_metadata or {}

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

        # CRITICAL FIX (Audit 2025112404 Finding 2): Correct model hierarchy detection
        # Model structure is model.backbone.{rgb_backbone,sonar_encoder}, not model.{rgb_backbone,sonar_backbone}
        # Used for conditional validity statistics reporting
        self.enabled_modalities = []

        # Check if model has backbone attribute (standard AquaSOT3DTracker architecture)
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            if hasattr(backbone, 'rgb_backbone') and backbone.rgb_backbone is not None:
                self.enabled_modalities.append('rgb')
            if hasattr(backbone, 'sonar_encoder') and backbone.sonar_encoder is not None:
                self.enabled_modalities.append('sonar')
        else:
            # Fallback for non-standard architectures or direct backbone access
            logger.warning(
                "Model has no 'backbone' attribute - modality detection may fail. "
                "Please verify model architecture."
            )
            if hasattr(model, 'rgb_backbone') and model.rgb_backbone is not None:
                self.enabled_modalities.append('rgb')
            if hasattr(model, 'sonar_encoder') and model.sonar_encoder is not None:
                self.enabled_modalities.append('sonar')

        logger.info(f"Trainer initialized: epochs={max_epochs}, amp={use_amp}, distributed={distributed}")
        logger.info(f"Enabled modalities: {self.enabled_modalities}")

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

            # CRITICAL FIX (Audit 2025111414 Finding 4): Respect eval_interval configuration
            # Validate based on eval_interval, but always validate on final epoch
            val_metrics = None
            should_validate = (
                self.val_loader is not None and
                ((epoch + 1) % self.eval_interval == 0 or epoch == self.max_epochs - 1)
            )
            if should_validate:
                val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Three-tier checkpoint saving
            if self.checkpoint_manager is not None and self.is_main_process:
                metrics = {**train_metrics}
                if val_metrics is not None:
                    metrics.update(val_metrics)

                # 1. Save "last" checkpoint every epoch (overwrite) - only if save_last is enabled
                # CRITICAL FIX (Audit 2025112408): Prevent race condition with best checkpoint saving
                # Pass is_best=False to prevent CheckpointManager from updating best_metric internally,
                # which would make the subsequent save_best check always fail
                if self.save_last:
                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=metrics,
                        filename='checkpoint_last.pth',
                        metadata=self.checkpoint_metadata,
                        is_best=False  # Prevent internal best_metric update
                    )

                # 2. Save "best" checkpoint if metric improves (overwrite) - only if save_best is enabled
                # CRITICAL FIX (Audit 2025111414 Finding 4 Regression): Only attempt best checkpoint
                # saving when validation metrics are available (to avoid "metric not found" errors)
                if self.save_best and val_metrics is not None:
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
                                filename='checkpoint_best.pth',
                                metadata=self.checkpoint_metadata
                            )
                            logger.info(f"Saved best checkpoint at epoch {epoch + 1} with {self.checkpoint_manager.metric_name}={current_metric:.4f}")

                # 3. Save interval checkpoint every N epochs (keep multiple)
                # CRITICAL FIX (Audit 2025112408): Consistency with save_last fix
                # Pass is_best=False to prevent unintended best_metric updates
                if (epoch + 1) % self.checkpoint_save_interval == 0:
                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=metrics,
                        filename=f'checkpoint_epoch_{epoch + 1}.pth',
                        metadata=self.checkpoint_metadata,
                        is_best=False  # Maintain consistency with save_last logic
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

        # CRITICAL (Audit 2025111903 Phase 5): Track validity statistics
        validity_stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'valid_rgb': 0,
            'valid_sonar': 0,
            'invalid_samples': 0
        }

        # CRITICAL FIX (Audit 2025112503 Remaining Issue 2): Get total batches for tail handling
        total_batches = len(self.train_loader)

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.max_epochs}',
            disable=not self.is_main_process
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)

            # ARCHITECTURE CHANGE (Template-Search Migration 2025112104):
            # CRITICAL FIX (Audit 2025112105): Template-search format only
            # Removed backward compatibility for frame sequence format (legacy temporal architecture)

            # Extract template-search pairs
            if 'rgb_template' in batch and batch['rgb_template'] is not None:
                # Template-search format (RGB available)
                batch_size = batch['rgb_template'].shape[0]
                rgb_template = batch['rgb_template']  # [B, 3, 256, 256]
                rgb_search = batch['rgb_search']  # [B, 3, 512, 512]
                sonar_template = batch.get('sonar_template', None)  # [B, 1, 256, 256]
                sonar_search = batch.get('sonar_search', None)  # [B, 1, 512, 512]
            elif 'sonar_template' in batch and batch['sonar_template'] is not None:
                # Template-search format (Sonar only)
                batch_size = batch['sonar_template'].shape[0]
                rgb_template = None
                rgb_search = None
                sonar_template = batch['sonar_template']  # [B, 1, 256, 256]
                sonar_search = batch['sonar_search']  # [B, 1, 512, 512]
            else:
                raise RuntimeError(
                    "Batch must contain template-search format with keys: "
                    "'rgb_template/rgb_search' or 'sonar_template/sonar_search'. "
                    "Got keys: " + str(list(batch.keys()))
                )

            # CRITICAL (Audit 2025111903 Phase 5): Track validity statistics for this batch
            validity_stats['total_samples'] += batch_size

            # Validity flags handling: support both template-search [B] and frame sequence [B, T] formats
            if 'is_valid' in batch:
                # Legacy format: single validity flag [B]
                valid_count = batch['is_valid'].sum().item()
                validity_stats['valid_samples'] += valid_count
                validity_stats['invalid_samples'] += (batch_size - valid_count)

            if 'is_valid_rgb' in batch:
                # Adaptive handling based on tensor shape
                is_valid_rgb = batch['is_valid_rgb']
                if is_valid_rgb.dim() == 2:
                    # Frame sequence format: [B, T], extract last frame (search frame)
                    valid_rgb = is_valid_rgb[:, -1]  # Use last frame as target
                else:
                    # Template-search format: [B], already target frame
                    valid_rgb = is_valid_rgb
                validity_stats['valid_rgb'] += valid_rgb.sum().item()

            if 'is_valid_sonar' in batch:
                # Adaptive handling based on tensor shape
                is_valid_sonar = batch['is_valid_sonar']
                if is_valid_sonar.dim() == 2:
                    # Frame sequence format: [B, T], extract last frame (search frame)
                    valid_sonar = is_valid_sonar[:, -1]  # Use last frame as target
                else:
                    # Template-search format: [B], already target frame
                    valid_sonar = is_valid_sonar
                validity_stats['valid_sonar'] += valid_sonar.sum().item()

            # Prepare ground truth boxes
            # ARCHITECTURE CHANGE (Template-Search Migration 2025112104):
            # Support both template-search and frame sequence GT box formats
            gt_boxes = {}

            # CRITICAL FIX (Audit 2025112306): Template-search architecture only
            # Remove frame sequence fallback with undefined target_idx
            # CRITICAL FIX (Audit 2025112312 - Single-Modality Visibility):
            # Use separate RGB and Sonar GT boxes to support single-modality scenarios
            if 'search_bbox_rgb' in batch and 'search_bbox_sonar' in batch:
                # Separate GT boxes (post-2025112312 format)
                gt_boxes['rgb'] = batch['search_bbox_rgb']
                gt_boxes['sonar'] = batch['search_bbox_sonar']
            elif 'search_bbox' in batch:
                # Legacy unified GT (backward compatibility)
                gt_boxes['rgb'] = batch['search_bbox']
                gt_boxes['sonar'] = batch['search_bbox']
            else:
                raise RuntimeError(
                    "Template-search architecture requires 'search_bbox_rgb' and 'search_bbox_sonar' in batch. "
                    f"Got keys: {list(batch.keys())}"
                )

            # CRITICAL FIX (Audit 2025111408 Finding 3):
            # Add pixel coordinates for quality estimators
            # CRITICAL FIX (Audit 2025112502 Finding 6): Support both old and new key names
            if 'rgb_boxes_pixels' in batch:
                gt_boxes['rgb_pixels'] = batch['rgb_boxes_pixels']
            elif 'search_bbox_rgb_pixels' in batch:
                gt_boxes['rgb_pixels'] = batch['search_bbox_rgb_pixels']

            if 'sonar_boxes_pixels' in batch:
                gt_boxes['sonar_pixels'] = batch['sonar_boxes_pixels']
            elif 'search_bbox_sonar_pixels' in batch:
                gt_boxes['sonar_pixels'] = batch['search_bbox_sonar_pixels']

            # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
            # Fusion GT generation REMOVED. Fusion box head no longer exists.
            # RGB and Sonar BoxHeads now use dual-view fused features with separate GT.

            # ARCHITECTURE CHANGE (Template-Search Migration 2025112104):
            # Metadata format differs between template-search and frame sequence
            # Template-search: metadata is already List[Dict] (B dicts)
            # Frame sequence: metadata is List[List[Dict]] (B lists of T dicts)
            metadata = {}

            if 'rgb_metadata' in batch:
                # Template-search format: already List[Dict]
                # Frame sequence format: List[List[Dict]]
                rgb_meta = batch['rgb_metadata']
                if isinstance(rgb_meta, list) and len(rgb_meta) > 0:
                    if isinstance(rgb_meta[0], dict):
                        # Template-search format: List[Dict]
                        metadata['rgb_search_metadata'] = rgb_meta
                    elif isinstance(rgb_meta[0], list):
                        # Frame sequence format: List[List[Dict]]
                        metadata['rgb_search_metadata'] = rgb_meta
                    else:
                        metadata['rgb_search_metadata'] = rgb_meta
                else:
                    metadata['rgb_search_metadata'] = rgb_meta

            if 'sonar_metadata' in batch:
                sonar_meta = batch['sonar_metadata']
                if isinstance(sonar_meta, list) and len(sonar_meta) > 0:
                    if isinstance(sonar_meta[0], dict):
                        # Template-search format: List[Dict]
                        metadata['sonar_search_metadata'] = sonar_meta
                    elif isinstance(sonar_meta[0], list):
                        # Frame sequence format: List[List[Dict]]
                        metadata['sonar_search_metadata'] = sonar_meta
                    else:
                        metadata['sonar_search_metadata'] = sonar_meta
                else:
                    metadata['sonar_search_metadata'] = sonar_meta

            # Legacy support
            if 'template_rgb_metadata' in batch:
                metadata['template_rgb_metadata'] = batch['template_rgb_metadata']
            if 'template_sonar_metadata' in batch:
                metadata['template_sonar_metadata'] = batch['template_sonar_metadata']

            # CRITICAL VALIDATION (Audit 2025111412 Finding 4):
            # Validate metadata list lengths before model forward
            for key, meta_list in metadata.items():
                if meta_list is not None and isinstance(meta_list, list):
                    if len(meta_list) != batch_size:
                        raise RuntimeError(
                            f"Trainer metadata validation failed: {key} has {len(meta_list)} "
                            f"entries but batch_size={batch_size}. Loader bug detected!"
                        )

            # CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX):
            # Extract has_object flags for confidence head training
            has_object_rgb = batch.get('has_object_rgb', None)
            has_object_sonar = batch.get('has_object_sonar', None)
            has_object = batch.get('has_object', None)

            # CRITICAL FIX (Audit 2025112401 Finding 2):
            # Extract validity masks to prevent occluded frames from contributing to loss
            is_valid_rgb = batch.get('is_valid_rgb', None)
            is_valid_sonar = batch.get('is_valid_sonar', None)

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        rgb_template,
                        rgb_search,
                        sonar_template,
                        sonar_search,
                        gt_boxes=gt_boxes,
                        return_losses=True,
                        metadata=metadata,
                        is_rgb_valid=is_valid_rgb,
                        is_sonar_valid=is_valid_sonar,
                        has_object_rgb=has_object_rgb,
                        has_object_sonar=has_object_sonar,
                        has_object=has_object
                    )
                    # Use LossManager for all losses (handles model-internal losses with scheduling)
                    total_loss, loss_dict = self.loss_manager.compute_losses(
                        outputs,
                        batch,
                        epoch=epoch
                    )
            else:
                outputs = self.model(
                    rgb_template,
                    rgb_search,
                    sonar_template,
                    sonar_search,
                    gt_boxes=gt_boxes,
                    return_losses=True,
                    metadata=metadata,
                    is_rgb_valid=is_valid_rgb,
                    is_sonar_valid=is_valid_sonar,
                    has_object_rgb=has_object_rgb,
                    has_object_sonar=has_object_sonar,
                    has_object=has_object
                )
                # Use LossManager for all losses (handles model-internal losses with scheduling)
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
            # CRITICAL FIX (Audit 2025112503 Remaining Issue 2): Step on last batch too
            # This ensures tail batches are not dropped, and avoids under-scaling
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == total_batches - 1:
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
                # CRITICAL (Audit 2025111903 Phase 5): Add validity rate to progress bar
                postfix = {
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                }
                if validity_stats['total_samples'] > 0:
                    validity_rate = validity_stats['valid_samples'] / validity_stats['total_samples'] * 100
                    postfix['valid%'] = f"{validity_rate:.1f}"
                pbar.set_postfix(postfix)

        # Average metrics
        averaged_metrics = {
            key: value / epoch_counts[key]
            for key, value in epoch_metrics.items()
        }

        # CRITICAL (Audit 2025111903 Phase 5 & Audit 2025112003): Add validity statistics to metrics
        # Only report validity rates for enabled modalities
        if validity_stats['total_samples'] > 0:
            averaged_metrics['validity_rate'] = validity_stats['valid_samples'] / validity_stats['total_samples']

            # CRITICAL FIX (Audit 2025112003): Conditional validity reporting
            # Only report RGB validity if RGB modality is enabled
            if 'rgb' in self.enabled_modalities and 'valid_rgb' in validity_stats:
                averaged_metrics['validity_rate_rgb'] = validity_stats['valid_rgb'] / validity_stats['total_samples']

            # Only report Sonar validity if Sonar modality is enabled
            if 'sonar' in self.enabled_modalities and 'valid_sonar' in validity_stats:
                averaged_metrics['validity_rate_sonar'] = validity_stats['valid_sonar'] / validity_stats['total_samples']

            averaged_metrics['invalid_samples'] = validity_stats['invalid_samples']

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

        # ARCHITECTURE CHANGE (Audit 2025112104): Temporal state reset removed
        # Template-search paradigm is stateless, no cross-sequence state leakage possible

        val_metrics = defaultdict(float)
        val_counts = defaultdict(int)

        # CRITICAL (Audit 2025111903 Phase 5): Track validity statistics in validation
        validity_stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'valid_rgb': 0,
            'valid_sonar': 0,
            'invalid_samples': 0
        }

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc='Validation',
                disable=not self.is_main_process
            ):
                # Move batch to device
                batch = self._to_device(batch)

                # ARCHITECTURE CHANGE (Template-Search Migration 2025112104):
                # CRITICAL FIX (Audit 2025112105): Template-search format only

                # Extract template-search pairs
                if 'rgb_template' in batch and batch['rgb_template'] is not None:
                    # Template-search format (RGB available)
                    batch_size = batch['rgb_template'].shape[0]
                    rgb_template = batch['rgb_template']  # [B, 3, 256, 256]
                    rgb_search = batch['rgb_search']  # [B, 3, 512, 512]
                    sonar_template = batch.get('sonar_template', None)  # [B, 1, 256, 256]
                    sonar_search = batch.get('sonar_search', None)  # [B, 1, 512, 512]
                elif 'sonar_template' in batch and batch['sonar_template'] is not None:
                    # Template-search format (Sonar only)
                    batch_size = batch['sonar_template'].shape[0]
                    rgb_template = None
                    rgb_search = None
                    sonar_template = batch['sonar_template']  # [B, 1, 256, 256]
                    sonar_search = batch['sonar_search']  # [B, 1, 512, 512]
                else:
                    raise RuntimeError(
                        "Batch must contain template-search format with keys: "
                        "'rgb_template/rgb_search' or 'sonar_template/sonar_search'. "
                        "Got keys: " + str(list(batch.keys()))
                    )

                # CRITICAL (Audit 2025111903 Phase 5): Track validity statistics for this batch
                validity_stats['total_samples'] += batch_size

                # Validity flags are per-frame [B, T], extract target frame [B]
                if 'is_valid' in batch:
                    valid_count = batch['is_valid'].sum().item()
                    validity_stats['valid_samples'] += valid_count
                    validity_stats['invalid_samples'] += (batch_size - valid_count)

                if 'is_valid_rgb' in batch:
                    # CRITICAL FIX (2025-11-21): Adaptive handling for template-search migration
                    # Support both [B, T] frame sequence and [B] template-search formats
                    is_valid_rgb = batch['is_valid_rgb']
                    if is_valid_rgb.dim() == 2:
                        # Frame sequence format: [B, T], extract last frame (search frame)
                        valid_rgb = is_valid_rgb[:, -1]  # Use last frame as target
                    else:
                        # Template-search format: [B], already target frame
                        valid_rgb = is_valid_rgb
                    validity_stats['valid_rgb'] += valid_rgb.sum().item()

                if 'is_valid_sonar' in batch:
                    # CRITICAL FIX (2025-11-21): Adaptive handling for template-search migration
                    # Support both [B, T] frame sequence and [B] template-search formats
                    is_valid_sonar = batch['is_valid_sonar']
                    if is_valid_sonar.dim() == 2:
                        # Frame sequence format: [B, T], extract last frame (search frame)
                        valid_sonar = is_valid_sonar[:, -1]  # Use last frame as target
                    else:
                        # Template-search format: [B], already target frame
                        valid_sonar = is_valid_sonar
                    validity_stats['valid_sonar'] += valid_sonar.sum().item()

                # Prepare ground truth boxes (for validation loss computation)
                # ARCHITECTURE CHANGE (Template-Search Migration 2025112104):
                # CRITICAL FIX (2025-11-21): Removed legacy frame sequence format support
                # Validation now exclusively uses template-search format with search_bbox
                gt_boxes = {}

                # CRITICAL FIX (Audit 2025112312 - Single-Modality Visibility):
                # Use separate RGB and Sonar GT boxes
                if 'search_bbox_rgb' in batch and 'search_bbox_sonar' in batch:
                    # Separate GT boxes (post-2025112312 format)
                    gt_boxes['rgb'] = batch['search_bbox_rgb']
                    gt_boxes['sonar'] = batch['search_bbox_sonar']
                elif 'search_bbox' in batch:
                    # Legacy unified GT (backward compatibility)
                    gt_boxes['rgb'] = batch['search_bbox']
                    gt_boxes['sonar'] = batch['search_bbox']
                else:
                    raise RuntimeError(
                        "Template-search architecture requires 'search_bbox_rgb' and 'search_bbox_sonar' in batch. "
                        f"Got keys: {list(batch.keys())}"
                    )

                # CRITICAL FIX (Audit 2025111408 Finding 3):
                # Add pixel coordinates for quality estimators
                # CRITICAL FIX (Audit 2025112502 Finding 6): Support both old and new key names
                if 'rgb_boxes_pixels' in batch:
                    gt_boxes['rgb_pixels'] = batch['rgb_boxes_pixels']
                elif 'search_bbox_rgb_pixels' in batch:
                    gt_boxes['rgb_pixels'] = batch['search_bbox_rgb_pixels']

                if 'sonar_boxes_pixels' in batch:
                    gt_boxes['sonar_pixels'] = batch['sonar_boxes_pixels']
                elif 'search_bbox_sonar_pixels' in batch:
                    gt_boxes['sonar_pixels'] = batch['search_bbox_sonar_pixels']

                # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
                # Fusion GT generation REMOVED. Fusion box head no longer exists.

                # ARCHITECTURE CHANGE (Template-Search Migration 2025112104):
                # Metadata format differs between template-search and frame sequence
                metadata = {}

                if 'rgb_metadata' in batch:
                    rgb_meta = batch['rgb_metadata']
                    if isinstance(rgb_meta, list) and len(rgb_meta) > 0:
                        if isinstance(rgb_meta[0], dict):
                            metadata['rgb_search_metadata'] = rgb_meta
                        elif isinstance(rgb_meta[0], list):
                            metadata['rgb_search_metadata'] = rgb_meta
                        else:
                            metadata['rgb_search_metadata'] = rgb_meta
                    else:
                        metadata['rgb_search_metadata'] = rgb_meta

                if 'sonar_metadata' in batch:
                    sonar_meta = batch['sonar_metadata']
                    if isinstance(sonar_meta, list) and len(sonar_meta) > 0:
                        if isinstance(sonar_meta[0], dict):
                            metadata['sonar_search_metadata'] = sonar_meta
                        elif isinstance(sonar_meta[0], list):
                            metadata['sonar_search_metadata'] = sonar_meta
                        else:
                            metadata['sonar_search_metadata'] = sonar_meta
                    else:
                        metadata['sonar_search_metadata'] = sonar_meta

                # Legacy support
                if 'template_rgb_metadata' in batch:
                    metadata['template_rgb_metadata'] = batch['template_rgb_metadata']
                if 'template_sonar_metadata' in batch:
                    metadata['template_sonar_metadata'] = batch['template_sonar_metadata']

                # CONTINUOUS FLOW DESIGN (Audit 2025112310 P0 FIX):
                # Extract has_object flags for confidence head training
                has_object_rgb = batch.get('has_object_rgb', None)
                has_object_sonar = batch.get('has_object_sonar', None)
                has_object = batch.get('has_object', None)

                # CRITICAL FIX (Audit 2025112401 Finding 2):
                # Extract validity masks to prevent occluded frames from contributing to loss
                is_valid_rgb = batch.get('is_valid_rgb', None)
                is_valid_sonar = batch.get('is_valid_sonar', None)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            rgb_template,
                            rgb_search,
                            sonar_template,
                            sonar_search,
                            gt_boxes=gt_boxes if gt_boxes else None,
                            return_losses=True if gt_boxes else False,
                            metadata=metadata,
                            is_rgb_valid=is_valid_rgb,
                            is_sonar_valid=is_valid_sonar,
                            has_object_rgb=has_object_rgb,
                            has_object_sonar=has_object_sonar,
                            has_object=has_object
                        )
                        # Use LossManager for all losses (handles model-internal losses with scheduling)
                        total_loss, loss_dict = self.loss_manager.compute_losses(
                            outputs,
                            batch,
                            epoch=epoch
                        )
                else:
                    outputs = self.model(
                        rgb_template,
                        rgb_search,
                        sonar_template,
                        sonar_search,
                        gt_boxes=gt_boxes if gt_boxes else None,
                        return_losses=True if gt_boxes else False,
                        metadata=metadata,
                        is_rgb_valid=is_valid_rgb,
                        is_sonar_valid=is_valid_sonar,
                        has_object_rgb=has_object_rgb,
                        has_object_sonar=has_object_sonar,
                        has_object=has_object
                    )
                    # Use LossManager for all losses (handles model-internal losses with scheduling)
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

        # CRITICAL (Audit 2025111903 Phase 5 & Audit 2025112003): Add validity statistics to validation metrics
        # Only report validity rates for enabled modalities
        if validity_stats['total_samples'] > 0:
            averaged_metrics['val_validity_rate'] = validity_stats['valid_samples'] / validity_stats['total_samples']

            # CRITICAL FIX (Audit 2025112003): Conditional validity reporting
            # Only report RGB validity if RGB modality is enabled
            if 'rgb' in self.enabled_modalities and 'valid_rgb' in validity_stats:
                averaged_metrics['val_validity_rate_rgb'] = validity_stats['valid_rgb'] / validity_stats['total_samples']

            # Only report Sonar validity if Sonar modality is enabled
            if 'sonar' in self.enabled_modalities and 'valid_sonar' in validity_stats:
                averaged_metrics['val_validity_rate_sonar'] = validity_stats['valid_sonar'] / validity_stats['total_samples']

            averaged_metrics['val_invalid_samples'] = validity_stats['invalid_samples']

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
            filename=filename,
            metadata=self.checkpoint_metadata
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
