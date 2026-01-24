"""
LTR Trainer from OSTrack.

Simplified for AquaSOT with wandb integration.
"""

import os
import time
import datetime
from collections import OrderedDict

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from .base_trainer import BaseTrainer
from lib.train.admin import AverageMeter, StatValue, TensorboardWriter


class LTRTrainer(BaseTrainer):
    """LTR-style trainer for AquaSOT."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None,
                 use_amp=False, cfg=None):
        """
        Args:
            actor: Actor for training
            loaders: List of data loaders [train_loader, val_loader]
            optimizer: Optimizer
            settings: Training settings
            lr_scheduler: Learning rate scheduler
            use_amp: Whether to use automatic mixed precision
            cfg: Configuration object (for wandb)
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()
        self.cfg = cfg

        # Initialize statistics
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_dir = os.path.join(settings.log_dir, 'tensorboard')
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = TensorboardWriter(
                tensorboard_dir,
                [loader.name for loader in loaders]
            )
        else:
            self.tensorboard_writer = None

        # Initialize wandb
        self.wandb_run = None
        if settings.local_rank in [-1, 0] and cfg is not None:
            self._init_wandb(cfg)

        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # Global step counter for wandb batch-level logging
        self.global_step = 0

    def _init_wandb(self, cfg):
        """Initialize wandb logging."""
        if not cfg.WANDB.ENABLED:
            return

        try:
            import wandb

            run_name = cfg.WANDB.NAME if cfg.WANDB.NAME else cfg.EXPERIMENT.NAME
            tags = list(cfg.EXPERIMENT.TAGS) + list(cfg.WANDB.TAGS)

            # Convert config to dict for wandb
            config_dict = {}
            for key in cfg.keys():
                if key != 'WANDB':
                    config_dict[key] = dict(cfg[key]) if hasattr(cfg[key], 'keys') else cfg[key]

            # Set wandb directory to outputs/wandb/
            wandb_dir = os.path.join(self.settings.save_dir, 'wandb')
            os.makedirs(wandb_dir, exist_ok=True)

            self.wandb_run = wandb.init(
                project=cfg.WANDB.PROJECT,
                entity=cfg.WANDB.ENTITY if cfg.WANDB.ENTITY else None,
                name=run_name,
                group=cfg.EXPERIMENT.GROUP,
                tags=tags if tags else None,
                notes=cfg.WANDB.NOTES if cfg.WANDB.NOTES else None,
                config=config_dict,
                mode=cfg.WANDB.MODE,
                resume="allow",
                dir=wandb_dir,
            )
            print(f"[wandb] Initialized: {self.wandb_run.url}")
        except Exception as e:
            print(f"[wandb] Failed to initialize: {e}")
            self.wandb_run = None

    def _set_default_settings(self):
        """Set default settings."""
        defaults = {
            'print_interval': 10,
            'print_stats': None,
            'description': '',
            'checkpoint_interval': 5,
            'local_rank': -1,
        }
        for param, default_value in defaults.items():
            if not hasattr(self.settings, param):
                setattr(self.settings, param, default_value)

    def train_epoch(self):
        """Train one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        # Write logs before resetting stats (critical order)
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()
            self._write_wandb()

        # Reset stats for next epoch (must be after logging)
        self._stats_new_epoch()

    def cycle_dataset(self, loader):
        """Run one cycle of training or validation."""
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()

            # Move data to device
            data = self._move_to_device(data)

            self.data_to_gpu_time = time.time()

            # Add epoch info
            data['epoch'] = self.epoch

            # Forward pass and loss
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # Backward pass
            if loader.training:
                self.optimizer.zero_grad()

                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.net.parameters(),
                            self.settings.grad_clip_norm
                        )
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.net.parameters(),
                            self.settings.grad_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # Update statistics
            batch_size = data['rgb_template'].shape[1]  # [N, B, C, H, W]
            self._update_stats(stats, batch_size, loader)

            # Print statistics
            self._print_stats(i, loader, batch_size)

            # Log to wandb (batch-level)
            if loader.training and self.settings.local_rank in [-1, 0]:
                self.global_step += 1
                self._write_wandb_batch(stats, loader.name)

        # Print epoch time
        epoch_time = self.prev_time - self.start_time
        print(f"Epoch Time: {datetime.timedelta(seconds=epoch_time)}")

    def _move_to_device(self, data):
        """Move data dict to device."""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
            elif isinstance(value, list):
                data[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for v in value]
        return data

    def _init_timing(self):
        """Initialize timing variables."""
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats, batch_size, loader):
        """Update statistics."""
        if loader.name not in self.stats or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({
                name: AverageMeter() for name in new_stats.keys()
            })

        # Add learning rate
        if loader.training and self.lr_scheduler is not None:
            try:
                lr_list = self.lr_scheduler.get_last_lr()
            except:
                lr_list = [self.optimizer.param_groups[0]['lr']]

            for i, lr in enumerate(lr_list):
                var_name = f'LearningRate/group{i}'
                if var_name not in self.stats[loader.name]:
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name]:
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        """Print statistics."""
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == len(loader):
            print_str = f'[{loader.name}: {self.epoch}, {i} / {len(loader)}] '
            print_str += f'FPS: {average_fps:.1f} ({batch_fps:.1f})  ,  '

            for name, val in self.stats[loader.name].items():
                if self.settings.print_stats is None or name in self.settings.print_stats:
                    if hasattr(val, 'avg'):
                        print_str += f'{name}: {val.avg:.5f}  ,  '

            print(print_str[:-5])

            # Write to log file
            if hasattr(self.settings, 'log_file'):
                with open(self.settings.log_file, 'a') as f:
                    f.write(print_str[:-5] + '\n')

    def _stats_new_epoch(self):
        """Reset statistics for new epoch."""
        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        """Write to tensorboard."""
        if self.tensorboard_writer is not None:
            if self.epoch == 1:
                self.tensorboard_writer.write_info(
                    getattr(self.settings, 'script_name', 'aquasot'),
                    self.settings.description
                )
            self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    def _write_wandb(self):
        """Write epoch summary to wandb."""
        if self.wandb_run is None:
            return

        try:
            import wandb

            log_dict = {'epoch': self.epoch}

            for loader_name, loader_stats in self.stats.items():
                if loader_stats is None:
                    continue
                for stat_name, stat_value in loader_stats.items():
                    key = f"{loader_name}/{stat_name}"
                    if hasattr(stat_value, 'avg'):
                        log_dict[key] = stat_value.avg
                    elif hasattr(stat_value, 'val'):
                        log_dict[key] = stat_value.val

            wandb.log(log_dict, step=self.global_step)
        except Exception as e:
            print(f"[wandb] Log error: {e}")

    def _write_wandb_batch(self, stats, loader_name):
        """Write batch-level stats to wandb."""
        if self.wandb_run is None:
            return

        try:
            import wandb

            log_dict = {
                'epoch': self.epoch,
                'global_step': self.global_step,
            }

            for stat_name, stat_value in stats.items():
                key = f"{loader_name}/{stat_name}"
                log_dict[key] = stat_value

            wandb.log(log_dict, step=self.global_step)
        except Exception as e:
            pass  # Silent fail for batch logging
