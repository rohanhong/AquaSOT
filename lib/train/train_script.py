"""
Training script for AquaSOT.

Based on OSTrack's train_script.py.
This is the main training logic that is called by run_training.py.
"""

import os
import importlib

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss

# Training pipeline
from lib.train.trainers import LTRTrainer
from lib.train.actors import AquaSOTActor
from lib.train.base_functions import update_settings, build_dataloaders, get_optimizer_scheduler

# Model
from lib.models.aquasot import build_aquasot

# Loss functions
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


def run(settings):
    """Main training function.

    Args:
        settings: Training settings object
    """
    settings.description = 'Training script for AquaSOT dual-stream RGB-Sonar tracker'

    # Load configuration
    if not os.path.exists(settings.cfg_file):
        raise ValueError(f"Config file not found: {settings.cfg_file}")

    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)

    # Override wandb config from command-line arguments
    if getattr(settings, 'wandb_enabled', False):
        cfg.WANDB.ENABLED = True
    if getattr(settings, 'wandb_project', None):
        cfg.WANDB.PROJECT = settings.wandb_project
    if getattr(settings, 'wandb_name', None):
        cfg.WANDB.NAME = settings.wandb_name
    if getattr(settings, 'wandb_entity', None):
        cfg.WANDB.ENTITY = settings.wandb_entity

    if settings.local_rank in [-1, 0]:
        print("=" * 60)
        print("Configuration:")
        print("=" * 60)
        for key in cfg.keys():
            print(f"{key}:")
            print(f"  {cfg[key]}")
        print("=" * 60)

    # Update settings from config
    update_settings(settings, cfg)

    # Setup logging
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(
        log_dir,
        f"{settings.script_name}-{settings.config_name}.log"
    )
    settings.log_dir = log_dir
    settings.checkpoint_dir = os.path.join(settings.save_dir, 'checkpoints')

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Build model
    net = build_aquasot(cfg, training=True)

    # Move to GPU
    net.cuda()

    # Wrap for distributed training
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device(f"cuda:{settings.local_rank}")
    else:
        settings.device = torch.device("cuda:0")

    # Loss functions
    focal_loss = FocalLoss()
    objective = {
        'giou': giou_loss,
        'l1': l1_loss,
        'focal': focal_loss,
    }

    # Loss weights from config
    loss_weight = {
        'giou': cfg.TRAIN.GIOU_WEIGHT,
        'l1': cfg.TRAIN.L1_WEIGHT,
        'focal': cfg.TRAIN.FOCAL_WEIGHT,
    }

    # Create actor
    actor = AquaSOTActor(
        net=net,
        objective=objective,
        loss_weight=loss_weight,
        settings=settings,
        cfg=cfg
    )

    # Create optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # Create trainer
    use_amp = getattr(cfg.TRAIN, 'AMP', False)
    trainer = LTRTrainer(
        actor,
        [loader_train, loader_val],
        optimizer,
        settings,
        lr_scheduler,
        use_amp=use_amp,
        cfg=cfg
    )

    # Start training
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
