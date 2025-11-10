#!/usr/bin/env python3
"""
Training Script for AquaSOT-3D

Main training script that integrates all components:
- Data loading
- Model building  
- Training loop with Phase 8 Trainer
- Checkpointing and logging

Usage:
    python scripts/train.py --config configs/train/train_uuv2.yaml
    python scripts/train.py --config configs/train/train_uuv2.yaml \\
        --opts training.lr=0.001 training.batch_size=32

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import logging

from aquasot3d.utils.config import load_config, save_config, setup_config_logging
from aquasot3d.dataloader import RGBSonarDataset, create_dataloader
from aquasot3d.models.tracker import AquaSOT3DTracker
from aquasot3d.training import Trainer, LossManager, build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='AquaSOT-3D Training Script')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--opts', nargs='+', default=[],
                        help='Override config options (e.g., training.lr=0.001)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    
    return parser.parse_args()


def setup_distributed(cfg):
    """Setup distributed training"""
    if cfg.system.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            gpu = int(os.environ['LOCAL_RANK'])
        else:
            rank = 0
            world_size = 1
            gpu = 0
        
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend=cfg.system.dist_backend,
            init_method=cfg.system.dist_url,
            world_size=world_size,
            rank=rank
        )
        
        return rank, world_size, gpu
    else:
        return 0, 1, 0


def build_model(cfg):
    """Build tracking model"""
    logger.info(f"Building model: {cfg.model.type}")
    
    # Build model based on config
    model = AquaSOT3DTracker(
        rgb_backbone_type=cfg.model.backbone.rgb.type,
        sonar_encoder_type=cfg.model.backbone.sonar.type,
        fusion_config=cfg.model.fusion,
        temporal_config=cfg.model.temporal,
        head_configs=cfg.model.heads
    )
    
    logger.info(f"Model built successfully")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model


def build_dataloaders(cfg):
    """Build training and validation dataloaders"""
    logger.info("Building dataloaders")
    
    # Training dataloader
    train_loader = create_dataloader(
        root=cfg.data.train_root,
        split=cfg.data.train_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle=True,
        augmentation=cfg.data.augmentation if cfg.data.augmentation.enabled else None
    )
    
    # Validation dataloader
    val_loader = create_dataloader(
        root=cfg.data.val_root,
        split=cfg.data.val_split,
        batch_size=cfg.training.val_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle=False,
        augmentation=None  # No augmentation for validation
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    setup_config_logging('INFO')
    
    # Load configuration
    logger.info("="*70)
    logger.info("AquaSOT-3D Training")
    logger.info("="*70)
    
    cfg = load_config(args.config, overrides=args.opts)
    
    # Override resume path if provided
    if args.resume is not None:
        cfg.training.resume = args.resume
    
    # Setup distributed training
    rank, world_size, gpu = setup_distributed(cfg)
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Experiment: {cfg.experiment.name}")
        logger.info(f"Output directory: {cfg.paths.work_dir}")
    
    # Create output directories
    if is_main_process:
        Path(cfg.paths.work_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        save_config(cfg, Path(cfg.paths.work_dir) / 'config.yaml')
    
    # Set random seed
    torch.manual_seed(cfg.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.system.seed)
    
    # Build model
    model = build_model(cfg)
    
    # Build dataloaders
    train_loader, val_loader = build_dataloaders(cfg)
    
    # Build loss manager
    loss_manager = LossManager(
        loss_weights=cfg.training.loss_weights,
        weight_schedule=cfg.training.loss_schedule.type if cfg.training.loss_schedule.enabled else None,
        schedule_config=cfg.training.loss_schedule if cfg.training.loss_schedule.enabled else None
    )
    
    # Build optimizer
    optimizer = build_optimizer(
        model,
        optimizer_type=cfg.training.optimizer.type,
        base_lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=cfg.training.optimizer.betas,
        use_layer_decay=cfg.training.optimizer.use_layer_decay,
        layer_decay_rate=cfg.training.optimizer.layer_decay_rate if cfg.training.optimizer.use_layer_decay else None
    )
    
    # Build scheduler
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=cfg.training.scheduler.type,
        num_epochs=cfg.training.num_epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        min_lr=cfg.training.scheduler.min_lr
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_manager=loss_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=cfg.system.device,
        use_amp=cfg.system.use_amp,
        grad_clip_norm=cfg.training.grad_clip.max_norm if cfg.training.grad_clip.enabled else None,
        checkpoint_dir=cfg.paths.checkpoint_dir,
        log_dir=cfg.paths.log_dir,
        log_interval=cfg.training.log_interval,
        val_interval=cfg.training.val_interval
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if cfg.training.resume:
        logger.info(f"Resuming from checkpoint: {cfg.training.resume}")
        start_epoch = trainer.load_checkpoint(
            cfg.training.resume,
            load_optimizer=cfg.training.load_optimizer,
            load_scheduler=cfg.training.load_scheduler
        )
    
    # Start training
    logger.info("="*70)
    logger.info("Starting training")
    logger.info("="*70)
    
    try:
        trainer.train(
            num_epochs=cfg.training.num_epochs,
            start_epoch=start_epoch
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if is_main_process:
            logger.info("Saving checkpoint before exit...")
            trainer.save_checkpoint(epoch=-1, is_best=False)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info("="*70)
    logger.info("Training completed successfully")
    logger.info("="*70)
    
    # Cleanup distributed
    if cfg.system.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
