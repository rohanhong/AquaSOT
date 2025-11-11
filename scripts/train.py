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
from aquasot3d.dataloader import RGBSonarDataset, TrackingDataset, create_dataloader
from aquasot3d.dataloader.loader import collate_fn_template_search
from aquasot3d.models.tracker import AquaSOT3DTracker
from aquasot3d.training import Trainer, LossManager, CheckpointManager, build_optimizer, build_scheduler

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
    """Build tracking model with proper config preprocessing"""
    logger.info(f"Building model: {cfg.model.type}")

    # Convert ConfigDict to plain dict for model building
    from aquasot3d.utils.config import _config_to_dict

    # Prepare backbone configs
    rgb_backbone_cfg = _config_to_dict(cfg.model.backbone.rgb) if cfg.model.backbone.rgb.get('type') not in [None, 'null'] else None
    sonar_backbone_cfg = _config_to_dict(cfg.model.backbone.sonar) if cfg.model.backbone.sonar.get('type') not in [None, 'null'] else None

    # Prepare fusion config
    fusion_cfg = _config_to_dict(cfg.model.fusion) if cfg.model.fusion.get('type') not in [None, 'null'] else None

    # Prepare temporal config (remove control parameters)
    temporal_cfg = None
    if cfg.model.temporal.get('enabled', True) and cfg.model.temporal.get('type') not in [None, 'null']:
        temporal_cfg = _config_to_dict(cfg.model.temporal)

    # Prepare head configs (remove 'type' and 'in_channels' which will be provided by tracker)
    def clean_head_config(head_cfg):
        """Remove parameters that tracker will provide explicitly"""
        if not head_cfg or head_cfg.get('type') in [None, 'null']:
            return None
        cfg_dict = _config_to_dict(head_cfg)
        cfg_dict.pop('type', None)
        cfg_dict.pop('in_channels', None)
        cfg_dict.pop('rgb_channels', None)
        cfg_dict.pop('sonar_channels', None)
        return cfg_dict

    box_head_cfg = clean_head_config(cfg.model.heads.get('rgb_box', {}))
    fusion_head_cfg = clean_head_config(cfg.model.heads.get('fusion_box', {}))
    weak3d_cfg = clean_head_config(cfg.model.heads.get('weak3d', {}))

    # Determine use_weak3d
    use_weak3d = cfg.model.heads.get('weak3d', {}).get('type') not in [None, 'null']

    # Build model
    model = AquaSOT3DTracker(
        rgb_backbone_config=rgb_backbone_cfg,
        sonar_backbone_config=sonar_backbone_cfg,
        fusion_config=fusion_cfg,
        temporal_config=temporal_cfg,
        box_head_config=box_head_cfg,
        fusion_head_config=fusion_head_cfg,
        weak3d_config=weak3d_cfg,
        use_temporal=cfg.model.temporal.get('enabled', True),
        use_weak3d=use_weak3d
    )

    logger.info(f"Model built successfully")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


def build_dataloaders(cfg):
    """Build training and validation dataloaders"""
    logger.info("Building dataloaders")

    # Create base datasets (single frames)
    train_base_dataset = RGBSonarDataset(
        data_root='data',
        sequence_name='uuv2',  # Will look in data/uuv2/light and data/uuv2/sonar
        return_occluded=True
    )

    val_base_dataset = RGBSonarDataset(
        data_root='data',
        sequence_name='uuv3',  # Will look in data/uuv3/light and data/uuv3/sonar
        return_occluded=True
    )

    # Build augmentation pipeline for training
    train_augmentation = None
    if cfg.data.get('augmentation', {}).get('enabled', False):
        from aquasot3d.dataloader.transforms.rgb_transforms import ColorJitter, RandomBlur, RandomOcclusion
        from aquasot3d.dataloader.transforms.sonar_transforms import SpeckleNoise, RadialScalingJitter, SectorOcclusion

        logger.info("Building data augmentation pipeline")

        class SimpleAugmentation:
            """Lightweight augmentation wrapper (no normalization or to_tensor)"""
            def __init__(self, rgb_transforms, sonar_transforms):
                self.rgb_transforms = rgb_transforms
                self.sonar_transforms = sonar_transforms

            def __call__(self, data):
                """Apply augmentations to RGB and Sonar images"""
                rgb_img = data.get('rgb_img')
                sonar_img = data.get('sonar_img')
                rgb_bbox = data.get('rgb_bbox')
                sonar_bbox = data.get('sonar_bbox')

                # Apply RGB transforms
                if rgb_img is not None:
                    for transform in self.rgb_transforms:
                        rgb_img = transform(rgb_img)

                # Apply Sonar transforms
                if sonar_img is not None:
                    for transform in self.sonar_transforms:
                        sonar_img = transform(sonar_img)

                return {
                    'rgb_img': rgb_img,
                    'sonar_img': sonar_img,
                    'rgb_bbox': rgb_bbox,
                    'sonar_bbox': sonar_bbox,
                }

        # Build RGB transforms
        rgb_transforms = []
        rgb_cfg = cfg.data.augmentation.get('rgb', {})
        if rgb_cfg.get('enable_color_jitter', False):
            rgb_transforms.append(ColorJitter(
                brightness=rgb_cfg.get('brightness', 0.2),
                contrast=rgb_cfg.get('contrast', 0.2),
                saturation=rgb_cfg.get('saturation', 0.2),
                hue=rgb_cfg.get('hue', 0.1),
                prob=rgb_cfg.get('color_jitter_prob', 0.5),
            ))
        if rgb_cfg.get('enable_blur', False):
            rgb_transforms.append(RandomBlur(
                prob=rgb_cfg.get('blur_prob', 0.3),
            ))
        if rgb_cfg.get('enable_occlusion', False):
            rgb_transforms.append(RandomOcclusion(
                min_size=rgb_cfg.get('occlusion_min_size', 0.05),
                max_size=rgb_cfg.get('occlusion_max_size', 0.3),
                prob=rgb_cfg.get('occlusion_prob', 0.2),
            ))

        # Build Sonar transforms
        sonar_transforms = []
        sonar_cfg = cfg.data.augmentation.get('sonar', {})
        if sonar_cfg.get('enable_speckle', False):
            sonar_transforms.append(SpeckleNoise(
                noise_level=sonar_cfg.get('speckle_level', 0.1),
                prob=sonar_cfg.get('speckle_prob', 0.5),
            ))
        if sonar_cfg.get('enable_radial_jitter', False):
            sonar_transforms.append(RadialScalingJitter(
                scale_range=tuple(sonar_cfg.get('radial_scale_range', [0.9, 1.1])),
                prob=sonar_cfg.get('radial_jitter_prob', 0.3),
            ))
        if sonar_cfg.get('enable_sector_occlusion', False):
            sonar_transforms.append(SectorOcclusion(
                min_angle=sonar_cfg.get('sector_min_angle', 10.0),
                max_angle=sonar_cfg.get('sector_max_angle', 40.0),
                prob=sonar_cfg.get('sector_occlusion_prob', 0.2),
            ))

        train_augmentation = SimpleAugmentation(rgb_transforms, sonar_transforms)
        logger.info(f"Augmentation enabled: {len(rgb_transforms)} RGB transforms, {len(sonar_transforms)} Sonar transforms")
    else:
        logger.info("Augmentation disabled")

    # Wrap with TrackingDataset for template-search pair sampling
    template_search_gap = cfg.data.get('template_search_gap', 30)
    template_size = cfg.data.get('template_size', 127)
    search_size = cfg.data.get('search_size', 255)

    train_dataset = TrackingDataset(
        base_dataset=train_base_dataset,
        template_search_gap=template_search_gap,
        template_size=template_size,
        search_size=search_size,
        augmentation=train_augmentation
    )
    val_dataset = TrackingDataset(
        base_dataset=val_base_dataset,
        template_search_gap=template_search_gap,
        template_size=template_size,
        search_size=search_size,
        augmentation=None  # No augmentation for validation
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders with template-search collate function
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn_template_search,
        distributed=cfg.system.distributed
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn_template_search,
        distributed=False  # No distributed for validation
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
    # Prepare optimizer parameters (separate control params from actual params)
    optimizer_params = {
        'model': model,
        'optimizer_type': cfg.training.optimizer.type,
        'base_lr': cfg.training.optimizer.lr,
        'weight_decay': cfg.training.optimizer.weight_decay,
        'betas': cfg.training.optimizer.betas,
    }

    # Add layer_decay only if use_layer_decay is enabled
    if cfg.training.optimizer.get('use_layer_decay', False):
        optimizer_params['layer_decay'] = cfg.training.optimizer.get('layer_decay_rate', 0.75)

    optimizer = build_optimizer(**optimizer_params)
    
    # Build scheduler
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=cfg.training.scheduler.type,
        num_epochs=cfg.training.num_epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        min_lr=cfg.training.scheduler.min_lr
    )

    # Build checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.paths.checkpoint_dir,
        max_to_keep=cfg.training.checkpoint.get('max_keep', 5),
        save_best_only=cfg.training.checkpoint.get('save_best_only', False),
        metric_name=cfg.training.checkpoint.get('metric', 'val_total'),
        mode=cfg.training.checkpoint.get('mode', 'min')
    )
    logger.info(f"CheckpointManager created: dir={cfg.paths.checkpoint_dir}, metric={checkpoint_manager.metric_name}")

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_manager=loss_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=checkpoint_manager,
        device=cfg.system.device,
        max_epochs=cfg.training.num_epochs,
        use_amp=cfg.system.use_amp,
        grad_clip_norm=cfg.training.grad_clip.max_norm if cfg.training.grad_clip.enabled else None,
        tensorboard_dir=cfg.paths.log_dir if cfg.training.get('tensorboard', True) else None,
        log_interval=cfg.training.log_interval,
        eval_interval=cfg.training.val_interval
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
        trainer.train(start_epoch=start_epoch)
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
