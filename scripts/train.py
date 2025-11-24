#!/usr/bin/env python3
"""
Training Script for AquaSOT-3D

Main training script that integrates all components:
- Data loading
- Model building  
- Training loop with Phase 8 Trainer
- Checkpointing and logging

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml \\
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
from aquasot3d.dataloader import RGBSonarDataset, MultiSequenceRGBSonarDataset, TemplateSearchDataset, template_search_collate_fn, create_dataloader
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

    # CRITICAL FIX (Audit 2025111704 Finding 2): Apply init_backbones checkpoint paths
    # If training.init_backbones is set, override backbone pretrained_path for Stage 3
    if hasattr(cfg.training, 'init_backbones') and cfg.training.init_backbones:
        init_backbones = cfg.training.init_backbones

        # Apply RGB checkpoint path
        if rgb_backbone_cfg and init_backbones.get('rgb'):
            rgb_checkpoint_path = init_backbones.rgb
            if rgb_checkpoint_path not in [None, 'null', '']:
                rgb_backbone_cfg['pretrained_path'] = rgb_checkpoint_path
                logger.info(f"Setting RGB backbone pretrained_path: {rgb_checkpoint_path}")

        # Apply Sonar checkpoint path
        if sonar_backbone_cfg and init_backbones.get('sonar'):
            sonar_checkpoint_path = init_backbones.sonar
            if sonar_checkpoint_path not in [None, 'null', '']:
                sonar_backbone_cfg['pretrained_path'] = sonar_checkpoint_path
                logger.info(f"Setting Sonar encoder pretrained_path: {sonar_checkpoint_path}")

    # Prepare fusion config
    fusion_cfg = _config_to_dict(cfg.model.fusion) if cfg.model.fusion.get('type') not in [None, 'null'] else None

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
    # ARCHITECTURE UPDATE (Audit 2025112313): fusion_head removed (dual-view fusion architecture)

    # CRITICAL FIX (Audit 2025112503 Remaining Issue 3): Extract allow_single_modality from config
    # This enables single-modality ablation experiments via YAML configuration
    allow_single_modality = cfg.model.get('allow_single_modality', False)

    # Build model
    model = AquaSOT3DTracker(
        rgb_backbone_config=rgb_backbone_cfg,
        sonar_backbone_config=sonar_backbone_cfg,
        fusion_config=fusion_cfg,
        box_head_config=box_head_cfg,
        allow_single_modality=allow_single_modality  # Pass flag to tracker
    )

    logger.info(f"Model built successfully")

    # Count parameters safely (skip uninitialized LazyModule parameters)
    try:
        # Check if parameter is initialized (for LazyModules)
        def is_initialized(p):
            if hasattr(p, 'is_initialized'):
                return p.is_initialized()
            return True

        total_params = sum(p.numel() for p in model.parameters() if is_initialized(p))
        logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Could not count parameters (LazyModules may need initialization): {e}")
        logger.info("Parameter count will be available after first forward pass")

    return model


def build_dataloaders(cfg):
    """Build training and validation dataloaders with multi-sequence support"""
    logger.info("Building dataloaders")

    # CRITICAL FIX (Audit 2025112404 - Modality Cleanup):
    # AquaSOT-3D is fundamentally a dual-sensor system
    # Both RGB and Sonar are always required - no single-sensor operation
    logger.info("Training with dual-sensor system (RGB + Sonar)")

    # Check if using new multi-sequence format or legacy single-sequence format
    use_multi_sequence = (
        hasattr(cfg.data, 'train_sequences') and
        cfg.data.train_sequences is not None and
        len(cfg.data.train_sequences) > 0
    )

    if use_multi_sequence:
        # Multi-sequence mode
        data_root = cfg.data.get('data_root', 'data')
        train_sequences = cfg.data.train_sequences
        val_sequences = cfg.data.get('val_sequences', [])

        logger.info(f"Multi-sequence mode: data_root={data_root}")
        logger.info(f"Train sequences: {train_sequences}")
        logger.info(f"Val sequences: {val_sequences}")

        # Create multi-sequence base datasets (always dual-sensor)
        train_base_dataset = MultiSequenceRGBSonarDataset(
            data_root=data_root,
            sequence_names=train_sequences,
            return_occluded=True,
            sampling_policy='frame_balanced'
        )

        # Log per-sequence information
        for seq_info in train_base_dataset.get_sequence_info():
            logger.info(
                f"  Train sequence '{seq_info['sequence_name']}': "
                f"{seq_info['num_valid_frames']} valid frames, "
                f"RGB {seq_info['rgb_resolution']}, "
                f"Sonar {seq_info['sonar_resolution']}"
            )

        if val_sequences:
            # Create validation dataset (always dual-sensor)
            val_base_dataset = MultiSequenceRGBSonarDataset(
                data_root=data_root,
                sequence_names=val_sequences,
                return_occluded=True,
                sampling_policy='frame_balanced'
            )

            for seq_info in val_base_dataset.get_sequence_info():
                logger.info(
                    f"  Val sequence '{seq_info['sequence_name']}': "
                    f"{seq_info['num_valid_frames']} valid frames"
                )
        else:
            logger.warning("No validation sequences specified")
            val_base_dataset = None

    else:
        # Legacy single-sequence mode (backward compatibility)
        logger.info("Single-sequence mode (legacy)")

        train_path = Path(cfg.data.train_root)
        val_path = Path(cfg.data.val_root)

        train_data_root = str(train_path.parent)
        train_sequence_name = train_path.name
        val_data_root = str(val_path.parent)
        val_sequence_name = val_path.name

        logger.info(f"Train data: {train_data_root}/{train_sequence_name}")
        logger.info(f"Val data: {val_data_root}/{val_sequence_name}")

        # Create single-sequence base datasets (always dual-sensor)
        train_base_dataset = RGBSonarDataset(
            data_root=train_data_root,
            sequence_name=train_sequence_name,
            return_occluded=True
        )

        val_base_dataset = RGBSonarDataset(
            data_root=val_data_root,
            sequence_name=val_sequence_name,
            return_occluded=True
        )

    # Build augmentation pipeline for training
    train_augmentation = None
    if cfg.data.get('augmentation', {}).get('enabled', False):
        from aquasot3d.dataloader.datasets.augmentation import build_augmentation

        logger.info("Building data augmentation pipeline")

        # Use bbox-aware augmentation builder
        aug_cfg = cfg.data.augmentation

        train_augmentation = build_augmentation(
            cfg=aug_cfg,
            is_train=True
        )

        logger.info(f"Augmentation type: {aug_cfg.get('type', 'TrackingAugmentationBboxAware')}")
    else:
        logger.info("Augmentation disabled")

    # Template-search dataset configuration
    template_size = cfg.data.get('template_size', 256)
    search_size = cfg.data.get('search_size', 512)
    template_search_gap_range = cfg.data.get('template_search_gap_range', [1, 30])
    template_update_policy = cfg.data.get('template_update_policy', 'first_frame')

    # Convert list to tuple for TemplateSearchDataset
    gap_tuple = tuple(template_search_gap_range) if isinstance(template_search_gap_range, list) else template_search_gap_range

    train_dataset = TemplateSearchDataset(
        base_dataset=train_base_dataset,
        template_size=template_size,
        search_size=search_size,
        template_search_gap=gap_tuple,  # Fixed: correct parameter name
        template_update_policy=template_update_policy,
        augmentation=train_augmentation,
        normalization_config=cfg.data.normalize  # CRITICAL FIX (Audit 2025112501 P2): Pass normalization config
    )

    logger.info(f"Template-search training: template={template_size}x{template_size}, "
                f"search={search_size}x{search_size}, gap_range={template_search_gap_range}, "
                f"update_policy={template_update_policy}")
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    # Create dataloaders with template-search collate function
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=template_search_collate_fn,
        distributed=cfg.system.distributed,
        prefetch_factor=cfg.data.get('prefetch_factor', None),
        persistent_workers=cfg.data.get('persistent_workers', None)
    )

    # Only create validation dataset/loader if base dataset exists
    if val_base_dataset is not None:
        # ARCHITECTURE CHANGE (Audit 2025112104): Template-search validation
        val_dataset = TemplateSearchDataset(
            base_dataset=val_base_dataset,
            template_size=template_size,
            search_size=search_size,
            template_search_gap=gap_tuple,  # Fixed: correct parameter name
            augmentation=None,  # No augmentation for validation
            normalization_config=cfg.data.normalize  # CRITICAL FIX (Audit 2025112501 P2): Pass normalization config
        )

        logger.info(f"Val dataset: {len(val_dataset)} samples")

        val_loader = create_dataloader(
            dataset=val_dataset,
            batch_size=cfg.training.val_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=template_search_collate_fn,  # Changed from collate_fn_sequence
            distributed=False,  # No distributed for validation
            prefetch_factor=cfg.data.get('prefetch_factor', None),
            persistent_workers=cfg.data.get('persistent_workers', None)
        )

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    else:
        val_loader = None
        logger.info(f"Train batches: {len(train_loader)}, Val disabled")

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

    # Add backbone_lr_multiplier if specified in config
    if cfg.training.optimizer.get('backbone_lr_multiplier') is not None:
        optimizer_params['backbone_lr_multiplier'] = cfg.training.optimizer.backbone_lr_multiplier

    # Add head_lr_multiplier if specified in config
    if cfg.training.optimizer.get('head_lr_multiplier') is not None:
        optimizer_params['head_lr_multiplier'] = cfg.training.optimizer.head_lr_multiplier

    # Add layer_decay only if use_layer_decay is enabled
    if cfg.training.optimizer.get('use_layer_decay', False):
        optimizer_params['layer_decay'] = cfg.training.optimizer.get('layer_decay_rate', 0.75)

    optimizer = build_optimizer(**optimizer_params)
    
    # Build scheduler
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=cfg.training.scheduler.type,
        max_epoch=cfg.training.num_epochs,  # Fix: use max_epoch parameter name
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        warmup_start_lr=cfg.training.scheduler.get('warmup_lr', 0),  # Fix: add warmup_start_lr
        eta_min=cfg.training.scheduler.min_lr  # Fix: use eta_min parameter name
    )

    # Build checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.paths.checkpoint_dir,
        max_to_keep=cfg.training.checkpoint.get('max_keep', 5),
        save_best_only=False,  # We'll handle save_best and save_last separately in Trainer
        metric_name=cfg.training.checkpoint.get('metric', 'val_total'),
        mode=cfg.training.checkpoint.get('mode', 'min')
    )
    logger.info(f"CheckpointManager created: dir={cfg.paths.checkpoint_dir}, metric={checkpoint_manager.metric_name}")

    # Prepare checkpoint metadata (CRITICAL FIX: Audit 2025111704 Finding 4)
    # Tag checkpoints with stage info for classification and automation
    # CRITICAL FIX (Audit 2025112404 - Modality Cleanup):
    # Always dual-sensor (RGB + Sonar) - no single-sensor operation
    checkpoint_metadata = {
        'stage': cfg.experiment.get('group', 'unknown'),
        'modalities': ['rgb', 'sonar'],  # Always both sensors
        'config_path': args.config,
        'experiment_name': cfg.experiment.name,
        'tags': cfg.experiment.get('tags', [])
    }
    logger.info(f"Checkpoint metadata: {checkpoint_metadata}")

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
        eval_interval=cfg.training.val_interval,
        checkpoint_save_interval=cfg.training.checkpoint.get('save_interval', 1),
        save_best=cfg.training.checkpoint.get('save_best', True),
        save_last=cfg.training.checkpoint.get('save_last', True),
        checkpoint_metadata=checkpoint_metadata
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
