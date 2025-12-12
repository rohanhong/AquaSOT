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
from aquasot3d.utils.config import _config_to_dict
# CRITICAL FIX (Audit 2025121202 Finding 4/16/26/27): Data validation utilities
from aquasot3d.utils.data_validation import (
    validate_sequences_or_raise,
    block_polar_data,
    get_validation_summary,
    EXPECTED_SONAR_RECT_RESOLUTION
)
# CRITICAL FIX (Audit 2025121202 Finding 13/21/24): Run manager utilities
from aquasot3d.utils.run_manager import (
    setup_train_run,
    detect_legacy_outputs,
    generate_run_id
)

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
    # NOTE: --force-legacy-single-modality removed (Audit 2025121202 Finding 1)
    # Single-modality is now completely blocked. Use modality_dropout for ablation.

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

    # CRITICAL FIX (Audit 2025121202 Finding 1): Block single-modality entry completely
    # Single-modality mode is REMOVED. For ablation studies, use modality_dropout.
    # modality_dropout simulates single-modality by zeroing one modality while preserving dual architecture.
    if cfg.model.get('allow_single_modality', False):
        raise ValueError(
            "BLOCKED: allow_single_modality is removed (Audit 2025121202 Finding 1).\n"
            "For single-modality ablation studies, use modality_dropout instead:\n"
            "  data.modality_dropout.enabled: true\n"
            "  data.modality_dropout.p_drop_sonar: 1.0  # For RGB-only\n"
            "  data.modality_dropout.p_drop_rgb: 1.0    # For Sonar-only\n"
            "See configs/ablation/ablation_rgb_only.yaml and ablation_sonar_only.yaml for examples."
        )
    allow_single_modality = False  # Always false, kept for tracker interface compatibility

    # CRITICAL FIX (Audit 2025121202 Finding 2): Wire modality_dropout from config
    # ModalityDropout randomly drops modalities during training to train single/zero-modality operation
    modality_dropout_cfg = None
    if hasattr(cfg.data, 'modality_dropout'):
        modality_dropout_cfg = _config_to_dict(cfg.data.modality_dropout)
        if modality_dropout_cfg.get('enabled', False):
            logger.info(f"Modality dropout enabled: {modality_dropout_cfg}")

    # Build model
    model = AquaSOT3DTracker(
        rgb_backbone_config=rgb_backbone_cfg,
        sonar_backbone_config=sonar_backbone_cfg,
        fusion_config=fusion_cfg,
        box_head_config=box_head_cfg,
        allow_single_modality=allow_single_modality,  # Pass flag to tracker (legacy only)
        modality_dropout=modality_dropout_cfg  # CRITICAL FIX: Pass modality dropout config
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

        # CRITICAL FIX (Audit 2025112504): Read visibility_policy from config
        # This controls which frames are used based on modality visibility:
        #   'all': Include ALL frames including zero-modality (0,0,0,0) - REQUIRED for confidence learning
        #   'any': At least one modality visible (excludes zero-modality) - insufficient for confidence heads
        #   'both': Both modalities visible (pure dual-modality training) - insufficient for conditional fusion
        visibility_policy = cfg.data.get('visibility_policy', 'both')
        logger.info(f"Using visibility_policy='{visibility_policy}' for training data filtering")

        # CRITICAL FIX (Audit 2025112507 Gap 2): Compatibility check for fusion type vs visibility policy
        # Conditional DCMA and confidence heads REQUIRE zero-modality frames (0,0,0,0) to learn suppression
        fusion_type = cfg.model.fusion.get('type', 'dcma')
        has_confidence_heads = (
            cfg.model.heads.get('rgb_box', {}).get('use_confidence_head', False) or
            cfg.model.heads.get('sonar_box', {}).get('use_confidence_head', False)
        )

        requires_all_policy = (
            fusion_type == 'conditional_dcma' or  # Needs single/zero-modality routing
            has_confidence_heads  # Needs zero-modality for has_object=False learning
        )

        if requires_all_policy and visibility_policy != 'all':
            logger.error(
                f"CRITICAL CONFIGURATION ERROR: fusion_type='{fusion_type}' "
                f"(has_confidence_heads={has_confidence_heads}) requires visibility_policy='all' "
                f"to learn zero-modality suppression, but got visibility_policy='{visibility_policy}'."
            )
            logger.error(
                "Without zero-modality frames (0,0,0,0), confidence heads cannot learn has_object=False, "
                "violating specification requirement. Auto-correcting to visibility_policy='all'."
            )
            visibility_policy = 'all'
            logger.warning(f"Overridden visibility_policy to 'all' for compatibility with {fusion_type}")

        if visibility_policy == 'all':
            logger.info(
                "visibility_policy='all' includes zero/single/dual-modality frames. "
                "Model will learn zero-modality suppression (has_object=False) and single-modality routing."
            )

        # Create multi-sequence base datasets (always dual-sensor)
        # IMPORTANT: Set return_occluded=False to prevent it from overriding visibility_policy
        train_base_dataset = MultiSequenceRGBSonarDataset(
            data_root=data_root,
            sequence_names=train_sequences,
            return_occluded=False,  # Use visibility_policy instead
            visibility_policy=visibility_policy,  # CRITICAL: Pass visibility_policy from config
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
            # Create validation dataset (same visibility_policy as training)
            val_base_dataset = MultiSequenceRGBSonarDataset(
                data_root=data_root,
                sequence_names=val_sequences,
                return_occluded=False,  # Use visibility_policy instead
                visibility_policy=visibility_policy,  # Same policy as training
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

        # CRITICAL FIX (Audit 2025112504): Read visibility_policy from config
        visibility_policy = cfg.data.get('visibility_policy', 'both')
        logger.info(f"Using visibility_policy='{visibility_policy}' for training data filtering")

        # CRITICAL FIX (Audit 2025112507 Gap 2): Compatibility check for fusion type vs visibility policy
        # (Same logic as multi-sequence mode above)
        fusion_type = cfg.model.fusion.get('type', 'dcma')
        has_confidence_heads = (
            cfg.model.heads.get('rgb_box', {}).get('use_confidence_head', False) or
            cfg.model.heads.get('sonar_box', {}).get('use_confidence_head', False)
        )

        requires_all_policy = (
            fusion_type == 'conditional_dcma' or
            has_confidence_heads
        )

        if requires_all_policy and visibility_policy != 'all':
            logger.error(
                f"CRITICAL CONFIGURATION ERROR: fusion_type='{fusion_type}' "
                f"(has_confidence_heads={has_confidence_heads}) requires visibility_policy='all' "
                f"to learn zero-modality suppression, but got visibility_policy='{visibility_policy}'."
            )
            logger.error(
                "Without zero-modality frames (0,0,0,0), confidence heads cannot learn has_object=False, "
                "violating specification requirement. Auto-correcting to visibility_policy='all'."
            )
            visibility_policy = 'all'
            logger.warning(f"Overridden visibility_policy to 'all' for compatibility with {fusion_type}")

        if visibility_policy == 'all':
            logger.info(
                "visibility_policy='all' includes zero/single/dual-modality frames. "
                "Model will learn zero-modality suppression (has_object=False) and single-modality routing."
            )

        # Create single-sequence base datasets (always dual-sensor)
        # IMPORTANT: Set return_occluded=False to prevent it from overriding visibility_policy
        train_base_dataset = RGBSonarDataset(
            data_root=train_data_root,
            sequence_name=train_sequence_name,
            return_occluded=False,  # Use visibility_policy instead
            visibility_policy=visibility_policy  # CRITICAL: Pass visibility_policy from config
        )

        val_base_dataset = RGBSonarDataset(
            data_root=val_data_root,
            sequence_name=val_sequence_name,
            return_occluded=False,  # Use visibility_policy instead
            visibility_policy=visibility_policy  # Same policy as training
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

    # Template-search dataset configuration (OSTrack-256 standard defaults)
    template_size = cfg.data.get('template_size', 128)  # OSTrack-256 standard
    search_size = cfg.data.get('search_size', 256)      # OSTrack-256 standard
    template_search_gap_range = cfg.data.get('template_search_gap_range', [1, 30])
    template_update_policy = cfg.data.get('template_update_policy', 'first_frame')

    # Convert list to tuple for TemplateSearchDataset
    gap_tuple = tuple(template_search_gap_range) if isinstance(template_search_gap_range, list) else template_search_gap_range

    # CRITICAL FIX (Audit 2025112701): Pass center_jitter_factor for online tracking
    # Without jitter, model learns to always predict center (0.5, 0.5), causing
    # evaluation failure when using previous prediction as search center
    center_jitter = cfg.data.get('center_jitter_factor', 0.25)  # OSTrack default: 0.25
    scale_jitter = cfg.data.get('scale_jitter_factor', 0.25)
    search_area_factor = cfg.data.get('search_area_factor', 4.0)
    template_area_factor = cfg.data.get('template_area_factor', 2.0)

    # CRITICAL FIX (Audit 2025121202 Finding 22/25): Pass quality thresholds from config
    quality_threshold_rgb = cfg.data.get('quality_threshold_rgb', 0.7)
    quality_threshold_sonar = cfg.data.get('quality_threshold_sonar', 0.7)

    # CRITICAL FIX (Audit 2025121202 Finding 22): Enable static quality filtering for training
    # Static quality filtering (bbox size, edge margin, aspect ratio) does NOT require model predictions
    # and helps training by selecting non-clipped, properly-sized templates.
    # Note: quality_threshold_rgb/sonar are EVAL-ONLY (require model predictions, not available in training)
    train_enable_quality_filter = cfg.data.get('train_enable_template_quality_filter', True)

    train_dataset = TemplateSearchDataset(
        base_dataset=train_base_dataset,
        template_size=template_size,
        search_size=search_size,
        template_search_gap=gap_tuple,  # Fixed: correct parameter name
        template_update_policy=template_update_policy,
        search_area_factor=search_area_factor,
        template_area_factor=template_area_factor,
        center_jitter_factor=center_jitter,  # CRITICAL: Enable training jitter
        scale_jitter_factor=scale_jitter,
        augmentation=train_augmentation,
        normalization_config=cfg.data.normalize,  # CRITICAL FIX (Audit 2025112501 P2): Pass normalization config
        # CRITICAL FIX (Audit 2025121202 Finding 22): Enable static quality filtering for training
        enable_template_quality_filter=train_enable_quality_filter,
        template_min_size=cfg.data.get('template_min_size', 5.0),
        template_edge_margin=cfg.data.get('template_edge_margin', 2),
        template_scan_frames=cfg.data.get('template_scan_frames', 20),
        # Note: quality_threshold_rgb/sonar are stored but EVAL-ONLY (no model predictions in training)
        quality_threshold_rgb=quality_threshold_rgb,
        quality_threshold_sonar=quality_threshold_sonar
    )

    logger.info(f"Training jitter enabled: center_jitter={center_jitter}, scale_jitter={scale_jitter}")

    # CRITICAL FIX (Audit 2025121202 Finding 22/28): Log effective configuration fields
    # Clarify which quality/threshold fields are active:
    # - enable_template_quality_filter: Static filtering (bbox size, edge margin) - ACTIVE in training
    # - quality_threshold_rgb/sonar: Model-based filtering - EVAL-ONLY (no predictions in training)
    # - validity_gating: Head-based routing - EVAL-ONLY (used in OnlineTrackingEvaluator)
    logger.info("Training quality configuration:")
    logger.info(f"  enable_template_quality_filter: {train_enable_quality_filter} (static: bbox size, edge margin)")
    logger.info(f"  template_min_size: {cfg.data.get('template_min_size', 5.0)}")
    logger.info(f"  template_edge_margin: {cfg.data.get('template_edge_margin', 2)}")
    logger.info(f"  template_scan_frames: {cfg.data.get('template_scan_frames', 20)}")
    logger.info(f"  quality_threshold_rgb: {quality_threshold_rgb} (EVAL-ONLY, stored)")
    logger.info(f"  quality_threshold_sonar: {quality_threshold_sonar} (EVAL-ONLY, stored)")
    logger.info(f"  template_update_policy: {template_update_policy}")
    logger.info(f"  validity_gating: EVAL-ONLY (see eval.yaml)")

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
        # NOTE: Validation uses no jitter (center_jitter=0) to evaluate on centered targets
        #
        # CRITICAL FIX (Audit 2025120401 Finding 3): Align validation template strategy with deployment
        # Validation MUST use 'first_frame' policy to match OnlineTrackingEvaluator protocol:
        #   - Training: 'mixformer' (random template selection for diverse learning)
        #   - Validation: 'first_frame' (consistent with online tracking initialization)
        # Using 'mixformer' for validation causes train-val metric mismatch because:
        #   - Val samples random templates (like training)
        #   - Online eval always uses first frame as initial template
        # This alignment ensures val metrics better predict online tracking performance.
        val_template_policy = cfg.data.get('val_template_update_policy', 'first_frame')
        logger.info(f"Validation template policy: {val_template_policy} (aligned with online tracking)")

        # CRITICAL FIX (Audit 2025120402 Finding 5): Enable template quality filtering for validation
        # This ensures val selects best quality template, consistent with OnlineTrackingEvaluator
        val_enable_quality_filter = cfg.data.get('val_enable_template_quality_filter', True)

        val_dataset = TemplateSearchDataset(
            base_dataset=val_base_dataset,
            template_size=template_size,
            search_size=search_size,
            template_search_gap=gap_tuple,  # Fixed: correct parameter name
            template_update_policy=val_template_policy,  # CRITICAL: Use first_frame for deployment alignment
            search_area_factor=search_area_factor,
            template_area_factor=template_area_factor,
            center_jitter_factor=0.0,  # No jitter for validation
            scale_jitter_factor=0.0,
            augmentation=None,  # No augmentation for validation
            normalization_config=cfg.data.normalize,  # CRITICAL FIX (Audit 2025112501 P2): Pass normalization config
            # CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering for validation
            enable_template_quality_filter=val_enable_quality_filter,
            template_min_size=cfg.data.get('template_min_size', 5.0),
            template_edge_margin=cfg.data.get('template_edge_margin', 2),
            template_scan_frames=cfg.data.get('template_scan_frames', 20),
            # CRITICAL FIX (Audit 2025121201 S3): Disable negative sampling for validation
            # Val/Test MUST use negative_sample_prob=0 to match online deployment distribution
            negative_sample_prob=0.0,
            # CRITICAL FIX (Audit 2025121202 Finding 22/25): Pass quality thresholds
            quality_threshold_rgb=quality_threshold_rgb,
            quality_threshold_sonar=quality_threshold_sonar
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

    # CRITICAL FIX (Audit 2025121202 Finding 4/16/26/27): Validate data before training
    rect_check_result = None
    data_paths = None
    if is_main_process:
        data_root = cfg.data.get('data_root', 'data')
        train_sequences = cfg.data.get('train_sequences', [])
        val_sequences = cfg.data.get('val_sequences', [])
        all_sequences = list(set(train_sequences + val_sequences))

        # CRITICAL FIX (Audit 2025121202 Finding 21/24/27): Record and validate data paths
        train_root = str(cfg.data.get('train_root') or data_root)
        val_root = str(cfg.data.get('val_root') or data_root)
        data_paths = {
            'data_root': str(data_root),
            'train_root': train_root,
            'val_root': val_root,
            'train_sequences': train_sequences,
            'val_sequences': val_sequences
        }

        # CRITICAL FIX (Audit 2025121202 Finding 27): Validate data root existence
        for root_name, root_path in [('data_root', data_root), ('train_root', train_root), ('val_root', val_root)]:
            if not Path(root_path).exists():
                raise FileNotFoundError(
                    f"Data root does not exist: {root_name}={root_path}\n"
                    f"Please verify the data path configuration in your config file."
                )
        logger.info(f"Data root validation passed: data_root={data_root}")

        # CRITICAL FIX (Audit 2025121202 Finding 27): Handle legacy single-sequence mode validation
        # In legacy mode, train_sequences is empty but train_root/val_root point to sequence directories
        if not all_sequences and (train_root != data_root or val_root != data_root):
            # Legacy mode: extract sequence names from paths
            legacy_sequences = []
            if train_root != data_root:
                train_seq_path = Path(train_root)
                legacy_sequences.append(train_seq_path.name)
                # Use parent as data_root for validation
                data_root = str(train_seq_path.parent)
            if val_root != data_root and val_root != train_root:
                val_seq_path = Path(val_root)
                legacy_sequences.append(val_seq_path.name)
            all_sequences = legacy_sequences
            logger.info(f"Legacy mode: extracted sequences {all_sequences} from train_root/val_root")

        if all_sequences:
            logger.info("Validating dataset rect preprocessing...")
            try:
                # Check rect data readiness and block polar data
                validate_sequences_or_raise(
                    data_root=data_root,
                    sequence_names=all_sequences,
                    strict=True  # Fail if rect data not ready
                )
                # CRITICAL FIX (Audit 2025121202 Finding 26): Block polar data usage
                for seq_name in all_sequences:
                    seq_path = Path(data_root) / seq_name
                    block_polar_data(seq_path)
                logger.info(f"Polar data check passed for {len(all_sequences)} sequences")
                # Log validation summary
                summary = get_validation_summary(data_root)
                logger.info(f"Data validation passed:\n{summary}")
                # CRITICAL FIX (Audit 2025121202 Finding 21/24): Record rect_check result
                rect_check_result = {
                    'passed': True,
                    'sequences_checked': all_sequences,
                    'data_root': str(data_root)
                }
            except Exception as e:
                logger.error(f"Data validation failed: {e}")
                logger.error(
                    "Please run preprocessing first: "
                    "python scripts/preprocess_sonar.py --all --data-root {data_root}"
                )
                raise

    # CRITICAL FIX (Audit 2025121202 Finding 13/21/24): Setup versioned output directory
    run_id = None
    if is_main_process:
        # Check for legacy outputs
        has_legacy, legacy_items = detect_legacy_outputs(cfg.paths.work_dir)
        if has_legacy:
            for item in legacy_items:
                logger.warning(f"Legacy output detected: {item}")
            logger.warning(
                "Consider migrating legacy outputs or using a different work_dir. "
                "New outputs will use run_id versioning."
            )

        # Generate run_id for versioned output
        run_id = generate_run_id()
        logger.info(f"Training run ID: {run_id}")

        # Setup versioned directories with rect_check and data_paths
        run_dir, checkpoint_dir, log_dir = setup_train_run(
            config=_config_to_dict(cfg),
            force=False,
            run_id=run_id,
            rect_check_result=rect_check_result,
            data_paths=data_paths
        )
        # Update config paths to versioned directories
        cfg.paths.work_dir = str(run_dir)
        cfg.paths.checkpoint_dir = str(checkpoint_dir)
        cfg.paths.log_dir = str(log_dir)
        logger.info(f"Versioned output directory: {run_dir}")

    # Create output directories (versioned)
    if is_main_process:
        Path(cfg.paths.work_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(cfg, Path(cfg.paths.work_dir) / 'config.yaml')

        # CRITICAL FIX (Audit 2025121202 Finding 20/28): Output effective configuration summary
        logger.info("="*70)
        logger.info("EFFECTIVE CONFIGURATION SUMMARY (Audit 2025121202 Finding 20/28)")
        logger.info("="*70)
        logger.info("[TRAINING SETTINGS - IN EFFECT]")
        logger.info(f"  batch_size: {cfg.training.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {cfg.training.gradient_accumulation_steps}")
        logger.info(f"  num_epochs: {cfg.training.num_epochs}")
        logger.info(f"  optimizer.lr: {cfg.training.optimizer.lr}")
        modality_dropout_cfg = cfg.data.get('modality_dropout', {})
        logger.info(f"  modality_dropout.enabled: {modality_dropout_cfg.get('enabled', False)}")
        if modality_dropout_cfg.get('enabled', False):
            logger.info(f"    p_drop_rgb: {modality_dropout_cfg.get('p_drop_rgb', 0.1)}")
            logger.info(f"    p_drop_sonar: {modality_dropout_cfg.get('p_drop_sonar', 0.1)}")
        # CRITICAL FIX (Audit 2025121202 Finding 22): Static quality filter for training
        train_quality_filter = cfg.data.get('train_enable_template_quality_filter', True)
        logger.info(f"  train_enable_template_quality_filter: {train_quality_filter} (static: bbox size, edge margin)")
        logger.info(f"  online_eval.enabled: {cfg.training.online_eval.get('enabled', False)}")
        logger.info(f"  fusion.type: {cfg.model.fusion.get('type', 'conditional_dcma')}")
        logger.info(f"  data_root: {data_root} (validated)")
        logger.info(f"  train_sequences: {len(train_sequences)} sequences")
        logger.info(f"  val_sequences: {len(val_sequences)} sequences")
        logger.info("[TRAINING SETTINGS - EVAL-ONLY (stored but not active)]")
        logger.info(f"  quality_threshold_rgb: {cfg.data.get('quality_threshold_rgb', 0.7)} (requires model predictions)")
        logger.info(f"  quality_threshold_sonar: {cfg.data.get('quality_threshold_sonar', 0.7)} (requires model predictions)")
        logger.info(f"  validity_gating: configured in eval.yaml (requires model outputs)")
        logger.info("[NOT IMPLEMENTED - CONFIG HAS NO EFFECT]")
        logger.info("  training.wandb: NOT IMPLEMENTED (use tensorboard instead)")
        logger.info("  FusionHead: REMOVED (use ConditionalDCMA)")
        logger.info("  allow_single_modality: REMOVED (use modality_dropout)")
        logger.info("  fusion_quality/fusion_weights: REMOVED (not collected)")
        logger.info("="*70)

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

    # Prepare optional online evaluation config (Audit Issue 11/12)
    online_eval_cfg = None
    if hasattr(cfg.training, 'online_eval') and cfg.training.online_eval.get('enabled', False):
        online_eval_cfg = _config_to_dict(cfg.training.online_eval)
        # Fill defaults from data/evaluation settings
        online_eval_cfg.setdefault('data_root', cfg.data.get('data_root', 'data'))
        online_eval_cfg.setdefault('sequences', cfg.data.get('val_sequences', []))
        online_eval_cfg.setdefault('template_size', cfg.data.get('template_size', 128))
        online_eval_cfg.setdefault('search_size', cfg.data.get('search_size', 256))
        online_eval_cfg.setdefault('template_area_factor', cfg.data.get('template_area_factor', 2.0))
        online_eval_cfg.setdefault('search_area_factor', cfg.data.get('search_area_factor', 4.0))
        # Pull normalization config
        online_eval_cfg.setdefault('normalize', _config_to_dict(cfg.data.normalize))
        # Template quality filtering defaults (align with evaluation config)
        online_eval_cfg.setdefault('enable_template_quality_filter', cfg.evaluation.get('enable_template_quality_filter', True))
        online_eval_cfg.setdefault('template_min_size', cfg.evaluation.get('template_min_size', 5.0))
        online_eval_cfg.setdefault('template_edge_margin', cfg.evaluation.get('template_edge_margin', 2))
        online_eval_cfg.setdefault('template_scan_frames', cfg.evaluation.get('template_scan_frames', 20))
        # CRITICAL FIX (Audit 2025121202 Finding 8/11/20): Template update from nested config
        template_update_cfg = cfg.evaluation.get('template_update', {})
        online_eval_cfg.setdefault('enable_template_update', template_update_cfg.get('enabled', cfg.evaluation.get('enable_template_update', True)))
        online_eval_cfg.setdefault('template_update_interval', template_update_cfg.get('interval', cfg.evaluation.get('template_update_interval', 10)))
        online_eval_cfg.setdefault('template_update_threshold', template_update_cfg.get('threshold', cfg.evaluation.get('template_update_threshold', 0.7)))

        # CRITICAL FIX (Audit 2025121202 Finding 3/5): Validity gating config
        validity_gating_cfg = cfg.evaluation.get('validity_gating', {})
        online_eval_cfg.setdefault('validity_gating', {
            'enabled': validity_gating_cfg.get('enabled', False),
            'rgb_threshold': validity_gating_cfg.get('rgb_threshold', 0.35),
            'sonar_threshold': validity_gating_cfg.get('sonar_threshold', 0.40)
        })

        # CRITICAL FIX (Audit 2025121202 Finding 9): Expected resolutions for fallback
        expected_sonar_res = cfg.evaluation.get('expected_sonar_resolution', [682, 376])
        expected_rgb_res = cfg.evaluation.get('expected_rgb_resolution', [1920, 1080])
        online_eval_cfg.setdefault('expected_sonar_resolution', (expected_sonar_res[1], expected_sonar_res[0]))  # H, W
        online_eval_cfg.setdefault('expected_rgb_resolution', (expected_rgb_res[1], expected_rgb_res[0]))  # H, W

        # CRITICAL FIX (Audit 2025121202 Finding 17): Metrics list from config
        online_eval_cfg.setdefault('metrics_list', cfg.evaluation.get('metrics', None))

        logger.info(f"Online evaluation enabled during training on sequences: {online_eval_cfg.get('sequences')}")

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
        checkpoint_metadata=checkpoint_metadata,
        online_eval_config=online_eval_cfg
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
