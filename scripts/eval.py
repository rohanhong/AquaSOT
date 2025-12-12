#!/usr/bin/env python3
"""
Evaluation Script for AquaSOT-3D

Evaluate trained model on test set using Phase 9 evaluation module.

Usage:
    python scripts/eval.py --config configs/eval.yaml \\
        --checkpoint outputs/train/checkpoints/best_model.pth

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
import logging
import numpy as np
import cv2

from aquasot3d.utils.config import load_config, save_config, setup_config_logging, _config_to_dict
from aquasot3d.dataloader import create_dataloader
from aquasot3d.models.tracker import AquaSOT3DTracker
from aquasot3d.evaluation import TrackingEvaluator, TrackingVisualizer, OnlineTrackingEvaluator
# CRITICAL FIX (Audit 2025121202 Finding 4/16/26/27): Data validation utilities
from aquasot3d.utils.data_validation import (
    validate_sequences_or_raise,
    get_validation_summary,
    block_polar_data,
    EXPECTED_SONAR_RECT_RESOLUTION
)
# CRITICAL FIX (Audit 2025121202 Finding 13/21/24): Run manager utilities
from aquasot3d.utils.run_manager import (
    setup_eval_run,
    detect_legacy_outputs,
    generate_run_id
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='AquaSOT-3D Evaluation Script')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to evaluation configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--opts', nargs='+', default=[],
                        help='Override config options')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output directory')

    # Visualization control
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument('--visualize', action='store_true', default=None,
                          help='Enable visualization (overrides config)')
    viz_group.add_argument('--no-visualize', action='store_true', default=None,
                          help='Disable visualization (overrides config)')

    # CRITICAL FIX (Audit 2025120401 Finding 7): Evaluation protocol selection
    # Default to 'online' which is the STANDARD tracking evaluation protocol
    parser.add_argument('--protocol', type=str, default='online',
                        choices=['online', 'offline'],
                        help='Evaluation protocol: '
                             'online=search centered on PREDICTION (standard), '
                             'offline=search centered on GT (ablation only)')

    # Template update control for online evaluation
    # CRITICAL FIX (Audit 2025120402 Finding 2-3): Clear baseline vs update mode
    template_group = parser.add_mutually_exclusive_group()
    template_group.add_argument('--enable-template-update', action='store_true', default=None,
                        help='Enable dynamic template update during online tracking (default)')
    template_group.add_argument('--no-template-update', '--baseline', action='store_true', default=None,
                        dest='no_template_update',
                        help='Disable template update for first-frame-fixed BASELINE evaluation. '
                             'Use this to compare with --enable-template-update for ablation.')

    return parser.parse_args()


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

    # Build model
    model = AquaSOT3DTracker(
        rgb_backbone_config=rgb_backbone_cfg,
        sonar_backbone_config=sonar_backbone_cfg,
        fusion_config=fusion_cfg,
        box_head_config=box_head_cfg
    )

    logger.info(f"Model built successfully")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


def load_model_checkpoint(model, checkpoint_path, device='cuda'):
    """Load model weights from checkpoint"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load state dict into model
    load_result = model.load_state_dict(state_dict, strict=False)

    # Report loading statistics
    logger.info(f"Checkpoint loading summary:")
    logger.info(f"  Total parameters in checkpoint: {len(state_dict)}")
    logger.info(f"  Total parameters in model: {len(model.state_dict())}")

    if len(load_result.missing_keys) > 0:
        logger.warning(f"  Missing {len(load_result.missing_keys)} parameters (using random initialization):")
        for key in load_result.missing_keys[:5]:
            logger.warning(f"    - {key}")
        if len(load_result.missing_keys) > 5:
            logger.warning(f"    ... and {len(load_result.missing_keys) - 5} more")

    if len(load_result.unexpected_keys) > 0:
        logger.warning(f"  Found {len(load_result.unexpected_keys)} unexpected keys (ignored)")

    # Log checkpoint metadata
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch {checkpoint['epoch'] + 1}")
    if 'best_metric' in checkpoint:
        logger.info(f"Best metric: {checkpoint['best_metric']}")

    model = model.to(device)
    model.eval()

    if len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0:
        logger.info("All checkpoint parameters loaded successfully!")
    else:
        logger.warning(f"Checkpoint loading completed with {len(load_result.missing_keys)} missing, {len(load_result.unexpected_keys)} unexpected keys")

    return model


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Setup logging
    setup_config_logging('INFO')
    
    logger.info("="*70)
    logger.info("AquaSOT-3D Evaluation")
    logger.info("="*70)
    
    # Load configuration
    cfg = load_config(args.config, overrides=args.opts)

    # Override output directory if specified
    if args.output:
        cfg.evaluation.output_dir = args.output
        cfg.paths.work_dir = args.output

    # Override checkpoint path
    cfg.evaluation.checkpoint = args.checkpoint

    # Override visualization setting if specified via CLI
    if args.visualize:
        cfg.evaluation.visualization.enabled = True
    elif args.no_visualize:
        cfg.evaluation.visualization.enabled = False
    
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {cfg.evaluation.output_dir}")

    # Log visualization status
    if cfg.evaluation.visualization.enabled:
        logger.info(f"Visualization: ENABLED")
        logger.info(f"  Visualization output will be saved to: {cfg.evaluation.visualization.save_dir}")
        logger.info(f"  Plot trajectories: {cfg.evaluation.visualization.get('plot_trajectories', False)}")
        logger.info(f"  Plot curves: {cfg.evaluation.visualization.get('plot_curves', False)}")
        logger.info(f"  Save failure cases: {cfg.evaluation.visualization.get('save_failure_cases', False)}")
    else:
        logger.info(f"Visualization: DISABLED")
        logger.info(f"  No visualization artifacts will be generated")
        logger.info(f"  Use --visualize flag to enable visualization")

    # CRITICAL FIX (Audit 2025121202 Finding 4/16/26/27): Validate data before evaluation
    data_root = cfg.data.get('data_root', 'data')
    test_sequences = cfg.data.get('test_sequences', [])
    if isinstance(test_sequences, str):
        test_sequences = [test_sequences]

    # CRITICAL FIX (Audit 2025121202 Finding 12): Enforce batch_size=1 for evaluation
    eval_batch_size = cfg.evaluation.get('batch_size', 1)
    if eval_batch_size != 1:
        logger.warning(
            f"Evaluation batch_size={eval_batch_size} specified, but online tracking "
            f"evaluation REQUIRES batch_size=1 for sequential frame processing. "
            f"Forcing batch_size=1."
        )
        cfg.evaluation.batch_size = 1
    logger.info(f"Evaluation batch_size: 1 (enforced for online tracking protocol)")

    # CRITICAL FIX (Audit 2025121202 Finding 21/24/27): Record and validate data paths
    rect_check_result = None
    test_root = str(cfg.data.get('test_root') or data_root)
    data_paths = {
        'data_root': str(data_root),
        'test_root': test_root,
        'test_sequences': test_sequences
    }

    # CRITICAL FIX (Audit 2025121202 Finding 27): Validate data root existence
    for root_name, root_path in [('data_root', data_root), ('test_root', test_root)]:
        if not Path(root_path).exists():
            raise FileNotFoundError(
                f"Data root does not exist: {root_name}={root_path}\n"
                f"Please verify the data path configuration in your config file."
            )
    logger.info(f"Data root validation passed: data_root={data_root}")

    # CRITICAL FIX (Audit 2025121202 Finding 27): Handle legacy single-sequence mode validation
    # In legacy mode, test_sequences is empty but test_root points to a sequence directory
    if not test_sequences and test_root != data_root:
        # Legacy mode: extract sequence name from path
        test_seq_path = Path(test_root)
        test_sequences = [test_seq_path.name]
        # Use parent as data_root for validation
        data_root = str(test_seq_path.parent)
        logger.info(f"Legacy mode: extracted sequence {test_sequences} from test_root")

    if test_sequences:
        logger.info("Validating evaluation data rect preprocessing...")
        try:
            validate_sequences_or_raise(
                data_root=data_root,
                sequence_names=test_sequences,
                strict=True  # Fail if rect data not ready
            )
            # CRITICAL FIX (Audit 2025121202 Finding 26): Block polar data usage
            for seq_name in test_sequences:
                seq_path = Path(data_root) / seq_name
                block_polar_data(seq_path)
            logger.info(f"Polar data check passed for {len(test_sequences)} sequences")
            summary = get_validation_summary(data_root)
            logger.info(f"Data validation passed:\n{summary}")
            # CRITICAL FIX (Audit 2025121202 Finding 21/24): Record rect_check result
            rect_check_result = {
                'passed': True,
                'sequences_checked': test_sequences,
                'data_root': str(data_root)
            }
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            logger.error(
                f"Please run preprocessing first: "
                f"python scripts/preprocess_sonar.py --all --data-root {data_root}"
            )
            raise

    # CRITICAL FIX (Audit 2025121202 Finding 13/21/24): Setup versioned output directory
    # Check for legacy outputs
    has_legacy, legacy_items = detect_legacy_outputs(cfg.evaluation.output_dir)
    if has_legacy:
        for item in legacy_items:
            logger.warning(f"Legacy output detected: {item}")

    # Generate run_id for versioned output
    run_id = generate_run_id()
    logger.info(f"Evaluation run ID: {run_id}")

    # Setup versioned directories with rect_check and data_paths
    run_dir, viz_dir = setup_eval_run(
        config=_config_to_dict(cfg),
        checkpoint_path=args.checkpoint,
        force=False,
        run_id=run_id,
        rect_check_result=rect_check_result,
        data_paths=data_paths
    )
    # Update config paths to versioned directories
    cfg.evaluation.output_dir = str(run_dir)
    if cfg.evaluation.visualization.enabled:
        cfg.evaluation.visualization.save_dir = str(viz_dir)
    logger.info(f"Versioned output directory: {run_dir}")

    # Create output directories (versioned)
    Path(cfg.evaluation.output_dir).mkdir(parents=True, exist_ok=True)
    if cfg.evaluation.visualization.enabled:
        Path(cfg.evaluation.visualization.save_dir).mkdir(parents=True, exist_ok=True)

    # Save eval config
    save_config(cfg, Path(cfg.evaluation.output_dir) / 'eval_config.yaml')
    
    # Build model from config
    device = cfg.system.device if hasattr(cfg.system, 'device') else 'cuda'
    model = build_model(cfg)

    # Load checkpoint weights
    model = load_model_checkpoint(model, args.checkpoint, device=device)
    
    # Build evaluation dataset
    logger.info("Building evaluation dataset")

    # Parse test data path correctly (fix for hardcoded path issue)
    from pathlib import Path as PathLib
    from aquasot3d.dataloader import RGBSonarDataset, EvaluationSequenceDataset
    from aquasot3d.dataloader.loader import collate_fn_template_search
    from torch.utils.data import ConcatDataset

    # Check if using multi-sequence format or legacy single-sequence format
    use_multi_sequence = (
        hasattr(cfg.data, 'test_sequences') and
        cfg.data.test_sequences is not None and
        len(cfg.data.test_sequences) > 0
    )

    if use_multi_sequence:
        # Multi-sequence evaluation mode with ONLINE TRACKING PROTOCOL
        data_root = cfg.data.get('data_root', 'data')
        test_sequences = cfg.data.test_sequences

        # Defensive check: ensure test_sequences is a list
        if isinstance(test_sequences, str):
            test_sequences = [test_sequences]

        logger.info(f"Multi-sequence ONLINE evaluation mode: data_root={data_root}")
        logger.info(f"Test sequences: {test_sequences}")
        logger.info("NOTE: Using STANDARD online tracking protocol (search centered on PREDICTION, not GT)")

        # Build base datasets for each sequence
        # CRITICAL: For online evaluation, we only need base_dataset (no EvaluationSequenceDataset)
        # OnlineTrackingEvaluator handles template/search cropping internally
        seq_name_to_base_dataset = {}
        total_frames = 0

        for seq_name in test_sequences:
            seq_base_dataset = RGBSonarDataset(
                data_root=data_root,
                sequence_name=seq_name,
                return_occluded=True  # Include all frames for evaluation (even 0000)
            )

            seq_name_to_base_dataset[seq_name] = seq_base_dataset
            seq_info = seq_base_dataset.get_sequence_info()
            total_frames += seq_base_dataset.num_frames

            logger.info(
                f"  Sequence '{seq_name}': {seq_base_dataset.num_frames} frames, "
                f"RGB {seq_info['rgb_resolution']}, Sonar {seq_info['sonar_resolution']}"
            )

        logger.info(f"Total evaluation frames across {len(test_sequences)} sequences: {total_frames}")

    else:
        # Legacy single-sequence evaluation mode (backward compatibility)
        logger.info("Single-sequence ONLINE evaluation mode (legacy)")
        logger.info("NOTE: Using STANDARD online tracking protocol (search centered on PREDICTION, not GT)")

        test_path = PathLib(cfg.data.test_root)
        if test_path.is_absolute():
            # Absolute path: split into root and sequence name
            data_root = str(test_path.parent)
            sequence_name = test_path.name
        else:
            # Relative path: assume format is "data/sequence_name"
            parts = test_path.parts
            if len(parts) >= 2:
                data_root = str(PathLib(*parts[:-1]))
                sequence_name = parts[-1]
            else:
                # Single directory name, assume it's in 'data/'
                data_root = 'data'
                sequence_name = str(test_path)

        logger.info(f"Data root: {data_root}")
        logger.info(f"Sequence name: {sequence_name}")

        # Create base dataset for online evaluation
        eval_base_dataset = RGBSonarDataset(
            data_root=data_root,
            sequence_name=sequence_name,
            return_occluded=True  # Include all frames for evaluation (even 0000)
        )

        # Create mapping from sequence_name to base_dataset
        seq_name_to_base_dataset = {
            sequence_name: eval_base_dataset
        }

        logger.info(f"Evaluation dataset: {eval_base_dataset.num_frames} frames")

    # === Create Tracking Evaluator based on protocol ===
    # CRITICAL FIX (Audit 2025120401 Finding 7): Add protocol selection
    eval_protocol = args.protocol
    logger.info(f"Evaluation protocol: {eval_protocol.upper()}")

    if eval_protocol == 'online':
        # STANDARD online tracking protocol:
        # - Template: Initialized from first frame GT
        # - Search region: Centered on PREVIOUS PREDICTION (not GT!)
        # This properly evaluates tracking robustness including drift recovery

        # CRITICAL FIX (Audit 2025121202 Finding 8/20): Read template_update from nested config
        # Support both nested template_update.* and legacy top-level keys
        template_update_cfg = cfg.evaluation.get('template_update', {})

        # Determine template update setting with precedence: CLI > nested config > legacy config
        if args.enable_template_update:
            enable_template_update = True
        elif args.no_template_update:
            enable_template_update = False
        elif 'enabled' in template_update_cfg:
            enable_template_update = template_update_cfg['enabled']
        else:
            # Legacy fallback
            enable_template_update = cfg.evaluation.get('enable_template_update', True)

        # Read interval and threshold from nested config with legacy fallback
        template_update_interval = template_update_cfg.get(
            'interval', cfg.evaluation.get('template_update_interval', 10)
        )
        template_update_threshold = template_update_cfg.get(
            'threshold', cfg.evaluation.get('template_update_threshold', 0.7)
        )

        logger.info(f"  Template update: {'ENABLED' if enable_template_update else 'DISABLED'}")
        if enable_template_update:
            logger.info(f"    Interval: {template_update_interval} frames, Threshold: {template_update_threshold}")
        logger.info(f"  Search center: PREVIOUS PREDICTION (standard tracking)")

        # CRITICAL FIX (Audit 2025112702): Use correct config key 'template_area_factor'
        # CRITICAL FIX (Audit 2025120402): Add template quality filtering parameters
        # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Pass validity gating config
        validity_gating_cfg = cfg.evaluation.get('validity_gating', {})

        # CRITICAL FIX (Audit 2025121202 Finding 9): Expected resolutions for fallback
        # Config uses W x H format, convert to (H, W) for evaluator
        expected_sonar_res_cfg = cfg.evaluation.get('expected_sonar_resolution', [682, 376])  # W, H
        expected_sonar_resolution = (expected_sonar_res_cfg[1], expected_sonar_res_cfg[0])  # Convert to H, W
        # RGB uses standard 1920x1080, but allow config override
        expected_rgb_res_cfg = cfg.evaluation.get('expected_rgb_resolution', [1920, 1080])  # W, H
        expected_rgb_resolution = (expected_rgb_res_cfg[1], expected_rgb_res_cfg[0])  # Convert to H, W
        evaluator = OnlineTrackingEvaluator(
            model=model,
            device=device,
            template_size=cfg.data.get('template_size', 128),
            search_size=cfg.data.get('search_size', 256),
            template_area_factor=cfg.data.get('template_area_factor', 2.0),
            search_area_factor=cfg.data.get('search_area_factor', 4.0),
            normalization_config=cfg.data.normalize,
            # CRITICAL FIX (Audit 2025120401 Finding 6): Pass template update config
            enable_template_update=enable_template_update,
            template_update_interval=template_update_interval,
            template_update_threshold=template_update_threshold,
            # CRITICAL FIX (Audit 2025120402 Finding 1-2): Template quality filtering
            template_min_size=cfg.evaluation.get('template_min_size', 5.0),
            template_edge_margin=cfg.evaluation.get('template_edge_margin', 2),
            template_scan_frames=cfg.evaluation.get('template_scan_frames', 20),
            enable_template_quality_filter=cfg.evaluation.get('enable_template_quality_filter', True),
            # CRITICAL FIX (Audit 2025121202 Finding 22): Quality thresholds for template selection
            quality_threshold_rgb=cfg.evaluation.get('quality_threshold_rgb', 0.7),
            quality_threshold_sonar=cfg.evaluation.get('quality_threshold_sonar', 0.7),
            # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Validity gating for online evaluation
            enable_validity_gating=validity_gating_cfg.get('enabled', False),
            rgb_validity_threshold=validity_gating_cfg.get('rgb_threshold', 0.35),
            sonar_validity_threshold=validity_gating_cfg.get('sonar_threshold', 0.40),
            # CRITICAL FIX (Audit 2025121202 Finding 9): Expected resolutions for fallback
            expected_rgb_resolution=expected_rgb_resolution,
            expected_sonar_resolution=expected_sonar_resolution,
            # CRITICAL FIX (Audit 2025121202 Finding 17): Pass metrics list from config
            metrics_list=cfg.evaluation.get('metrics', None)
        )
    else:
        # OFFLINE protocol (ablation only):
        # - Search region: Centered on GT (not realistic deployment scenario)
        # WARNING: This is NOT standard tracking evaluation!
        logger.warning("OFFLINE protocol selected - search centered on GT (NOT realistic!)")
        logger.warning("This mode is for ABLATION ONLY, not for reporting tracking performance")

        # Use OnlineTrackingEvaluator but note this is for consistency
        # Future: could implement a dedicated OfflineEvaluator if needed
        # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Pass validity gating config
        validity_gating_cfg = cfg.evaluation.get('validity_gating', {})

        # CRITICAL FIX (Audit 2025121202 Finding 9): Expected resolutions for fallback
        expected_sonar_res_cfg = cfg.evaluation.get('expected_sonar_resolution', [682, 376])
        expected_sonar_resolution = (expected_sonar_res_cfg[1], expected_sonar_res_cfg[0])
        expected_rgb_res_cfg = cfg.evaluation.get('expected_rgb_resolution', [1920, 1080])
        expected_rgb_resolution = (expected_rgb_res_cfg[1], expected_rgb_res_cfg[0])
        evaluator = OnlineTrackingEvaluator(
            model=model,
            device=device,
            template_size=cfg.data.get('template_size', 128),
            search_size=cfg.data.get('search_size', 256),
            template_area_factor=cfg.data.get('template_area_factor', 2.0),
            search_area_factor=cfg.data.get('search_area_factor', 4.0),
            normalization_config=cfg.data.normalize,
            enable_template_update=False,  # No update for offline
            # CRITICAL FIX (Audit 2025120402 Finding 1-2): Template quality filtering
            template_min_size=cfg.evaluation.get('template_min_size', 5.0),
            template_edge_margin=cfg.evaluation.get('template_edge_margin', 2),
            template_scan_frames=cfg.evaluation.get('template_scan_frames', 20),
            enable_template_quality_filter=cfg.evaluation.get('enable_template_quality_filter', True),
            # CRITICAL FIX (Audit 2025121202 Finding 22): Quality thresholds for template selection
            quality_threshold_rgb=cfg.evaluation.get('quality_threshold_rgb', 0.7),
            quality_threshold_sonar=cfg.evaluation.get('quality_threshold_sonar', 0.7),
            # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Validity gating for offline evaluation
            enable_validity_gating=validity_gating_cfg.get('enabled', False),
            rgb_validity_threshold=validity_gating_cfg.get('rgb_threshold', 0.35),
            sonar_validity_threshold=validity_gating_cfg.get('sonar_threshold', 0.40),
            # CRITICAL FIX (Audit 2025121202 Finding 9): Expected resolutions for fallback
            expected_rgb_resolution=expected_rgb_resolution,
            expected_sonar_resolution=expected_sonar_resolution,
            # CRITICAL FIX (Audit 2025121202 Finding 17): Pass metrics list from config
            metrics_list=cfg.evaluation.get('metrics', None)
        )

    # CRITICAL FIX (Audit 2025121202 Finding 20/28): Output effective configuration summary
    logger.info("="*70)
    logger.info("EFFECTIVE CONFIGURATION SUMMARY (Audit 2025121202 Finding 20/28)")
    logger.info("="*70)

    # Section 1: Settings actively in effect during evaluation
    logger.info("[EVALUATION SETTINGS - IN EFFECT]")
    logger.info(f"  eval_protocol: {eval_protocol}")
    logger.info(f"  batch_size: 1 (ENFORCED - online tracking requirement)")
    logger.info(f"  enable_template_update: {enable_template_update}")
    logger.info(f"  validity_gating.enabled: {validity_gating_cfg.get('enabled', False)}")
    if validity_gating_cfg.get('enabled', False):
        logger.info(f"    rgb_validity_threshold: {validity_gating_cfg.get('rgb_threshold', 0.35)}")
        logger.info(f"    sonar_validity_threshold: {validity_gating_cfg.get('sonar_threshold', 0.40)}")
    logger.info(f"  quality_threshold_rgb: {cfg.evaluation.get('quality_threshold_rgb', 0.7)}")
    logger.info(f"  quality_threshold_sonar: {cfg.evaluation.get('quality_threshold_sonar', 0.7)}")
    logger.info(f"  expected_sonar_resolution: {expected_sonar_res_cfg} (W,H)")
    logger.info(f"  template_scan_frames: {cfg.evaluation.get('template_scan_frames', 20)}")
    logger.info(f"  enable_template_quality_filter: {cfg.evaluation.get('enable_template_quality_filter', True)}")
    logger.info(f"  metrics: {cfg.evaluation.get('metrics', ['success_auc', 'precision_20', 'mean_iou'])}")
    logger.info(f"  save_metrics: {cfg.evaluation.get('save_metrics', True)}")
    logger.info(f"  save_predictions: {cfg.evaluation.get('save_predictions', True)}")
    logger.info(f"  visualization.enabled: {cfg.evaluation.visualization.enabled}")

    # Section 2: Conditional settings (require visualization.enabled)
    create_videos_cfg = cfg.evaluation.visualization.get('create_videos', False)
    if cfg.evaluation.visualization.enabled:
        logger.info("[CONDITIONAL SETTINGS - REQUIRES visualization.enabled=true]")
        logger.info(f"  visualization.create_videos: {create_videos_cfg}")
        if create_videos_cfg:
            logger.info(f"  visualization.video_fps: {cfg.evaluation.visualization.get('video_fps', 10)}")
    elif create_videos_cfg:
        logger.warning("[CONDITIONAL SETTINGS - NOT ACTIVE]")
        logger.warning(f"  visualization.create_videos: {create_videos_cfg} <- IGNORED (visualization.enabled=false)")

    # Section 3: Deprecated/removed settings (config has no effect)
    logger.info("[DEPRECATED - CONFIG HAS NO EFFECT]")
    logger.info("  model.allow_single_modality: REMOVED (use data.modality_dropout instead)")
    logger.info("  model.heads.fusion: REMOVED (FusionHead deprecated, use ConditionalDCMA)")

    logger.info("="*70)

    # Run evaluation
    logger.info("="*70)
    logger.info(f"Starting {eval_protocol.upper()} Tracking Evaluation")
    logger.info("="*70)

    results = evaluator.evaluate_all_sequences(seq_name_to_base_dataset, verbose=True)
    
    # Print results
    logger.info("="*70)
    logger.info("Evaluation Results")
    logger.info("="*70)
    
    # ARCHITECTURE UPDATE (Audit 2025112313): fusion removed (dual-view fusion architecture)
    for modality in ['rgb', 'sonar']:
        if modality in results['overall']:
            metrics = results['overall'][modality]
            logger.info(f"\n{modality.upper()} Metrics:")
            logger.info(f"  Success AUC:      {metrics.get('success_auc_mean', 0.0):.4f}")
            logger.info(f"  Precision@20:     {metrics.get('precision_20_mean', 0.0):.4f}")
            logger.info(f"  Avg Center Error: {metrics.get('avg_center_error_mean', 0.0):.2f} px")
            logger.info(f"  Mean IoU:         {metrics.get('mean_iou_mean', 0.0):.4f}")

    # Save metrics and predictions
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CRITICAL FIX (Audit 2025121202 Finding 19): Handle save_report configuration
    # The save_metrics method includes report generation, but we explicitly check save_report
    # to ensure configuration consistency and transparency
    save_report = cfg.evaluation.get('save_report', True)
    save_metrics = cfg.evaluation.get('save_metrics', True)

    if save_metrics:
        logger.info("\nSaving evaluation metrics...")
        evaluator.save_metrics(cfg.evaluation.output_dir, timestamp=timestamp)
        if save_report:
            logger.info(f"  Report generated in: {cfg.evaluation.output_dir}/metrics/")

    if not save_metrics and save_report:
        # User wants report but not metrics - still generate report via save_metrics
        logger.info("\nGenerating evaluation report...")
        logger.warning("save_report=true but save_metrics=false; generating report anyway")
        evaluator.save_metrics(cfg.evaluation.output_dir, timestamp=timestamp)

    if cfg.evaluation.get('save_predictions', True):
        logger.info("Saving raw predictions...")
        evaluator.save_predictions(cfg.evaluation.output_dir, timestamp=timestamp)

    # Visualization
    if cfg.evaluation.visualization.enabled:
        logger.info("\n"+"="*70)
        logger.info("Creating visualizations")
        logger.info("="*70)

        from aquasot3d.evaluation import compute_success_curve, compute_precision_curve
        import numpy as np

        visualizer = TrackingVisualizer(
            save_dir=cfg.evaluation.visualization.save_dir,
            dpi=cfg.evaluation.visualization.get('dpi', 150)
        )

        # Aggregate predictions across all sequences for curve computation
        all_rgb_pred, all_rgb_gt = [], []
        all_sonar_pred, all_sonar_gt = [], []

        for seq_name, seq_data in results['sequences'].items():
            preds = seq_data.get('predictions', {})
            all_rgb_pred.extend(preds.get('rgb_pred', []))
            all_rgb_gt.extend(preds.get('rgb_gt', []))
            all_sonar_pred.extend(preds.get('sonar_pred', []))
            all_sonar_gt.extend(preds.get('sonar_gt', []))

        # Compute and plot success curves
        if cfg.evaluation.visualization.get('plot_curves', True):
            logger.info("Plotting success and precision curves...")

            rgb_success_curve = None
            sonar_success_curve = None
            rgb_precision_curve = None
            sonar_precision_curve = None

            if all_rgb_pred and all_rgb_gt:
                rgb_success_curve = compute_success_curve(all_rgb_pred, all_rgb_gt)
                rgb_precision_curve = compute_precision_curve(all_rgb_pred, all_rgb_gt)

            if all_sonar_pred and all_sonar_gt:
                sonar_success_curve = compute_success_curve(all_sonar_pred, all_sonar_gt)
                sonar_precision_curve = compute_precision_curve(all_sonar_pred, all_sonar_gt)

            # Plot curves
            visualizer.plot_success_curves(
                rgb_curve=rgb_success_curve,
                sonar_curve=sonar_success_curve,
                save_name='success_curves.png'
            )
            logger.info("  Saved: success_curves.png")

            visualizer.plot_precision_curves(
                rgb_curve=rgb_precision_curve,
                sonar_curve=sonar_precision_curve,
                save_name='precision_curves.png'
            )
            logger.info("  Saved: precision_curves.png")

        # Get best and worst sequences
        best_sequences = evaluator.get_best_sequences(
            n=cfg.evaluation.visualization.get('num_best', 5)
        )
        worst_sequences = evaluator.get_worst_sequences(
            n=cfg.evaluation.visualization.get('num_worst', 5)
        )

        logger.info(f"\nTop {len(best_sequences)} sequences:")
        for name, auc in best_sequences:
            logger.info(f"  {name}: {auc:.4f}")

        logger.info(f"\nBottom {len(worst_sequences)} sequences:")
        for name, auc in worst_sequences:
            logger.info(f"  {name}: {auc:.4f}")

        # Frame-level visualization with bounding boxes on ORIGINAL full-resolution images
        # CRITICAL FIX: Previously used search crop (256x256), now uses original images
        if cfg.evaluation.visualization.get('save_failure_cases', True):
            logger.info("\nGenerating frame visualizations on ORIGINAL images...")

            from tqdm import tqdm

            # Visualization output scale (for speed optimization)
            # 0.5 = half resolution, 1.0 = original resolution
            vis_scale = cfg.evaluation.visualization.get('output_scale', 0.5)

            seq_items = list(results['sequences'].items())
            for seq_name, seq_data in tqdm(seq_items, desc="Sequences", unit="seq"):
                seq_vis_dir = Path(cfg.evaluation.visualization.save_dir) / 'frames' / seq_name
                seq_vis_dir.mkdir(parents=True, exist_ok=True)

                preds = seq_data.get('predictions', {})
                rgb_preds = preds.get('rgb_pred', [])
                sonar_preds = preds.get('sonar_pred', [])
                # CRITICAL FIX: Use GT from predictions (already in original coordinates)
                rgb_gts = preds.get('rgb_gt', [])
                sonar_gts = preds.get('sonar_gt', [])

                # Get base dataset for this sequence to load original images
                if seq_name not in seq_name_to_base_dataset:
                    logger.warning(f"No base dataset found for sequence '{seq_name}', skipping visualization")
                    continue

                base_dataset = seq_name_to_base_dataset[seq_name]
                num_frames = min(len(rgb_preds), base_dataset.num_frames)

                # Get sequence info once for placeholder dimensions
                seq_info = base_dataset.get_sequence_info()
                rgb_w, rgb_h = seq_info['rgb_resolution']  # (width, height)
                sonar_w, sonar_h = seq_info['sonar_resolution']  # (width, height)

                # Helper to filter out occlusion marker [0,0,0,0]
                def is_valid_bbox(bbox):
                    """Check if bbox is valid (not None and not all zeros)."""
                    if bbox is None:
                        return False
                    bbox_arr = np.asarray(bbox)
                    return not np.all(bbox_arr == 0)

                # Helper to scale image and bbox for visualization output
                def scale_image_and_bbox(img, bbox, scale_factor):
                    """Scale image and bbox by scale_factor (e.g., 0.5 = half size)."""
                    if scale_factor is None or scale_factor <= 0 or scale_factor >= 1.0:
                        return img, bbox
                    cur_h, cur_w = img.shape[:2]
                    new_h = int(cur_h * scale_factor)
                    new_w = int(cur_w * scale_factor)
                    img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    if bbox is not None:
                        bbox_scaled = np.array(bbox) * scale_factor
                    else:
                        bbox_scaled = None
                    return img_scaled, bbox_scaled

                seq_frames_combined = []

                def _validate_bbox_coords(bbox, img_shape, name):
                    """Raise if bbox is outside reasonable bounds of the image."""
                    if bbox is None:
                        return
                    h_img, w_img = img_shape[:2]
                    x, y, w, h = bbox
                    if not np.isfinite([x, y, w, h]).all():
                        raise ValueError(f"{seq_name} frame {frame_idx}: {name} has NaN/Inf bbox {bbox}")
                    if w <= 0 or h <= 0:
                        raise ValueError(f"{seq_name} frame {frame_idx}: {name} has non-positive size {bbox}")
                    # Allow small tolerance outside image (up to 50%)
                    if (x + w) < -0.5 * w_img or x > 1.5 * w_img or (y + h) < -0.5 * h_img or y > 1.5 * h_img:
                        raise ValueError(
                            f"{seq_name} frame {frame_idx}: {name} bbox {bbox} is outside image bounds "
                            f"({w_img}x{h_img}) - check coordinate space."
                        )

                for frame_idx in tqdm(range(num_frames), desc=f"  {seq_name}", unit="frame", leave=False):
                    # Load ORIGINAL full-resolution images from base dataset
                    frame_data = base_dataset._get_frame_data(frame_idx)

                    # Get RGB image (original resolution, e.g., 1920x1080)
                    # NOTE: _load_image already converts BGR to RGB, so no conversion needed
                    rgb_img = frame_data['rgb_img']
                    if rgb_img is None:
                        # Create placeholder for occluded frames
                        rgb_img = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)

                    # Get Sonar image (original resolution, e.g., 682x376)
                    # NOTE: _load_image returns sonar as (H, W, 1), need to convert to RGB
                    sonar_img = frame_data['sonar_img']
                    if sonar_img is not None:
                        # Convert grayscale (H, W, 1) to RGB for visualization
                        if len(sonar_img.shape) == 3 and sonar_img.shape[2] == 1:
                            sonar_img = cv2.cvtColor(sonar_img.squeeze(-1), cv2.COLOR_GRAY2RGB)
                        elif len(sonar_img.shape) == 2:
                            sonar_img = cv2.cvtColor(sonar_img, cv2.COLOR_GRAY2RGB)
                    else:
                        # Create placeholder for occluded frames
                        sonar_img = np.zeros((sonar_h, sonar_w, 3), dtype=np.uint8)

                    # Get predictions (already in ORIGINAL image coordinates)
                    rgb_pred = rgb_preds[frame_idx] if frame_idx < len(rgb_preds) else None
                    sonar_pred = sonar_preds[frame_idx] if frame_idx < len(sonar_preds) else None

                    # Get GT (already in ORIGINAL image coordinates from evaluator)
                    rgb_gt = rgb_gts[frame_idx] if frame_idx < len(rgb_gts) else None
                    sonar_gt = sonar_gts[frame_idx] if frame_idx < len(sonar_gts) else None

                    # Filter out occlusion markers [0,0,0,0]
                    # When GT is occlusion marker, also skip pred (no target to evaluate)
                    rgb_gt_valid = is_valid_bbox(rgb_gt)
                    sonar_gt_valid = is_valid_bbox(sonar_gt)

                    rgb_pred = rgb_pred if (rgb_gt_valid and is_valid_bbox(rgb_pred)) else None
                    sonar_pred = sonar_pred if (sonar_gt_valid and is_valid_bbox(sonar_pred)) else None
                    rgb_gt = rgb_gt if rgb_gt_valid else None
                    sonar_gt = sonar_gt if sonar_gt_valid else None

                    # Downscale images and bboxes for faster visualization output
                    # This does NOT affect evaluation metrics (computed in original coordinates)
                    rgb_img, rgb_pred = scale_image_and_bbox(rgb_img, rgb_pred, vis_scale)
                    rgb_img, rgb_gt = scale_image_and_bbox(rgb_img, rgb_gt, None)  # img already scaled
                    if rgb_gt is not None and vis_scale and 0 < vis_scale < 1.0:
                        rgb_gt = np.array(rgb_gt) * vis_scale

                    sonar_img, sonar_pred = scale_image_and_bbox(sonar_img, sonar_pred, vis_scale)
                    sonar_img, sonar_gt = scale_image_and_bbox(sonar_img, sonar_gt, None)  # img already scaled
                    if sonar_gt is not None and vis_scale and 0 < vis_scale < 1.0:
                        sonar_gt = np.array(sonar_gt) * vis_scale

                    # Draw RGB frame with predictions and GT
                    _validate_bbox_coords(rgb_pred, rgb_img.shape, "RGB pred")
                    _validate_bbox_coords(rgb_gt, rgb_img.shape, "RGB GT")
                    rgb_vis = visualizer.draw_tracking_result(
                        rgb_img.copy(),
                        pred_box=rgb_pred,
                        gt_box=rgb_gt,
                        frame_id=frame_idx,
                        pred_format='xywh',
                        gt_format='xywh'
                    )

                    # Draw Sonar frame with predictions and GT
                    _validate_bbox_coords(sonar_pred, sonar_img.shape, "Sonar pred")
                    _validate_bbox_coords(sonar_gt, sonar_img.shape, "Sonar GT")
                    sonar_vis = visualizer.draw_tracking_result(
                        sonar_img.copy(),
                        pred_box=sonar_pred,
                        gt_box=sonar_gt,
                        frame_id=frame_idx,
                        pred_format='xywh',
                        gt_format='xywh'
                    )

                    # Resize sonar to match RGB height for side-by-side display
                    vis_rgb_h_cur, vis_rgb_w_cur = rgb_vis.shape[:2]
                    vis_sonar_h_cur, vis_sonar_w_cur = sonar_vis.shape[:2]
                    if vis_sonar_h_cur != vis_rgb_h_cur:
                        scale = vis_rgb_h_cur / vis_sonar_h_cur
                        new_sonar_w = int(vis_sonar_w_cur * scale)
                        sonar_vis = cv2.resize(sonar_vis, (new_sonar_w, vis_rgb_h_cur), interpolation=cv2.INTER_LINEAR)

                    # Combine RGB and Sonar side by side
                    combined = np.hstack([rgb_vis, sonar_vis])
                    seq_frames_combined.append(combined)

                    # Save frame
                    frame_path = seq_vis_dir / f'frame_{frame_idx:04d}.png'
                    cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

                # Create video if frames exist
                if cfg.evaluation.visualization.get('create_videos', False) and len(seq_frames_combined) > 0:
                    video_path = visualizer.create_video_from_frames(
                        seq_frames_combined,
                        f'{seq_name}_tracking.mp4',
                        fps=cfg.evaluation.visualization.get('video_fps', 10)
                    )
                    if video_path:
                        logger.info(f"  Video saved: {video_path}")

                logger.info(f"  Saved {len(seq_frames_combined)} frames for {seq_name}")

    logger.info("\n"+"="*70)
    logger.info("Evaluation completed successfully")
    logger.info(f"Results saved to: {cfg.evaluation.output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
