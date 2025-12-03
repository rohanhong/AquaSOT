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

from aquasot3d.utils.config import load_config, save_config, setup_config_logging
from aquasot3d.dataloader import create_dataloader
from aquasot3d.models.tracker import AquaSOT3DTracker
from aquasot3d.evaluation import TrackingEvaluator, TrackingVisualizer, OnlineTrackingEvaluator

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
    parser.add_argument('--enable-template-update', action='store_true', default=None,
                        help='Enable dynamic template update during online tracking')
    parser.add_argument('--no-template-update', action='store_true', default=None,
                        help='Disable template update (use first frame only)')

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

    # Create output directories
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

        # Determine template update setting
        if args.enable_template_update:
            enable_template_update = True
        elif args.no_template_update:
            enable_template_update = False
        else:
            # Default from config or True
            enable_template_update = cfg.evaluation.get('enable_template_update', True)

        logger.info(f"  Template update: {'ENABLED' if enable_template_update else 'DISABLED'}")
        logger.info(f"  Search center: PREVIOUS PREDICTION (standard tracking)")

        # CRITICAL FIX (Audit 2025112702): Use correct config key 'template_area_factor'
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
            template_update_interval=cfg.evaluation.get('template_update_interval', 10),
            template_update_threshold=cfg.evaluation.get('template_update_threshold', 0.7)
        )
    else:
        # OFFLINE protocol (ablation only):
        # - Search region: Centered on GT (not realistic deployment scenario)
        # WARNING: This is NOT standard tracking evaluation!
        logger.warning("OFFLINE protocol selected - search centered on GT (NOT realistic!)")
        logger.warning("This mode is for ABLATION ONLY, not for reporting tracking performance")

        # Use OnlineTrackingEvaluator but note this is for consistency
        # Future: could implement a dedicated OfflineEvaluator if needed
        evaluator = OnlineTrackingEvaluator(
            model=model,
            device=device,
            template_size=cfg.data.get('template_size', 128),
            search_size=cfg.data.get('search_size', 256),
            template_area_factor=cfg.data.get('template_area_factor', 2.0),
            search_area_factor=cfg.data.get('search_area_factor', 4.0),
            normalization_config=cfg.data.normalize,
            enable_template_update=False  # No update for offline
        )

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

    if cfg.evaluation.get('save_metrics', True):
        logger.info("\nSaving evaluation metrics...")
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
                    rgb_vis = visualizer.draw_tracking_result(
                        rgb_img.copy(),
                        pred_box=rgb_pred,
                        gt_box=rgb_gt,
                        frame_id=frame_idx,
                        pred_format='xywh',
                        gt_format='xywh'
                    )

                    # Draw Sonar frame with predictions and GT
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
