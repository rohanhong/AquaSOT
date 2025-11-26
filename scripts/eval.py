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
from aquasot3d.evaluation import TrackingEvaluator, TrackingVisualizer

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
        # Multi-sequence evaluation mode
        data_root = cfg.data.get('data_root', 'data')
        test_sequences = cfg.data.test_sequences

        # Defensive check: ensure test_sequences is a list
        if isinstance(test_sequences, str):
            test_sequences = [test_sequences]

        logger.info(f"Multi-sequence evaluation mode: data_root={data_root}")
        logger.info(f"Test sequences: {test_sequences}")

        # Build evaluation dataset for each sequence
        eval_datasets = []
        total_frames = 0

        for seq_name in test_sequences:
            # CRITICAL FIX (Audit 2025112504 Finding 2):
            # Create base dataset with return_occluded=True to include all frames for evaluation,
            # but EvaluationSequenceDataset will use template_visibility_policy='any' to ensure
            # template is selected from frames with at least one modality visible
            seq_base_dataset = RGBSonarDataset(
                data_root=data_root,
                sequence_name=seq_name,
                return_occluded=True  # Include all frames for evaluation (even 0000)
            )

            # CRITICAL FIX (Audit 2025111703 Finding 4): Read template_update config
            template_update_enabled = cfg.evaluation.get('template_update', {}).get('enabled', False)
            template_update_interval = cfg.evaluation.get('template_update', {}).get('interval', 10)

            # CRITICAL FIX (Audit 2025112504 Finding 2): Read template_visibility_policy
            # Default to 'any' to prevent template selection from 0000 frames
            template_visibility_policy = cfg.evaluation.get('template', {}).get('visibility_policy', 'any')

            # Create evaluation dataset for this sequence
            # CRITICAL FIX (Audit 2025112601): Pass context_factor for bbox-based template cropping
            seq_eval_dataset = EvaluationSequenceDataset(
                base_dataset=seq_base_dataset,
                template_size=cfg.data.get('template_size', 128),  # OSTrack-256 standard
                search_size=cfg.data.get('search_size', 256),      # OSTrack-256 standard
                template_frame='first',
                enable_update=template_update_enabled,
                update_interval=template_update_interval,
                normalization_config=cfg.data.normalize,
                template_visibility_policy=template_visibility_policy,
                context_factor=cfg.data.get('context_factor', 2.0)
            )

            eval_datasets.append(seq_eval_dataset)
            seq_info = seq_base_dataset.get_sequence_info()
            total_frames += len(seq_eval_dataset)

            logger.info(
                f"  Sequence '{seq_name}': {len(seq_eval_dataset)} frames, "
                f"RGB {seq_info['rgb_resolution']}, Sonar {seq_info['sonar_resolution']}"
            )

        # Concatenate all sequence datasets
        eval_dataset = ConcatDataset(eval_datasets)

        logger.info(f"Total evaluation frames across {len(test_sequences)} sequences: {total_frames}")
        logger.info(f"Using sequential evaluation (template=first frame per sequence)")
        if template_update_enabled:
            logger.info(f"Template update: ENABLED (interval={template_update_interval} frames)")
        else:
            logger.info(f"Template update: DISABLED (fixed template)")

    else:
        # Legacy single-sequence evaluation mode (backward compatibility)
        logger.info("Single-sequence evaluation mode (legacy)")

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

        # CRITICAL FIX (Audit 2025112504 Finding 2):
        # Create base dataset with return_occluded=True to include all frames for evaluation,
        # but EvaluationSequenceDataset will use template_visibility_policy='any' to ensure
        # template is selected from frames with at least one modality visible
        eval_base_dataset = RGBSonarDataset(
            data_root=data_root,
            sequence_name=sequence_name,
            return_occluded=True  # Include all frames for evaluation (even 0000)
        )

        # CRITICAL FIX (Audit 2025111703 Finding 4): Read template_update config
        template_update_enabled = cfg.evaluation.get('template_update', {}).get('enabled', False)
        template_update_interval = cfg.evaluation.get('template_update', {}).get('interval', 10)

        # CRITICAL FIX (Audit 2025112504 Finding 2): Read template_visibility_policy
        # Default to 'any' to prevent template selection from 0000 frames
        template_visibility_policy = cfg.evaluation.get('template', {}).get('visibility_policy', 'any')

        # Use EvaluationSequenceDataset for sequential evaluation (not random sampling)
        # This ensures reproducible, protocol-compliant evaluation
        # CRITICAL FIX (Audit 2025112601): Pass context_factor for bbox-based template cropping
        eval_dataset = EvaluationSequenceDataset(
            base_dataset=eval_base_dataset,
            template_size=cfg.data.get('template_size', 128),  # OSTrack-256 standard
            search_size=cfg.data.get('search_size', 256),      # OSTrack-256 standard
            template_frame='first',  # Use first frame as template (RGBS50 protocol)
            enable_update=template_update_enabled,
            update_interval=template_update_interval,
            normalization_config=cfg.data.normalize,
            template_visibility_policy=template_visibility_policy,
            context_factor=cfg.data.get('context_factor', 2.0)
        )

        logger.info(f"Evaluation dataset: {len(eval_dataset)} frames")
        logger.info(f"Using sequential evaluation (template=first frame)")
        if template_update_enabled:
            logger.info(f"Template update: ENABLED (interval={template_update_interval} frames)")
        else:
            logger.info(f"Template update: DISABLED (fixed template)")

    # Create dataloader
    eval_loader = create_dataloader(
        dataset=eval_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,  # CRITICAL: No shuffle for evaluation
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn_template_search,
        distributed=False
    )

    logger.info(f"Evaluation batches: {len(eval_loader)}")

    # Parse additional metrics from config
    additional_metrics = []
    if hasattr(cfg.evaluation, 'metrics') and cfg.evaluation.metrics:
        # Extract non-standard metrics from config
        standard_metrics = {'success_auc', 'precision_20', 'avg_center_error', 'mean_iou', 'success_0.5'}
        for metric_name in cfg.evaluation.metrics:
            if metric_name not in standard_metrics:
                additional_metrics.append(metric_name)

    logger.info(f"Standard metrics will be computed")
    if additional_metrics:
        logger.info(f"Additional metrics: {additional_metrics}")

    # Create evaluator
    evaluator = TrackingEvaluator(
        model=model,
        dataloader=eval_loader,
        device=device,
        output_dir=cfg.evaluation.output_dir,
        save_predictions=cfg.evaluation.save_predictions,
        verbose=True,
        search_size=cfg.data.get('search_size', 256),
        additional_metrics=additional_metrics
    )
    
    # Run evaluation
    logger.info("="*70)
    logger.info("Starting evaluation")
    logger.info("="*70)
    
    results = evaluator.evaluate()
    
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

        # Frame-level visualization with bounding boxes
        if cfg.evaluation.visualization.get('save_failure_cases', True):
            logger.info("\nGenerating frame visualizations with bounding boxes...")

            # Denormalization parameters (ImageNet for RGB, 0.5 for Sonar)
            rgb_mean = np.array(cfg.data.normalize.get('rgb_mean', [0.485, 0.456, 0.406]))
            rgb_std = np.array(cfg.data.normalize.get('rgb_std', [0.229, 0.224, 0.225]))
            sonar_mean = cfg.data.normalize.get('sonar_mean', [0.5])[0]
            sonar_std = cfg.data.normalize.get('sonar_std', [0.5])[0]

            for seq_name, seq_data in results['sequences'].items():
                seq_vis_dir = Path(cfg.evaluation.visualization.save_dir) / 'frames' / seq_name
                seq_vis_dir.mkdir(parents=True, exist_ok=True)

                preds = seq_data.get('predictions', {})
                rgb_preds = preds.get('rgb_pred', [])
                sonar_preds = preds.get('sonar_pred', [])
                rgb_gts = preds.get('rgb_gt', [])
                sonar_gts = preds.get('sonar_gt', [])

                # Re-iterate through dataset to get frames for this sequence
                seq_frames_rgb = []
                seq_frames_sonar = []

                for i, sample in enumerate(eval_dataset):
                    if sample.get('sequence_name', '') != seq_name:
                        continue
                    if i >= len(rgb_preds):
                        break

                    # Denormalize RGB image
                    rgb_img = sample['rgb_search'].numpy()  # [3, H, W]
                    rgb_img = rgb_img.transpose(1, 2, 0)  # [H, W, 3]
                    rgb_img = rgb_img * rgb_std + rgb_mean
                    rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)

                    # Denormalize Sonar image
                    sonar_img = sample['sonar_search'].numpy()  # [1, H, W]
                    sonar_img = sonar_img.squeeze(0)  # [H, W]
                    sonar_img = sonar_img * sonar_std + sonar_mean
                    sonar_img = (sonar_img * 255).clip(0, 255).astype(np.uint8)
                    sonar_img = cv2.cvtColor(sonar_img, cv2.COLOR_GRAY2RGB)

                    # Get predictions and GT (in letterbox space for visualization)
                    rgb_pred = preds.get('rgb_pred', [None])[i] if i < len(preds.get('rgb_pred', [])) else None
                    sonar_pred = preds.get('sonar_pred', [None])[i] if i < len(preds.get('sonar_pred', [])) else None
                    rgb_gt_pixels = sample.get('search_bbox_rgb_pixels')
                    sonar_gt_pixels = sample.get('search_bbox_sonar_pixels')

                    # Convert GT to numpy if tensor
                    if rgb_gt_pixels is not None and hasattr(rgb_gt_pixels, 'numpy'):
                        rgb_gt_pixels = rgb_gt_pixels.numpy()
                    if sonar_gt_pixels is not None and hasattr(sonar_gt_pixels, 'numpy'):
                        sonar_gt_pixels = sonar_gt_pixels.numpy()

                    # Transform predictions to letterbox space for visualization
                    rgb_meta = sample.get('rgb_search_metadata', {})
                    sonar_meta = sample.get('sonar_search_metadata', {})

                    def transform_to_letterbox(bbox_orig, meta):
                        if bbox_orig is None:
                            return None
                        x, y, w, h = bbox_orig
                        scale = meta.get('scale', 1.0)
                        pad_left = meta.get('pad_left', 0)
                        pad_top = meta.get('pad_top', 0)
                        return np.array([
                            x * scale + pad_left,
                            y * scale + pad_top,
                            w * scale,
                            h * scale
                        ])

                    rgb_pred_lb = transform_to_letterbox(rgb_pred, rgb_meta)
                    sonar_pred_lb = transform_to_letterbox(sonar_pred, sonar_meta)

                    # Draw RGB frame
                    rgb_vis = visualizer.draw_tracking_result(
                        rgb_img.copy(),
                        pred_box=rgb_pred_lb,
                        gt_box=rgb_gt_pixels,
                        frame_id=i,
                        pred_format='xywh',
                        gt_format='xywh'
                    )

                    # Draw Sonar frame
                    sonar_vis = visualizer.draw_tracking_result(
                        sonar_img.copy(),
                        pred_box=sonar_pred_lb,
                        gt_box=sonar_gt_pixels,
                        frame_id=i,
                        pred_format='xywh',
                        gt_format='xywh'
                    )

                    # Combine RGB and Sonar side by side
                    combined = np.hstack([rgb_vis, sonar_vis])
                    seq_frames_rgb.append(combined)

                    # Save frame
                    frame_path = seq_vis_dir / f'frame_{i:04d}.png'
                    cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

                # Create video if frames exist
                if cfg.evaluation.visualization.get('create_videos', False) and len(seq_frames_rgb) > 0:
                    video_path = visualizer.create_video_from_frames(
                        seq_frames_rgb,
                        f'{seq_name}_tracking.mp4',
                        fps=cfg.evaluation.visualization.get('video_fps', 10)
                    )
                    if video_path:
                        logger.info(f"  Video saved: {video_path}")

                logger.info(f"  Saved {len(seq_frames_rgb)} frames for {seq_name}")

    logger.info("\n"+"="*70)
    logger.info("Evaluation completed successfully")
    logger.info(f"Results saved to: {cfg.evaluation.output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
