#!/usr/bin/env python3
"""
Demo Inference Script for AquaSOT-3D

Run real-time tracking on a sequence with visualization.
Reuses OnlineTrackingEvaluator for consistent behavior with training/evaluation.

CRITICAL FIX (Audit 2025120401 Finding 8, 11): Fully implemented demo script
that reuses OnlineTrackingEvaluator for correct preprocessing, metadata passing,
and output key handling.

Usage:
    # Run on a sequence from the dataset
    python scripts/demo.py --checkpoint outputs/train/checkpoints/best.pth \\
        --sequence uuv1 --data-root data

    # Save visualization video
    python scripts/demo.py --checkpoint outputs/train/checkpoints/best.pth \\
        --sequence uuv1 --output demo_output.mp4

    # With config file for model architecture
    python scripts/demo.py --checkpoint outputs/train/checkpoints/best.pth \\
        --sequence uuv1 --config configs/eval.yaml

Author: AquaSOT-3D Team
Date: 2025-11-10
Updated: 2025-12-04 (Audit 2025120401 Finding 11 - Complete rewrite)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import argparse
import torch
import cv2
import numpy as np
import logging
from tqdm import tqdm
from typing import Tuple

from aquasot3d.models.tracker import AquaSOT3DTracker
from aquasot3d.dataloader import RGBSonarDataset
from aquasot3d.evaluation import OnlineTrackingEvaluator
from aquasot3d.utils.config import load_config, setup_config_logging, _config_to_dict

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='AquaSOT-3D Demo - Online Tracking Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with a sequence
    python scripts/demo.py --checkpoint best.pth --sequence uuv1

    # Save visualization video
    python scripts/demo.py --checkpoint best.pth --sequence uuv1 --output tracking.mp4

    # Use config file for model architecture
    python scripts/demo.py --checkpoint best.pth --sequence uuv1 --config configs/eval.yaml
"""
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Sequence name to process (e.g., uuv1)')

    # Data paths
    parser.add_argument('--data-root', type=str, default='data',
                        help='Root directory containing sequences (default: data)')
    parser.add_argument('--config', type=str, default='configs/eval.yaml',
                        help='Config file for model architecture (default: configs/eval.yaml)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path. If not provided, displays live window')
    parser.add_argument('--output-scale', type=float, default=0.5,
                        help='Scale factor for output (default: 0.5 = half resolution)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Output video FPS (default: 10)')

    # Display options
    parser.add_argument('--no-display', action='store_true',
                        help='Disable live display (useful for headless servers)')
    parser.add_argument('--show-confidence', action='store_true',
                        help='Show confidence scores on visualization')

    # Tracking options
    parser.add_argument('--enable-template-update', action='store_true', default=True,
                        help='Enable dynamic template update (default: True)')
    parser.add_argument('--no-template-update', action='store_true',
                        help='Disable template update')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (default: cuda)')

    return parser.parse_args()


def build_model(cfg, checkpoint_path: str, device: str) -> AquaSOT3DTracker:
    """Build and load model from checkpoint"""
    logger.info("Building model from config")

    # Prepare backbone configs
    rgb_backbone_cfg = _config_to_dict(cfg.model.backbone.rgb) if cfg.model.backbone.rgb.get('type') not in [None, 'null'] else None
    sonar_backbone_cfg = _config_to_dict(cfg.model.backbone.sonar) if cfg.model.backbone.sonar.get('type') not in [None, 'null'] else None

    # Prepare fusion config
    fusion_cfg = _config_to_dict(cfg.model.fusion) if cfg.model.fusion.get('type') not in [None, 'null'] else None

    # Prepare head config
    def clean_head_config(head_cfg):
        if not head_cfg or head_cfg.get('type') in [None, 'null']:
            return None
        cfg_dict = _config_to_dict(head_cfg)
        cfg_dict.pop('type', None)
        cfg_dict.pop('in_channels', None)
        cfg_dict.pop('rgb_channels', None)
        cfg_dict.pop('sonar_channels', None)
        return cfg_dict

    box_head_cfg = clean_head_config(cfg.model.heads.get('rgb_box', {}))

    # Build model
    model = AquaSOT3DTracker(
        rgb_backbone_config=rgb_backbone_cfg,
        sonar_backbone_config=sonar_backbone_cfg,
        fusion_config=fusion_cfg,
        box_head_config=box_head_cfg
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(f"Missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys: {len(load_result.unexpected_keys)}")

    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def draw_bbox(image: np.ndarray, bbox: np.ndarray, color: Tuple[int, int, int],
              label: str = None, thickness: int = 2) -> np.ndarray:
    """Draw bounding box on image"""
    if bbox is None or np.sum(np.abs(bbox)) == 0:
        return image

    x, y, w, h = bbox.astype(int)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(image, (x, y - text_h - 5), (x + text_w + 5, y), color, -1)
        cv2.putText(image, label, (x + 2, y - 3), font, font_scale, (255, 255, 255), 1)

    return image


def main():
    """Main demo function"""
    args = parse_args()

    # Setup logging
    setup_config_logging('INFO')

    logger.info("=" * 70)
    logger.info("AquaSOT-3D Demo - Online Tracking")
    logger.info("=" * 70)

    # Load config
    cfg = load_config(args.config)
    device = args.device

    # Build model
    model = build_model(cfg, args.checkpoint, device)

    # Load sequence
    logger.info(f"Loading sequence: {args.sequence} from {args.data_root}")
    dataset = RGBSonarDataset(
        data_root=args.data_root,
        sequence_name=args.sequence,
        return_occluded=True
    )
    logger.info(f"Sequence loaded: {dataset.num_frames} frames")

    # Determine template update setting
    enable_template_update = args.enable_template_update and not args.no_template_update

    # Create OnlineTrackingEvaluator for consistent behavior
    # CRITICAL FIX (Audit 2025120401 Finding 11):
    # Reuse OnlineTrackingEvaluator to ensure:
    # - Correct preprocessing (normalization, padding values)
    # - Correct metadata passing to GeometryGatedAttention
    # - Correct output key handling (rgb_boxes, sonar_boxes)
    # - Consistent behavior with training/evaluation
    evaluator = OnlineTrackingEvaluator(
        model=model,
        device=device,
        template_size=cfg.data.get('template_size', 128),
        search_size=cfg.data.get('search_size', 256),
        template_area_factor=cfg.data.get('template_area_factor', 2.0),
        search_area_factor=cfg.data.get('search_area_factor', 4.0),
        normalization_config=cfg.data.normalize,
        enable_template_update=enable_template_update,
        template_update_interval=cfg.evaluation.get('template_update_interval', 10),
        template_update_threshold=cfg.evaluation.get('template_update_threshold', 0.7)
    )

    logger.info(f"Template update: {'ENABLED' if enable_template_update else 'DISABLED'}")

    # Run evaluation to get predictions
    logger.info("Running tracking...")
    results = evaluator.evaluate_sequence(dataset, verbose=True)

    # Extract predictions
    rgb_preds = results.get('rgb_pred', [])
    sonar_preds = results.get('sonar_pred', [])
    rgb_gts = results.get('rgb_gt', [])
    sonar_gts = results.get('sonar_gt', [])

    logger.info(f"Tracking complete: {len(rgb_preds)} frames processed")

    # Create visualization
    logger.info("Creating visualization...")

    # Get sequence info for frame dimensions
    seq_info = dataset.get_sequence_info()
    rgb_w, rgb_h = seq_info['rgb_resolution']
    sonar_w, sonar_h = seq_info['sonar_resolution']

    # Setup video writer if output specified
    video_writer = None
    if args.output:
        out_rgb_w = int(rgb_w * args.output_scale)
        out_rgb_h = int(rgb_h * args.output_scale)
        scale_sonar = out_rgb_h / sonar_h
        out_sonar_w = int(sonar_w * scale_sonar)

        frame_width = out_rgb_w + out_sonar_w
        frame_height = out_rgb_h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (frame_width, frame_height))
        logger.info(f"Video writer initialized: {frame_width}x{frame_height} @ {args.fps}fps")

    # Generate visualization frames
    num_frames = min(len(rgb_preds), dataset.num_frames)

    frame_iter = range(num_frames)
    if not args.no_display or args.output:
        frame_iter = tqdm(frame_iter, desc="Visualizing", unit="frame")

    for frame_idx in frame_iter:
        # Load original images
        frame_data = dataset._get_frame_data(frame_idx)
        rgb_img = frame_data['rgb_img']
        sonar_img = frame_data['sonar_img']

        # Handle None images
        if rgb_img is None:
            rgb_img = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)
        if sonar_img is None:
            sonar_img = np.zeros((sonar_h, sonar_w), dtype=np.uint8)

        # Convert sonar to RGB for visualization
        if sonar_img.ndim == 2:
            sonar_vis = cv2.cvtColor(sonar_img, cv2.COLOR_GRAY2RGB)
        elif sonar_img.ndim == 3 and sonar_img.shape[2] == 1:
            sonar_vis = cv2.cvtColor(sonar_img.squeeze(-1), cv2.COLOR_GRAY2RGB)
        else:
            sonar_vis = sonar_img.copy()

        rgb_vis = rgb_img.copy()

        # Get predictions and GT
        rgb_pred = rgb_preds[frame_idx] if frame_idx < len(rgb_preds) else None
        sonar_pred = sonar_preds[frame_idx] if frame_idx < len(sonar_preds) else None
        rgb_gt = rgb_gts[frame_idx] if frame_idx < len(rgb_gts) else None
        sonar_gt = sonar_gts[frame_idx] if frame_idx < len(sonar_gts) else None

        # Check for valid boxes (not occlusion marker 0,0,0,0)
        def is_valid(bbox):
            return bbox is not None and np.sum(np.abs(bbox)) > 0

        # Draw GT (green)
        if is_valid(rgb_gt):
            rgb_vis = draw_bbox(rgb_vis, rgb_gt, (0, 255, 0), "GT")
        if is_valid(sonar_gt):
            sonar_vis = draw_bbox(sonar_vis, sonar_gt, (0, 255, 0), "GT")

        # Draw predictions (red)
        if is_valid(rgb_pred):
            rgb_vis = draw_bbox(rgb_vis, rgb_pred, (255, 0, 0), "Pred")
        if is_valid(sonar_pred):
            sonar_vis = draw_bbox(sonar_vis, sonar_pred, (255, 0, 0), "Pred")

        # Scale for output
        out_rgb_w = int(rgb_w * args.output_scale)
        out_rgb_h = int(rgb_h * args.output_scale)
        rgb_vis = cv2.resize(rgb_vis, (out_rgb_w, out_rgb_h))

        # Scale sonar to match RGB height
        scale_sonar = out_rgb_h / sonar_h
        out_sonar_w = int(sonar_w * scale_sonar)
        sonar_vis = cv2.resize(sonar_vis, (out_sonar_w, out_rgb_h))

        # Combine side by side
        combined = np.hstack([rgb_vis, sonar_vis])

        # Add frame info
        cv2.putText(combined, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write to video
        if video_writer:
            video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        # Display
        if not args.no_display:
            cv2.imshow("AquaSOT-3D Demo", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                logger.info("User interrupted")
                break

    # Cleanup
    if video_writer:
        video_writer.release()
        logger.info(f"Video saved to: {args.output}")

    if not args.no_display:
        cv2.destroyAllWindows()

    logger.info("Demo completed")


if __name__ == '__main__':
    main()
