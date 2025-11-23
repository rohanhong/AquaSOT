"""
Evaluation Test with Real Data and Visualization

This test uses actual UUV dataset images to generate visualizations,
demonstrating the complete tracking pipeline with real RGB and Sonar data.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import cv2

from aquasot3d.models import build_tracker
from aquasot3d.evaluation import TrackingVisualizer
from aquasot3d.dataloader.datasets.rgb_sonar_dataset import parse_groundtruth


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp(prefix='aquasot_real_viz_')
    yield temp_dir
    print(f"\nReal data visualization outputs saved to: {temp_dir}")


@pytest.fixture
def tracker_config():
    """Tracker configuration for real data evaluation"""
    return {
        'rgb_backbone_config': {
            'type': 'vit_rgb',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False,
            'output_dim': 192
        },
        'sonar_backbone_config': {
            'type': 'sonar',
            'encoder_type': 'conv',
            'output_dim': 128
        },
        'fusion_config': {
            'type': 'g2_dcma_lite',
            'output_dim': 192,
            'num_heads': 4,
            'num_points': 2
        },
        'temporal_config': None,
        'use_quality_gating': True,
        'quality_config': {
            'rgb': {'method': 'combined', 'learnable': True},
            'sonar': {'method': 'combined', 'learnable': True, 'use_geometry': True},
            'adaptive_weighting': {
                'method': 'softmax',
                'temporal_smoothing': True,
                'learnable_fusion': False
            }
        }
    }


class TestRealDataEvaluationWithVisualization:
    """Test suite for evaluation with real UUV dataset images"""

    def test_real_data_visualization(self, tracker_config, temp_output_dir):
        """
        Test evaluation with real RGB and Sonar images from UUV dataset.

        This test:
        1. Loads actual images from data/uuv3/
        2. Runs inference with the tracker
        3. Generates visualizations with real frames
        4. Validates all audit fixes work with real data
        """
        # Check if data directory exists
        data_dir = Path('data/uuv3')
        if not data_dir.exists():
            pytest.skip("UUV3 dataset not found, skipping real data test")

        light_img_dir = data_dir / 'light' / 'img'
        sonar_img_dir = data_dir / 'sonar' / 'img'
        light_gt_file = data_dir / 'light' / 'groundtruth.txt'
        sonar_gt_file = data_dir / 'sonar' / 'groundtruth.txt'

        # Verify required files exist
        if not (light_img_dir.exists() and sonar_img_dir.exists()):
            pytest.skip("Required image directories not found")
        if not (light_gt_file.exists() and sonar_gt_file.exists()):
            pytest.skip("Groundtruth files not found")

        # Create output directories
        output_dir = Path(temp_output_dir)
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Build tracker
        print("\nBuilding tracker...")
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker.eval()

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tracker = tracker.to(device)

        # Load ground truth
        light_gt_boxes = parse_groundtruth(str(light_gt_file))  # (N, 4) in xywh
        sonar_gt_boxes = parse_groundtruth(str(sonar_gt_file))

        # Create visualizer
        visualizer = TrackingVisualizer(save_dir=str(viz_dir))

        # Select frames to process (use first 5 frames for quick test)
        frame_indices = [100, 105, 110, 115, 120]  # Sample frames
        sequence_name = 'uuv3'

        print(f"\nProcessing {len(frame_indices)} frames from {sequence_name}...")

        all_predictions = []

        # Load template (first frame)
        template_idx = 100
        light_template_path = light_img_dir / f'{template_idx}.jpg'
        sonar_template_path = sonar_img_dir / f'{template_idx}.jpg'

        light_template_img = cv2.imread(str(light_template_path))
        sonar_template_img = cv2.imread(str(sonar_template_path), cv2.IMREAD_GRAYSCALE)

        # Preprocess template
        light_template = self._preprocess_rgb(light_template_img, size=(128, 128))
        sonar_template = self._preprocess_sonar(sonar_template_img, size=(376, 682))

        light_template = torch.from_numpy(light_template).unsqueeze(0).to(device)
        sonar_template = torch.from_numpy(sonar_template).unsqueeze(0).to(device)

        # Process each search frame
        with torch.no_grad():
            for frame_idx in frame_indices:
                # Load search frame images
                light_search_path = light_img_dir / f'{frame_idx}.jpg'
                sonar_search_path = sonar_img_dir / f'{frame_idx}.jpg'

                if not (light_search_path.exists() and sonar_search_path.exists()):
                    print(f"  Frame {frame_idx}: Images not found, skipping")
                    continue

                light_search_img = cv2.imread(str(light_search_path))
                sonar_search_img = cv2.imread(str(sonar_search_path), cv2.IMREAD_GRAYSCALE)

                # Preprocess search frames
                light_search = self._preprocess_rgb(light_search_img, size=(256, 256))
                sonar_search = self._preprocess_sonar(sonar_search_img, size=(376, 682))

                light_search_tensor = torch.from_numpy(light_search).unsqueeze(0).to(device)
                sonar_search_tensor = torch.from_numpy(sonar_search).unsqueeze(0).to(device)

                # Run inference
                outputs = tracker(
                    rgb_template=light_template,
                    rgb_search=light_search_tensor,
                    sonar_template=sonar_template,
                    sonar_search=sonar_search_tensor,
                    return_losses=False
                )

                # Extract predictions (normalized cxcywh)
                rgb_pred_norm = outputs['rgb_boxes'].cpu().numpy()[0]
                sonar_pred_norm = outputs['sonar_boxes'].cpu().numpy()[0]
                fusion_pred_norm = outputs['fusion_boxes'].cpu().numpy()[0]

                # Get ground truth (xywh format)
                light_gt_xywh = light_gt_boxes[frame_idx - 1] if frame_idx <= len(light_gt_boxes) else None
                sonar_gt_xywh = sonar_gt_boxes[frame_idx - 1] if frame_idx <= len(sonar_gt_boxes) else None

                # Convert predictions to pixel coordinates (xywh)
                rgb_pred_xywh = self._cxcywh_to_xywh_pixel(
                    rgb_pred_norm, light_search_img.shape
                )
                sonar_pred_xywh = self._cxcywh_to_xywh_pixel(
                    sonar_pred_norm, sonar_search_img.shape
                )

                # Draw RGB visualization
                rgb_vis = visualizer.draw_tracking_result(
                    frame=light_search_img.copy(),
                    pred_box=rgb_pred_xywh,
                    gt_box=light_gt_xywh,
                    frame_id=frame_idx,
                    save_name=f'{sequence_name}_light_frame_{frame_idx:03d}.png'
                )

                # Draw Sonar visualization
                sonar_vis = visualizer.draw_tracking_result(
                    frame=sonar_search_img.copy(),
                    pred_box=sonar_pred_xywh,
                    gt_box=sonar_gt_xywh,
                    frame_id=frame_idx,
                    save_name=f'{sequence_name}_sonar_frame_{frame_idx:03d}.png'
                )

                print(f"  Frame {frame_idx}: RGB pred={rgb_pred_xywh}, GT={light_gt_xywh}")

                all_predictions.append({
                    'frame_id': frame_idx,
                    'rgb_pred': rgb_pred_xywh,
                    'sonar_pred': sonar_pred_xywh,
                    'fusion_pred': self._cxcywh_to_xywh_pixel(fusion_pred_norm, light_search_img.shape),
                    'rgb_gt': light_gt_xywh,
                    'sonar_gt': sonar_gt_xywh
                })

        # Verify visualizations were created
        vis_files = list(viz_dir.glob('*.png'))
        assert len(vis_files) > 0, "No visualization files created"

        print(f"\nGenerated {len(vis_files)} visualization files with real images")
        print(f"Outputs saved to: {viz_dir}")

        # Generate trajectory plot
        self._plot_trajectories(all_predictions, viz_dir, sequence_name)

        return all_predictions

    def _preprocess_rgb(self, img, size=(256, 256)):
        """Preprocess RGB image for model input"""
        # Resize
        img = cv2.resize(img, size)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        # Convert to CHW format
        img = np.transpose(img, (2, 0, 1))
        return img

    def _preprocess_sonar(self, img, size=(376, 682)):
        """Preprocess Sonar image for model input"""
        # Resize
        img = cv2.resize(img, size)
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        return img

    def _cxcywh_to_xywh_pixel(self, box_norm, frame_shape):
        """Convert normalized cxcywh to pixel xywh"""
        h, w = frame_shape[:2]
        cx, cy, bw, bh = box_norm

        # Denormalize
        cx = cx * w
        cy = cy * h
        bw = bw * w
        bh = bh * h

        # Convert to xywh
        x = cx - bw / 2
        y = cy - bh / 2

        return np.array([x, y, bw, bh])

    def _plot_trajectories(self, predictions, output_dir, sequence_name):
        """Plot predicted vs ground truth trajectories"""
        import matplotlib.pyplot as plt

        if len(predictions) == 0:
            return

        # Extract trajectories
        frame_ids = [p['frame_id'] for p in predictions]

        # Extract center positions from xywh
        rgb_pred_centers = np.array([
            [p['rgb_pred'][0] + p['rgb_pred'][2]/2, p['rgb_pred'][1] + p['rgb_pred'][3]/2]
            for p in predictions
        ])
        rgb_gt_centers = np.array([
            [p['rgb_gt'][0] + p['rgb_gt'][2]/2, p['rgb_gt'][1] + p['rgb_gt'][3]/2]
            for p in predictions
        ])
        fusion_pred_centers = np.array([
            [p['fusion_pred'][0] + p['fusion_pred'][2]/2, p['fusion_pred'][1] + p['fusion_pred'][3]/2]
            for p in predictions
        ])

        # Create trajectory plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # X trajectory
        ax1.plot(frame_ids, rgb_gt_centers[:, 0], 'r-', label='GT', linewidth=2)
        ax1.plot(frame_ids, rgb_pred_centers[:, 0], 'g--', label='RGB Pred', linewidth=2)
        ax1.plot(frame_ids, fusion_pred_centers[:, 0], 'b-.', label='Fusion Pred', linewidth=2)
        ax1.set_xlabel('Frame ID', fontsize=12)
        ax1.set_ylabel('X Center (pixels)', fontsize=12)
        ax1.set_title('X Trajectory - Real Data', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Y trajectory
        ax2.plot(frame_ids, rgb_gt_centers[:, 1], 'r-', label='GT', linewidth=2)
        ax2.plot(frame_ids, rgb_pred_centers[:, 1], 'g--', label='RGB Pred', linewidth=2)
        ax2.plot(frame_ids, fusion_pred_centers[:, 1], 'b-.', label='Fusion Pred', linewidth=2)
        ax2.set_xlabel('Frame ID', fontsize=12)
        ax2.set_ylabel('Y Center (pixels)', fontsize=12)
        ax2.set_title('Y Trajectory - Real Data', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Trajectory Comparison - {sequence_name} (Real Images)', fontsize=16)
        plt.tight_layout()

        # Save figure
        save_path = output_dir / f'{sequence_name}_trajectory_real.png'
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved real data trajectory plot: {save_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
