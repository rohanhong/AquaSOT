"""
Visualization Tools for AquaSOT-3D

This module provides visualization utilities for tracking results including
bounding box overlays, trajectory plotting, attention maps, and failure analysis.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)


class TrackingVisualizer:
    """
    Visualizer for tracking results.

    This class provides methods to visualize tracking predictions, ground truth,
    trajectories, and attention maps.

    Args:
        save_dir: Directory to save visualizations
        dpi: DPI for saved figures
        font_size: Font size for labels

    Example:
        >>> visualizer = TrackingVisualizer(save_dir='outputs/vis')
        >>> visualizer.draw_tracking_result(frame, pred_box, gt_box)
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        dpi: int = 100,
        font_size: int = 12
    ):
        self.save_dir = Path(save_dir) if save_dir else None
        self.dpi = dpi
        self.font_size = font_size

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'pred': (0, 255, 0),      # Green for predictions
            'gt': (255, 0, 0),        # Red for ground truth
            'fusion': (255, 255, 0),  # Yellow for fusion
            'rgb': (0, 255, 255),     # Cyan for RGB
            'sonar': (255, 0, 255)    # Magenta for Sonar
        }

    def draw_tracking_result(
        self,
        frame: np.ndarray,
        pred_box: Optional[np.ndarray] = None,
        gt_box: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        save_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw tracking result on a frame.

        Args:
            frame: Input frame (H, W, 3) or (H, W)
            pred_box: Predicted bounding box [x, y, w, h] or [cx, cy, w, h]
            gt_box: Ground truth bounding box
            frame_id: Frame number (for annotation)
            save_name: Filename to save (optional)

        Returns:
            Annotated frame (always RGB format)

        Example:
            >>> vis_frame = visualizer.draw_tracking_result(frame, pred_box, gt_box, frame_id=0)
        """
        # Copy frame to avoid modifying original
        vis_frame = frame.copy()

        # Convert grayscale to RGB if needed (high-generalization image format handling)
        if len(vis_frame.shape) == 2:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2RGB)

        # Draw ground truth (red)
        if gt_box is not None:
            vis_frame = self._draw_bbox(vis_frame, gt_box, self.colors['gt'], label='GT')

        # Draw prediction (green)
        if pred_box is not None:
            vis_frame = self._draw_bbox(vis_frame, pred_box, self.colors['pred'], label='Pred')

        # Add frame ID
        if frame_id is not None:
            cv2.putText(
                vis_frame,
                f'Frame: {frame_id}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            cv2.imwrite(str(save_path), vis_frame)

        return vis_frame

    def draw_multi_modal_result(
        self,
        rgb_frame: np.ndarray,
        sonar_frame: np.ndarray,
        rgb_pred: Optional[np.ndarray] = None,
        sonar_pred: Optional[np.ndarray] = None,
        fusion_pred: Optional[np.ndarray] = None,
        rgb_gt: Optional[np.ndarray] = None,
        sonar_gt: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        save_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw multi-modal tracking results side-by-side.

        Args:
            rgb_frame: RGB frame
            sonar_frame: Sonar frame
            rgb_pred: RGB prediction
            sonar_pred: Sonar prediction
            fusion_pred: Fusion prediction
            rgb_gt: RGB ground truth
            sonar_gt: Sonar ground truth
            frame_id: Frame ID
            save_name: Save filename

        Returns:
            Combined visualization frame
        """
        # Draw on RGB frame
        rgb_vis = rgb_frame.copy()
        if len(rgb_vis.shape) == 2:
            rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_GRAY2RGB)
        if rgb_gt is not None:
            rgb_vis = self._draw_bbox(rgb_vis, rgb_gt, self.colors['gt'], label='GT')
        if rgb_pred is not None:
            rgb_vis = self._draw_bbox(rgb_vis, rgb_pred, self.colors['rgb'], label='RGB')
        if fusion_pred is not None:
            rgb_vis = self._draw_bbox(rgb_vis, fusion_pred, self.colors['fusion'], label='Fusion', thickness=2)

        # Draw on Sonar frame
        sonar_vis = sonar_frame.copy()
        if len(sonar_vis.shape) == 2:
            sonar_vis = cv2.cvtColor(sonar_vis, cv2.COLOR_GRAY2RGB)
        if sonar_gt is not None:
            sonar_vis = self._draw_bbox(sonar_vis, sonar_gt, self.colors['gt'], label='GT')
        if sonar_pred is not None:
            sonar_vis = self._draw_bbox(sonar_vis, sonar_pred, self.colors['sonar'], label='Sonar')
        if fusion_pred is not None:
            sonar_vis = self._draw_bbox(sonar_vis, fusion_pred, self.colors['fusion'], label='Fusion', thickness=2)

        # Add labels
        cv2.putText(rgb_vis, 'RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(sonar_vis, 'Sonar', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if frame_id is not None:
            cv2.putText(rgb_vis, f'Frame: {frame_id}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Concatenate horizontally
        combined = np.hstack([rgb_vis, sonar_vis])

        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            cv2.imwrite(str(save_path), combined)

        return combined

    def plot_trajectory_3d(
        self,
        centers_3d: List[np.ndarray],
        gt_centers_3d: Optional[List[np.ndarray]] = None,
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot 3D center trajectory.

        Args:
            centers_3d: List of predicted 3D centers (X, Y, Z)
            gt_centers_3d: List of ground truth 3D centers (optional)
            save_name: Filename to save

        Returns:
            Matplotlib figure

        Example:
            >>> fig = visualizer.plot_trajectory_3d(pred_centers, gt_centers)
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Convert to numpy arrays
        if len(centers_3d) > 0:
            pred_centers = np.array(centers_3d)
            ax.plot(
                pred_centers[:, 0],
                pred_centers[:, 1],
                pred_centers[:, 2],
                'g-',
                linewidth=2,
                label='Predicted',
                marker='o',
                markersize=4
            )

        if gt_centers_3d is not None and len(gt_centers_3d) > 0:
            gt_centers = np.array(gt_centers_3d)
            ax.plot(
                gt_centers[:, 0],
                gt_centers[:, 1],
                gt_centers[:, 2],
                'r-',
                linewidth=2,
                label='Ground Truth',
                marker='s',
                markersize=4
            )

        ax.set_xlabel('X (right) [m]', fontsize=self.font_size)
        ax.set_ylabel('Y (down) [m]', fontsize=self.font_size)
        ax.set_zlabel('Z (forward) [m]', fontsize=self.font_size)
        ax.set_title('3D Trajectory', fontsize=self.font_size + 2, fontweight='bold')
        ax.legend(fontsize=self.font_size)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_success_curves(
        self,
        rgb_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sonar_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fusion_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot success curves for all modalities.

        Args:
            rgb_curve: (thresholds, success_rates) for RGB (optional)
            sonar_curve: (thresholds, success_rates) for Sonar (optional)
            fusion_curve: (thresholds, success_rates) for Fusion (optional)
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curves (only if provided)
        if rgb_curve is not None:
            ax.plot(rgb_curve[0], rgb_curve[1], 'c-', linewidth=2, label='RGB')
        if sonar_curve is not None:
            ax.plot(sonar_curve[0], sonar_curve[1], 'm-', linewidth=2, label='Sonar')
        if fusion_curve is not None:
            ax.plot(fusion_curve[0], fusion_curve[1], 'y-', linewidth=3, label='Fusion')

        ax.set_xlabel('Overlap Threshold', fontsize=self.font_size)
        ax.set_ylabel('Success Rate', fontsize=self.font_size)
        ax.set_title('Success Plot', fontsize=self.font_size + 2, fontweight='bold')
        ax.legend(fontsize=self.font_size)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_precision_curves(
        self,
        rgb_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sonar_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fusion_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot precision curves for all modalities.

        Args:
            rgb_curve: (thresholds, precisions) for RGB (optional)
            sonar_curve: (thresholds, precisions) for Sonar (optional)
            fusion_curve: (thresholds, precisions) for Fusion (optional)
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curves (only if provided)
        if rgb_curve is not None:
            ax.plot(rgb_curve[0], rgb_curve[1], 'c-', linewidth=2, label='RGB')
        if sonar_curve is not None:
            ax.plot(sonar_curve[0], sonar_curve[1], 'm-', linewidth=2, label='Sonar')
        if fusion_curve is not None:
            ax.plot(fusion_curve[0], fusion_curve[1], 'y-', linewidth=3, label='Fusion')

        ax.set_xlabel('Location Error Threshold (pixels)', fontsize=self.font_size)
        ax.set_ylabel('Precision', fontsize=self.font_size)
        ax.set_title('Precision Plot', fontsize=self.font_size + 2, fontweight='bold')
        ax.legend(fontsize=self.font_size)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()

        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def visualize_attention_map(
        self,
        frame: np.ndarray,
        attention_map: np.ndarray,
        save_name: Optional[str] = None,
        alpha: float = 0.5,
        pred_box: Optional[np.ndarray] = None,
        gt_box: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize attention map overlaid on frame.

        Args:
            frame: Input frame (H, W, 3) or (H, W)
            attention_map: Attention weights (H, W)
            save_name: Filename to save
            alpha: Transparency for attention overlay (0-1)
            pred_box: Optional prediction box to draw
            gt_box: Optional ground truth box to draw

        Returns:
            Visualization frame with attention overlay (high-generalization implementation)
        """
        # Convert frame to RGB for visualization (high-generalization format handling)
        if len(frame.shape) == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            vis_frame = frame.copy()

        # Resize attention map if needed
        if attention_map.shape[:2] != frame.shape[:2]:
            attention_map = cv2.resize(attention_map, (frame.shape[1], frame.shape[0]))

        # Normalize attention map to 0-255
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        attention_normalized = (attention_normalized * 255).astype(np.uint8)

        # Apply colormap (jet)
        attention_colored = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)

        # Blend with original frame
        result = cv2.addWeighted(vis_frame, 1 - alpha, attention_colored, alpha, 0)

        # Draw bounding boxes if provided
        if gt_box is not None:
            result = self._draw_bbox(result, gt_box, self.colors['gt'], label='GT')
        if pred_box is not None:
            result = self._draw_bbox(result, pred_box, self.colors['pred'], label='Pred')

        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            cv2.imwrite(str(save_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        return result

    def create_failure_case_report(
        self,
        sequence_name: str,
        frames: List[np.ndarray],
        pred_boxes: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        metrics: Dict[str, float],
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Create failure case analysis visualization.

        Args:
            sequence_name: Name of the sequence
            frames: List of frames
            pred_boxes: Predicted boxes
            gt_boxes: Ground truth boxes
            metrics: Metrics dictionary
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        # Select key frames (start, middle, end, and worst frame)
        num_frames = len(frames)
        frame_indices = [0, num_frames // 2, num_frames - 1]

        # Find worst frame (highest center error)
        from aquasot3d.evaluation.metrics import compute_center_error
        errors = [compute_center_error(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]
        worst_idx = int(np.argmax(errors))
        if worst_idx not in frame_indices:
            frame_indices.append(worst_idx)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for i, idx in enumerate(frame_indices):
            if i >= 4:
                break

            # Draw tracking result
            vis_frame = self.draw_tracking_result(
                frames[idx],
                pred_boxes[idx],
                gt_boxes[idx]
            )

            # Display
            axes[i].imshow(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'Frame {idx} (Error: {errors[idx]:.2f}px)', fontsize=self.font_size)
            axes[i].axis('off')

        # Add title with metrics
        title = f'Failure Analysis: {sequence_name}\n'
        title += f"AUC: {metrics.get('success_auc', 0):.4f}, "
        title += f"Precision@20: {metrics.get('precision_20', 0):.4f}, "
        title += f"Avg Error: {metrics.get('avg_center_error', 0):.2f}px"
        fig.suptitle(title, fontsize=self.font_size + 4, fontweight='bold')

        plt.tight_layout()

        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        color: Tuple[int, int, int],
        label: str = '',
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding box on frame.

        Args:
            frame: Input frame
            bbox: Bounding box in [x, y, w, h] format (top-left corner + size)
            color: BGR color tuple
            label: Label text
            thickness: Line thickness

        Returns:
            Frame with bounding box

        Note:
            This function expects xywh format (top-left corner + width/height),
            which is the standard format from groundtruth.txt files.
        """
        # Convert bbox to xyxy format
        if len(bbox) == 4:
            # xywh format (top-left corner + width + height)
            x, y, w, h = bbox
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
        else:
            return frame

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if label:
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_y = max(y1, label_size[1] + 10)
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0], label_y),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

        return frame

    def visualize_batch_results(
        self,
        rgb_frames: List[np.ndarray],
        sonar_frames: List[np.ndarray],
        predictions: Dict[str, List[np.ndarray]],
        save_frames: bool = False,
        prefix: str = 'frame'
    ) -> List[np.ndarray]:
        """
        Visualize batch of multi-modal tracking results.

        Args:
            rgb_frames: List of RGB frames
            sonar_frames: List of sonar frames
            predictions: Dictionary of predictions with keys:
                - 'rgb_pred': RGB predictions
                - 'sonar_pred': Sonar predictions
                - 'fusion_pred': Fusion predictions
                - 'rgb_gt': RGB ground truth (optional)
                - 'sonar_gt': Sonar ground truth (optional)
            save_frames: Whether to save individual frames
            prefix: Prefix for saved frame filenames

        Returns:
            List of visualization frames
        """
        results = []
        num_frames = len(rgb_frames)

        for i in range(num_frames):
            # Get predictions for this frame
            rgb_pred = predictions.get('rgb_pred', [None] * num_frames)[i]
            sonar_pred = predictions.get('sonar_pred', [None] * num_frames)[i]
            fusion_pred = predictions.get('fusion_pred', [None] * num_frames)[i]
            rgb_gt = predictions.get('rgb_gt', [None] * num_frames)[i]
            sonar_gt = predictions.get('sonar_gt', [None] * num_frames)[i]

            # Draw multi-modal result
            vis_frame = self.draw_multi_modal_result(
                rgb_frames[i],
                sonar_frames[i],
                rgb_pred=rgb_pred,
                sonar_pred=sonar_pred,
                fusion_pred=fusion_pred,
                rgb_gt=rgb_gt,
                sonar_gt=sonar_gt,
                frame_id=i,
                save_name=f'{prefix}_{i:04d}.png' if save_frames else None
            )
            results.append(vis_frame)

        return results

    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_filename: str,
        fps: int = 30
    ) -> Optional[str]:
        """
        Create video from list of frames.

        Args:
            frames: List of frames (H, W, 3)
            output_filename: Output video filename (relative to save_dir)
            fps: Frames per second

        Returns:
            Output path if successful, None otherwise
        """
        if len(frames) == 0:
            logger.warning("No frames to save")
            return None

        if not self.save_dir:
            logger.warning("No save_dir specified, cannot create video")
            return None

        output_path = self.save_dir / output_filename
        height, width = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            writer.write(frame_bgr)

        writer.release()
        logger.info(f"Saved video to: {output_path}")
        return str(output_path)


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
):
    """
    Create video from list of frames.

    Args:
        frames: List of frames (H, W, 3)
        output_path: Output video file path
        fps: Frames per second

    Example:
        >>> create_video_from_frames(vis_frames, 'tracking_result.mp4', fps=30)
    """
    if len(frames) == 0:
        logger.warning("No frames to save")
        return

    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
    logger.info(f"Saved video to: {output_path}")
