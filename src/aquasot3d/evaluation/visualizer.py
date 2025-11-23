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

from ..utils.bbox_utils import convert_bbox_format

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
            'rgb': (0, 255, 255),     # Cyan for RGB
            'sonar': (255, 0, 255)    # Magenta for Sonar
        }

    def draw_tracking_result(
        self,
        frame: np.ndarray,
        pred_box: Optional[np.ndarray] = None,
        gt_box: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        save_name: Optional[str] = None,
        pred_format: str = 'xywh',
        gt_format: str = 'xywh'
    ) -> np.ndarray:
        """
        Draw tracking result on a frame.

        ENHANCEMENT (Audit 2025111405 Requirement #2): Added format parameters to explicitly
        specify bbox coordinate format. Supports auto-detection of normalized coordinates.

        Args:
            frame: Input frame (H, W, 3) or (H, W)
            pred_box: Predicted bounding box in specified format (pixel or normalized coords)
            gt_box: Ground truth bounding box in specified format (pixel or normalized coords)
            frame_id: Frame number (for annotation)
            save_name: Filename to save (optional)
            pred_format: Format of pred_box - 'xywh', 'cxcywh', or 'xyxy'. Default: 'xywh'.
            gt_format: Format of gt_box - 'xywh', 'cxcywh', or 'xyxy'. Default: 'xywh'.

        Returns:
            Annotated frame (always RGB format)

        Example:
            >>> # Groundtruth in pixel xywh format
            >>> vis = visualizer.draw_tracking_result(frame, pred_box, gt_box, frame_id=0)
            >>>
            >>> # Model prediction in normalized cxcywh format
            >>> vis = visualizer.draw_tracking_result(
            ...     frame, pred_box, gt_box,
            ...     pred_format='cxcywh'  # Will auto-detect and denormalize
            ... )

        Note:
            - Bboxes with values in [0,1] are auto-detected as normalized and converted to pixels
            - A warning is logged when auto-detection occurs
            - To suppress warnings, explicitly denormalize before passing to this function
        """
        # Copy frame to avoid modifying original
        vis_frame = frame.copy()

        # Convert grayscale to RGB if needed (high-generalization image format handling)
        if len(vis_frame.shape) == 2:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2RGB)

        # Draw ground truth (red) with explicit format
        if gt_box is not None:
            vis_frame = self._draw_bbox(vis_frame, gt_box, self.colors['gt'], label='GT', format=gt_format)

        # Draw prediction (green) with explicit format
        if pred_box is not None:
            vis_frame = self._draw_bbox(vis_frame, pred_box, self.colors['pred'], label='Pred', format=pred_format)

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
        rgb_gt: Optional[np.ndarray] = None,
        sonar_gt: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        save_name: Optional[str] = None,
        bbox_format: str = 'xywh'
    ) -> np.ndarray:
        """
        Draw multi-modal tracking results side-by-side.

        ENHANCEMENT (Audit 2025111405 Requirement #2): Added bbox_format parameter
        to explicitly specify coordinate format for all bboxes.

        ARCHITECTURE UPDATE (Audit 2025112313): Fusion removed (dual-view fusion
        architecture). RGB and Sonar predictions now use dual-view fused features.

        Args:
            rgb_frame: RGB frame
            sonar_frame: Sonar frame
            rgb_pred: RGB prediction (uses rgb_view_fused features)
            sonar_pred: Sonar prediction (uses sonar_view_fused features)
            rgb_gt: RGB ground truth
            sonar_gt: Sonar ground truth
            frame_id: Frame ID
            save_name: Save filename
            bbox_format: Format for all bboxes - 'xywh', 'cxcywh', or 'xyxy'. Default: 'xywh'.

        Returns:
            Combined visualization frame
        """
        # Draw on RGB frame
        rgb_vis = rgb_frame.copy()
        if len(rgb_vis.shape) == 2:
            rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_GRAY2RGB)
        if rgb_gt is not None:
            rgb_vis = self._draw_bbox(rgb_vis, rgb_gt, self.colors['gt'], label='GT', format=bbox_format)
        if rgb_pred is not None:
            rgb_vis = self._draw_bbox(rgb_vis, rgb_pred, self.colors['rgb'], label='RGB', format=bbox_format)

        # Draw on Sonar frame
        sonar_vis = sonar_frame.copy()
        if len(sonar_vis.shape) == 2:
            sonar_vis = cv2.cvtColor(sonar_vis, cv2.COLOR_GRAY2RGB)
        if sonar_gt is not None:
            sonar_vis = self._draw_bbox(sonar_vis, sonar_gt, self.colors['gt'], label='GT', format=bbox_format)
        if sonar_pred is not None:
            sonar_vis = self._draw_bbox(sonar_vis, sonar_pred, self.colors['sonar'], label='Sonar', format=bbox_format)

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
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot success curves for RGB and Sonar modalities.

        ARCHITECTURE UPDATE (Audit 2025112313): Fusion removed (dual-view fusion).

        Args:
            rgb_curve: (thresholds, success_rates) for RGB (optional)
            sonar_curve: (thresholds, success_rates) for Sonar (optional)
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
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot precision curves for RGB and Sonar modalities.

        ARCHITECTURE UPDATE (Audit 2025112313): Fusion removed (dual-view fusion).

        Args:
            rgb_curve: (thresholds, precisions) for RGB (optional)
            sonar_curve: (thresholds, precisions) for Sonar (optional)
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
        gt_box: Optional[np.ndarray] = None,
        bbox_format: str = 'xywh'
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
            bbox_format: Format for bboxes - 'xywh', 'cxcywh', or 'xyxy'. Default: 'xywh'.

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

        # Draw bounding boxes if provided with explicit format
        if gt_box is not None:
            result = self._draw_bbox(result, gt_box, self.colors['gt'], label='GT', format=bbox_format)
        if pred_box is not None:
            result = self._draw_bbox(result, pred_box, self.colors['pred'], label='Pred', format=bbox_format)

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
        thickness: int = 2,
        format: str = 'xywh',
        auto_detect_normalized: bool = True
    ) -> np.ndarray:
        """
        Draw bounding box on frame with automatic format conversion and coordinate space detection.

        ENHANCEMENT (Audit 2025111405 Requirement #2): This function now auto-detects normalized
        coordinates and logs warnings when format mismatches are detected. This prevents silent
        failures when passing model outputs (normalized cxcywh) without specifying format.

        Args:
            frame: Input frame
            bbox: Bounding box in specified format (pixel or normalized coordinates)
            color: BGR color tuple
            label: Label text
            thickness: Line thickness
            format: Bbox format - 'xywh' (top-left + size), 'cxcywh' (center + size),
                    or 'xyxy' (corners). Default: 'xywh' for backward compatibility.
            auto_detect_normalized: If True, automatically detect and convert normalized
                    coordinates [0,1] to pixel coordinates. Default: True.

        Returns:
            Frame with bounding box

        Example:
            >>> # Model output in normalized cxcywh format (auto-detected)
            >>> vis = self._draw_bbox(frame, pred_norm_cxcywh, color, format='cxcywh')
            >>> # Groundtruth in pixel xywh format
            >>> vis = self._draw_bbox(frame, gt_xywh, color, format='xywh')

        Note:
            When auto_detect_normalized=True and bbox values are in [0,1] range,
            they are automatically denormalized to pixel coordinates using frame shape.
            A warning is logged to alert about implicit coordinate space conversion.
        """
        if len(bbox) != 4:
            return frame

        # Auto-detect normalized coordinates (values in [0, 1] range)
        bbox_copy = bbox.copy() if isinstance(bbox, np.ndarray) else np.array(bbox)
        is_normalized = auto_detect_normalized and np.all((bbox_copy >= 0) & (bbox_copy <= 1.5))

        if is_normalized:
            # Bbox appears to be normalized - denormalize to pixel coordinates
            img_h, img_w = frame.shape[:2]
            logger.warning(
                f"Auto-detected normalized bbox coordinates (values in [0,1]). "
                f"Converting to pixel space using frame shape ({img_h}x{img_w}). "
                f"Original bbox: {bbox_copy[:4]}, format: {format}. "
                f"To avoid this warning, explicitly denormalize bboxes before visualization "
                f"or pass pixel coordinates directly."
            )

            # Denormalize based on current format
            from ..utils.bbox_utils import denormalize_bbox
            # Reshape to 2D if needed (denormalize_bbox expects shape (N, 4))
            is_1d = bbox_copy.ndim == 1
            if is_1d:
                bbox_copy = bbox_copy.reshape(1, -1)
            bbox_copy = denormalize_bbox(bbox_copy, (img_h, img_w), format=format)
            if is_1d:
                bbox_copy = bbox_copy.flatten()

        # Convert to xywh format if needed
        if format != 'xywh':
            try:
                # Reshape to 2D if needed (convert_bbox_format expects shape (N, 4))
                is_1d_convert = bbox_copy.ndim == 1
                if is_1d_convert:
                    bbox_copy = bbox_copy.reshape(1, -1)
                bbox_copy = convert_bbox_format(bbox_copy, src_format=format, dst_format='xywh')
                if is_1d_convert:
                    bbox_copy = bbox_copy.flatten()
            except Exception as e:
                logger.warning(f"Failed to convert bbox from {format} to xywh: {e}")
                return frame

        # Extract xywh coordinates
        x, y, w, h = bbox_copy
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

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
            rgb_gt = predictions.get('rgb_gt', [None] * num_frames)[i]
            sonar_gt = predictions.get('sonar_gt', [None] * num_frames)[i]

            # Draw multi-modal result
            vis_frame = self.draw_multi_modal_result(
                rgb_frames[i],
                sonar_frames[i],
                rgb_pred=rgb_pred,
                sonar_pred=sonar_pred,
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

    @staticmethod
    def estimate_3d_position(
        rgb_box: Optional[np.ndarray] = None,
        sonar_box: Optional[np.ndarray] = None,
        rgb_fov_h: float = 47.0,
        rgb_fov_v: float = 27.0,
        sonar_fov_h: float = 130.0,
        default_depth: float = 5.0
    ) -> Optional[np.ndarray]:
        """
        Simple post-processing function to estimate 3D position from 2D predictions.

        This function provides a rough 3D coordinate estimate for visualization purposes
        only. It does NOT participate in training or model feedback.

        NOTE: Without extrinsic calibration, 3D estimates are approximate and should
        only be used for qualitative visualization, not quantitative evaluation.

        Args:
            rgb_box: RGB bounding box in CXCYWH format, normalized [0,1] (cx, cy, w, h)
            sonar_box: Sonar bounding box in CXCYWH format, normalized [0,1] (cx, cy, w, h)
            rgb_fov_h: RGB camera horizontal FOV in degrees (default: 47.0)
            rgb_fov_v: RGB camera vertical FOV in degrees (default: 27.0)
            sonar_fov_h: Sonar horizontal FOV in degrees (default: 130.0)
            default_depth: Default depth in meters when only one modality is available (default: 5.0)

        Returns:
            Estimated 3D position [x, y, depth] in meters, or None if insufficient data
            - x: horizontal displacement (left-right)
            - y: vertical displacement (up-down)
            - depth: forward distance from sensors

        Method:
            1. Convert normalized 2D centers to angular displacements using FOV
            2. Estimate depth from bbox size (larger = closer, inverse relationship)
            3. Calculate X and Y using depth and angles
            4. Fuse RGB and Sonar estimates with simple averaging

        Example:
            >>> rgb_box = np.array([0.5, 0.5, 0.2, 0.15])  # Center, 20% width, 15% height
            >>> sonar_box = np.array([0.5, 0.6, 0.25, 0.3])
            >>> pos_3d = TrackingVisualizer.estimate_3d_position(rgb_box, sonar_box)
            >>> print(f"Estimated position: x={pos_3d[0]:.2f}m, y={pos_3d[1]:.2f}m, depth={pos_3d[2]:.2f}m")
        """
        if rgb_box is None and sonar_box is None:
            return None

        # Convert FOV to radians
        rgb_fov_h_rad = np.deg2rad(rgb_fov_h)
        rgb_fov_v_rad = np.deg2rad(rgb_fov_v)
        sonar_fov_h_rad = np.deg2rad(sonar_fov_h)

        estimates = []

        # Process RGB estimate
        if rgb_box is not None:
            cx, cy, w, h = rgb_box

            # Estimate depth from bbox size (inverse relationship: larger = closer)
            # Assume object is roughly 1m in size, depth inversely proportional to bbox area
            bbox_area = w * h  # Normalized area [0, 1]
            if bbox_area > 0:
                depth_rgb = 1.0 / np.sqrt(bbox_area)  # Rough depth estimate
            else:
                depth_rgb = default_depth

            # Convert normalized center to angular displacement
            # cx=0.5 means centered (angle=0), cx=0 means left edge, cx=1 means right edge
            angle_h_rgb = (cx - 0.5) * rgb_fov_h_rad  # Horizontal angle
            angle_v_rgb = (0.5 - cy) * rgb_fov_v_rad  # Vertical angle (inverted: y=0 is top)

            # Calculate X, Y from depth and angles
            x_rgb = depth_rgb * np.tan(angle_h_rgb)
            y_rgb = depth_rgb * np.tan(angle_v_rgb)

            estimates.append(np.array([x_rgb, y_rgb, depth_rgb]))

        # Process Sonar estimate
        if sonar_box is not None:
            cx, cy, w, h = sonar_box

            # Estimate depth from bbox size
            bbox_area = w * h
            if bbox_area > 0:
                depth_sonar = 1.0 / np.sqrt(bbox_area)
            else:
                depth_sonar = default_depth

            # Sonar only provides horizontal angle (no vertical angle)
            angle_h_sonar = (cx - 0.5) * sonar_fov_h_rad

            # Calculate X from sonar (Y uses default or RGB estimate)
            x_sonar = depth_sonar * np.tan(angle_h_sonar)

            # For Y, use middle of vertical range (sonar doesn't provide vertical angle)
            y_sonar = 0.0

            estimates.append(np.array([x_sonar, y_sonar, depth_sonar]))

        # Fuse estimates with simple averaging
        if len(estimates) == 2:
            # Both RGB and Sonar available: average them
            pos_3d = np.mean(estimates, axis=0)
        else:
            # Only one modality available
            pos_3d = estimates[0]

        return pos_3d


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
