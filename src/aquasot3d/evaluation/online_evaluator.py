"""
Online Tracking Evaluator for AquaSOT-3D

This module implements the STANDARD online tracking evaluation protocol:
- Template: Initialized from first frame GT bbox
- Search region: Centered on PREVIOUS FRAME PREDICTION (not GT!)
- This simulates real-world tracking where GT is unknown during inference

Key difference from offline evaluation:
- Offline: Search centered on GT (unrealistic, gives model unfair advantage)
- Online: Search centered on previous prediction (realistic, tests tracking robustness)

Reference implementations:
- OSTrack: https://github.com/botaoye/OSTrack
- PyTracking: https://github.com/visionml/pytracking
- GOT-10k toolkit: https://github.com/got-10k/toolkit

Author: AquaSOT-3D Team
Date: 2025-11-27
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from aquasot3d.evaluation.metrics import compute_sequence_metrics, aggregate_metrics

logger = logging.getLogger(__name__)


class OnlineTrackingEvaluator:
    """
    Standard online tracking evaluator following VOT/OTB/LaSOT protocols.

    Key features:
    1. First frame: Initialize template from GT, set initial state = GT
    2. Subsequent frames: Crop search region centered on PREVIOUS PREDICTION
    3. Model predicts target location within search region
    4. Prediction mapped back to original image coordinates
    5. Update state for next frame

    This properly evaluates tracking robustness including:
    - Target re-detection after occlusion
    - Drift recovery
    - Scale adaptation

    Args:
        model: Tracking model to evaluate
        device: Device to run evaluation on
        template_size: Template crop size (default: 128)
        search_size: Search region crop size (default: 256)
        template_area_factor: Template crop area factor (default: 2.0)
        search_area_factor: Search crop area factor (default: 4.0)
        normalization_config: Normalization parameters for RGB and Sonar
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        template_size: int = 128,
        search_size: int = 256,
        template_area_factor: float = 2.0,
        search_area_factor: float = 4.0,
        normalization_config: Optional[Dict] = None
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor

        # Normalization parameters
        norm_cfg = normalization_config or {}
        self.rgb_mean = torch.tensor(
            norm_cfg.get('rgb_mean', [0.485, 0.456, 0.406])
        ).view(3, 1, 1)
        self.rgb_std = torch.tensor(
            norm_cfg.get('rgb_std', [0.229, 0.224, 0.225])
        ).view(3, 1, 1)
        self.sonar_mean = torch.tensor(
            norm_cfg.get('sonar_mean', [0.5])
        ).view(1, 1, 1)
        self.sonar_std = torch.tensor(
            norm_cfg.get('sonar_std', [0.5])
        ).view(1, 1, 1)

        # Store results for get_best/worst_sequences
        self._last_results = None

        logger.info(
            f"OnlineTrackingEvaluator initialized: "
            f"template_size={template_size}, search_size={search_size}, "
            f"search_area_factor={search_area_factor}"
        )

    def evaluate_sequence(
        self,
        base_dataset,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single sequence using online tracking protocol.

        Args:
            base_dataset: RGBSonarDataset instance for this sequence
            verbose: Whether to show progress bar

        Returns:
            Dictionary with predictions and ground truth for each modality
        """
        self.model.eval()
        self.model = self.model.to(self.device)

        num_frames = base_dataset.num_frames
        sequence_name = getattr(base_dataset, 'sequence_name', 'unknown')

        # Initialize prediction storage
        predictions = {
            'rgb_pred': [],
            'sonar_pred': [],
            'rgb_gt': [],
            'sonar_gt': []
        }

        # === FRAME 0: Initialize tracker ===
        frame0_data = base_dataset._get_frame_data(0)

        # Get first frame GT bboxes
        rgb_gt_bbox = frame0_data['rgb_bbox'].copy()
        sonar_gt_bbox = frame0_data['sonar_bbox'].copy()

        # Initialize tracking state with GT (standard protocol)
        # State = predicted bbox from previous frame (or GT for frame 0)
        rgb_state = rgb_gt_bbox.copy()
        sonar_state = sonar_gt_bbox.copy()

        # Crop and cache templates from first frame
        rgb_template, rgb_template_mask = self._crop_region(
            frame0_data['rgb_img'],
            rgb_gt_bbox,
            self.template_size,
            self.template_area_factor,
            is_sonar=False
        )
        sonar_template, sonar_template_mask = self._crop_region(
            frame0_data['sonar_img'],
            sonar_gt_bbox,
            self.template_size,
            self.template_area_factor,
            is_sonar=True
        )

        # Move templates to device
        rgb_template = rgb_template.unsqueeze(0).to(self.device)
        sonar_template = sonar_template.unsqueeze(0).to(self.device)

        # For frame 0, prediction = GT (initialization)
        predictions['rgb_pred'].append(rgb_gt_bbox.copy())
        predictions['sonar_pred'].append(sonar_gt_bbox.copy())
        predictions['rgb_gt'].append(rgb_gt_bbox.copy())
        predictions['sonar_gt'].append(sonar_gt_bbox.copy())

        # === FRAMES 1 to N-1: Online tracking ===
        frame_iter = range(1, num_frames)
        if verbose:
            frame_iter = tqdm(frame_iter, desc=f"  {sequence_name}", unit="frame", leave=False)

        with torch.no_grad():
            for frame_idx in frame_iter:
                frame_data = base_dataset._get_frame_data(frame_idx)

                # Get current frame GT (for evaluation only, not used for tracking!)
                rgb_gt_bbox = frame_data['rgb_bbox'].copy()
                sonar_gt_bbox = frame_data['sonar_bbox'].copy()

                # === CRITICAL: Crop search region centered on PREVIOUS PREDICTION ===
                # This is the key difference from offline evaluation!
                rgb_search, rgb_search_meta = self._crop_region(
                    frame_data['rgb_img'],
                    rgb_state,  # Use PREVIOUS PREDICTION, not GT!
                    self.search_size,
                    self.search_area_factor,
                    is_sonar=False,
                    return_metadata=True
                )
                sonar_search, sonar_search_meta = self._crop_region(
                    frame_data['sonar_img'],
                    sonar_state,  # Use PREVIOUS PREDICTION, not GT!
                    self.search_size,
                    self.search_area_factor,
                    is_sonar=True,
                    return_metadata=True
                )

                # Move to device and add batch dimension
                rgb_search = rgb_search.unsqueeze(0).to(self.device)
                sonar_search = sonar_search.unsqueeze(0).to(self.device)

                # === Model forward pass ===
                outputs = self.model(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search,
                    return_losses=False
                )

                # === Extract and denormalize predictions ===
                # Model outputs normalized cxcywh in [0, 1]
                rgb_pred_norm = outputs['rgb_boxes'][0].cpu().numpy()  # [cx, cy, w, h]
                sonar_pred_norm = outputs['sonar_boxes'][0].cpu().numpy()

                # Transform from normalized crop space to original image space
                rgb_pred_orig = self._transform_pred_to_original(
                    rgb_pred_norm, rgb_search_meta
                )
                sonar_pred_orig = self._transform_pred_to_original(
                    sonar_pred_norm, sonar_search_meta
                )

                # === Update state for next frame ===
                # CRITICAL: Use current prediction as next frame's search center
                if self._is_valid_bbox(rgb_pred_orig):
                    rgb_state = rgb_pred_orig.copy()
                # else: keep previous state (handle tracking failures gracefully)

                if self._is_valid_bbox(sonar_pred_orig):
                    sonar_state = sonar_pred_orig.copy()

                # Store predictions and GT
                predictions['rgb_pred'].append(rgb_pred_orig)
                predictions['sonar_pred'].append(sonar_pred_orig)
                predictions['rgb_gt'].append(rgb_gt_bbox)
                predictions['sonar_gt'].append(sonar_gt_bbox)

        return predictions

    def _crop_region(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        output_size: int,
        area_factor: float,
        is_sonar: bool = False,
        return_metadata: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Crop region centered on bbox following OSTrack methodology.

        Args:
            image: Input image [H, W, C] or [H, W]
            bbox: Center bbox [x, y, w, h] in XYWH pixel format
            output_size: Output crop size
            area_factor: Crop size multiplier
            is_sonar: Whether this is sonar image
            return_metadata: Whether to return crop metadata

        Returns:
            crop_tensor: Normalized crop [C, H, W]
            metadata (optional): Dict with crop parameters for inverse transform
        """
        # Handle None or invalid image
        if image is None:
            num_channels = 1 if is_sonar else 3
            crop_tensor = torch.zeros(num_channels, output_size, output_size)
            if return_metadata:
                metadata = {
                    'crop_center': (0, 0),
                    'crop_sz': output_size,
                    'resize_factor': 1.0,
                    'crop_x1': 0,
                    'crop_y1': 0,
                    'original_size': (output_size, output_size)
                }
                return crop_tensor, metadata
            return crop_tensor, None

        H, W = image.shape[:2]
        x, y, w, h = bbox

        # Handle degenerate bbox (e.g., occlusion marker [0,0,0,0])
        if w <= 0 or h <= 0:
            w = max(w, 10)
            h = max(h, 10)

        # Compute bbox center
        cx = x + w / 2
        cy = y + h / 2

        # Compute crop size: sqrt(w*h) * area_factor
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * area_factor))
        crop_sz = max(crop_sz, 1)

        # Compute crop boundaries
        x1 = int(round(cx - crop_sz / 2))
        y1 = int(round(cy - crop_sz / 2))
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        # Compute padding amounts
        x1_pad = max(0, -x1)
        x2_pad = max(0, x2 - W)
        y1_pad = max(0, -y1)
        y2_pad = max(0, y2 - H)

        # Clamp to valid region
        x1_valid = max(0, x1)
        x2_valid = min(W, x2)
        y1_valid = max(0, y1)
        y2_valid = min(H, y2)

        # Extract valid region
        if x2_valid <= x1_valid or y2_valid <= y1_valid:
            # Completely outside image
            if image.ndim == 2:
                crop_padded = np.full((crop_sz, crop_sz), 127, dtype=np.uint8)
            else:
                avg_color = image.mean(axis=(0, 1)).astype(np.uint8)
                crop_padded = np.full((crop_sz, crop_sz, image.shape[2]), avg_color, dtype=np.uint8)
        else:
            crop = image[y1_valid:y2_valid, x1_valid:x2_valid]

            if image.ndim == 2:
                pad_value = int(image.mean())
            else:
                pad_value = tuple(int(c) for c in image.mean(axis=(0, 1)))

            crop_padded = cv2.copyMakeBorder(
                crop, y1_pad, y2_pad, x1_pad, x2_pad,
                cv2.BORDER_CONSTANT, value=pad_value
            )

        # Resize to output size
        resize_factor = output_size / crop_sz
        crop_resized = cv2.resize(
            crop_padded, (output_size, output_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to tensor and normalize
        if crop_resized.ndim == 2:
            crop_tensor = torch.from_numpy(crop_resized).unsqueeze(0).float() / 255.0
            crop_tensor = (crop_tensor - self.sonar_mean) / self.sonar_std
        else:
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = (crop_tensor - self.rgb_mean) / self.rgb_std

        if return_metadata:
            metadata = {
                'crop_center': (cx, cy),
                'crop_sz': crop_sz,
                'resize_factor': resize_factor,
                'crop_x1': x1,
                'crop_y1': y1,
                'original_size': (H, W),
                'output_size': output_size
            }
            return crop_tensor, metadata

        return crop_tensor, None

    def _transform_pred_to_original(
        self,
        pred_norm: np.ndarray,
        crop_metadata: Dict
    ) -> np.ndarray:
        """
        Transform normalized prediction from crop space to original image space.

        Args:
            pred_norm: Predicted bbox [cx, cy, w, h] normalized to [0, 1]
            crop_metadata: Crop parameters from _crop_region

        Returns:
            bbox in original image coordinates [x, y, w, h] (XYWH format)
        """
        output_size = crop_metadata['output_size']
        resize_factor = crop_metadata['resize_factor']
        crop_x1 = crop_metadata['crop_x1']
        crop_y1 = crop_metadata['crop_y1']

        # Step 1: Denormalize from [0, 1] to crop pixel coordinates
        cx_crop = pred_norm[0] * output_size
        cy_crop = pred_norm[1] * output_size
        w_crop = pred_norm[2] * output_size
        h_crop = pred_norm[3] * output_size

        # Step 2: Undo resize (crop pixels -> original crop size)
        cx_orig_crop = cx_crop / resize_factor
        cy_orig_crop = cy_crop / resize_factor
        w_orig = w_crop / resize_factor
        h_orig = h_crop / resize_factor

        # Step 3: Add crop offset to get original image coordinates
        cx_orig = cx_orig_crop + crop_x1
        cy_orig = cy_orig_crop + crop_y1

        # Step 4: Convert from cxcywh to xywh
        x_orig = cx_orig - w_orig / 2
        y_orig = cy_orig - h_orig / 2

        return np.array([x_orig, y_orig, w_orig, h_orig], dtype=np.float32)

    def _is_valid_bbox(self, bbox: np.ndarray, min_size: float = 1.0) -> bool:
        """Check if bbox is valid (not degenerate)."""
        if bbox is None:
            return False
        if len(bbox) < 4:
            return False
        x, y, w, h = bbox
        return w >= min_size and h >= min_size

    def evaluate_all_sequences(
        self,
        sequence_datasets: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate multiple sequences and aggregate results.

        Args:
            sequence_datasets: Dict mapping sequence_name -> base_dataset
            verbose: Whether to show progress

        Returns:
            Dictionary with per-sequence and overall results
        """
        all_results = {}
        all_metrics = []

        seq_iter = sequence_datasets.items()
        if verbose:
            seq_iter = tqdm(list(seq_iter), desc="Sequences", unit="seq")

        for seq_name, base_dataset in seq_iter:
            logger.info(f"Evaluating sequence: {seq_name}")

            # Run online tracking evaluation
            predictions = self.evaluate_sequence(base_dataset, verbose=verbose)

            # Compute metrics for each modality
            seq_metrics = {}

            # RGB metrics
            if predictions['rgb_pred'] and predictions['rgb_gt']:
                rgb_metrics = compute_sequence_metrics(
                    predictions['rgb_pred'],
                    predictions['rgb_gt'],
                    modality='rgb',
                    additional_metrics=['precision_10', 'precision_5', 'success_0.7']
                )
                seq_metrics['rgb'] = rgb_metrics

            # Sonar metrics
            if predictions['sonar_pred'] and predictions['sonar_gt']:
                sonar_metrics = compute_sequence_metrics(
                    predictions['sonar_pred'],
                    predictions['sonar_gt'],
                    modality='sonar',
                    additional_metrics=['precision_10', 'precision_5', 'success_0.7']
                )
                seq_metrics['sonar'] = sonar_metrics

            all_results[seq_name] = {
                'predictions': predictions,
                'metrics': seq_metrics
            }
            all_metrics.append(seq_metrics)

        # Aggregate metrics per modality
        overall_metrics = {}

        # Collect RGB metrics from all sequences
        rgb_seq_metrics = [m.get('rgb', {}) for m in all_metrics if 'rgb' in m]
        if rgb_seq_metrics:
            overall_metrics['rgb'] = aggregate_metrics(rgb_seq_metrics, modality='rgb')

        # Collect Sonar metrics from all sequences
        sonar_seq_metrics = [m.get('sonar', {}) for m in all_metrics if 'sonar' in m]
        if sonar_seq_metrics:
            overall_metrics['sonar'] = aggregate_metrics(sonar_seq_metrics, modality='sonar')

        # Store results for get_best/worst_sequences
        self._last_results = {
            'overall': overall_metrics,
            'sequences': all_results
        }

        return self._last_results

    def get_best_sequences(self, n: int = 5, metric: str = 'success_auc') -> List[Tuple[str, float]]:
        """
        Get top N best performing sequences.

        Args:
            n: Number of sequences to return
            metric: Metric to rank by (default: 'success_auc')

        Returns:
            List of (sequence_name, metric_value) tuples sorted descending
        """
        if self._last_results is None:
            logger.warning("No evaluation results available. Run evaluate_all_sequences first.")
            return []

        seq_scores = []
        for seq_name, seq_data in self._last_results['sequences'].items():
            metrics = seq_data.get('metrics', {})
            # Try RGB first, then Sonar
            rgb_metrics = metrics.get('rgb', {})
            sonar_metrics = metrics.get('sonar', {})

            score = rgb_metrics.get(metric, sonar_metrics.get(metric, 0.0))
            seq_scores.append((seq_name, score))

        # Sort descending by score
        seq_scores.sort(key=lambda x: x[1], reverse=True)
        return seq_scores[:n]

    def get_worst_sequences(self, n: int = 5, metric: str = 'success_auc') -> List[Tuple[str, float]]:
        """
        Get top N worst performing sequences.

        Args:
            n: Number of sequences to return
            metric: Metric to rank by (default: 'success_auc')

        Returns:
            List of (sequence_name, metric_value) tuples sorted ascending
        """
        if self._last_results is None:
            logger.warning("No evaluation results available. Run evaluate_all_sequences first.")
            return []

        seq_scores = []
        for seq_name, seq_data in self._last_results['sequences'].items():
            metrics = seq_data.get('metrics', {})
            # Try RGB first, then Sonar
            rgb_metrics = metrics.get('rgb', {})
            sonar_metrics = metrics.get('sonar', {})

            score = rgb_metrics.get(metric, sonar_metrics.get(metric, 0.0))
            seq_scores.append((seq_name, score))

        # Sort ascending by score (worst first)
        seq_scores.sort(key=lambda x: x[1], reverse=False)
        return seq_scores[:n]

    def save_metrics(self, output_dir: Union[str, Path], timestamp: str = None) -> None:
        """
        Save evaluation metrics to files.

        Args:
            output_dir: Directory to save metrics
            timestamp: Optional timestamp for file naming
        """
        import json
        from datetime import datetime

        if self._last_results is None:
            logger.warning("No results to save. Run evaluate_all_sequences first.")
            return

        output_dir = Path(output_dir)
        metrics_dir = output_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save overall metrics
        overall_metrics_file = metrics_dir / f'overall_metrics_{timestamp}.json'
        with open(overall_metrics_file, 'w') as f:
            json.dump(self._last_results['overall'], f, indent=2, default=float)
        logger.info(f"Saved overall metrics to {overall_metrics_file}")

        # Save per-sequence metrics
        seq_metrics_file = metrics_dir / f'sequence_metrics_{timestamp}.json'
        seq_metrics = {}
        for seq_name, seq_data in self._last_results['sequences'].items():
            seq_metrics[seq_name] = seq_data.get('metrics', {})
        with open(seq_metrics_file, 'w') as f:
            json.dump(seq_metrics, f, indent=2, default=float)
        logger.info(f"Saved sequence metrics to {seq_metrics_file}")

        # Save summary report
        report_file = metrics_dir / f'eval_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("AquaSOT-3D Online Tracking Evaluation Report\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 60 + "\n\n")

            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 40 + "\n")
            overall = self._last_results['overall']
            for modality in ['rgb', 'sonar']:
                if modality in overall:
                    f.write(f"\n{modality.upper()}:\n")
                    for metric, value in overall[modality].items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")

            # Per-sequence metrics
            f.write("\n\nPER-SEQUENCE METRICS\n")
            f.write("-" * 40 + "\n")
            for seq_name, seq_data in self._last_results['sequences'].items():
                f.write(f"\n{seq_name}:\n")
                metrics = seq_data.get('metrics', {})
                for modality in ['rgb', 'sonar']:
                    if modality in metrics:
                        f.write(f"  {modality.upper()}:\n")
                        for metric, value in metrics[modality].items():
                            if isinstance(value, (int, float)):
                                f.write(f"    {metric}: {value:.4f}\n")

        logger.info(f"Saved evaluation report to {report_file}")

    def save_predictions(self, output_dir: Union[str, Path], timestamp: str = None) -> None:
        """
        Save raw predictions for each sequence.

        Args:
            output_dir: Directory to save predictions
            timestamp: Optional timestamp for file naming
        """
        import json
        from datetime import datetime

        if self._last_results is None:
            logger.warning("No results to save. Run evaluate_all_sequences first.")
            return

        output_dir = Path(output_dir)
        pred_dir = output_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for seq_name, seq_data in self._last_results['sequences'].items():
            predictions = seq_data.get('predictions', {})
            seq_file = pred_dir / f'{seq_name}_predictions_{timestamp}.json'

            # Convert numpy arrays to lists for JSON serialization
            pred_json = {}
            for key, values in predictions.items():
                if isinstance(values, list):
                    pred_json[key] = [
                        v.tolist() if hasattr(v, 'tolist') else v
                        for v in values
                    ]
                else:
                    pred_json[key] = values

            with open(seq_file, 'w') as f:
                json.dump(pred_json, f, indent=2)

        logger.info(f"Saved predictions for {len(self._last_results['sequences'])} sequences")
