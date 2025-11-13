"""
Tracking Evaluator for AquaSOT-3D

This module implements the evaluation pipeline for tracking performance assessment,
including sequence-level tracking, metric aggregation, and benchmark reporting.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import json
import logging
from tqdm import tqdm
from collections import defaultdict

from aquasot3d.evaluation.metrics import (
    compute_sequence_metrics,
    aggregate_metrics,
    compute_multi_modal_metrics,
    format_metrics_report
)

logger = logging.getLogger(__name__)


class TrackingEvaluator:
    """
    Evaluator for tracking performance assessment.

    This class handles sequence-level tracking evaluation, computing metrics
    for RGB, Sonar, and Fusion predictions separately.

    Args:
        model: Tracking model to evaluate
        dataloader: DataLoader for evaluation dataset
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        save_predictions: Whether to save per-frame predictions
        verbose: Whether to print detailed progress

    Example:
        >>> evaluator = TrackingEvaluator(
        ...     model=tracker,
        ...     dataloader=val_loader,
        ...     device='cuda',
        ...     output_dir='outputs/evaluation'
        ... )
        >>> results = evaluator.evaluate()
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = 'cuda',
        output_dir: Optional[str] = None,
        save_predictions: bool = False,
        verbose: bool = True,
        search_size: int = 256,
        additional_metrics: Optional[List[str]] = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_predictions = save_predictions
        self.verbose = verbose
        self.search_size = search_size
        self.additional_metrics = additional_metrics or []

        # Create output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.sequence_results = {}
        self.overall_metrics = {}

        logger.info(f"TrackingEvaluator initialized: device={device}, output_dir={output_dir}")
        if self.additional_metrics:
            logger.info(f"Additional metrics: {self.additional_metrics}")

    def evaluate(self) -> Dict[str, Any]:
        """
        Run complete evaluation on all sequences.

        Returns:
            Dictionary containing overall results and per-sequence metrics

        Example:
            >>> results = evaluator.evaluate()
            >>> print(f"Fusion AUC: {results['overall']['fusion']['success_auc_mean']:.4f}")
        """
        self.model.eval()
        self.model = self.model.to(self.device)

        logger.info("Starting evaluation...")

        # Accumulate predictions by sequence
        sequence_predictions = defaultdict(lambda: {
            'rgb_pred': [],
            'sonar_pred': [],
            'fusion_pred': [],
            'rgb_gt': [],
            'sonar_gt': []
        })

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(
                self.dataloader,
                desc="Evaluating",
                disable=not self.verbose
            )):
                # Get sequence name (extract from list if batch contains multiple samples)
                sequence_name_raw = batch.get('sequence_name', f'seq_{batch_idx}')

                # Handle both list and string formats
                if isinstance(sequence_name_raw, list):
                    # Extract first element and verify all elements are the same
                    if len(sequence_name_raw) > 0:
                        sequence_name = sequence_name_raw[0]
                        # Sanity check: all samples in batch should be from same sequence
                        if not all(s == sequence_name for s in sequence_name_raw):
                            logger.warning(
                                f"Batch {batch_idx} contains samples from multiple sequences: "
                                f"{set(sequence_name_raw)}. Using first: {sequence_name}"
                            )
                    else:
                        sequence_name = f'seq_{batch_idx}'
                else:
                    sequence_name = sequence_name_raw

                # Track batch (returns per-frame predictions)
                batch_results = self.track_sequence(batch)

                # Accumulate predictions for this sequence
                for key in ['rgb_pred', 'sonar_pred', 'fusion_pred', 'rgb_gt', 'sonar_gt']:
                    if key in batch_results:
                        sequence_predictions[sequence_name][key].extend(batch_results[key])

        # Compute metrics for each sequence
        sequence_metrics = []

        for sequence_name, predictions in sequence_predictions.items():
            # Compute metrics for this sequence
            seq_metrics = self.compute_sequence_metrics(predictions)

            # Store results
            self.sequence_results[sequence_name] = {
                'predictions': predictions,
                'metrics': seq_metrics
            }

            sequence_metrics.append(seq_metrics)

            # Save per-sequence results if requested
            if self.save_predictions and self.output_dir:
                self.save_sequence_results(sequence_name, predictions, seq_metrics)

        # Aggregate metrics across all sequences
        self.overall_metrics = self.aggregate_all_metrics(sequence_metrics)

        # Generate report
        if self.verbose:
            self.print_summary()

        # Save overall results
        if self.output_dir:
            self.save_overall_results()

        logger.info("Evaluation completed")

        return {
            'overall': self.overall_metrics,
            'sequences': self.sequence_results
        }

    def _denormalize_bbox(self, bbox: np.ndarray, image_size: int) -> np.ndarray:
        """
        Convert normalized bbox [0, 1] to pixel coordinates.

        Args:
            bbox: Bounding box in normalized coordinates [cx, cy, w, h]
            image_size: Image size in pixels

        Returns:
            Bounding box in pixel coordinates
        """
        if bbox is None:
            return None

        # Check if bbox needs denormalization (max value < 2 suggests normalized coords)
        if np.max(np.abs(bbox)) < 2.0:
            # Denormalize: multiply by image_size
            return bbox * image_size
        else:
            # Already in pixel coordinates
            return bbox

    def track_sequence(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[np.ndarray]]:
        """
        Track a single sequence and return predictions in pixel coordinates.

        IMPORTANT: This method now returns boxes in pixel coordinates (not normalized),
        which is required for correct evaluation metrics (precision@20, center error).

        Args:
            batch: Batch data containing sequence information

        Returns:
            Dictionary with predictions and ground truth (in pixel coordinates)
        """
        # Move batch to device
        rgb_template = batch['rgb_template'].to(self.device)
        rgb_search = batch['rgb_search'].to(self.device)
        sonar_template = batch['sonar_template'].to(self.device)
        sonar_search = batch['sonar_search'].to(self.device)

        # Run model
        outputs = self.model(rgb_template, rgb_search, sonar_template, sonar_search)

        # Extract predictions
        predictions = {
            'rgb_pred': [],
            'sonar_pred': [],
            'fusion_pred': [],
            'rgb_gt': [],
            'sonar_gt': []
        }

        # Get predictions (model outputs plural keys: rgb_boxes, sonar_boxes, fusion_boxes)
        # Denormalize boxes from [0,1] to pixel coordinates
        if 'rgb_boxes' in outputs:
            rgb_boxes = outputs['rgb_boxes'].cpu().numpy()
            predictions['rgb_pred'] = [
                self._denormalize_bbox(rgb_boxes[i], self.search_size)
                for i in range(len(rgb_boxes))
            ]

        if 'sonar_boxes' in outputs:
            sonar_boxes = outputs['sonar_boxes'].cpu().numpy()
            predictions['sonar_pred'] = [
                self._denormalize_bbox(sonar_boxes[i], self.search_size)
                for i in range(len(sonar_boxes))
            ]

        if 'fusion_boxes' in outputs:
            fusion_boxes = outputs['fusion_boxes'].cpu().numpy()
            predictions['fusion_pred'] = [
                self._denormalize_bbox(fusion_boxes[i], self.search_size)
                for i in range(len(fusion_boxes))
            ]

        # Get ground truth (support both singular and plural forms)
        # Also denormalize ground truth boxes
        rgb_gt_key = 'rgb_box' if 'rgb_box' in batch else 'rgb_boxes'
        if rgb_gt_key in batch:
            rgb_gt = batch[rgb_gt_key].cpu().numpy()
            predictions['rgb_gt'] = [
                self._denormalize_bbox(rgb_gt[i], self.search_size)
                for i in range(len(rgb_gt))
            ]

        sonar_gt_key = 'sonar_box' if 'sonar_box' in batch else 'sonar_boxes'
        if sonar_gt_key in batch:
            sonar_gt = batch[sonar_gt_key].cpu().numpy()
            predictions['sonar_gt'] = [
                self._denormalize_bbox(sonar_gt[i], self.search_size)
                for i in range(len(sonar_gt))
            ]

        return predictions

    def compute_sequence_metrics(
        self,
        predictions: Dict[str, List[np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for a single sequence.

        Args:
            predictions: Dictionary with predictions and ground truth

        Returns:
            Metrics dictionary for all modalities
        """
        # Compute metrics for each modality (pass additional_metrics)
        metrics = compute_multi_modal_metrics(
            predictions.get('rgb_pred', []),
            predictions.get('sonar_pred', []),
            predictions.get('fusion_pred', []),
            predictions.get('rgb_gt', []),
            predictions.get('sonar_gt', []),
            additional_metrics=self.additional_metrics
        )

        return metrics

    def aggregate_all_metrics(
        self,
        sequence_metrics: List[Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across all sequences.

        Args:
            sequence_metrics: List of per-sequence metrics

        Returns:
            Aggregated metrics for each modality
        """
        aggregated = {}

        # Aggregate for each modality
        for modality in ['rgb', 'sonar', 'fusion']:
            # Extract metrics for this modality
            modality_metrics = [
                seq_metrics[modality]
                for seq_metrics in sequence_metrics
                if modality in seq_metrics
            ]

            # Aggregate
            if len(modality_metrics) > 0:
                aggregated[modality] = aggregate_metrics(modality_metrics, modality)

        return aggregated

    def save_sequence_results(
        self,
        sequence_name: str,
        predictions: Dict[str, List[np.ndarray]],
        metrics: Dict[str, Dict[str, float]]
    ):
        """
        Save per-sequence results to file.

        Args:
            sequence_name: Name of the sequence
            predictions: Predictions dictionary
            metrics: Metrics dictionary
        """
        if not self.output_dir:
            return

        seq_dir = self.output_dir / 'sequences' / sequence_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        if self.save_predictions:
            predictions_file = seq_dir / 'predictions.npz'
            np.savez(
                predictions_file,
                **{k: np.array(v) for k, v in predictions.items()}
            )

        # Save metrics
        metrics_file = seq_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.debug(f"Saved results for sequence: {sequence_name}")

    def save_overall_results(self):
        """Save overall evaluation results."""
        if not self.output_dir:
            return

        # Save overall metrics
        metrics_file = self.output_dir / 'overall_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.overall_metrics, f, indent=2)

        # Save detailed report
        report_file = self.output_dir / 'evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write(self.generate_report())

        logger.info(f"Saved evaluation results to: {self.output_dir}")

    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        for modality in ['rgb', 'sonar', 'fusion']:
            if modality not in self.overall_metrics:
                continue

            metrics = self.overall_metrics[modality]
            print(f"\n{modality.upper()} Results:")
            print("-" * 50)
            print(f"  Success (AUC):      {metrics.get('success_auc_mean', 0.0):.4f} ± {metrics.get('success_auc_std', 0.0):.4f}")
            print(f"  Precision@20:       {metrics.get('precision_20_mean', 0.0):.4f} ± {metrics.get('precision_20_std', 0.0):.4f}")
            print(f"  Avg Center Error:   {metrics.get('avg_center_error_mean', 0.0):.2f} ± {metrics.get('avg_center_error_std', 0.0):.2f} px")
            print(f"  Mean IoU:           {metrics.get('mean_iou_mean', 0.0):.4f} ± {metrics.get('mean_iou_std', 0.0):.4f}")

        print("\n" + "=" * 70)

    def generate_report(self) -> str:
        """
        Generate detailed evaluation report.

        Returns:
            Formatted report string
        """
        lines = []

        lines.append("=" * 70)
        lines.append("AquaSOT-3D Tracking Evaluation Report")
        lines.append("=" * 70)
        lines.append("")

        # Overall statistics
        lines.append("Overall Statistics:")
        lines.append("-" * 50)
        total_sequences = len(self.sequence_results)
        lines.append(f"  Total Sequences: {total_sequences}")

        if 'fusion' in self.overall_metrics:
            total_frames = self.overall_metrics['fusion'].get('total_frames', 0)
            lines.append(f"  Total Frames: {total_frames}")

        lines.append("")

        # Per-modality results
        for modality in ['rgb', 'sonar', 'fusion']:
            if modality not in self.overall_metrics:
                continue

            metrics = self.overall_metrics[modality]

            lines.append(f"{modality.upper()} Performance:")
            lines.append("-" * 50)
            lines.append(f"  Success (AUC):")
            lines.append(f"    Mean:   {metrics.get('success_auc_mean', 0.0):.4f}")
            lines.append(f"    Median: {metrics.get('success_auc_median', 0.0):.4f}")
            lines.append(f"    Std:    {metrics.get('success_auc_std', 0.0):.4f}")
            lines.append("")
            lines.append(f"  Precision@20:")
            lines.append(f"    Mean:   {metrics.get('precision_20_mean', 0.0):.4f}")
            lines.append(f"    Median: {metrics.get('precision_20_median', 0.0):.4f}")
            lines.append(f"    Std:    {metrics.get('precision_20_std', 0.0):.4f}")
            lines.append("")
            lines.append(f"  Average Center Error:")
            lines.append(f"    Mean:   {metrics.get('avg_center_error_mean', 0.0):.2f} px")
            lines.append(f"    Median: {metrics.get('avg_center_error_median', 0.0):.2f} px")
            lines.append(f"    Std:    {metrics.get('avg_center_error_std', 0.0):.2f} px")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def get_best_sequences(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N sequences by fusion AUC.

        Args:
            n: Number of sequences to return

        Returns:
            List of (sequence_name, auc) tuples
        """
        seq_aucs = [
            (name, result['metrics']['fusion']['success_auc'])
            for name, result in self.sequence_results.items()
            if 'fusion' in result['metrics']
        ]

        # Sort by AUC descending
        seq_aucs.sort(key=lambda x: x[1], reverse=True)

        return seq_aucs[:n]

    def get_worst_sequences(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get bottom N sequences by fusion AUC (for failure analysis).

        Args:
            n: Number of sequences to return

        Returns:
            List of (sequence_name, auc) tuples
        """
        seq_aucs = [
            (name, result['metrics']['fusion']['success_auc'])
            for name, result in self.sequence_results.items()
            if 'fusion' in result['metrics']
        ]

        # Sort by AUC ascending
        seq_aucs.sort(key=lambda x: x[1])

        return seq_aucs[:n]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to use
        output_dir: Output directory for results

    Returns:
        Evaluation results dictionary

    Example:
        >>> results = evaluate_model(tracker, val_loader, 'cuda', 'outputs/eval')
    """
    evaluator = TrackingEvaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        save_predictions=True,
        verbose=True
    )

    return evaluator.evaluate()
