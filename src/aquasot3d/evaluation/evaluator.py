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

        # CRITICAL FIX (Audit 2025111701 Finding 5): Enforce batch_size=1 for evaluation
        # Batched evaluation breaks sequence boundary management when sequences span multiple batches
        if dataloader.batch_size is not None and dataloader.batch_size > 1:
            raise ValueError(
                f"Evaluation requires batch_size=1 to maintain sequence boundaries. "
                f"Got batch_size={dataloader.batch_size}. Set evaluation.batch_size=1 in config."
            )

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
            'rgb_gt': [],
            'sonar_gt': []
        })

        current_sequence = None

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

                if current_sequence is None or sequence_name != current_sequence:
                    logger.debug(f"Sequence boundary detected: {current_sequence} -> {sequence_name}")
                    current_sequence = sequence_name

                # Track batch (returns per-frame predictions)
                batch_results = self.track_sequence(batch)

                # Accumulate predictions for this sequence
                for key in ['rgb_pred', 'sonar_pred', 'rgb_gt', 'sonar_gt']:
                    if key in batch_results:
                        sequence_predictions[sequence_name][key].extend(batch_results[key])

        # Compute metrics for each sequence
        sequence_metrics = []

        for sequence_name, predictions in sequence_predictions.items():
            # Compute metrics for this sequence
            seq_metrics = self.compute_sequence_metrics(predictions)

            # CRITICAL FIX (Audit 2025111413 Finding 3):
            # Add per-sequence diagnostics with quality trends for failure analysis
            diagnostics = {
                'num_frames': {
                    'rgb': len(predictions.get('rgb_pred', [])),
                    'sonar': len(predictions.get('sonar_pred', []))
                },
                'modality_availability': {
                    'has_rgb': len(predictions.get('rgb_pred', [])) > 0,
                    'has_sonar': len(predictions.get('sonar_pred', [])) > 0
                },
                'quality_trends': self._compute_quality_diagnostics(predictions)
            }

            # Store results with diagnostics
            self.sequence_results[sequence_name] = {
                'predictions': predictions,
                'metrics': seq_metrics,
                'diagnostics': diagnostics  # New field for debugging
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

    def _compute_padding_ratio(self, crop_metadata: Dict) -> float:
        """
        Compute the padding ratio of a crop region.

        CRITICAL FIX (Audit 2025120902 Issue 9 Revised):
        Matches online_evaluator._compute_padding_ratio() for consistency.

        Padding ratio indicates what fraction of the crop lies outside
        the original image boundary. High padding (>0.3) suggests the
        target may be occluded or out-of-frame.

        Args:
            crop_metadata: Dictionary containing crop parameters:
                - crop_x1, crop_y1: Top-left corner of crop (can be negative)
                - crop_sz: Size of crop window
                - original_width, original_height: Source image dimensions

        Returns:
            Padding ratio in [0, 1], where 0 = fully valid, 1 = fully padded
        """
        if not crop_metadata:
            return 0.0  # Assume valid if no metadata

        # Extract crop parameters
        crop_x1 = crop_metadata.get('crop_x1', 0)
        crop_y1 = crop_metadata.get('crop_y1', 0)
        crop_sz = crop_metadata.get('crop_sz', 256)
        orig_w = crop_metadata.get('original_width', crop_sz)
        orig_h = crop_metadata.get('original_height', crop_sz)

        # Compute crop boundaries
        crop_x2 = crop_x1 + crop_sz
        crop_y2 = crop_y1 + crop_sz

        # Compute padding on each side (how much lies outside image)
        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - orig_w)
        pad_bottom = max(0, crop_y2 - orig_h)

        # Compute padded area (approximate rectangle intersection)
        # Simplified: sum of padded margins times crop size
        pad_area = (pad_left + pad_right) * crop_sz + (pad_top + pad_bottom) * max(0, crop_sz - pad_left - pad_right)
        total_area = crop_sz * crop_sz

        padding_ratio = pad_area / max(total_area, 1)
        return min(1.0, padding_ratio)

    def _compute_quality_diagnostics(self, predictions: Dict) -> Dict:
        """
        Compute quality score diagnostics for failure analysis.

        CRITICAL FIX (Audit 2025111413 Finding 3): Extract quality trends
        to enable automated failure mode detection when fusion underperforms.

        Args:
            predictions: Predictions dict with quality scores

        Returns:
            Quality diagnostics dict with statistics
        """
        import numpy as np

        diagnostics = {}

        # RGB quality statistics
        if predictions.get('rgb_quality'):
            rgb_q = np.array(predictions['rgb_quality'])
            diagnostics['rgb_quality'] = {
                'mean': float(np.mean(rgb_q)),
                'std': float(np.std(rgb_q)),
                'min': float(np.min(rgb_q)),
                'max': float(np.max(rgb_q))
            }

        # Sonar quality statistics
        if predictions.get('sonar_quality'):
            sonar_q = np.array(predictions['sonar_quality'])
            diagnostics['sonar_quality'] = {
                'mean': float(np.mean(sonar_q)),
                'std': float(np.std(sonar_q)),
                'min': float(np.min(sonar_q)),
                'max': float(np.max(sonar_q))
            }

        # Fusion quality statistics
        if predictions.get('fusion_quality'):
            fusion_q = np.array(predictions['fusion_quality'])
            diagnostics['fusion_quality'] = {
                'mean': float(np.mean(fusion_q)),
                'std': float(np.std(fusion_q)),
                'min': float(np.min(fusion_q)),
                'max': float(np.max(fusion_q))
            }

        # Adaptive fusion weights
        if predictions.get('fusion_weights'):
            weights = np.array(predictions['fusion_weights'])
            if weights.ndim == 2 and weights.shape[1] == 2:
                diagnostics['fusion_weights'] = {
                    'rgb_mean': float(np.mean(weights[:, 0])),
                    'sonar_mean': float(np.mean(weights[:, 1])),
                    'rgb_std': float(np.std(weights[:, 0])),
                    'sonar_std': float(np.std(weights[:, 1]))
                }

        return diagnostics

    def _denormalize_bbox(
        self,
        bbox: np.ndarray,
        image_size: int,
        scale_x: float = None,
        scale_y: float = None,
        pad_x: int = 0,
        pad_y: int = 0,
        original_size: tuple = None,
        output_format: str = 'xywh',
        # CRITICAL FIX (Audit 2025112603): Support target-centered cropping
        crop_x1: int = None,
        crop_y1: int = None,
        resize_factor: float = None
    ) -> np.ndarray:
        """
        Convert normalized bbox [cx, cy, w, h] in [0,1] to PHYSICAL pixel coordinates.

        CRITICAL FIX (Audit 2025112603): Now supports both letterbox and target-centered cropping.
        - Letterbox mode: Uses scale_x, scale_y, pad_x, pad_y
        - Target-centered crop mode: Uses crop_x1, crop_y1, resize_factor

        Args:
            bbox: Bounding box in normalized coordinates [cx, cy, w, h] in [0, 1]
            image_size: Resized image size (e.g., 256)
            scale_x: X-axis scale factor (letterbox mode)
            scale_y: Y-axis scale factor (letterbox mode)
            pad_x: Left padding offset (letterbox mode)
            pad_y: Top padding offset (letterbox mode)
            original_size: Original image size (H, W) for final reconstruction
            output_format: Output format - 'xywh' or 'cxcywh'
            crop_x1: Crop origin X (target-centered crop mode, can be negative)
            crop_y1: Crop origin Y (target-centered crop mode, can be negative)
            resize_factor: output_size / crop_sz (target-centered crop mode)

        Returns:
            Bounding box in PHYSICAL pixel coordinates [x, y, w, h] or [cx, cy, w, h]
        """
        if bbox is None:
            return None

        # Check if bbox needs denormalization (max value < 2 suggests normalized coords)
        if np.max(np.abs(bbox)) >= 2.0:
            # Already in pixel coordinates
            return bbox

        # Step 1: Denormalize from [0, 1] to output image coordinates
        bbox_output = bbox * image_size  # [cx, cy, w, h] in output pixels

        # CRITICAL FIX (Audit 2025112603): Detect target-centered crop mode
        # If crop_x1/crop_y1/resize_factor are provided, use target-centered crop logic
        if crop_x1 is not None and crop_y1 is not None and resize_factor is not None:
            # Target-centered cropping mode
            # bbox_output is in crop pixel coordinates (0 to output_size)
            # Convert back to original image coordinates:
            # x_orig = x_crop / resize_factor + crop_x1
            cx_orig = bbox_output[0] / resize_factor + crop_x1
            cy_orig = bbox_output[1] / resize_factor + crop_y1
            w_orig = bbox_output[2] / resize_factor
            h_orig = bbox_output[3] / resize_factor
        else:
            # Legacy letterbox mode
            # Step 2: Remove letterbox padding offsets
            cx_scaled = bbox_output[0] - pad_x
            cy_scaled = bbox_output[1] - pad_y
            w_scaled = bbox_output[2]
            h_scaled = bbox_output[3]

            # Step 3: Undo per-axis scaling
            if scale_x is None:
                scale_x = 1.0
            if scale_y is None:
                scale_y = 1.0

            cx_orig = cx_scaled / scale_x
            cy_orig = cy_scaled / scale_y
            w_orig = w_scaled / scale_x
            h_orig = h_scaled / scale_y

        # Step 4: Clip to original image boundaries (if provided)
        if original_size is not None:
            orig_h, orig_w = original_size
            cx_orig = np.clip(cx_orig, 0, orig_w)
            cy_orig = np.clip(cy_orig, 0, orig_h)
            w_orig = np.clip(w_orig, 0, orig_w)
            h_orig = np.clip(h_orig, 0, orig_h)

        # Step 5: Convert to requested output format
        if output_format == 'xywh':
            x = cx_orig - w_orig / 2
            y = cy_orig - h_orig / 2
            return np.array([x, y, w_orig, h_orig], dtype=np.float32)
        elif output_format == 'cxcywh':
            return np.array([cx_orig, cy_orig, w_orig, h_orig], dtype=np.float32)
        else:
            raise ValueError(f"Unknown output_format: {output_format}. Expected 'xywh' or 'cxcywh'")

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

        # CRITICAL FIX (Audit 2025111411 Finding 4):
        # Extract metadata for quality estimators (must be done BEFORE model forward)
        # Support per-sample metadata (list of dicts) for batch_size > 1
        rgb_metadata_list = batch.get('rgb_search_metadata', [])
        sonar_metadata_list = batch.get('sonar_search_metadata', [])
        template_rgb_metadata = batch.get('template_rgb_metadata', [])
        template_sonar_metadata = batch.get('template_sonar_metadata', [])

        # Handle both list (new format) and dict (legacy format) for backward compatibility
        if isinstance(rgb_metadata_list, dict):
            rgb_metadata_list = [rgb_metadata_list]
        if isinstance(sonar_metadata_list, dict):
            sonar_metadata_list = [sonar_metadata_list]
        if isinstance(template_rgb_metadata, dict):
            template_rgb_metadata = [template_rgb_metadata]
        if isinstance(template_sonar_metadata, dict):
            template_sonar_metadata = [template_sonar_metadata]

        # Package metadata for tracker
        metadata = {
            'rgb_search_metadata': rgb_metadata_list,
            'sonar_search_metadata': sonar_metadata_list,
            'template_rgb_metadata': template_rgb_metadata,
            'template_sonar_metadata': template_sonar_metadata,
        }

        # CRITICAL FIX (Audit 2025120902 Issue 9 + Revised): Self-inferred validity
        # Original fix removed GT visibility (correct - no GT leakage).
        # However, ConditionalG2DCMA needs validity hints to bypass unreliable modalities.
        # Without validity hints, fusion always assumes both modalities are valid,
        # causing model to use corrupted features from heavily-padded crops.
        #
        # Solution: Self-infer validity from padding ratio (NOT from GT!)
        # This matches online evaluation and real deployment behavior.
        #
        # Validity inference: If padding_ratio >= PADDING_THRESHOLD, modality is invalid
        # because the crop is mostly outside the image boundary.
        VALIDITY_PADDING_THRESHOLD = 0.3  # Same threshold as online evaluator

        batch_size = rgb_template.shape[0]

        # Compute padding-based validity for each sample in batch
        inferred_rgb_valid = []
        inferred_sonar_valid = []

        for i in range(batch_size):
            # Compute RGB padding ratio
            rgb_meta = rgb_metadata_list[i] if i < len(rgb_metadata_list) else {}
            rgb_padding_ratio = self._compute_padding_ratio(rgb_meta)
            rgb_valid = rgb_padding_ratio < VALIDITY_PADDING_THRESHOLD

            # Compute Sonar padding ratio
            sonar_meta = sonar_metadata_list[i] if i < len(sonar_metadata_list) else {}
            sonar_padding_ratio = self._compute_padding_ratio(sonar_meta)
            sonar_valid = sonar_padding_ratio < VALIDITY_PADDING_THRESHOLD

            inferred_rgb_valid.append(rgb_valid)
            inferred_sonar_valid.append(sonar_valid)

        # Convert to tensors
        is_rgb_valid = torch.tensor(inferred_rgb_valid, dtype=torch.bool, device=self.device)
        is_sonar_valid = torch.tensor(inferred_sonar_valid, dtype=torch.bool, device=self.device)

        # CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised): Extract attention masks
        # Attention masks mark valid (non-padded) regions in search crops
        # The batch contains pre-computed masks from the dataloader
        rgb_att_mask = batch.get('rgb_att_mask', None)
        sonar_att_mask = batch.get('sonar_att_mask', None)
        # CRITICAL FIX (Audit 2025120909): Extract template attention masks
        rgb_template_att_mask = batch.get('rgb_template_att_mask', None)
        sonar_template_att_mask = batch.get('sonar_template_att_mask', None)

        # Move masks to device if they exist
        if rgb_att_mask is not None:
            rgb_att_mask = rgb_att_mask.to(self.device)
        if sonar_att_mask is not None:
            sonar_att_mask = sonar_att_mask.to(self.device)
        if rgb_template_att_mask is not None:
            rgb_template_att_mask = rgb_template_att_mask.to(self.device)
        if sonar_template_att_mask is not None:
            sonar_template_att_mask = sonar_template_att_mask.to(self.device)

        # Run model with SELF-INFERRED validity (not GT!)
        # This enables ConditionalG2DCMA to properly bypass unreliable modalities
        outputs = self.model(
            rgb_template, rgb_search, sonar_template, sonar_search,
            is_rgb_valid=is_rgb_valid,  # Self-inferred from padding ratio
            is_sonar_valid=is_sonar_valid,  # Self-inferred from padding ratio
            # has_object_*: Not needed in evaluation (only for training loss)
            metadata=metadata,
            return_losses=False,  # Evaluation mode (no loss computation)
            rgb_att_mask=rgb_att_mask,
            sonar_att_mask=sonar_att_mask,
            rgb_template_att_mask=rgb_template_att_mask,
            sonar_template_att_mask=sonar_template_att_mask
        )

        # Extract predictions with quality scores
        predictions = {
            'rgb_pred': [],
            'sonar_pred': [],
            'rgb_gt': [],
            'sonar_gt': [],
            # CRITICAL FIX (Audit 2025111413 Finding 3): Collect quality trends
            'rgb_quality': [],
            'sonar_quality': [],
            'fusion_quality': [],
            'fusion_weights': [],
            # CRITICAL FIX (Audit 2025112101 Phase 4.2): Collect scenario labels
            'scenario_labels': []
        }

        # Get predictions (model outputs plural keys: rgb_boxes, sonar_boxes, fusion_boxes)
        # Denormalize boxes from [0,1] to physical pixel coordinates using per-sample metadata
        #
        # BEST PRACTICE (Audit 2025111411 Finding 3): Use type-safe bbox wrappers at boundaries
        # to prevent format confusion. Model outputs normalized cxcywh [0,1], but evaluation
        # expects xywh pixels. Type wrappers make this conversion explicit:
        #   from aquasot3d.utils import BBoxCXCYWH, BBoxXYWH
        #   pred_norm_cxcywh = BBoxCXCYWH(outputs['rgb_boxes'])  # Model output
        #   pred_xywh = pred_norm_cxcywh.to_xywh()                # Convert to xywh format
        #   pred_pixels = denormalize_bbox(pred_xywh.tensor(), ...)  # Denormalize
        #   pred_bbox = BBoxXYWH(pred_pixels)                     # Type-safe pixel coords
        #
        if 'rgb_boxes' in outputs:
            # PRODUCTION USE (Audit 2025111411 Finding 3):
            # Use type-safe wrappers to make format conversion explicit
            from aquasot3d.utils import BBoxCXCYWH, BBoxXYWH

            rgb_boxes = outputs['rgb_boxes'].cpu()  # Keep as tensor for wrapper

            # CRITICAL FIX (Audit 2025111920): Keep cxcywh format for denormalization
            # _denormalize_bbox() expects cxcywh normalized input (see line 314-321)
            # Previous code incorrectly converted to xywh, causing format mismatch
            rgb_boxes_cxcywh = BBoxCXCYWH(rgb_boxes)
            rgb_boxes_np = rgb_boxes_cxcywh.numpy()  # Keep as cxcywh

            predictions['rgb_pred'] = [
                self._denormalize_bbox(
                    rgb_boxes_np[i],  # cxcywh normalized [0,1]
                    self.search_size,
                    scale_x=rgb_metadata_list[i].get('scale_x') if i < len(rgb_metadata_list) else None,
                    scale_y=rgb_metadata_list[i].get('scale_y') if i < len(rgb_metadata_list) else None,
                    pad_x=rgb_metadata_list[i].get('pad_x', 0) if i < len(rgb_metadata_list) else 0,
                    pad_y=rgb_metadata_list[i].get('pad_y', 0) if i < len(rgb_metadata_list) else 0,
                    original_size=(
                        rgb_metadata_list[i].get('original_height'),
                        rgb_metadata_list[i].get('original_width')
                    ) if i < len(rgb_metadata_list) else None,
                    # CRITICAL FIX (Audit 2025112603): Pass target-centered crop params
                    crop_x1=rgb_metadata_list[i].get('crop_x1') if i < len(rgb_metadata_list) else None,
                    crop_y1=rgb_metadata_list[i].get('crop_y1') if i < len(rgb_metadata_list) else None,
                    resize_factor=rgb_metadata_list[i].get('resize_factor') if i < len(rgb_metadata_list) else None
                )
                for i in range(len(rgb_boxes_np))
            ]

        if 'sonar_boxes' in outputs:
            # CRITICAL FIX (Audit 2025111920): Keep cxcywh format for denormalization
            from aquasot3d.utils import BBoxCXCYWH

            sonar_boxes = outputs['sonar_boxes'].cpu()
            sonar_boxes_cxcywh = BBoxCXCYWH(sonar_boxes)
            sonar_boxes_np = sonar_boxes_cxcywh.numpy()  # Keep as cxcywh

            predictions['sonar_pred'] = [
                self._denormalize_bbox(
                    sonar_boxes_np[i],  # cxcywh normalized [0,1]
                    self.search_size,
                    scale_x=sonar_metadata_list[i].get('scale_x') if i < len(sonar_metadata_list) else None,
                    scale_y=sonar_metadata_list[i].get('scale_y') if i < len(sonar_metadata_list) else None,
                    pad_x=sonar_metadata_list[i].get('pad_x', 0) if i < len(sonar_metadata_list) else 0,
                    pad_y=sonar_metadata_list[i].get('pad_y', 0) if i < len(sonar_metadata_list) else 0,
                    original_size=(
                        sonar_metadata_list[i].get('original_height'),
                        sonar_metadata_list[i].get('original_width')
                    ) if i < len(sonar_metadata_list) else None,
                    # CRITICAL FIX (Audit 2025112603): Pass target-centered crop params
                    crop_x1=sonar_metadata_list[i].get('crop_x1') if i < len(sonar_metadata_list) else None,
                    crop_y1=sonar_metadata_list[i].get('crop_y1') if i < len(sonar_metadata_list) else None,
                    resize_factor=sonar_metadata_list[i].get('resize_factor') if i < len(sonar_metadata_list) else None
                )
                for i in range(len(sonar_boxes_np))
            ]

        # ARCHITECTURE UPDATE (Audit 2025112313): fusion_boxes completely removed
        # RGB and Sonar boxes now use dual-view fused features (no separate fusion head)

        # Get ground truth boxes
        # CRITICAL FIX (Audit 2025112601): Use ORIGINAL space GT, NOT letterbox space
        # Predictions are denormalized to ORIGINAL image coordinates (1920x1080 for RGB, 682x376 for sonar)
        # GT must also be in ORIGINAL space for correct IoU/precision computation
        #
        # Key insight: search_bbox_rgb_pixels is in LETTERBOX space (256x256)
        #              search_bbox_rgb is in ORIGINAL space (1920x1080)
        # We need ORIGINAL space for both pred and GT!

        # RGB ground truth - prefer original space coordinates
        rgb_gt_key = 'search_bbox_rgb' if 'search_bbox_rgb' in batch else 'rgb_boxes'
        if rgb_gt_key in batch:
            rgb_gt = batch[rgb_gt_key]
            if isinstance(rgb_gt, torch.Tensor):
                rgb_gt = rgb_gt.cpu().numpy()
            # Check if in original space (values > search_size suggests original coords)
            if np.max(np.abs(rgb_gt[0])) >= self.search_size:
                # Already in original space - use directly
                predictions['rgb_gt'] = [rgb_gt[i] for i in range(len(rgb_gt))]
            else:
                # Likely normalized or letterbox space - need to denormalize
                # Use metadata to convert back to original space
                predictions['rgb_gt'] = [
                    self._denormalize_bbox(
                        rgb_gt[i],
                        self.search_size,
                        scale_x=rgb_metadata_list[i].get('scale_x') if i < len(rgb_metadata_list) else None,
                        scale_y=rgb_metadata_list[i].get('scale_y') if i < len(rgb_metadata_list) else None,
                        pad_x=rgb_metadata_list[i].get('pad_x', 0) if i < len(rgb_metadata_list) else 0,
                        pad_y=rgb_metadata_list[i].get('pad_y', 0) if i < len(rgb_metadata_list) else 0,
                        original_size=(
                            rgb_metadata_list[i].get('original_height'),
                            rgb_metadata_list[i].get('original_width')
                        ) if i < len(rgb_metadata_list) else None,
                        # CRITICAL FIX (Audit 2025112603): Pass target-centered crop params
                        crop_x1=rgb_metadata_list[i].get('crop_x1') if i < len(rgb_metadata_list) else None,
                        crop_y1=rgb_metadata_list[i].get('crop_y1') if i < len(rgb_metadata_list) else None,
                        resize_factor=rgb_metadata_list[i].get('resize_factor') if i < len(rgb_metadata_list) else None
                    )
                    for i in range(len(rgb_gt))
                ]

        # Sonar ground truth - prefer original space coordinates
        sonar_gt_key = 'search_bbox_sonar' if 'search_bbox_sonar' in batch else 'sonar_boxes'
        if sonar_gt_key in batch:
            sonar_gt = batch[sonar_gt_key]
            if isinstance(sonar_gt, torch.Tensor):
                sonar_gt = sonar_gt.cpu().numpy()
            # Check if in original space
            if np.max(np.abs(sonar_gt[0])) >= self.search_size:
                # Already in original space - use directly
                predictions['sonar_gt'] = [sonar_gt[i] for i in range(len(sonar_gt))]
            else:
                # Likely normalized or letterbox space - need to denormalize
                predictions['sonar_gt'] = [
                    self._denormalize_bbox(
                        sonar_gt[i],
                        self.search_size,
                        scale_x=sonar_metadata_list[i].get('scale_x') if i < len(sonar_metadata_list) else None,
                        scale_y=sonar_metadata_list[i].get('scale_y') if i < len(sonar_metadata_list) else None,
                        pad_x=sonar_metadata_list[i].get('pad_x', 0) if i < len(sonar_metadata_list) else 0,
                        pad_y=sonar_metadata_list[i].get('pad_y', 0) if i < len(sonar_metadata_list) else 0,
                        original_size=(
                            sonar_metadata_list[i].get('original_height'),
                            sonar_metadata_list[i].get('original_width')
                        ) if i < len(sonar_metadata_list) else None,
                        # CRITICAL FIX (Audit 2025112603): Pass target-centered crop params
                        crop_x1=sonar_metadata_list[i].get('crop_x1') if i < len(sonar_metadata_list) else None,
                        crop_y1=sonar_metadata_list[i].get('crop_y1') if i < len(sonar_metadata_list) else None,
                        resize_factor=sonar_metadata_list[i].get('resize_factor') if i < len(sonar_metadata_list) else None
                    )
                    for i in range(len(sonar_gt))
                ]

        # CRITICAL FIX (Audit 2025111413 Finding 3): Extract quality scores and fusion weights
        if 'rgb_quality' in outputs:
            predictions['rgb_quality'] = outputs['rgb_quality'].cpu().numpy().tolist()
        if 'sonar_quality' in outputs:
            predictions['sonar_quality'] = outputs['sonar_quality'].cpu().numpy().tolist()
        if 'fusion_quality' in outputs:
            predictions['fusion_quality'] = outputs['fusion_quality'].cpu().numpy().tolist()
        if 'fusion_weights' in outputs:
            predictions['fusion_weights'] = outputs['fusion_weights'].cpu().numpy().tolist()

        # =====================================================================
        # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
        # Use head validity outputs for prediction gating (replaces padding-only)
        # =====================================================================
        # The box head now outputs learned validity predictions that combine
        # padding ratio and objectness for SOTA-aligned fusion routing.
        # Priority for gating: validity_final > validity > confidence > quality
        #
        # Thresholds (from audit spec):
        # - RGB validity: 0.35 (more lenient, RGB typically more reliable)
        # - Sonar validity: 0.4 (stricter, sonar more prone to false positives)
        # - Fallback confidence: 0.3 (same as before)
        RGB_VALIDITY_THRESHOLD = 0.35
        SONAR_VALIDITY_THRESHOLD = 0.4
        CONFIDENCE_VALIDITY_THRESHOLD = 0.3  # Fallback when validity not available

        # Extract RGB validity from head outputs (preferred) or fallback to confidence
        rgb_valid_values = []
        if 'rgb_validity' in outputs and outputs['rgb_validity'] is not None:
            # CRITICAL FIX: Use head validity (includes padding + objectness learning)
            rgb_valid_values = outputs['rgb_validity'].cpu().numpy().tolist()
            rgb_threshold = RGB_VALIDITY_THRESHOLD
        elif 'rgb_objectness' in outputs and outputs['rgb_objectness'] is not None:
            rgb_valid_values = outputs['rgb_objectness'].cpu().numpy().tolist()
            rgb_threshold = CONFIDENCE_VALIDITY_THRESHOLD
        elif 'rgb_confidence' in outputs and outputs['rgb_confidence'] is not None:
            rgb_valid_values = outputs['rgb_confidence'].cpu().numpy().tolist()
            rgb_threshold = CONFIDENCE_VALIDITY_THRESHOLD
        elif 'rgb_quality' in outputs and outputs['rgb_quality'] is not None:
            rgb_valid_values = outputs['rgb_quality'].cpu().numpy().tolist()
            rgb_threshold = CONFIDENCE_VALIDITY_THRESHOLD
        else:
            rgb_threshold = CONFIDENCE_VALIDITY_THRESHOLD

        # Extract Sonar validity from head outputs (preferred) or fallback to confidence
        sonar_valid_values = []
        if 'sonar_validity' in outputs and outputs['sonar_validity'] is not None:
            # CRITICAL FIX: Use head validity (includes padding + objectness learning)
            sonar_valid_values = outputs['sonar_validity'].cpu().numpy().tolist()
            sonar_threshold = SONAR_VALIDITY_THRESHOLD
        elif 'sonar_objectness' in outputs and outputs['sonar_objectness'] is not None:
            sonar_valid_values = outputs['sonar_objectness'].cpu().numpy().tolist()
            sonar_threshold = CONFIDENCE_VALIDITY_THRESHOLD
        elif 'sonar_confidence' in outputs and outputs['sonar_confidence'] is not None:
            sonar_valid_values = outputs['sonar_confidence'].cpu().numpy().tolist()
            sonar_threshold = CONFIDENCE_VALIDITY_THRESHOLD
        elif 'sonar_quality' in outputs and outputs['sonar_quality'] is not None:
            sonar_valid_values = outputs['sonar_quality'].cpu().numpy().tolist()
            sonar_threshold = CONFIDENCE_VALIDITY_THRESHOLD
        else:
            sonar_threshold = CONFIDENCE_VALIDITY_THRESHOLD

        # Apply validity/confidence gating to predictions
        # Replace bbox with None if validity is below threshold
        if rgb_valid_values and len(rgb_valid_values) == len(predictions.get('rgb_pred', [])):
            for i, val in enumerate(rgb_valid_values):
                if val < rgb_threshold:
                    predictions['rgb_pred'][i] = None  # Mark as invalid prediction

        if sonar_valid_values and len(sonar_valid_values) == len(predictions.get('sonar_pred', [])):
            for i, val in enumerate(sonar_valid_values):
                if val < sonar_threshold:
                    predictions['sonar_pred'][i] = None  # Mark as invalid prediction

        # Store validity values for analysis/debugging
        predictions['rgb_validity'] = rgb_valid_values
        predictions['sonar_validity'] = sonar_valid_values

        # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
        # Classify each frame into modality availability scenarios using HEAD VALIDITY
        # This uses learned validity (not GT or padding-only) for consistent evaluation
        if rgb_valid_values and sonar_valid_values:
            # Convert validity values to boolean using thresholds
            for i in range(min(len(rgb_valid_values), len(sonar_valid_values))):
                rgb_valid = rgb_valid_values[i] >= rgb_threshold
                sonar_valid = sonar_valid_values[i] >= sonar_threshold

                # Classify scenario based on head validity
                if rgb_valid and sonar_valid:
                    scenario = 'dual'
                elif rgb_valid and not sonar_valid:
                    scenario = 'single_rgb'
                elif not rgb_valid and sonar_valid:
                    scenario = 'single_sonar'
                else:
                    scenario = 'zero'

                predictions['scenario_labels'].append(scenario)
        elif 'is_rgb_valid' in outputs and 'is_sonar_valid' in outputs:
            # Fallback to passed-in validity (padding-based) if head validity not available
            is_rgb_valid = outputs['is_rgb_valid'].cpu()  # [B] boolean tensor
            is_sonar_valid = outputs['is_sonar_valid'].cpu()  # [B] boolean tensor

            # Convert to scenario labels
            for i in range(len(is_rgb_valid)):
                rgb_valid = is_rgb_valid[i].item()
                sonar_valid = is_sonar_valid[i].item()

                # Classify scenario
                if rgb_valid and sonar_valid:
                    scenario = 'dual'
                elif rgb_valid and not sonar_valid:
                    scenario = 'single_rgb'
                elif not rgb_valid and sonar_valid:
                    scenario = 'single_sonar'
                else:
                    scenario = 'zero'

                predictions['scenario_labels'].append(scenario)

        return predictions

    def compute_sequence_metrics(
        self,
        predictions: Dict[str, List[np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for a single sequence.

        CRITICAL FIX (Audit 2025112101 Phase 4.2):
        Now computes both standard and per-scenario metrics.

        Args:
            predictions: Dictionary with predictions and ground truth

        Returns:
            Metrics dictionary for all modalities (standard + per-scenario)
        """
        # Import scenario metric functions
        from aquasot3d.evaluation.metrics import compute_scenario_metrics

        # ARCHITECTURE UPDATE (Audit 2025112313): fusion completely removed
        # Compute standard metrics for RGB and Sonar only
        from aquasot3d.evaluation.metrics import compute_sequence_metrics

        metrics = {}
        metrics['rgb'] = compute_sequence_metrics(
            predictions.get('rgb_pred', []),
            predictions.get('rgb_gt', []),
            'rgb',
            additional_metrics=self.additional_metrics
        )
        metrics['sonar'] = compute_sequence_metrics(
            predictions.get('sonar_pred', []),
            predictions.get('sonar_gt', []),
            'sonar',
            additional_metrics=self.additional_metrics
        )

        # CRITICAL FIX (Audit 2025112101 Phase 4.2):
        # Compute per-scenario metrics if scenario labels available
        if 'scenario_labels' in predictions and len(predictions['scenario_labels']) > 0:
            # Compute per-scenario metrics for RGB (primary output in new architecture)
            from aquasot3d.evaluation.metrics import compute_scenario_metrics
            rgb_scenario_metrics = compute_scenario_metrics(
                predictions.get('rgb_pred', []),
                predictions.get('rgb_gt', []),
                predictions.get('scenario_labels', []),
                modality='rgb',
                additional_metrics=self.additional_metrics
            )

            # Add scenario metrics to the results
            metrics['rgb_scenarios'] = rgb_scenario_metrics

            # Compute per-scenario metrics for Sonar
            sonar_scenario_metrics = compute_scenario_metrics(
                predictions.get('sonar_pred', []),
                predictions.get('sonar_gt', []),
                predictions.get('scenario_labels', []),
                modality='sonar',
                additional_metrics=self.additional_metrics
            )
            metrics['sonar_scenarios'] = sonar_scenario_metrics

        return metrics

    def aggregate_all_metrics(
        self,
        sequence_metrics: List[Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across all sequences.

        CRITICAL FIX (Audit 2025112101 Phase 4.2):
        Now aggregates both standard and per-scenario metrics.

        Args:
            sequence_metrics: List of per-sequence metrics

        Returns:
            Aggregated metrics for each modality (standard + per-scenario)
        """
        from aquasot3d.evaluation.metrics import aggregate_scenario_metrics

        aggregated = {}

        # CRITICAL FIX (Audit 2025120902 Issue 18): Remove 'fusion' from modality list
        # Fusion boxes were removed in architecture update (Audit 2025112313)
        # Previous code still iterated ['rgb', 'sonar', 'fusion'] causing errors
        # Now only aggregate metrics for RGB and Sonar modalities
        for modality in ['rgb', 'sonar']:  # Removed 'fusion'
            # Extract metrics for this modality
            modality_metrics = [
                seq_metrics[modality]
                for seq_metrics in sequence_metrics
                if modality in seq_metrics
            ]

            # Aggregate
            if len(modality_metrics) > 0:
                aggregated[modality] = aggregate_metrics(modality_metrics, modality)

        # CRITICAL FIX (Audit 2025112101 Phase 4.2):
        # Aggregate per-scenario metrics
        # CRITICAL FIX (Audit 2025120902 Issue 18): Remove 'fusion' from modality list
        for modality in ['rgb', 'sonar']:  # Removed 'fusion'
            scenario_key = f'{modality}_scenarios'

            # Extract scenario metrics for this modality
            scenario_metrics_list = [
                seq_metrics[scenario_key]
                for seq_metrics in sequence_metrics
                if scenario_key in seq_metrics
            ]

            # Aggregate scenario metrics
            if len(scenario_metrics_list) > 0:
                aggregated[scenario_key] = aggregate_scenario_metrics(
                    scenario_metrics_list,
                    modality=modality
                )

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

        # CRITICAL FIX (Audit 2025120902 Issue 18): Remove 'fusion' from modality list
        for modality in ['rgb', 'sonar']:  # Removed 'fusion'
            if modality not in self.overall_metrics:
                continue

            metrics = self.overall_metrics[modality]
            print(f"\n{modality.upper()} Results:")
            print("-" * 50)
            print(f"  Success (AUC):      {metrics.get('success_auc_mean', 0.0):.4f} ± {metrics.get('success_auc_std', 0.0):.4f}")
            print(f"  Precision@20:       {metrics.get('precision_20_mean', 0.0):.4f} ± {metrics.get('precision_20_std', 0.0):.4f}")
            print(f"  Avg Center Error:   {metrics.get('avg_center_error_mean', 0.0):.2f} ± {metrics.get('avg_center_error_std', 0.0):.2f} px")
            print(f"  Mean IoU:           {metrics.get('mean_iou_mean', 0.0):.4f} ± {metrics.get('mean_iou_std', 0.0):.4f}")

        # CRITICAL FIX (Audit 2025120902 Issue 18): Print per-scenario metrics for RGB (primary)
        # Previously printed fusion_scenarios which no longer exists
        if 'rgb_scenarios' in self.overall_metrics:
            print("\n" + "=" * 70)
            print("PER-SCENARIO PERFORMANCE (RGB)")
            print("=" * 70)

            scenario_metrics = self.overall_metrics['rgb_scenarios']

            for scenario, scenario_name in [
                ('dual', 'Dual-Modality (RGB + Sonar)'),
                ('single_rgb', 'Single-RGB'),
                ('single_sonar', 'Single-Sonar'),
                ('zero', 'Zero-Modality')
            ]:
                if scenario in scenario_metrics:
                    m = scenario_metrics[scenario]
                    num_seqs = m.get('num_sequences', 0)

                    print(f"\n{scenario_name}:")
                    print("-" * 50)

                    if num_seqs == 0:
                        print("  No samples in this scenario")
                        continue

                    print(f"  Success (AUC):      {m.get('success_auc_mean', 0.0):.4f} ± {m.get('success_auc_std', 0.0):.4f}")
                    print(f"  Precision@20:       {m.get('precision_20_mean', 0.0):.4f} ± {m.get('precision_20_std', 0.0):.4f}")
                    print(f"  Avg Center Error:   {m.get('avg_center_error_mean', 0.0):.2f} ± {m.get('avg_center_error_std', 0.0):.2f} px")
                    print(f"  Mean IoU:           {m.get('mean_iou_mean', 0.0):.4f} ± {m.get('mean_iou_std', 0.0):.4f}")

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

        # CRITICAL FIX (Audit 2025120902 Issue 18): Use 'rgb' instead of 'fusion'
        if 'rgb' in self.overall_metrics:
            total_frames = self.overall_metrics['rgb'].get('total_frames', 0)
            lines.append(f"  Total Frames: {total_frames}")

        lines.append("")

        # Per-modality results
        # CRITICAL FIX (Audit 2025120902 Issue 18): Remove 'fusion' from modality list
        for modality in ['rgb', 'sonar']:  # Removed 'fusion'
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
