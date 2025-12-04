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

    # Confidence-based search area expansion parameters (mainstream defaults)
    CONFIDENCE_HIGH = 0.5       # High confidence threshold
    CONFIDENCE_LOW = 0.3        # Low confidence threshold
    EXPANSION_FACTOR = 2.0      # Maximum expansion: 2x normal search area

    # Template quality thresholds (Audit 2025120402)
    # Used to filter out incomplete/edge-clipped templates during initialization
    TEMPLATE_MIN_SIZE = 5.0             # Minimum bbox dimension (pixels)
    TEMPLATE_EDGE_MARGIN = 2            # Minimum distance from image edge (pixels)
    TEMPLATE_SCAN_FRAMES = 20           # Number of frames to scan for best template

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        template_size: int = 128,
        search_size: int = 256,
        template_area_factor: float = 2.0,
        search_area_factor: float = 4.0,
        normalization_config: Optional[Dict] = None,
        # Search area expansion for tracking recovery
        enable_search_expansion: bool = True,
        confidence_high: float = None,
        confidence_low: float = None,
        max_expansion_factor: float = None,
        # CRITICAL FIX (Audit 2025120401 Finding 6): Template update configuration
        enable_template_update: bool = True,
        template_update_interval: int = 10,  # Update template every N frames
        template_update_threshold: float = 0.7,  # Minimum confidence to update
        # CRITICAL FIX (Audit 2025120402 Finding 1-2): Template quality filtering
        template_min_size: float = None,        # Min bbox dimension (pixels)
        template_edge_margin: int = None,       # Min distance from image edge (pixels)
        template_scan_frames: int = None,       # Frames to scan for best template
        enable_template_quality_filter: bool = True  # Enable quality-based template selection
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self._base_search_area_factor = search_area_factor  # Store original

        # Search area expansion for tracking recovery
        self.enable_search_expansion = enable_search_expansion
        self.confidence_high = confidence_high or self.CONFIDENCE_HIGH
        self.confidence_low = confidence_low or self.CONFIDENCE_LOW
        self.max_expansion_factor = max_expansion_factor or self.EXPANSION_FACTOR

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

        # CRITICAL FIX (Audit 2025120401 Finding 6): Template update parameters
        self.enable_template_update = enable_template_update
        self.template_update_interval = template_update_interval
        self.template_update_threshold = template_update_threshold

        # CRITICAL FIX (Audit 2025120402 Finding 1-2): Template quality filtering
        self.template_min_size = template_min_size or self.TEMPLATE_MIN_SIZE
        self.template_edge_margin = template_edge_margin or self.TEMPLATE_EDGE_MARGIN
        self.template_scan_frames = template_scan_frames or self.TEMPLATE_SCAN_FRAMES
        self.enable_template_quality_filter = enable_template_quality_filter

        logger.info(
            f"OnlineTrackingEvaluator initialized: "
            f"template_size={template_size}, search_size={search_size}, "
            f"search_area_factor={search_area_factor}, "
            f"enable_search_expansion={enable_search_expansion}, "
            f"enable_template_update={enable_template_update}, "
            f"enable_template_quality_filter={enable_template_quality_filter}"
        )

    def evaluate_sequence(
        self,
        base_dataset,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single sequence using online tracking protocol.

        CRITICAL FIX (Audit 2025120301 Finding 6):
        - Added visibility check for template initialization
        - Searches for first valid (non-zero bbox) frame to initialize template
        - Handles sequences that start with occlusion gracefully

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

        # === CRITICAL FIX (Audit 2025120402 Finding 1-2): Template quality filtering ===
        # Select the best template frame based on completeness and quality,
        # not just the first valid frame. This handles edge-clipped/incomplete templates.
        if self.enable_template_quality_filter:
            template_frame_idx, selection_reason = self._select_best_template_frame(
                base_dataset, num_frames
            )
        else:
            # Fallback: use first valid frame (original behavior for --baseline mode)
            template_frame_idx, selection_reason = self._find_first_valid_frame(
                base_dataset, num_frames
            )

        if template_frame_idx < 0:
            # No valid frame found - entire sequence has no visible target
            logger.warning(
                f"Sequence {sequence_name}: No valid frame found for template initialization. "
                f"All frames have zero/invalid bbox."
            )
            # Return empty predictions (no tracking possible)
            return predictions

        if template_frame_idx > 0 or selection_reason != "dual_complete":
            logger.info(
                f"Sequence {sequence_name}: Template selected from frame {template_frame_idx} "
                f"(reason: {selection_reason})"
            )

        # CRITICAL FIX (Audit 2025120403 Issue 8 & 21): Store None for pre-init frames
        # Before initialization, no prediction is possible, so store None (not [0,0,0,0])
        # This ensures metrics.py _is_valid_bbox() properly skips these frames
        for skip_idx in range(template_frame_idx):
            skip_frame_data = base_dataset._get_frame_data(skip_idx)
            predictions['rgb_pred'].append(None)  # No prediction possible before init
            predictions['sonar_pred'].append(None)
            predictions['rgb_gt'].append(skip_frame_data['rgb_bbox'].copy())
            predictions['sonar_gt'].append(skip_frame_data['sonar_bbox'].copy())

        # === Initialize tracker from first visible frame ===
        frame0_data = base_dataset._get_frame_data(template_frame_idx)

        # Get template frame GT bboxes
        rgb_gt_bbox = frame0_data['rgb_bbox'].copy()
        sonar_gt_bbox = frame0_data['sonar_bbox'].copy()

        # CRITICAL FIX (Audit 2025120403 Issue 6): NO cross-modality bbox borrowing
        # Cross-modal fallback breaks uncalibrated multi-modal tracking
        # because RGB and Sonar may have different coordinate systems.
        # Instead, handle missing modality by skipping or using center fallback.
        rgb_is_valid = np.sum(np.abs(rgb_gt_bbox)) > 0
        sonar_is_valid = np.sum(np.abs(sonar_gt_bbox)) > 0

        if not rgb_is_valid and not sonar_is_valid:
            # Neither modality valid - cannot initialize
            logger.warning(
                f"Sequence {sequence_name}: Frame {template_frame_idx} has no valid bbox. "
                f"Searching for next valid frame..."
            )
            # Find next valid frame
            for next_idx in range(template_frame_idx + 1, num_frames):
                next_data = base_dataset._get_frame_data(next_idx)
                if np.sum(np.abs(next_data['rgb_bbox'])) > 0 or np.sum(np.abs(next_data['sonar_bbox'])) > 0:
                    template_frame_idx = next_idx
                    frame0_data = next_data
                    rgb_gt_bbox = frame0_data['rgb_bbox'].copy()
                    sonar_gt_bbox = frame0_data['sonar_bbox'].copy()
                    rgb_is_valid = np.sum(np.abs(rgb_gt_bbox)) > 0
                    sonar_is_valid = np.sum(np.abs(sonar_gt_bbox)) > 0
                    break
            else:
                logger.warning(f"Sequence {sequence_name}: No valid frames found at all")
                return predictions

        # Use image center as fallback for missing modality (not cross-modal bbox)
        if not rgb_is_valid:
            rgb_img = frame0_data['rgb_img']
            H, W = rgb_img.shape[:2]
            rgb_gt_bbox = np.array([W/4, H/4, W/2, H/2], dtype=np.float32)
            logger.debug(f"RGB init: using image center fallback due to invalid GT")

        if not sonar_is_valid:
            sonar_img = frame0_data['sonar_img']
            H, W = sonar_img.shape[:2]
            sonar_gt_bbox = np.array([W/4, H/4, W/2, H/2], dtype=np.float32)
            logger.debug(f"Sonar init: using image center fallback due to invalid GT")

        # Initialize tracking state with GT (standard protocol)
        # State = predicted bbox from previous frame (or GT for frame 0)
        rgb_state = rgb_gt_bbox.copy()
        sonar_state = sonar_gt_bbox.copy()

        # Crop and cache templates from first visible frame
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

        # For template frame, prediction = GT (initialization)
        predictions['rgb_pred'].append(rgb_gt_bbox.copy())
        predictions['sonar_pred'].append(sonar_gt_bbox.copy())
        predictions['rgb_gt'].append(rgb_gt_bbox.copy())
        predictions['sonar_gt'].append(sonar_gt_bbox.copy())

        # Initialize confidence tracking for search area expansion
        # Start with high confidence (just initialized from GT)
        rgb_conf_prev = 1.0
        sonar_conf_prev = 1.0
        rgb_search_factor = self._base_search_area_factor
        sonar_search_factor = self._base_search_area_factor

        # Track expansion statistics
        expansion_count = 0

        # CRITICAL FIX (Audit 2025120401 Finding 6): Template update tracking
        template_update_count = 0
        frames_since_update = 0

        # === FRAMES template_frame_idx+1 to N-1: Online tracking ===
        # CRITICAL FIX (Audit 2025120301 Finding 6): Start from next frame after template
        start_frame = template_frame_idx + 1
        frame_iter = range(start_frame, num_frames)
        if verbose:
            frame_iter = tqdm(frame_iter, desc=f"  {sequence_name}", unit="frame", leave=False)

        with torch.no_grad():
            for frame_idx in frame_iter:
                frame_data = base_dataset._get_frame_data(frame_idx)

                # Get current frame GT (for evaluation only, not used for tracking!)
                rgb_gt_bbox = frame_data['rgb_bbox'].copy()
                sonar_gt_bbox = frame_data['sonar_bbox'].copy()

                # === Compute dynamic search factors based on previous confidence ===
                rgb_search_factor, sonar_search_factor = self._compute_search_factor(
                    rgb_conf_prev, sonar_conf_prev
                )

                # Track if expansion is active
                if rgb_search_factor > self._base_search_area_factor * 1.1:
                    expansion_count += 1

                # === CRITICAL: Crop search region centered on PREVIOUS PREDICTION ===
                # This is the key difference from offline evaluation!
                rgb_search, rgb_search_meta = self._crop_region(
                    frame_data['rgb_img'],
                    rgb_state,  # Use PREVIOUS PREDICTION, not GT!
                    self.search_size,
                    rgb_search_factor,  # Dynamic search factor
                    is_sonar=False,
                    return_metadata=True
                )
                sonar_search, sonar_search_meta = self._crop_region(
                    frame_data['sonar_img'],
                    sonar_state,  # Use PREVIOUS PREDICTION, not GT!
                    self.search_size,
                    sonar_search_factor,  # Dynamic search factor
                    is_sonar=True,
                    return_metadata=True
                )

                # Move to device and add batch dimension
                rgb_search = rgb_search.unsqueeze(0).to(self.device)
                sonar_search = sonar_search.unsqueeze(0).to(self.device)

                # === CRITICAL FIX (Audit 2025120401 Finding 1): Build metadata for fusion ===
                # GeometryGatedAttention requires sonar_letterbox_metadata for correct radial binning
                # Without this, geometric gating degrades to no-geometry baseline
                metadata = {
                    'rgb_search_metadata': [rgb_search_meta],
                    'sonar_search_metadata': [sonar_search_meta],  # List format for tracker.py
                    'sonar_metadata': [sonar_search_meta]  # Alternative key used by tracker
                }

                # === CRITICAL FIX (Audit 2025120403 Issue 4): NO GT visibility in model forward ===
                # Previous implementation leaked GT occlusion status to model, masking true routing failures.
                # Model must predict objectness from features alone, not from GT hints.
                # Occlusion status is determined AFTER prediction using confidence/objectness scores.
                rgb_gt_occluded = np.sum(np.abs(rgb_gt_bbox)) == 0
                sonar_gt_occluded = np.sum(np.abs(sonar_gt_bbox)) == 0

                # CRITICAL FIX (Audit 2025120403 Issue 4):
                # DO NOT pass GT-derived is_rgb_valid/is_sonar_valid to model!
                # Model should always assume both modalities are valid during inference.
                # Routing decisions must come from model's own confidence predictions.

                # === Model forward pass ===
                # CRITICAL FIX (Audit 2025120401 Finding 1): Pass metadata to model
                # CRITICAL FIX (Audit 2025120403 Issue 4): Remove GT validity leakage
                outputs = self.model(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search,
                    # is_rgb_valid=REMOVED - model must infer from features
                    # is_sonar_valid=REMOVED - model must infer from features
                    return_losses=False,
                    metadata=metadata
                )

                # === Extract and denormalize predictions ===
                # Model outputs normalized cxcywh in [0, 1]
                rgb_pred_norm = outputs['rgb_boxes'][0].cpu().numpy()  # [cx, cy, w, h]
                sonar_pred_norm = outputs['sonar_boxes'][0].cpu().numpy()

                # CRITICAL FIX (Audit 2025120403 Issue 9): Use objectness, not quality
                # Quality scores indicate "how good" the prediction is but can be high even
                # when the model is confidently wrong (locked on wrong peak).
                # Objectness indicates "is there an object" which better triggers recovery.
                #
                # Priority order:
                # 1. objectness - direct object presence indicator
                # 2. confidence - alternative name for objectness in some models
                # 3. quality - fallback (less reliable for recovery triggering)

                # RGB confidence extraction
                if 'rgb_objectness' in outputs and outputs['rgb_objectness'] is not None:
                    rgb_conf_prev = outputs['rgb_objectness'][0].cpu().item()
                elif 'rgb_confidence' in outputs and outputs['rgb_confidence'] is not None:
                    rgb_conf_prev = outputs['rgb_confidence'][0].cpu().item()
                elif 'rgb_quality' in outputs and outputs['rgb_quality'] is not None:
                    # Fallback to quality (less reliable)
                    rgb_conf_prev = outputs['rgb_quality'][0].cpu().item()
                else:
                    rgb_conf_prev = 0.5  # Conservative default triggers moderate expansion

                # Sonar confidence extraction
                if 'sonar_objectness' in outputs and outputs['sonar_objectness'] is not None:
                    sonar_conf_prev = outputs['sonar_objectness'][0].cpu().item()
                elif 'sonar_confidence' in outputs and outputs['sonar_confidence'] is not None:
                    sonar_conf_prev = outputs['sonar_confidence'][0].cpu().item()
                elif 'sonar_quality' in outputs and outputs['sonar_quality'] is not None:
                    sonar_conf_prev = outputs['sonar_quality'][0].cpu().item()
                else:
                    sonar_conf_prev = 0.5

                # Transform from normalized crop space to original image space
                rgb_pred_orig = self._transform_pred_to_original(
                    rgb_pred_norm, rgb_search_meta
                )
                sonar_pred_orig = self._transform_pred_to_original(
                    sonar_pred_norm, sonar_search_meta
                )

                # CRITICAL FIX (Audit 2025120403 Issue 2 & 7): Sanitize predictions
                # Clamp to image bounds, reject NaN/Inf, use fallback for invalid predictions
                rgb_img_size = (frame_data['rgb_img'].shape[0], frame_data['rgb_img'].shape[1])
                sonar_img_size = (frame_data['sonar_img'].shape[0], frame_data['sonar_img'].shape[1])

                rgb_pred_sanitized = self._sanitize_bbox(
                    rgb_pred_orig, rgb_img_size, fallback_bbox=rgb_state.copy()
                )
                sonar_pred_sanitized = self._sanitize_bbox(
                    sonar_pred_orig, sonar_img_size, fallback_bbox=sonar_state.copy()
                )

                # === Update state for next frame ===
                # CRITICAL FIX (Audit 2025120403 Issue 2 & 7): Use sanitized predictions
                # State update based on sanitized prediction validity and confidence

                # RGB state update - use confidence-based decision (not GT occlusion)
                if rgb_pred_sanitized is not None and rgb_conf_prev >= self.confidence_low:
                    rgb_state = rgb_pred_sanitized.copy()
                # else: keep previous state (low confidence or invalid prediction)

                # Sonar state update
                if sonar_pred_sanitized is not None and sonar_conf_prev >= self.confidence_low:
                    sonar_state = sonar_pred_sanitized.copy()

                # CRITICAL FIX (Audit 2025120403 Issue 8 & 21): Store None for occluded frames
                # Instead of storing [0,0,0,0], store None to properly signal "no prediction".
                # metrics.py _is_valid_bbox() will skip None entries during metric computation.
                # This follows standard tracking benchmarks (LaSOT, GOT-10k) protocol.
                if rgb_gt_occluded:
                    predictions['rgb_pred'].append(None)  # Not [0,0,0,0]
                else:
                    predictions['rgb_pred'].append(rgb_pred_sanitized)

                if sonar_gt_occluded:
                    predictions['sonar_pred'].append(None)  # Not [0,0,0,0]
                else:
                    predictions['sonar_pred'].append(sonar_pred_sanitized)

                predictions['rgb_gt'].append(rgb_gt_bbox)
                predictions['sonar_gt'].append(sonar_gt_bbox)

                # === CRITICAL FIX (Audit 2025120401 Finding 6): Template update ===
                # Update template periodically when tracking is confident
                # This aligns with MixFormer-style dynamic template update during training
                frames_since_update += 1

                if self.enable_template_update:
                    # CRITICAL FIX (Audit 2025120403 Issue 9): Use objectness, not GT occlusion
                    # Template update decision based on model confidence, not GT visibility
                    avg_conf = (rgb_conf_prev + sonar_conf_prev) / 2.0

                    # Check if we should update template (confidence-based, not GT-based)
                    should_update = (
                        frames_since_update >= self.template_update_interval and
                        avg_conf >= self.template_update_threshold and
                        rgb_pred_sanitized is not None and
                        sonar_pred_sanitized is not None
                    )

                    if should_update:
                        # Update templates using sanitized prediction as bbox
                        # CRITICAL FIX (Audit 2025120403): Use sanitized predictions
                        new_rgb_template, _ = self._crop_region(
                            frame_data['rgb_img'],
                            rgb_pred_sanitized,  # Use sanitized prediction, not raw
                            self.template_size,
                            self.template_area_factor,
                            is_sonar=False
                        )
                        new_sonar_template, _ = self._crop_region(
                            frame_data['sonar_img'],
                            sonar_pred_sanitized,  # Use sanitized prediction, not raw
                            self.template_size,
                            self.template_area_factor,
                            is_sonar=True
                        )

                        # Update cached templates
                        rgb_template = new_rgb_template.unsqueeze(0).to(self.device)
                        sonar_template = new_sonar_template.unsqueeze(0).to(self.device)

                        # Reset counter
                        frames_since_update = 0
                        template_update_count += 1

        # Log expansion statistics
        if expansion_count > 0:
            logger.debug(
                f"Sequence {sequence_name}: search expansion triggered in "
                f"{expansion_count}/{num_frames-1} frames ({100*expansion_count/(num_frames-1):.1f}%)"
            )

        # Log template update statistics
        if template_update_count > 0:
            logger.debug(
                f"Sequence {sequence_name}: template updated {template_update_count} times"
            )

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

        CRITICAL FIX (Audit 2025120301 Finding 5):
        - Added NaN/Inf/extreme value protection for bbox
        - Added crop_sz upper limit to prevent memory explosion
        - Returns safe fallback crop when bbox is invalid

        CRITICAL FIX (Audit 2025120401 Finding 1-2):
        - Returns letterbox-format metadata compatible with GeometryGatedAttention
        - Keys: scale, pad_top, pad_bottom, original_height, original_width, new_size
        - This enables correct radial binning in geometric gating fusion

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
                                 AND letterbox-format metadata for GeometryGatedAttention
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

        # CRITICAL FIX (Audit 2025120301 Finding 5):
        # Validate bbox - protect against NaN/Inf/extreme values
        x, y, w, h = bbox
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(w) or not np.isfinite(h):
            logger.warning(f"Invalid bbox detected: {bbox}, using image center fallback")
            x, y, w, h = W / 4, H / 4, W / 2, H / 2  # Fallback to image center

        # Clamp extreme values
        max_dim = max(H, W)
        x = np.clip(x, -max_dim, max_dim * 2)
        y = np.clip(y, -max_dim, max_dim * 2)
        w = np.clip(w, 1, max_dim * 2)
        h = np.clip(h, 1, max_dim * 2)

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

        # CRITICAL FIX (Audit 2025120301 Finding 5):
        # Limit crop_sz to prevent memory explosion from drifted predictions
        # Cap at 2x max(H, W) with absolute maximum of 2048
        max_crop_sz = min(2 * max(H, W), 2048)
        if crop_sz > max_crop_sz:
            logger.warning(
                f"Crop size {crop_sz} exceeds limit {max_crop_sz}, clamping. "
                f"bbox={[x, y, w, h]}, area_factor={area_factor}"
            )
            crop_sz = max_crop_sz

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
            # CRITICAL FIX (Audit 2025120401 Finding 10):
            # Return crop metadata in unified format. The tracker.py fuse_features()
            # will automatically convert this to letterbox format using
            # crop_metadata_to_letterbox() for GeometryGatedAttention compatibility.
            #
            # This ensures consistent metadata handling across training/validation/online.

            metadata = {
                # Crop metadata for bbox inverse transform
                'crop_center': (cx, cy),
                'crop_sz': crop_sz,
                'resize_factor': resize_factor,
                'crop_x1': x1,
                'crop_y1': y1,
                'original_size': (H, W),
                'output_size': output_size,
                # Padding in crop space (before resize) - needed for letterbox conversion
                'pad_left': x1_pad,
                'pad_top': y1_pad,
                'pad_right': x2_pad,
                'pad_bottom': y2_pad,
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

        DESIGN NOTE (Audit 2025120403 Issue 14 & 17):
        The crop_x1/crop_y1 values already encode padding information:
        - When crop extends beyond image left edge: crop_x1 is negative
        - The magnitude of negative crop_x1 equals pad_left
        - Therefore: original_x = crop_x + crop_x1 (no separate padding subtraction)

        This follows OSTrack's inverse transform methodology where crop_x1
        represents the crop origin in original image coordinates (can be negative).

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
        # NOTE: crop_x1/crop_y1 already encode padding offset (negative when padded)
        # So this single addition handles both normal and padded cases correctly.
        cx_orig = cx_orig_crop + crop_x1
        cy_orig = cy_orig_crop + crop_y1

        # Step 4: Convert from cxcywh to xywh
        x_orig = cx_orig - w_orig / 2
        y_orig = cy_orig - h_orig / 2

        return np.array([x_orig, y_orig, w_orig, h_orig], dtype=np.float32)

    def _sanitize_bbox(
        self,
        bbox: np.ndarray,
        image_size: Tuple[int, int],
        fallback_bbox: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Sanitize bbox: reject NaN/Inf, clamp to image bounds.

        CRITICAL FIX (Audit 2025120403 Issue 2 & 7):
        - Filters out NaN/Inf predictions that cause metric explosion
        - Clamps bbox to image boundaries to prevent drift
        - Returns fallback or None for invalid inputs

        Args:
            bbox: Input bbox [x, y, w, h] in XYWH format
            image_size: Image dimensions (H, W)
            fallback_bbox: Optional fallback bbox if input is invalid

        Returns:
            Sanitized bbox or fallback/None if input is invalid
        """
        if bbox is None:
            return fallback_bbox

        # Check for NaN/Inf
        if not np.all(np.isfinite(bbox)):
            logger.warning(f"NaN/Inf detected in bbox: {bbox}, using fallback")
            return fallback_bbox

        x, y, w, h = bbox
        H, W = image_size

        # Reject completely invalid dimensions
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid bbox dimensions: w={w}, h={h}, using fallback")
            return fallback_bbox

        # Clamp dimensions to reasonable range (max 2x image size)
        w = np.clip(w, 1.0, W * 2)
        h = np.clip(h, 1.0, H * 2)

        # Clamp position to keep bbox mostly within image
        # Allow partial overlap (up to 50% outside)
        x = np.clip(x, -w * 0.5, W - w * 0.5)
        y = np.clip(y, -h * 0.5, H - h * 0.5)

        return np.array([x, y, w, h], dtype=np.float32)

    def _is_valid_bbox(self, bbox: np.ndarray, min_size: float = 1.0) -> bool:
        """
        Check if bbox is valid (not degenerate, no NaN/Inf).

        CRITICAL FIX (Audit 2025120403 Issue 2 & 7):
        - Added NaN/Inf check

        Args:
            bbox: Bounding box [x, y, w, h]
            min_size: Minimum dimension size

        Returns:
            True if bbox is valid for tracking
        """
        if bbox is None:
            return False
        if len(bbox) < 4:
            return False
        # CRITICAL FIX (Audit 2025120403 Issue 2 & 7): Check NaN/Inf
        if not np.all(np.isfinite(bbox)):
            return False
        x, y, w, h = bbox
        return w >= min_size and h >= min_size

    def _evaluate_template_quality(
        self,
        bbox: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[bool, float]:
        """
        Evaluate template quality based on completeness and edge proximity.

        CRITICAL FIX (Audit 2025120402 Finding 1-2):
        Checks if bbox is:
        1. Valid (non-zero, reasonable size)
        2. Complete (not clipped at image edges)
        3. Has sufficient size for tracking

        Args:
            bbox: Bounding box [x, y, w, h] in XYWH format
            image_size: Image dimensions (H, W)

        Returns:
            Tuple of (is_complete, quality_score)
            - is_complete: True if bbox meets all quality criteria
            - quality_score: 0.0-1.0 score for ranking (higher = better)
        """
        if bbox is None:
            return False, 0.0

        x, y, w, h = bbox
        H, W = image_size

        # Check 1: Valid bbox (non-zero)
        if np.sum(np.abs(bbox)) == 0:
            return False, 0.0

        # Check 2: Minimum size requirement
        if w < self.template_min_size or h < self.template_min_size:
            return False, 0.1  # Small but valid

        # Check 3: Edge proximity (not clipped)
        margin = self.template_edge_margin
        left_margin = x
        top_margin = y
        right_margin = W - (x + w)
        bottom_margin = H - (y + h)

        is_edge_clipped = (
            left_margin < margin or
            top_margin < margin or
            right_margin < margin or
            bottom_margin < margin
        )

        if is_edge_clipped:
            # Compute how much is clipped (for scoring)
            min_margin = min(left_margin, top_margin, right_margin, bottom_margin)
            clip_penalty = max(0.0, 1.0 - (margin - min_margin) / margin)
            return False, 0.3 + 0.3 * clip_penalty  # Clipped but visible

        # Compute quality score based on:
        # - Size (larger is better, normalized by image size)
        # - Centrality (more centered is often better for initial template)
        # - Aspect ratio (moderate aspect ratio is better)

        # Size score (0.3 weight)
        bbox_area = w * h
        image_area = H * W
        area_ratio = bbox_area / image_area
        # Optimal range: 1%-10% of image area
        if area_ratio < 0.01:
            size_score = area_ratio / 0.01
        elif area_ratio < 0.10:
            size_score = 1.0
        else:
            size_score = max(0.5, 1.0 - (area_ratio - 0.10) / 0.20)

        # Margin score (0.3 weight) - prefer centered targets
        margin_score = min(left_margin, top_margin, right_margin, bottom_margin) / max(W, H)
        margin_score = min(1.0, margin_score * 10)  # Normalize

        # Aspect ratio score (0.2 weight) - penalize extreme aspect ratios
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect <= 2.0:
            aspect_score = 1.0
        elif aspect <= 4.0:
            aspect_score = 1.0 - (aspect - 2.0) / 4.0
        else:
            aspect_score = 0.5

        # Combined quality score
        quality_score = 0.3 * size_score + 0.4 * margin_score + 0.3 * aspect_score
        quality_score = 0.7 + 0.3 * quality_score  # Base score of 0.7 for complete templates

        return True, quality_score

    def _select_best_template_frame(
        self,
        base_dataset,
        num_frames: int
    ) -> Tuple[int, str]:
        """
        Select the best template frame from the first N frames.

        CRITICAL FIX (Audit 2025120402 Finding 1):
        Scans first N frames to find the one with highest quality template,
        considering both RGB and Sonar bbox quality.

        Args:
            base_dataset: Dataset instance
            num_frames: Total number of frames in sequence

        Returns:
            Tuple of (frame_index, selection_reason)
        """
        scan_limit = min(self.template_scan_frames, num_frames)

        best_frame_idx = -1
        best_quality = -1.0
        best_reason = "no_valid_frame"

        # Track candidate frames
        candidates = []

        for frame_idx in range(scan_limit):
            frame_data = base_dataset._get_frame_data(frame_idx)
            rgb_bbox = frame_data['rgb_bbox']
            sonar_bbox = frame_data['sonar_bbox']

            # Get image sizes
            rgb_img = frame_data.get('rgb_img')
            sonar_img = frame_data.get('sonar_img')

            rgb_size = (rgb_img.shape[0], rgb_img.shape[1]) if rgb_img is not None else (480, 640)
            sonar_size = (sonar_img.shape[0], sonar_img.shape[1]) if sonar_img is not None else (512, 512)

            # Evaluate quality for both modalities
            rgb_complete, rgb_quality = self._evaluate_template_quality(rgb_bbox, rgb_size)
            sonar_complete, sonar_quality = self._evaluate_template_quality(sonar_bbox, sonar_size)

            # Combined quality: require at least one modality to be valid
            rgb_valid = rgb_bbox is not None and np.sum(np.abs(rgb_bbox)) > 0
            sonar_valid = sonar_bbox is not None and np.sum(np.abs(sonar_bbox)) > 0

            if not (rgb_valid or sonar_valid):
                continue  # Skip fully invalid frames

            # Compute combined quality score
            # Prefer frames where both modalities are complete
            if rgb_complete and sonar_complete:
                combined_quality = (rgb_quality + sonar_quality) / 2.0 + 0.2  # Bonus for dual-complete
                reason = "dual_complete"
            elif rgb_complete:
                combined_quality = rgb_quality * 0.8
                reason = "rgb_complete"
            elif sonar_complete:
                combined_quality = sonar_quality * 0.8
                reason = "sonar_complete"
            else:
                # Neither complete but at least one valid
                combined_quality = max(rgb_quality, sonar_quality) * 0.5
                reason = "partial_valid"

            candidates.append((frame_idx, combined_quality, reason))

            if combined_quality > best_quality:
                best_quality = combined_quality
                best_frame_idx = frame_idx
                best_reason = reason

        # If no complete template found, fall back to first valid frame
        if best_frame_idx < 0:
            # Scan again for any valid frame (original behavior)
            for frame_idx in range(num_frames):
                frame_data = base_dataset._get_frame_data(frame_idx)
                rgb_bbox = frame_data['rgb_bbox']
                sonar_bbox = frame_data['sonar_bbox']

                rgb_valid = rgb_bbox is not None and np.sum(np.abs(rgb_bbox)) > 0
                sonar_valid = sonar_bbox is not None and np.sum(np.abs(sonar_bbox)) > 0

                if rgb_valid or sonar_valid:
                    return frame_idx, "first_valid_fallback"

            return -1, "no_valid_frame"

        return best_frame_idx, best_reason

    def _find_first_valid_frame(
        self,
        base_dataset,
        num_frames: int
    ) -> Tuple[int, str]:
        """
        Find the first frame with at least one valid modality (original behavior).

        This is the fallback when enable_template_quality_filter=False.
        Does NOT consider edge proximity or size constraints.

        Args:
            base_dataset: Dataset instance
            num_frames: Total number of frames in sequence

        Returns:
            Tuple of (frame_index, selection_reason)
        """
        for frame_idx in range(num_frames):
            frame_data = base_dataset._get_frame_data(frame_idx)
            rgb_bbox = frame_data['rgb_bbox']
            sonar_bbox = frame_data['sonar_bbox']

            rgb_valid = rgb_bbox is not None and np.sum(np.abs(rgb_bbox)) > 0
            sonar_valid = sonar_bbox is not None and np.sum(np.abs(sonar_bbox)) > 0

            if rgb_valid or sonar_valid:
                return frame_idx, "first_valid"

        return -1, "no_valid_frame"

    def _compute_search_factor(
        self,
        rgb_conf: float,
        sonar_conf: float
    ) -> Tuple[float, float]:
        """
        Compute dynamic search area factors based on confidence scores.

        Uses mainstream strategy: expand search area when confidence is low
        to enable target re-detection after tracking loss.

        Reference: SiamFC++, DiMP use similar progressive expansion strategies.

        Args:
            rgb_conf: RGB branch confidence score [0, 1]
            sonar_conf: Sonar branch confidence score [0, 1]

        Returns:
            Tuple of (rgb_search_factor, sonar_search_factor)
        """
        if not self.enable_search_expansion:
            return self._base_search_area_factor, self._base_search_area_factor

        base = self._base_search_area_factor
        max_factor = base * self.max_expansion_factor

        def compute_factor(conf: float) -> float:
            """Compute search factor for single modality."""
            if conf >= self.confidence_high:
                # High confidence: use normal search area
                return base
            elif conf >= self.confidence_low:
                # Medium confidence: linear interpolation to 1.5x
                ratio = (self.confidence_high - conf) / (self.confidence_high - self.confidence_low)
                return base * (1.0 + 0.5 * ratio)
            else:
                # Low confidence: expand to maximum (2x)
                return max_factor

        rgb_factor = compute_factor(rgb_conf)
        sonar_factor = compute_factor(sonar_conf)

        return rgb_factor, sonar_factor

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
