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

    # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Configurable validity thresholds
    # Default thresholds for state update and template update decisions
    RGB_VALIDITY_THRESHOLD_DEFAULT = 0.35    # RGB validity threshold (more lenient)
    SONAR_VALIDITY_THRESHOLD_DEFAULT = 0.40  # Sonar validity threshold (stricter)

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
        enable_template_quality_filter: bool = True,  # Enable quality-based template selection
        # CRITICAL FIX (Audit 2025121202 Finding 22): Quality thresholds for template selection
        quality_threshold_rgb: float = 0.7,     # Min quality score for RGB template candidate
        quality_threshold_sonar: float = 0.7,   # Min quality score for Sonar template candidate
        # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Configurable validity gating
        rgb_validity_threshold: float = None,   # RGB validity threshold for state/template update
        sonar_validity_threshold: float = None, # Sonar validity threshold for state/template update
        enable_validity_gating: bool = True,    # Master switch for validity gating
        # CRITICAL FIX (Audit 2025121202 Finding 9): Expected resolutions for fallback
        expected_rgb_resolution: Tuple[int, int] = None,    # (H, W) expected RGB resolution
        expected_sonar_resolution: Tuple[int, int] = None,  # (H, W) expected Sonar resolution
        # CRITICAL FIX (Audit 2025121202 Finding 17): Configurable metrics list
        metrics_list: Optional[List[str]] = None            # List of metrics to compute
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

        # CRITICAL FIX (Audit 2025121202 Finding 22): Quality thresholds for template selection
        self.quality_threshold_rgb = quality_threshold_rgb
        self.quality_threshold_sonar = quality_threshold_sonar

        # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Configurable validity gating
        self.rgb_validity_threshold = rgb_validity_threshold if rgb_validity_threshold is not None else self.RGB_VALIDITY_THRESHOLD_DEFAULT
        self.sonar_validity_threshold = sonar_validity_threshold if sonar_validity_threshold is not None else self.SONAR_VALIDITY_THRESHOLD_DEFAULT
        self.enable_validity_gating = enable_validity_gating

        # CRITICAL FIX (Audit 2025121202 Finding 9): Expected resolutions for fallback
        # Use spec-compliant defaults: RGB 1920x1080, Sonar 682x376 (W x H -> stored as H, W)
        self.expected_rgb_resolution = expected_rgb_resolution or (1080, 1920)  # H, W
        self.expected_sonar_resolution = expected_sonar_resolution or (376, 682)  # H, W

        # CRITICAL FIX (Audit 2025121202 Finding 17): Configurable metrics list
        # Default metrics if not provided: precision_10, precision_5, success_0.7
        # These are "additional" metrics beyond the default set in compute_sequence_metrics
        self.metrics_list = metrics_list  # Full config list, will be split into default/additional

        # CRITICAL FIX (Audit 2025120902 Issue 3): Lost-state detector parameters
        # When tracking confidence stays low for too long, trigger global re-detection
        self.lost_state_threshold = 5  # Consecutive low-conf frames to trigger lost state
        self.global_search_expansion = 4.0  # Expansion factor for global re-detection
        self._consecutive_low_conf_rgb = 0  # Counter for RGB
        self._consecutive_low_conf_sonar = 0  # Counter for Sonar
        self._lost_state_count = 0  # Count of lost state recoveries

        # CRITICAL FIX (Audit 2025120909): Template bank and recovery freeze
        # Template bank: Store multiple templates to allow fallback during drift
        # Recovery freeze: Prevent template updates during lost state recovery
        self.template_bank_size = 3  # Number of templates to store
        self.recovery_freeze_frames = 10  # Frames to freeze template after lost recovery
        self._template_bank_rgb = []  # List of (template_tensor, metadata, mask)
        self._template_bank_sonar = []  # List of (template_tensor, metadata, mask)
        self._frames_since_recovery = float('inf')  # Frames since last lost recovery
        self._in_recovery_freeze = False  # Flag to prevent template updates

        logger.info(
            f"OnlineTrackingEvaluator initialized: "
            f"template_size={template_size}, search_size={search_size}, "
            f"search_area_factor={search_area_factor}, "
            f"enable_search_expansion={enable_search_expansion}, "
            f"enable_template_update={enable_template_update}, "
            f"enable_template_quality_filter={enable_template_quality_filter}, "
            f"enable_validity_gating={enable_validity_gating}, "
            f"rgb_validity_threshold={self.rgb_validity_threshold}, "
            f"sonar_validity_threshold={self.sonar_validity_threshold}, "
            f"expected_rgb_resolution={self.expected_rgb_resolution}, "
            f"expected_sonar_resolution={self.expected_sonar_resolution}"
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

        # CRITICAL FIX (Audit 2025121202 Finding 12): Verify batch_size=1
        # Online tracking requires sequential frame processing with single batch
        logger.debug(f"Sequence {sequence_name}: evaluating {num_frames} frames (batch_size=1)")

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

        # Capture template metadata for fusion module
        rgb_template, rgb_template_meta = self._crop_region(
            frame0_data['rgb_img'],
            rgb_gt_bbox,
            self.template_size,
            self.template_area_factor,
            is_sonar=False,
            return_metadata=True  # CRITICAL: Capture metadata
        )
        sonar_template, sonar_template_meta = self._crop_region(
            frame0_data['sonar_img'],
            sonar_gt_bbox,
            self.template_size,
            self.template_area_factor,
            is_sonar=True,
            return_metadata=True  # CRITICAL: Capture metadata
        )

        # CRITICAL FIX (Audit 2025120909): Generate template attention masks
        rgb_template_att_mask = self._generate_attention_mask(rgb_template_meta, output_size=self.template_size)
        sonar_template_att_mask = self._generate_attention_mask(sonar_template_meta, output_size=self.template_size)

        # Move templates to device
        rgb_template = rgb_template.unsqueeze(0).to(self.device)
        sonar_template = sonar_template.unsqueeze(0).to(self.device)

        # CRITICAL FIX (Audit 2025120909): Initialize template bank with first templates
        # Store initial templates as the first entries in the bank
        self._template_bank_rgb = [(rgb_template.clone(), rgb_template_meta.copy(), rgb_template_att_mask.clone())]
        self._template_bank_sonar = [(sonar_template.clone(), sonar_template_meta.copy(), sonar_template_att_mask.clone())]
        self._frames_since_recovery = float('inf')  # Reset recovery counter
        self._in_recovery_freeze = False  # Not in recovery at start

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

                # CRITICAL FIX (Audit 2025120902 Issue 3): Global re-detection override
                # When lost state triggered, use much larger search area for recovery
                if hasattr(self, '_trigger_global_search') and self._trigger_global_search:
                    rgb_search_factor = self.global_search_expansion
                    sonar_search_factor = self.global_search_expansion
                    self._trigger_global_search = False  # Reset flag
                    logger.debug(
                        f"Frame {frame_idx}: Global search triggered with factor={self.global_search_expansion}"
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

                # Build metadata for fusion module
                metadata = {
                    'rgb_search_metadata': [rgb_search_meta],
                    'sonar_search_metadata': [sonar_search_meta],  # List format for tracker.py
                    'sonar_metadata': [sonar_search_meta],  # Alternative key used by tracker
                    # CRITICAL FIX (Audit 2025120909): Add template metadata
                    'rgb_template_metadata': [rgb_template_meta],
                    'sonar_template_metadata': [sonar_template_meta],
                }

                # === CRITICAL FIX (Audit 2025120902 Issue 7): Self-inferred validity for fusion ===
                # Instead of removing is_valid (which disables conditional fusion),
                # derive validity from previous frame's confidence and current padding ratio.
                # This enables ConditionalDCMA to properly bypass unreliable modalities.

                # Check GT occlusion status (for prediction storage, NOT for model input)
                rgb_gt_occluded = np.sum(np.abs(rgb_gt_bbox)) == 0
                sonar_gt_occluded = np.sum(np.abs(sonar_gt_bbox)) == 0

                # Compute padding ratios to detect off-screen crops
                rgb_padding_ratio_pre = self._compute_padding_ratio(rgb_search_meta)
                sonar_padding_ratio_pre = self._compute_padding_ratio(sonar_search_meta)

                # Self-infer validity from confidence + padding (NOT from GT!)
                # CRITICAL FIX (Audit 2025121201 S1): Use >= threshold (not >) for consistency
                # Validity threshold: conf >= 0.5 AND padding < 0.3
                VALIDITY_CONF_THRESHOLD = 0.5
                VALIDITY_PADDING_THRESHOLD = 0.3

                inferred_rgb_valid = (
                    rgb_conf_prev >= VALIDITY_CONF_THRESHOLD and
                    rgb_padding_ratio_pre < VALIDITY_PADDING_THRESHOLD
                )
                inferred_sonar_valid = (
                    sonar_conf_prev >= VALIDITY_CONF_THRESHOLD and
                    sonar_padding_ratio_pre < VALIDITY_PADDING_THRESHOLD
                )

                # Convert to tensor for model forward
                is_rgb_valid_tensor = torch.tensor([inferred_rgb_valid], dtype=torch.bool, device=self.device)
                is_sonar_valid_tensor = torch.tensor([inferred_sonar_valid], dtype=torch.bool, device=self.device)

                # === Model forward pass ===
                # CRITICAL FIX (Audit 2025120902 Issue 7): Pass self-inferred validity
                # This enables ConditionalDCMA to bypass unreliable modalities
                # CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised): Pass attention masks
                # Generate attention masks from crop metadata for proper padding handling
                rgb_att_mask = self._generate_attention_mask(rgb_search_meta, output_size=self.search_size)
                sonar_att_mask = self._generate_attention_mask(sonar_search_meta, output_size=self.search_size)

                # CRITICAL FIX (Audit 2025121202 Finding 12): Verify batch_size=1 at runtime
                # This catches any accidental batching that would corrupt online tracking
                assert rgb_search.shape[0] == 1, f"batch_size must be 1, got {rgb_search.shape[0]}"
                assert sonar_search.shape[0] == 1, f"batch_size must be 1, got {sonar_search.shape[0]}"

                # CRITICAL FIX (Audit 2025120909): Pass TEMPLATE attention masks
                # Template masks mark valid (non-padded) regions in template crops
                outputs = self.model(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search,
                    is_rgb_valid=is_rgb_valid_tensor,    # Self-inferred, NOT GT
                    is_sonar_valid=is_sonar_valid_tensor,  # Self-inferred, NOT GT
                    return_losses=False,
                    metadata=metadata,
                    rgb_att_mask=rgb_att_mask,  # Search attention mask
                    sonar_att_mask=sonar_att_mask,  # Search attention mask
                    rgb_template_att_mask=rgb_template_att_mask,  # Template attention mask
                    sonar_template_att_mask=sonar_template_att_mask  # Template attention mask
                )

                # === Extract and denormalize predictions ===
                # Model outputs normalized cxcywh in [0, 1]
                rgb_pred_norm = outputs['rgb_boxes'][0].cpu().numpy()  # [cx, cy, w, h]
                sonar_pred_norm = outputs['sonar_boxes'][0].cpu().numpy()

                # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
                # Use head validity outputs for state/template update decisions
                # Priority order:
                # 1. validity_final - learned validity (combines padding + objectness)
                # 2. validity - learned validity (before fallback rule)
                # 3. objectness - direct object presence indicator
                # 4. confidence - alternative name for objectness
                # 5. quality - fallback (less reliable)
                #
                # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Use configurable thresholds
                # Thresholds are now constructor parameters, defaulting to class constants:
                # - RGB validity: 0.35 (more lenient)
                # - Sonar validity: 0.4 (stricter)

                # CRITICAL FIX (Audit 2025121201 S1): RGB validity/confidence extraction
                # Validity priority: validity_final -> validity -> None (fallback to conf threshold)
                # Confidence priority: objectness -> pred_confidence/confidence -> quality -> 0.5
                rgb_validity_value = None
                rgb_is_valid_head = None
                for vkey in ['rgb_validity', 'rgb_validity_final']:
                    if vkey in outputs and outputs[vkey] is not None:
                        rgb_validity_value = outputs[vkey][0].cpu().item()
                        # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Use configurable threshold
                        rgb_is_valid_head = rgb_validity_value >= self.rgb_validity_threshold if self.enable_validity_gating else True
                        break

                rgb_conf_prev = 0.5  # Default
                for ckey in ['rgb_objectness', 'rgb_confidence', 'rgb_quality']:
                    if ckey in outputs and outputs[ckey] is not None:
                        rgb_conf_prev = outputs[ckey][0].cpu().item()
                        break

                # CRITICAL FIX (Audit 2025121201 S1): Sonar validity/confidence extraction
                # Same priority as RGB
                sonar_validity_value = None
                sonar_is_valid_head = None
                for vkey in ['sonar_validity', 'sonar_validity_final']:
                    if vkey in outputs and outputs[vkey] is not None:
                        sonar_validity_value = outputs[vkey][0].cpu().item()
                        # CRITICAL FIX (Audit 2025121202 Finding 3, 5): Use configurable threshold
                        sonar_is_valid_head = sonar_validity_value >= self.sonar_validity_threshold if self.enable_validity_gating else True
                        break

                sonar_conf_prev = 0.5  # Default
                for ckey in ['sonar_objectness', 'sonar_confidence', 'sonar_quality']:
                    if ckey in outputs and outputs[ckey] is not None:
                        sonar_conf_prev = outputs[ckey][0].cpu().item()
                        break

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
                # CRITICAL FIX (Audit 2025120902 Issue 1): Padding-aware state update
                # Skip state update when crop has high padding ratio (>0.3) to prevent drift
                # Also implements cross-modality re-anchor for recovery

                # Compute padding ratios for both modalities
                rgb_padding_ratio = self._compute_padding_ratio(rgb_search_meta)
                sonar_padding_ratio = self._compute_padding_ratio(sonar_search_meta)

                # Thresholds for state update decisions
                PADDING_THRESHOLD = 0.3  # Skip update if padding > 30%
                CONF_THRESHOLD = self.confidence_low  # From config

                # Track last valid states for cross-modal re-anchor
                if not hasattr(self, '_last_valid_rgb_state'):
                    self._last_valid_rgb_state = rgb_state.copy()
                if not hasattr(self, '_last_valid_sonar_state'):
                    self._last_valid_sonar_state = sonar_state.copy()

                # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
                # Use head validity for state update decisions when available
                # Fallback to confidence + padding if head validity not available

                # RGB state update - use head validity if available, else fallback
                if rgb_is_valid_head is not None:
                    # Use learned head validity (preferred - single forward)
                    rgb_update_valid = (
                        rgb_pred_sanitized is not None and
                        rgb_is_valid_head
                    )
                else:
                    # Fallback to confidence + padding (backward compatibility)
                    rgb_update_valid = (
                        rgb_pred_sanitized is not None and
                        rgb_conf_prev >= CONF_THRESHOLD and
                        rgb_padding_ratio < PADDING_THRESHOLD
                    )

                if rgb_update_valid:
                    rgb_state = rgb_pred_sanitized.copy()
                    self._last_valid_rgb_state = rgb_state.copy()
                # else: keep previous state

                # Sonar state update with cross-modal re-anchor
                # Use head validity if available, else fallback
                if sonar_is_valid_head is not None:
                    # Use learned head validity (preferred - single forward)
                    sonar_update_valid = (
                        sonar_pred_sanitized is not None and
                        sonar_is_valid_head
                    )
                else:
                    # Fallback to confidence + padding
                    sonar_update_valid = (
                        sonar_pred_sanitized is not None and
                        sonar_conf_prev >= CONF_THRESHOLD and
                        sonar_padding_ratio < PADDING_THRESHOLD
                    )

                if sonar_update_valid:
                    sonar_state = sonar_pred_sanitized.copy()
                    self._last_valid_sonar_state = sonar_state.copy()
                elif not sonar_update_valid:
                    # CRITICAL FIX (Audit 2025120902 Issue 1): Cross-modal re-anchor
                    # When sonar is unreliable (invalid head validity or low conf), re-anchor to RGB
                    # This prevents unbounded sonar drift by using RGB as primary anchor
                    if rgb_update_valid:
                        # Re-center sonar search on RGB state to avoid divergence
                        # Note: This assumes RGB and Sonar have similar FOV, which may not be exact
                        # but is better than letting sonar drift off-screen
                        sonar_invalid_reason = (
                            f"validity={sonar_validity_value:.2f}" if sonar_validity_value is not None
                            else f"conf={sonar_conf_prev:.2f}, padding={sonar_padding_ratio:.2f}"
                        )
                        logger.debug(
                            f"Frame {frame_idx}: Sonar re-anchoring to RGB ({sonar_invalid_reason})"
                        )
                        # Use scaled RGB state as sonar anchor (approximate cross-modal alignment)
                        # This follows PrDiMP/OSTrack multi-cam primary-modality anchor pattern
                        sonar_state = self._last_valid_rgb_state.copy()
                    # else: keep previous sonar state

                # === CRITICAL FIX (Audit 2025120902 - Head-Validity Solution): Lost-state detector ===
                # Track consecutive invalid frames to detect tracking failure
                # Use head validity when available, else fallback to confidence
                if rgb_is_valid_head is not None:
                    rgb_is_invalid = not rgb_is_valid_head
                else:
                    rgb_is_invalid = rgb_conf_prev < CONF_THRESHOLD

                if sonar_is_valid_head is not None:
                    sonar_is_invalid = not sonar_is_valid_head
                else:
                    sonar_is_invalid = sonar_conf_prev < CONF_THRESHOLD

                if rgb_is_invalid:
                    self._consecutive_low_conf_rgb += 1
                else:
                    self._consecutive_low_conf_rgb = 0

                if sonar_is_invalid:
                    self._consecutive_low_conf_sonar += 1
                else:
                    self._consecutive_low_conf_sonar = 0

                # Check if in lost state (both modalities consistently low confidence)
                in_lost_state = (
                    self._consecutive_low_conf_rgb >= self.lost_state_threshold and
                    self._consecutive_low_conf_sonar >= self.lost_state_threshold
                )

                if in_lost_state:
                    # TRIGGER GLOBAL RE-DETECTION
                    # Expand search area to cover larger region centered on last valid position
                    # Use image center as fallback if no valid position
                    logger.warning(
                        f"Frame {frame_idx}: LOST STATE detected "
                        f"(rgb_low_conf_streak={self._consecutive_low_conf_rgb}, "
                        f"sonar_low_conf_streak={self._consecutive_low_conf_sonar}). "
                        f"Triggering global re-detection."
                    )

                    # Reset states to image center for global search
                    H_rgb, W_rgb = rgb_img_size
                    H_sonar, W_sonar = sonar_img_size

                    # Use larger bbox centered at image center for global search
                    # Size is 25% of image to give model a starting point
                    rgb_state = np.array([
                        W_rgb * 0.375, H_rgb * 0.375,  # x, y at 37.5%
                        W_rgb * 0.25, H_rgb * 0.25     # w, h at 25%
                    ], dtype=np.float32)

                    sonar_state = np.array([
                        W_sonar * 0.375, H_sonar * 0.375,
                        W_sonar * 0.25, H_sonar * 0.25
                    ], dtype=np.float32)

                    # Mark for expanded search on next iteration
                    # This is done by temporarily boosting search_area_factor
                    self._trigger_global_search = True
                    self._lost_state_count += 1

                    # CRITICAL FIX (Audit 2025120909): Template recovery freeze
                    # After lost state, freeze template updates to prevent drift contamination
                    self._in_recovery_freeze = True
                    self._frames_since_recovery = 0

                    # CRITICAL FIX (Audit 2025120909): Revert to oldest template in bank
                    # When in lost state, the current template may be corrupted by drift.
                    # Revert to the first (oldest) template in the bank which was set during init.
                    if self._template_bank_rgb:
                        rgb_template, rgb_template_meta, rgb_template_att_mask = self._template_bank_rgb[0]
                        logger.debug(f"Frame {frame_idx}: Reverted to initial RGB template from bank")
                    if self._template_bank_sonar:
                        sonar_template, sonar_template_meta, sonar_template_att_mask = self._template_bank_sonar[0]
                        logger.debug(f"Frame {frame_idx}: Reverted to initial Sonar template from bank")

                    # Reset confidence to high to force state update on next frame
                    rgb_conf_prev = 0.9
                    sonar_conf_prev = 0.9

                    # Reset counters
                    self._consecutive_low_conf_rgb = 0
                    self._consecutive_low_conf_sonar = 0

                # CRITICAL FIX (Audit 2025120403 Issue 8 & 21): Store None for occluded frames
                # Instead of storing [0,0,0,0], store None to properly signal "no prediction".
                # metrics.py _is_valid_bbox() will skip None entries during metric computation.
                # This follows standard tracking benchmarks (LaSOT, GOT-10k) protocol.
                #
                # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
                # Also store None when head validity is below threshold (invalid prediction)
                # Priority: GT occlusion > head validity > store prediction
                if rgb_gt_occluded:
                    predictions['rgb_pred'].append(None)  # GT occluded
                elif rgb_is_valid_head is not None and not rgb_is_valid_head:
                    predictions['rgb_pred'].append(None)  # Head validity invalid
                else:
                    predictions['rgb_pred'].append(rgb_pred_sanitized)

                if sonar_gt_occluded:
                    predictions['sonar_pred'].append(None)  # GT occluded
                elif sonar_is_valid_head is not None and not sonar_is_valid_head:
                    predictions['sonar_pred'].append(None)  # Head validity invalid
                else:
                    predictions['sonar_pred'].append(sonar_pred_sanitized)

                # Store validity values for analysis/debugging
                if 'rgb_validity' not in predictions:
                    predictions['rgb_validity'] = []
                if 'sonar_validity' not in predictions:
                    predictions['sonar_validity'] = []
                predictions['rgb_validity'].append(rgb_validity_value if rgb_validity_value is not None else rgb_conf_prev)
                predictions['sonar_validity'].append(sonar_validity_value if sonar_validity_value is not None else sonar_conf_prev)

                predictions['rgb_gt'].append(rgb_gt_bbox)
                predictions['sonar_gt'].append(sonar_gt_bbox)

                # === CRITICAL FIX (Audit 2025120902 Issue 2): Template update ===
                # Update template periodically when tracking is confident
                # Uses min_conf (not avg) + IoU check to prevent high-bias sonar from corrupting template
                frames_since_update += 1

                # CRITICAL FIX (Audit 2025120909): Update recovery freeze counter
                if self._in_recovery_freeze:
                    self._frames_since_recovery += 1
                    if self._frames_since_recovery >= self.recovery_freeze_frames:
                        # Recovery period ended - allow template updates again
                        self._in_recovery_freeze = False
                        logger.debug(
                            f"Frame {frame_idx}: Recovery freeze ended after {self._frames_since_recovery} frames"
                        )

                if self.enable_template_update and not self._in_recovery_freeze:
                    # CRITICAL FIX (Audit 2025120902 - Head-Validity Solution):
                    # Use head validity for template update gating when available
                    # Fallback to min_conf when head validity not available

                    # Determine if both modalities are valid for template update
                    if rgb_is_valid_head is not None and sonar_is_valid_head is not None:
                        # Use learned head validity (preferred - single forward)
                        both_valid_for_update = rgb_is_valid_head and sonar_is_valid_head
                        validity_for_log = f"rgb_validity={rgb_validity_value:.2f}, sonar_validity={sonar_validity_value:.2f}"
                    else:
                        # Fallback to min confidence threshold
                        min_conf = min(rgb_conf_prev, sonar_conf_prev)
                        both_valid_for_update = min_conf >= self.template_update_threshold
                        validity_for_log = f"min_conf={min_conf:.2f}"

                    # IoU check: new predictions should overlap reasonably with previous states
                    # This prevents drift accumulation during template updates
                    rgb_iou_with_prev = self._compute_bbox_iou(rgb_pred_sanitized, rgb_state)
                    sonar_iou_with_prev = self._compute_bbox_iou(sonar_pred_sanitized, sonar_state)
                    MIN_IOU_FOR_UPDATE = 0.2  # At least 20% overlap required

                    # Boundary check: ensure predictions are not at image edge
                    rgb_in_bounds = True
                    sonar_in_bounds = True
                    if rgb_pred_sanitized is not None:
                        rx, ry, rw, rh = rgb_pred_sanitized
                        margin = 5  # pixels
                        rgb_in_bounds = (rx >= margin and ry >= margin and
                                        rx + rw <= rgb_img_size[1] - margin and
                                        ry + rh <= rgb_img_size[0] - margin)
                    if sonar_pred_sanitized is not None:
                        sx, sy, sw, sh = sonar_pred_sanitized
                        sonar_in_bounds = (sx >= margin and sy >= margin and
                                          sx + sw <= sonar_img_size[1] - margin and
                                          sy + sh <= sonar_img_size[0] - margin)

                    # Check if we should update template (validity/conf + IoU + boundary checks)
                    should_update = (
                        frames_since_update >= self.template_update_interval and
                        both_valid_for_update and  # Use head validity or min_conf
                        rgb_pred_sanitized is not None and
                        sonar_pred_sanitized is not None and
                        rgb_iou_with_prev >= MIN_IOU_FOR_UPDATE and  # IoU check
                        sonar_iou_with_prev >= MIN_IOU_FOR_UPDATE and
                        rgb_in_bounds and sonar_in_bounds  # Boundary check
                    )

                    if should_update:
                        # Update templates using sanitized prediction as bbox
                        # CRITICAL FIX (Audit 2025120403): Use sanitized predictions
                        # CRITICAL FIX (Audit 2025120909): Capture template metadata on update
                        new_rgb_template, new_rgb_template_meta = self._crop_region(
                            frame_data['rgb_img'],
                            rgb_pred_sanitized,  # Use sanitized prediction, not raw
                            self.template_size,
                            self.template_area_factor,
                            is_sonar=False,
                            return_metadata=True  # CRITICAL: Capture metadata
                        )
                        new_sonar_template, new_sonar_template_meta = self._crop_region(
                            frame_data['sonar_img'],
                            sonar_pred_sanitized,  # Use sanitized prediction, not raw
                            self.template_size,
                            self.template_area_factor,
                            is_sonar=True,
                            return_metadata=True  # CRITICAL: Capture metadata
                        )

                        # Update cached templates
                        rgb_template = new_rgb_template.unsqueeze(0).to(self.device)
                        sonar_template = new_sonar_template.unsqueeze(0).to(self.device)

                        # CRITICAL FIX (Audit 2025120909): Update template metadata and attention masks
                        rgb_template_meta = new_rgb_template_meta
                        sonar_template_meta = new_sonar_template_meta
                        rgb_template_att_mask = self._generate_attention_mask(rgb_template_meta, output_size=self.template_size)
                        sonar_template_att_mask = self._generate_attention_mask(sonar_template_meta, output_size=self.template_size)

                        # CRITICAL FIX (Audit 2025120909): Add to template bank (FIFO, keep oldest at index 0)
                        # Keep initial template at index 0, append new ones at the end
                        # When bank is full, remove oldest non-initial template
                        if len(self._template_bank_rgb) < self.template_bank_size:
                            # Bank not full - just append
                            self._template_bank_rgb.append((rgb_template.clone(), rgb_template_meta.copy(), rgb_template_att_mask.clone()))
                            self._template_bank_sonar.append((sonar_template.clone(), sonar_template_meta.copy(), sonar_template_att_mask.clone()))
                        else:
                            # Bank full - remove oldest non-initial (index 1), keep initial (index 0)
                            if len(self._template_bank_rgb) > 1:
                                self._template_bank_rgb.pop(1)
                                self._template_bank_sonar.pop(1)
                            self._template_bank_rgb.append((rgb_template.clone(), rgb_template_meta.copy(), rgb_template_att_mask.clone()))
                            self._template_bank_sonar.append((sonar_template.clone(), sonar_template_meta.copy(), sonar_template_att_mask.clone()))

                        # Reset counter
                        frames_since_update = 0
                        template_update_count += 1
                        logger.debug(
                            f"Frame {frame_idx}: Template updated (bank_size_rgb={len(self._template_bank_rgb)}) "
                            f"({validity_for_log}, rgb_iou={rgb_iou_with_prev:.2f}, "
                            f"sonar_iou={sonar_iou_with_prev:.2f})"
                        )

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
        - Returns letterbox-format metadata for fusion module compatibility
        - Keys: scale, pad_top, pad_bottom, original_height, original_width, new_size

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
                                 AND letterbox-format metadata for fusion module
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
            # Return crop metadata in unified format for fusion module compatibility.
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
        Sanitize bbox: reject NaN/Inf, STRICTLY clamp to image bounds.

        CRITICAL FIX (Audit 2025120902 Issue 10):
        - Previous soft clamp allowed x in [-0.5*w, W-0.5*w], letting drifted states persist
        - Now uses STRICT clamping: x in [0, W-w], y in [0, H-h]
        - Returns None (triggers recovery) if clamp distance exceeds 25% of box dimension
        - This prevents off-screen drift from contaminating template updates

        Args:
            bbox: Input bbox [x, y, w, h] in XYWH format
            image_size: Image dimensions (H, W)
            fallback_bbox: Optional fallback bbox if input is invalid

        Returns:
            Sanitized bbox, fallback, or None if severely off-screen (triggers recovery)
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

        # Clamp dimensions to reasonable range (max 1.5x image size)
        w_original = w
        h_original = h
        w = np.clip(w, 1.0, W * 1.5)
        h = np.clip(h, 1.0, H * 1.5)

        # CRITICAL FIX (Audit 2025120902 Issue 10): STRICT boundary clamping
        # Force bbox to stay FULLY inside image bounds
        x_original = x
        y_original = y
        x = np.clip(x, 0.0, max(0.0, W - w))
        y = np.clip(y, 0.0, max(0.0, H - h))

        # Calculate clamp distance - if too large, bbox was severely off-screen
        # Return None to trigger recovery path instead of using corrupted state
        x_clamp_dist = abs(x - x_original)
        y_clamp_dist = abs(y - y_original)

        # If clamp moved bbox by more than 25% of its dimension, it was off-screen
        # This indicates tracking failure - return None to force recovery
        if x_clamp_dist > 0.25 * w_original or y_clamp_dist > 0.25 * h_original:
            logger.debug(
                f"Bbox severely off-screen: original=({x_original:.1f}, {y_original:.1f}), "
                f"clamped=({x:.1f}, {y:.1f}), clamp_dist=({x_clamp_dist:.1f}, {y_clamp_dist:.1f}), "
                f"returning None to trigger recovery"
            )
            return None  # Force recovery path

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

    def _compute_padding_ratio(self, crop_metadata: Dict) -> float:
        """
        Compute the padding ratio of a crop region.

        CRITICAL FIX (Audit 2025120902 Issue 1):
        Detects when crop extends significantly beyond image bounds.
        High padding ratio indicates unreliable features (mostly background/padding).

        Args:
            crop_metadata: Crop metadata dict with pad_* keys

        Returns:
            Padding ratio in [0, 1]. 0 = no padding, 1 = all padding
        """
        if crop_metadata is None:
            return 0.0

        # Get padding amounts (in crop space before resize)
        pad_left = crop_metadata.get('pad_left', 0)
        pad_right = crop_metadata.get('pad_right', 0)
        pad_top = crop_metadata.get('pad_top', 0)
        pad_bottom = crop_metadata.get('pad_bottom', 0)

        # Get crop size
        crop_sz = crop_metadata.get('crop_sz', 256)
        if crop_sz <= 0:
            return 0.0

        # Total padding area vs crop area
        total_padding = pad_left + pad_right + pad_top + pad_bottom
        # Approximate padding area ratio (using perimeter-based estimate)
        # More accurate would be: (pad_left + pad_right) * crop_sz + (pad_top + pad_bottom) * (crop_sz - pad_left - pad_right)
        # Simplified: total padding pixels / total crop pixels
        pad_area = (pad_left + pad_right) * crop_sz + (pad_top + pad_bottom) * max(0, crop_sz - pad_left - pad_right)
        total_area = crop_sz * crop_sz

        padding_ratio = pad_area / max(total_area, 1)
        return min(1.0, padding_ratio)

    def _generate_attention_mask(self, crop_metadata: Dict, output_size: int = 256) -> torch.Tensor:
        """
        Generate attention mask from crop metadata.

        CRITICAL FIX (Audit 2025120902 Issue 15, 17 Revised):
        Creates attention mask for padded regions to pass to the model backbone.
        The mask marks valid (non-padded) regions as 1 and padded regions as 0.

        Args:
            crop_metadata: Dictionary containing crop parameters:
                - crop_x1, crop_y1: Crop origin (can be negative if padded)
                - crop_sz: Original crop size before resize
                - original_width, original_height: Source image dimensions
                - output_size: Final output size after resize
            output_size: Target mask size (default: 256 for search region)

        Returns:
            Attention mask tensor [1, output_size, output_size] where 1=valid, 0=padded
        """
        if not crop_metadata:
            # No metadata - assume all valid
            return torch.ones((1, output_size, output_size), dtype=torch.float32, device=self.device)

        # Extract crop parameters
        crop_x1 = crop_metadata.get('crop_x1', 0)
        crop_y1 = crop_metadata.get('crop_y1', 0)
        crop_sz = crop_metadata.get('crop_sz', output_size)
        orig_w = crop_metadata.get('original_width', crop_metadata.get('original_size', (output_size, output_size))[1])
        orig_h = crop_metadata.get('original_height', crop_metadata.get('original_size', (output_size, output_size))[0])

        # Compute crop boundaries
        crop_x2 = crop_x1 + crop_sz
        crop_y2 = crop_y1 + crop_sz

        # Compute padding on each side (in crop space before resize)
        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - orig_w)
        pad_bottom = max(0, crop_y2 - orig_h)

        # Scale padding to output size
        resize_factor = output_size / crop_sz
        pad_left_out = int(pad_left * resize_factor)
        pad_top_out = int(pad_top * resize_factor)
        pad_right_out = int(pad_right * resize_factor)
        pad_bottom_out = int(pad_bottom * resize_factor)

        # Create mask: 1 for valid, 0 for padded
        mask = torch.ones((1, output_size, output_size), dtype=torch.float32, device=self.device)

        if pad_top_out > 0:
            mask[:, :pad_top_out, :] = 0
        if pad_bottom_out > 0:
            mask[:, -pad_bottom_out:, :] = 0
        if pad_left_out > 0:
            mask[:, :, :pad_left_out] = 0
        if pad_right_out > 0:
            mask[:, :, -pad_right_out:] = 0

        return mask

    def _compute_bbox_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute IoU between two bboxes in XYWH format.

        CRITICAL FIX (Audit 2025120902 Issue 2):
        Used for template update validation - only update if IoU with previous state > threshold.

        Args:
            bbox1: First bbox [x, y, w, h]
            bbox2: Second bbox [x, y, w, h]

        Returns:
            IoU value in [0, 1]
        """
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to xyxy
        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

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

        # CRITICAL FIX (Audit 2025121202 Finding 9/25): Validate resolution at first frame
        resolution_validated = False

        for frame_idx in range(scan_limit):
            frame_data = base_dataset._get_frame_data(frame_idx)
            rgb_bbox = frame_data['rgb_bbox']
            sonar_bbox = frame_data['sonar_bbox']

            # Get image sizes
            rgb_img = frame_data.get('rgb_img')
            sonar_img = frame_data.get('sonar_img')

            # CRITICAL FIX (Audit 2025121202 Finding 9): Use configured expected resolution
            # instead of hardcoded values for fallback when image is None
            rgb_size = (rgb_img.shape[0], rgb_img.shape[1]) if rgb_img is not None else self.expected_rgb_resolution
            sonar_size = (sonar_img.shape[0], sonar_img.shape[1]) if sonar_img is not None else self.expected_sonar_resolution

            # CRITICAL FIX (Audit 2025121202 Finding 9/25): Hard validate resolution at first valid frame
            if not resolution_validated and (rgb_img is not None or sonar_img is not None):
                resolution_validated = True
                # Validate sonar resolution (CRITICAL - geometric consistency requires exact match)
                if sonar_img is not None:
                    actual_sonar = (sonar_img.shape[0], sonar_img.shape[1])
                    if actual_sonar != self.expected_sonar_resolution:
                        raise ValueError(
                            f"SONAR RESOLUTION MISMATCH (Audit 2025121202 Finding 9): "
                            f"expected {self.expected_sonar_resolution} (H,W), got {actual_sonar}.\n"
                            f"Sonar rect data must be exactly 682x376 for geometric consistency.\n"
                            f"Run: python scripts/preprocess_sonar.py --all --width 682 --height 376"
                        )
                # Validate RGB resolution (warning only - less critical)
                if rgb_img is not None:
                    actual_rgb = (rgb_img.shape[0], rgb_img.shape[1])
                    if actual_rgb != self.expected_rgb_resolution:
                        logger.warning(
                            f"RGB resolution mismatch: expected {self.expected_rgb_resolution} (H,W), "
                            f"got {actual_rgb}. Template quality scoring may be affected."
                        )

            # Evaluate quality for both modalities
            rgb_complete, rgb_quality = self._evaluate_template_quality(rgb_bbox, rgb_size)
            sonar_complete, sonar_quality = self._evaluate_template_quality(sonar_bbox, sonar_size)

            # CRITICAL FIX (Audit 2025121202 Finding 22): Apply quality thresholds
            # A modality is only considered "high quality" if it meets the threshold
            rgb_meets_threshold = rgb_complete and rgb_quality >= self.quality_threshold_rgb
            sonar_meets_threshold = sonar_complete and sonar_quality >= self.quality_threshold_sonar

            # Combined quality: require at least one modality to be valid
            rgb_valid = rgb_bbox is not None and np.sum(np.abs(rgb_bbox)) > 0
            sonar_valid = sonar_bbox is not None and np.sum(np.abs(sonar_bbox)) > 0

            if not (rgb_valid or sonar_valid):
                continue  # Skip fully invalid frames

            # Compute combined quality score
            # Prefer frames where both modalities meet quality thresholds
            if rgb_meets_threshold and sonar_meets_threshold:
                combined_quality = (rgb_quality + sonar_quality) / 2.0 + 0.2  # Bonus for dual-complete
                reason = "dual_high_quality"
            elif rgb_meets_threshold:
                combined_quality = rgb_quality * 0.8
                reason = "rgb_high_quality"
            elif sonar_meets_threshold:
                combined_quality = sonar_quality * 0.8
                reason = "sonar_high_quality"
            elif rgb_complete and sonar_complete:
                # Both complete but below threshold
                combined_quality = (rgb_quality + sonar_quality) / 2.0
                reason = "dual_complete_below_threshold"
            elif rgb_complete:
                combined_quality = rgb_quality * 0.6
                reason = "rgb_complete_below_threshold"
            elif sonar_complete:
                combined_quality = sonar_quality * 0.6
                reason = "sonar_complete_below_threshold"
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

        # CRITICAL FIX (Audit 2025121202 Finding 17): Use configured metrics list
        # Default metrics computed by compute_sequence_metrics: success_auc, precision_20,
        # avg_center_error, mean_iou, success_0.5. Extract additional metrics from config.
        default_metrics = {'success_auc', 'precision_20', 'avg_center_error', 'mean_iou', 'success_0.5'}
        if self.metrics_list:
            additional_metrics = [m for m in self.metrics_list if m not in default_metrics]
            logger.debug(f"Using configured metrics. Additional metrics: {additional_metrics}")
        else:
            # Fallback to default additional metrics
            additional_metrics = ['precision_10', 'precision_5', 'success_0.7']
            logger.debug(f"Using default additional metrics: {additional_metrics}")

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
                    additional_metrics=additional_metrics
                )
                seq_metrics['rgb'] = rgb_metrics

            # Sonar metrics
            if predictions['sonar_pred'] and predictions['sonar_gt']:
                sonar_metrics = compute_sequence_metrics(
                    predictions['sonar_pred'],
                    predictions['sonar_gt'],
                    modality='sonar',
                    additional_metrics=additional_metrics
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

            # CRITICAL FIX (Audit 2025121202 Finding 17): Record configuration used
            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Template size: {self.template_size}\n")
            f.write(f"Search size: {self.search_size}\n")
            f.write(f"Template update: {self.enable_template_update}\n")
            if self.enable_template_update:
                f.write(f"  Interval: {self.template_update_interval}\n")
                f.write(f"  Threshold: {self.template_update_threshold}\n")
            f.write(f"Validity gating: {self.enable_validity_gating}\n")
            if self.enable_validity_gating:
                f.write(f"  RGB threshold: {self.rgb_validity_threshold}\n")
                f.write(f"  Sonar threshold: {self.sonar_validity_threshold}\n")
            # Record metrics list
            default_metrics = ['success_auc', 'precision_20', 'avg_center_error', 'mean_iou', 'success_0.5']
            if self.metrics_list:
                additional = [m for m in self.metrics_list if m not in default_metrics]
            else:
                additional = ['precision_10', 'precision_5', 'success_0.7']
            f.write(f"Metrics (default): {default_metrics}\n")
            f.write(f"Metrics (additional): {additional}\n")
            f.write("\n")

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
