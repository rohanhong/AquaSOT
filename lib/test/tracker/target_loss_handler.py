"""
Target loss detection and recovery handler for AquaSOT.

Implements dual-modal independent state machines for RGB and Sonar modalities.
Each modality tracks its own state (STABLE/UNCERTAIN/LOST) independently,
allowing Sonar to continue tracking when target leaves RGB FOV.

Reference: NA-OSTrack OcclusionHandler (external/NA-OSTrack/occlusion_handler/)
"""

import numpy as np
from enum import Enum, auto


class TrackingState(Enum):
    """Tracking state for each modality."""
    STABLE = auto()      # Target confidently tracked
    UNCERTAIN = auto()   # Low confidence, proceed with caution
    LOST = auto()        # Target lost, hold last stable position


class ModalityState:
    """Independent state tracking for a single modality (RGB or Sonar)."""

    def __init__(self):
        self.state = TrackingState.STABLE
        self.lost_counter = 0
        self.found_counter = 0
        self.last_stable_box = None  # [cx, cy, w, h]
        self.last_stable_frame = 0   # Frame number when last STABLE

    def reset(self, init_box):
        """Reset state with initial box."""
        self.state = TrackingState.STABLE
        self.lost_counter = 0
        self.found_counter = 0
        self.last_stable_box = np.array(init_box).copy()
        self.last_stable_frame = 0


class DualModalTargetLossHandler:
    """
    Handles target loss detection and recovery for dual-modal tracking.

    Key features:
    1. Independent state machines for RGB and Sonar
    2. Score-based target loss detection with hysteresis
    3. Search region validation to prevent drift
    4. Primary output selection based on modality states and scores
    """

    def __init__(self, cfg):
        """
        Initialize handler with config parameters.

        Args:
            cfg: Configuration object with TEST.TARGET_LOSS section
        """
        self.enabled = cfg.TEST.TARGET_LOSS.ENABLED

        # Score thresholds
        self.not_found_threshold = cfg.TEST.TARGET_LOSS.NOT_FOUND_THRESHOLD
        self.uncertain_threshold = cfg.TEST.TARGET_LOSS.UNCERTAIN_THRESHOLD
        self.stable_threshold = cfg.TEST.TARGET_LOSS.STABLE_THRESHOLD

        # Validate threshold ordering
        assert self.not_found_threshold < self.uncertain_threshold, (
            f"not_found_threshold ({self.not_found_threshold}) must be < "
            f"uncertain_threshold ({self.uncertain_threshold})"
        )
        assert self.uncertain_threshold <= self.stable_threshold, (
            f"uncertain_threshold ({self.uncertain_threshold}) must be <= "
            f"stable_threshold ({self.stable_threshold})"
        )

        # Hysteresis parameters
        self.lost_confirm_frames = cfg.TEST.TARGET_LOSS.LOST_CONFIRM_FRAMES
        self.found_confirm_frames = cfg.TEST.TARGET_LOSS.FOUND_CONFIRM_FRAMES
        self.quick_recovery_frames = cfg.TEST.TARGET_LOSS.QUICK_RECOVERY_FRAMES

        # Search region validation
        self.search_region_k = cfg.TEST.TARGET_LOSS.SEARCH_REGION_K
        self.use_search_region = cfg.TEST.TARGET_LOSS.USE_SEARCH_REGION

        # Primary output selection
        self.primary_priority = cfg.TEST.TARGET_LOSS.PRIMARY_PRIORITY

        # Per-modality state
        self.rgb_modality = ModalityState()
        self.sonar_modality = ModalityState()

        # Image shapes for search region validation
        self.rgb_image_shape = None   # (H, W)
        self.sonar_image_shape = None  # (H, W)

        # Modality-specific last stable boxes (keep coordinates separate)
        # IMPORTANT: RGB and Sonar have different coordinate systems (1920x1080 vs 682x376)
        # Never mix coordinates between modalities
        self.rgb_last_stable_fallback = None
        self.sonar_last_stable_fallback = None

        # Track last stable scores for analysis when both modalities are LOST
        self.rgb_last_stable_score = 1.0
        self.sonar_last_stable_score = 1.0

        # Frame counter for tracking which modality was last stable
        self.frame_counter = 0

    def initialize(self, rgb_init_box, sonar_init_box, rgb_image_shape, sonar_image_shape):
        """
        Initialize handler with first frame info.

        Args:
            rgb_init_box: Initial RGB box [cx, cy, w, h]
            sonar_init_box: Initial Sonar box [cx, cy, w, h]
            rgb_image_shape: RGB image shape (H, W)
            sonar_image_shape: Sonar image shape (H, W)
        """
        self.rgb_modality.reset(rgb_init_box)
        self.sonar_modality.reset(sonar_init_box)

        self.rgb_image_shape = rgb_image_shape
        self.sonar_image_shape = sonar_image_shape

        # Initialize modality-specific fallbacks (keep coordinate systems separate)
        self.rgb_last_stable_fallback = np.array(rgb_init_box).copy()
        self.sonar_last_stable_fallback = np.array(sonar_init_box).copy()

        # Reset stable scores (first frame assumed stable with score 1.0)
        self.rgb_last_stable_score = 1.0
        self.sonar_last_stable_score = 1.0

        # Reset frame counter
        self.frame_counter = 0

    def check(self, rgb_pred_box, sonar_pred_box, rgb_score, sonar_score,
              rgb_prev_box, sonar_prev_box):
        """
        Check tracking validity for both modalities independently.

        Args:
            rgb_pred_box: RGB prediction [cx, cy, w, h]
            sonar_pred_box: Sonar prediction [cx, cy, w, h]
            rgb_score: RGB confidence score
            sonar_score: Sonar confidence score
            rgb_prev_box: Previous RGB state box [cx, cy, w, h]
            sonar_prev_box: Previous Sonar state box [cx, cy, w, h]

        Returns:
            dict with keys:
                - rgb_box: Output RGB box
                - sonar_box: Output Sonar box
                - primary_box: Primary output box (best available)
                - rgb_should_update: Whether to update RGB state
                - sonar_should_update: Whether to update Sonar state
                - rgb_state: RGB TrackingState
                - sonar_state: Sonar TrackingState
        """
        # Increment frame counter
        self.frame_counter += 1

        if not self.enabled:
            return {
                'rgb_box': rgb_pred_box,
                'sonar_box': sonar_pred_box,
                'primary_box': rgb_pred_box,
                'rgb_should_update': True,
                'sonar_should_update': True,
                'rgb_state': TrackingState.STABLE,
                'sonar_state': TrackingState.STABLE,
            }

        # 1. Check RGB modality independently
        rgb_output, rgb_should_update = self._check_modality(
            self.rgb_modality, rgb_pred_box, rgb_score,
            rgb_prev_box, self.rgb_image_shape
        )

        # 2. Check Sonar modality independently
        sonar_output, sonar_should_update = self._check_modality(
            self.sonar_modality, sonar_pred_box, sonar_score,
            sonar_prev_box, self.sonar_image_shape
        )

        # 3. Update modality-specific fallback boxes, scores, and timestamps
        # IMPORTANT: Never mix RGB and Sonar coordinates
        if self.rgb_modality.state == TrackingState.STABLE:
            self.rgb_last_stable_fallback = rgb_output.copy()
            self.rgb_last_stable_score = rgb_score
            self.rgb_modality.last_stable_frame = self.frame_counter
        if self.sonar_modality.state == TrackingState.STABLE:
            self.sonar_last_stable_fallback = sonar_output.copy()
            self.sonar_last_stable_score = sonar_score
            self.sonar_modality.last_stable_frame = self.frame_counter

        # 4. Select primary output
        primary_output = self._select_primary(
            rgb_output, sonar_output,
            self.rgb_modality.state, self.sonar_modality.state,
            rgb_score, sonar_score
        )

        return {
            'rgb_box': rgb_output,
            'sonar_box': sonar_output,
            'primary_box': primary_output,
            'rgb_should_update': rgb_should_update,
            'sonar_should_update': sonar_should_update,
            'rgb_state': self.rgb_modality.state,
            'sonar_state': self.sonar_modality.state,
        }

    def _check_modality(self, modality, pred_box, score, prev_box, image_shape):
        """
        Check single modality tracking validity.

        Args:
            modality: ModalityState instance
            pred_box: Predicted box [cx, cy, w, h]
            score: Confidence score
            prev_box: Previous state box [cx, cy, w, h]
            image_shape: Image shape (H, W)

        Returns:
            output_box: Validated output box
            should_update: Whether to update tracker state
        """
        # Classify score into flag
        flag = self._classify_score(score)

        # Search region validation
        in_region = True
        if self.use_search_region and prev_box is not None and image_shape is not None:
            in_region = self._in_search_region(prev_box, pred_box, image_shape)

        # Determine if target is lost this frame
        target_lost = flag in ['not_found', 'uncertain'] or not in_region

        # Update state machine with hysteresis
        self._update_state_machine(modality, target_lost, score)

        # Determine output based on state
        output_box, should_update = self._get_output(modality, pred_box)

        return output_box, should_update

    def _classify_score(self, score):
        """Classify score into flag string.

        Score classification with three-tier thresholds:
        - not_found: score < not_found_threshold (0.25)
        - uncertain: not_found_threshold <= score < uncertain_threshold (0.40)
        - marginal: uncertain_threshold <= score < stable_threshold (0.50)
        - stable: score >= stable_threshold (0.50)

        Note: Both 'uncertain' and 'marginal' are treated as uncertain states
        in the state machine, but marginal scores are closer to stable.
        """
        if score < self.not_found_threshold:
            return 'not_found'
        elif score < self.uncertain_threshold:
            return 'uncertain'
        elif score >= self.stable_threshold:
            return 'stable'
        else:
            # marginal zone: uncertain_threshold <= score < stable_threshold
            return 'uncertain'

    def _in_search_region(self, prev_box, pred_box, image_shape):
        """
        Check if prediction center is within valid search region.

        Args:
            prev_box: Previous box [cx, cy, w, h]
            pred_box: Predicted box [cx, cy, w, h]
            image_shape: Image shape (H, W)

        Returns:
            bool: True if prediction is within search region
        """
        cx_prev, cy_prev, w_prev, h_prev = prev_box
        cx_pred, cy_pred = pred_box[0], pred_box[1]

        # Calculate search region bounds (K * target size)
        margin_x = w_prev * self.search_region_k / 2
        margin_y = h_prev * self.search_region_k / 2

        x_min = max(0, cx_prev - margin_x)
        x_max = min(image_shape[1], cx_prev + margin_x)
        y_min = max(0, cy_prev - margin_y)
        y_max = min(image_shape[0], cy_prev + margin_y)

        return x_min <= cx_pred <= x_max and y_min <= cy_pred <= y_max

    def _update_state_machine(self, modality, target_lost, score):
        """
        Update modality state machine with hysteresis.

        Args:
            modality: ModalityState instance
            target_lost: Whether target appears lost this frame
            score: Confidence score
        """
        if target_lost:
            modality.found_counter = 0
            modality.lost_counter += 1

            if modality.state == TrackingState.STABLE:
                if modality.lost_counter >= self.lost_confirm_frames:
                    modality.state = TrackingState.LOST
                else:
                    modality.state = TrackingState.UNCERTAIN
            elif modality.state == TrackingState.UNCERTAIN:
                if modality.lost_counter >= self.lost_confirm_frames:
                    modality.state = TrackingState.LOST
            # If already LOST, stay LOST

        else:
            modality.lost_counter = 0
            modality.found_counter += 1

            if modality.state == TrackingState.LOST:
                if modality.found_counter >= self.found_confirm_frames:
                    modality.state = TrackingState.STABLE
                else:
                    modality.state = TrackingState.UNCERTAIN
            elif modality.state == TrackingState.UNCERTAIN:
                # Quick recovery from uncertain with good frames
                if modality.found_counter >= self.quick_recovery_frames:
                    modality.state = TrackingState.STABLE

    def _get_output(self, modality, pred_box):
        """
        Get output box and update decision based on modality state.

        Args:
            modality: ModalityState instance
            pred_box: Predicted box [cx, cy, w, h]

        Returns:
            output_box: Output box to use
            should_update: Whether to update tracker state
        """
        if modality.state == TrackingState.STABLE:
            # Confident tracking: use prediction, update last stable
            modality.last_stable_box = pred_box.copy()
            return pred_box, True

        elif modality.state == TrackingState.UNCERTAIN:
            # Uncertain: use prediction but with caution
            # Still allow state update to track movement
            return pred_box, True

        else:  # LOST
            # Lost: hold last stable position
            return modality.last_stable_box.copy(), False

    def _select_primary(self, rgb_box, sonar_box, rgb_state, sonar_state,
                        rgb_score, sonar_score):
        """
        Select primary output box based on modality states and scores.

        Args:
            rgb_box: RGB output box
            sonar_box: Sonar output box
            rgb_state: RGB TrackingState
            sonar_state: Sonar TrackingState
            rgb_score: RGB confidence score
            sonar_score: Sonar confidence score

        Returns:
            Primary output box
        """
        if rgb_state == TrackingState.STABLE and sonar_state == TrackingState.STABLE:
            # Both valid: use priority setting
            if self.primary_priority == "rgb_first":
                return rgb_box
            elif self.primary_priority == "sonar_first":
                return sonar_box
            else:  # higher_score
                return rgb_box if rgb_score >= sonar_score else sonar_box

        elif rgb_state == TrackingState.STABLE:
            # Only RGB is stable
            return rgb_box

        elif sonar_state == TrackingState.STABLE:
            # Only Sonar is stable (target may be outside RGB FOV)
            return sonar_box

        else:
            # Both lost: use fallback from the modality that was last stable
            # This provides the most recent reliable position estimate
            rgb_last_frame = self.rgb_modality.last_stable_frame
            sonar_last_frame = self.sonar_modality.last_stable_frame

            if sonar_last_frame > rgb_last_frame:
                # Sonar was stable more recently (target may have left RGB FOV)
                return self.sonar_last_stable_fallback.copy()
            else:
                # RGB was stable more recently, or both equal (default to RGB)
                return self.rgb_last_stable_fallback.copy()
