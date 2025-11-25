"""
Online Tracker for Template-Search Inference

This module provides online tracking interface with template update strategies
following mainstream SOT methods (OSTrack/TransT/ARTrack).

Implements class-agnostic tracking with dynamic template refresh.

CRITICAL FIX (Audit 2025112508):
- Fixed letterbox metadata generation and transmission to model
- Fixed conditional fusion routing via confidence-driven validity inference
- Fixed preprocessing consistency (padding values, normalization from config)

Author: AquaSOT-3D Team
Date: 2025-11-25
"""

import torch
import numpy as np
from typing import Dict, Literal, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# CRITICAL FIX (Audit 2025112508 Finding 1 & 3):
# Import shared preprocessing utilities for consistency
from aquasot3d.utils.preprocessing import preprocess_for_inference
from aquasot3d.utils.config import load_config


@dataclass
class TemplateBuffer:
    """
    Buffer for storing template frames and metadata.

    CRITICAL FIX (Audit 2025112508 Finding 1):
    Added metadata storage for letterbox transformation.
    """
    rgb_template: torch.Tensor  # [1, 3, H, W]
    sonar_template: torch.Tensor  # [1, 1, H, W]
    template_bbox_rgb: np.ndarray  # [4] in CXCYWH normalized format
    template_bbox_sonar: np.ndarray  # [4] in CXCYWH normalized format
    frame_id_rgb: int  # Frame ID when RGB template was last updated
    frame_id_sonar: int  # Frame ID when Sonar template was last updated
    quality_score_rgb: Optional[float] = None  # RGB quality score [0, 1]
    quality_score_sonar: Optional[float] = None  # Sonar quality score [0, 1]
    # CRITICAL FIX (Audit 2025112508 Finding 1): Letterbox metadata
    rgb_template_metadata: Optional[Dict] = None
    sonar_template_metadata: Optional[Dict] = None


class OnlineTracker:
    """
    Online tracking with template update strategies.

    This class wraps AquaSOT3DTracker to provide online inference interface
    with per-frame template management, following OSTrack/TransT paradigm.

    Template Update Strategies:
    - first_frame: Always use first frame as template (baseline, no updates)
    - recent: Update template every N frames with latest prediction
    - keyframe: Update template based on quality/motion thresholds
    - mixformer: MixFormer-style quality-driven updates with independent RGB/Sonar updating

    Args:
        model: AquaSOT3DTracker instance
        template_size: Template image size (default: 256)
        search_size: Search image size (default: 512)
        update_strategy: Template update strategy (default: 'first_frame')
        update_interval: Frames between updates for 'recent' strategy (default: 10)
        quality_threshold_rgb: RGB quality threshold for 'mixformer'/'keyframe' (default: 0.7)
        quality_threshold_sonar: Sonar quality threshold for 'mixformer'/'keyframe' (default: 0.7)
        device: Device for inference (default: 'cuda')

    Example:
        >>> tracker = OnlineTracker(model, update_strategy='first_frame')
        >>> tracker.init(first_frame_rgb, first_frame_sonar, init_bbox_rgb, init_bbox_sonar)
        >>> for frame_rgb, frame_sonar in video:
        >>>     bbox_rgb, bbox_sonar, bbox_fusion = tracker.track(frame_rgb, frame_sonar)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config_path: Optional[str] = None,
        template_size: int = 256,
        search_size: int = 512,
        update_strategy: Literal['first_frame', 'recent', 'keyframe', 'mixformer'] = 'first_frame',
        update_interval: int = 10,
        quality_threshold_rgb: float = 0.7,
        quality_threshold_sonar: float = 0.7,
        confidence_threshold: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize OnlineTracker.

        CRITICAL FIX (Audit 2025112508):
        - Added config_path to load normalization parameters
        - Added confidence_threshold for modality validity inference
        - Now stores metadata for geometry-gated fusion

        Args:
            model: AquaSOT3DTracker instance
            config_path: Path to config file (for normalization params). If None, uses defaults.
            template_size: Template image size (default: 256)
            search_size: Search image size (default: 512)
            update_strategy: Template update strategy
            update_interval: Frames between updates for 'recent' strategy
            quality_threshold_rgb: RGB quality threshold for 'mixformer'/'keyframe'
            quality_threshold_sonar: Sonar quality threshold for 'mixformer'/'keyframe'
            confidence_threshold: Confidence threshold for validity inference (Audit 2025112508 Finding 2)
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()

        self.template_size = template_size
        self.search_size = search_size
        self.update_strategy = update_strategy
        self.update_interval = update_interval
        self.quality_threshold_rgb = quality_threshold_rgb
        self.quality_threshold_sonar = quality_threshold_sonar
        self.device = device

        # CRITICAL FIX (Audit 2025112508 Finding 2):
        # Confidence threshold for modality validity inference
        self.confidence_threshold = confidence_threshold

        # CRITICAL FIX (Audit 2025112508 Finding 3):
        # Load normalization parameters from config (not hardcoded)
        if config_path is not None:
            cfg = load_config(config_path)
            self.rgb_mean = cfg.data.normalize.rgb_mean
            self.rgb_std = cfg.data.normalize.rgb_std
            self.sonar_mean = cfg.data.normalize.sonar_mean[0] if isinstance(cfg.data.normalize.sonar_mean, list) else cfg.data.normalize.sonar_mean
            self.sonar_std = cfg.data.normalize.sonar_std[0] if isinstance(cfg.data.normalize.sonar_std, list) else cfg.data.normalize.sonar_std
        else:
            # Default to standard values if no config provided
            self.rgb_mean = [0.485, 0.456, 0.406]
            self.rgb_std = [0.229, 0.224, 0.225]
            self.sonar_mean = 0.5
            self.sonar_std = 0.5

        # Tracking state
        self.template_buffer: Optional[TemplateBuffer] = None
        self.frame_count = 0
        self.is_initialized = False

        # CRITICAL FIX (Audit 2025112508 Finding 1):
        # Storage for current frame metadata (needed for model forward pass)
        self.current_rgb_search_metadata: Optional[Dict] = None
        self.current_sonar_search_metadata: Optional[Dict] = None

    def init(
        self,
        rgb_frame: np.ndarray,
        sonar_frame: np.ndarray,
        bbox_rgb: np.ndarray,
        bbox_sonar: np.ndarray
    ):
        """
        Initialize tracker with first frame.

        CRITICAL FIX (Audit 2025112508 Finding 1):
        Now captures and stores letterbox metadata.

        Args:
            rgb_frame: RGB image [H, W, 3] in uint8 format
            sonar_frame: Sonar image [H, W] in uint8 format
            bbox_rgb: Initial RGB bbox [4] in CXCYWH normalized format
            bbox_sonar: Initial sonar bbox [4] in CXCYWH normalized format
        """
        # CRITICAL FIX: Preprocess frames and capture metadata
        rgb_template, rgb_template_metadata = preprocess_for_inference(
            rgb_frame, self.template_size, is_rgb=True,
            rgb_mean=self.rgb_mean, rgb_std=self.rgb_std,
            device=self.device
        )
        sonar_template, sonar_template_metadata = preprocess_for_inference(
            sonar_frame, self.template_size, is_rgb=False,
            sonar_mean=self.sonar_mean, sonar_std=self.sonar_std,
            device=self.device
        )

        # Store initial template with metadata
        self.template_buffer = TemplateBuffer(
            rgb_template=rgb_template,
            sonar_template=sonar_template,
            template_bbox_rgb=bbox_rgb.copy(),
            template_bbox_sonar=bbox_sonar.copy(),
            frame_id_rgb=0,
            frame_id_sonar=0,
            quality_score_rgb=1.0,
            quality_score_sonar=1.0,
            rgb_template_metadata=rgb_template_metadata,
            sonar_template_metadata=sonar_template_metadata
        )

        self.frame_count = 0
        self.is_initialized = True

    def track(
        self,
        rgb_frame: np.ndarray,
        sonar_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track object in current frame.

        CRITICAL FIX (Audit 2025112508):
        - Now generates and passes letterbox metadata to model (Finding 1)
        - Implements confidence-driven validity inference (Finding 2)
        - Uses unified preprocessing with correct padding/normalization (Finding 3)

        Args:
            rgb_frame: RGB image [H, W, 3] in uint8 format
            sonar_frame: Sonar image [H, W] in uint8 format

        Returns:
            Tuple of (bbox_rgb, bbox_sonar, bbox_fusion) in CXCYWH normalized format
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")

        self.frame_count += 1

        # CRITICAL FIX (Audit 2025112508 Finding 1 & 3):
        # Preprocess search frames with metadata capture
        # Uses shared letterbox implementation with correct padding (RGB=114, Sonar=127)
        # and config-driven normalization
        rgb_search, rgb_search_metadata = preprocess_for_inference(
            rgb_frame, self.search_size, is_rgb=True,
            rgb_mean=self.rgb_mean, rgb_std=self.rgb_std,
            device=self.device
        )
        sonar_search, sonar_search_metadata = preprocess_for_inference(
            sonar_frame, self.search_size, is_rgb=False,
            sonar_mean=self.sonar_mean, sonar_std=self.sonar_std,
            device=self.device
        )

        # Store metadata for current frame
        self.current_rgb_search_metadata = rgb_search_metadata
        self.current_sonar_search_metadata = sonar_search_metadata

        # CRITICAL FIX (Audit 2025112508 Finding 1):
        # Prepare metadata dict for model (geometry-gated fusion needs this)
        metadata = {
            'rgb_search_metadata': rgb_search_metadata,
            'sonar_search_metadata': sonar_search_metadata,
            'rgb_template_metadata': self.template_buffer.rgb_template_metadata,
            'sonar_template_metadata': self.template_buffer.sonar_template_metadata
        }

        # CRITICAL FIX (Audit 2025112508 Finding 2):
        # DO NOT pass is_rgb_valid/is_sonar_valid here - let them default to None
        # This allows ConditionalG2DCMA.forward() to automatically infer validity
        # from learned confidence estimators, enabling proper single/zero-modality routing.
        #
        # Flow: OnlineTracker (validity=None)
        #       → AquaSOT3DTracker.forward (passes None through)
        #       → ConditionalG2DCMA.forward (infers validity from confidence)
        #       → Correct routing (dual/single-RGB/single-Sonar/zero)
        with torch.no_grad():
            outputs = self.model(
                rgb_template=self.template_buffer.rgb_template,
                rgb_search=rgb_search,
                sonar_template=self.template_buffer.sonar_template,
                sonar_search=sonar_search,
                return_losses=False,
                metadata=metadata
                # NOTE: is_rgb_valid and is_sonar_valid NOT passed (default to None)
                # This enables automatic validity inference in ConditionalG2DCMA
            )

        # Extract predictions (CXCYWH normalized)
        # ARCHITECTURE UPDATE (Audit 2025112313): fusion_boxes removed (dual-view fusion)
        # RGB and Sonar boxes now use dual-view fused features. Use RGB as primary output.
        bbox_rgb = outputs['rgb_boxes'][0].cpu().numpy()
        bbox_sonar = outputs['sonar_boxes'][0].cpu().numpy()
        bbox_fusion = bbox_rgb  # Use RGB as primary output (dual-view fused features)

        # Extract quality scores for each modality
        quality_score_rgb = None
        quality_score_sonar = None

        if 'rgb_quality' in outputs:
            quality_score_rgb = outputs['rgb_quality'][0].item()

        if 'sonar_quality' in outputs:
            quality_score_sonar = outputs['sonar_quality'][0].item()

        # CRITICAL FIX (Audit 2025112502 Finding 4): Extract confidence (objectness) scores
        # Confidence represents has_object probability, distinct from quality (IoU-based)
        # Used for robust template update decisions in occluded/missing target scenarios
        confidence_rgb = None
        confidence_sonar = None

        if 'rgb_confidence' in outputs:
            confidence_rgb = outputs['rgb_confidence'][0].item()

        if 'sonar_confidence' in outputs:
            confidence_sonar = outputs['sonar_confidence'][0].item()

        # Update template based on strategy
        # CRITICAL FIX: Use modality-specific bboxes for independent template updates
        # This ensures MixFormer-style independent RGB/Sonar template updating works correctly
        # ENHANCED (Audit 2025112502 Finding 4): Pass confidence scores for robust update decisions
        self._update_template(
            rgb_frame=rgb_frame,
            sonar_frame=sonar_frame,
            bbox_rgb=bbox_rgb,
            bbox_sonar=bbox_sonar,
            quality_score_rgb=quality_score_rgb,
            quality_score_sonar=quality_score_sonar,
            confidence_rgb=confidence_rgb,
            confidence_sonar=confidence_sonar
        )

        return bbox_rgb, bbox_sonar, bbox_fusion

    def _update_template(
        self,
        rgb_frame: np.ndarray,
        sonar_frame: np.ndarray,
        bbox_rgb: np.ndarray,
        bbox_sonar: np.ndarray,
        quality_score_rgb: Optional[float],
        quality_score_sonar: Optional[float],
        confidence_rgb: Optional[float] = None,
        confidence_sonar: Optional[float] = None
    ):
        """
        Update template according to update strategy.

        ENHANCED (Audit 2025112502 Finding 4): Added confidence parameters for robust updates.

        For 'mixformer' strategy, RGB and Sonar templates are updated independently
        based on their respective quality scores, allowing robust tracking even when
        one modality degrades (e.g., RGB occlusion but Sonar clear).
        """
        if self.update_strategy == 'first_frame':
            return

        elif self.update_strategy == 'recent':
            if self.frame_count % self.update_interval == 0:
                # CRITICAL FIX (Audit 2025112508): Use shared preprocessing with metadata
                rgb_template, rgb_template_metadata = preprocess_for_inference(
                    rgb_frame, self.template_size, is_rgb=True,
                    rgb_mean=self.rgb_mean, rgb_std=self.rgb_std,
                    device=self.device
                )
                sonar_template, sonar_template_metadata = preprocess_for_inference(
                    sonar_frame, self.template_size, is_rgb=False,
                    sonar_mean=self.sonar_mean, sonar_std=self.sonar_std,
                    device=self.device
                )

                self.template_buffer = TemplateBuffer(
                    rgb_template=rgb_template,
                    sonar_template=sonar_template,
                    template_bbox_rgb=bbox_rgb.copy(),
                    template_bbox_sonar=bbox_sonar.copy(),
                    frame_id_rgb=self.frame_count,
                    frame_id_sonar=self.frame_count,
                    quality_score_rgb=quality_score_rgb,
                    quality_score_sonar=quality_score_sonar,
                    rgb_template_metadata=rgb_template_metadata,
                    sonar_template_metadata=sonar_template_metadata
                )

        elif self.update_strategy == 'keyframe':
            avg_quality = None
            if quality_score_rgb is not None and quality_score_sonar is not None:
                avg_quality = (quality_score_rgb + quality_score_sonar) / 2.0

            if avg_quality is not None and avg_quality > max(self.quality_threshold_rgb, self.quality_threshold_sonar):
                # CRITICAL FIX (Audit 2025112508): Use shared preprocessing with metadata
                rgb_template, rgb_template_metadata = preprocess_for_inference(
                    rgb_frame, self.template_size, is_rgb=True,
                    rgb_mean=self.rgb_mean, rgb_std=self.rgb_std,
                    device=self.device
                )
                sonar_template, sonar_template_metadata = preprocess_for_inference(
                    sonar_frame, self.template_size, is_rgb=False,
                    sonar_mean=self.sonar_mean, sonar_std=self.sonar_std,
                    device=self.device
                )

                self.template_buffer = TemplateBuffer(
                    rgb_template=rgb_template,
                    sonar_template=sonar_template,
                    template_bbox_rgb=bbox_rgb.copy(),
                    template_bbox_sonar=bbox_sonar.copy(),
                    frame_id_rgb=self.frame_count,
                    frame_id_sonar=self.frame_count,
                    quality_score_rgb=quality_score_rgb,
                    quality_score_sonar=quality_score_sonar,
                    rgb_template_metadata=rgb_template_metadata,
                    sonar_template_metadata=sonar_template_metadata
                )

        elif self.update_strategy == 'mixformer':
            # CRITICAL FIX (Audit 2025112502 Finding 4): Enhanced template update with confidence
            # Confidence (objectness) prevents updates on false positives or missing targets
            # Quality (IoU) ensures accurate localization before updating
            # Both must be high for safe template update
            update_rgb = False
            update_sonar = False

            # RGB template update: require BOTH high quality AND high confidence
            if quality_score_rgb is not None and quality_score_rgb > self.quality_threshold_rgb:
                if confidence_rgb is not None and confidence_rgb > 0.7:
                    update_rgb = True
                elif confidence_rgb is None:
                    # Fallback: if confidence not available, use quality alone (backward compatibility)
                    update_rgb = True

            # Sonar template update: require BOTH high quality AND high confidence
            if quality_score_sonar is not None and quality_score_sonar > self.quality_threshold_sonar:
                if confidence_sonar is not None and confidence_sonar > 0.7:
                    update_sonar = True
                elif confidence_sonar is None:
                    # Fallback: if confidence not available, use quality alone (backward compatibility)
                    update_sonar = True

            if update_rgb or update_sonar:
                # Start with current buffer values
                new_rgb_template = self.template_buffer.rgb_template
                new_sonar_template = self.template_buffer.sonar_template
                new_bbox_rgb = self.template_buffer.template_bbox_rgb
                new_bbox_sonar = self.template_buffer.template_bbox_sonar
                new_frame_id_rgb = self.template_buffer.frame_id_rgb
                new_frame_id_sonar = self.template_buffer.frame_id_sonar
                new_quality_rgb = self.template_buffer.quality_score_rgb
                new_quality_sonar = self.template_buffer.quality_score_sonar
                new_rgb_metadata = self.template_buffer.rgb_template_metadata
                new_sonar_metadata = self.template_buffer.sonar_template_metadata

                # CRITICAL FIX (Audit 2025112508): Use shared preprocessing with metadata
                if update_rgb:
                    new_rgb_template, new_rgb_metadata = preprocess_for_inference(
                        rgb_frame, self.template_size, is_rgb=True,
                        rgb_mean=self.rgb_mean, rgb_std=self.rgb_std,
                        device=self.device
                    )
                    new_bbox_rgb = bbox_rgb.copy()
                    new_frame_id_rgb = self.frame_count
                    new_quality_rgb = quality_score_rgb

                if update_sonar:
                    new_sonar_template, new_sonar_metadata = preprocess_for_inference(
                        sonar_frame, self.template_size, is_rgb=False,
                        sonar_mean=self.sonar_mean, sonar_std=self.sonar_std,
                        device=self.device
                    )
                    new_bbox_sonar = bbox_sonar.copy()
                    new_frame_id_sonar = self.frame_count
                    new_quality_sonar = quality_score_sonar

                self.template_buffer = TemplateBuffer(
                    rgb_template=new_rgb_template,
                    sonar_template=new_sonar_template,
                    template_bbox_rgb=new_bbox_rgb,
                    template_bbox_sonar=new_bbox_sonar,
                    frame_id_rgb=new_frame_id_rgb,
                    frame_id_sonar=new_frame_id_sonar,
                    quality_score_rgb=new_quality_rgb,
                    quality_score_sonar=new_quality_sonar,
                    rgb_template_metadata=new_rgb_metadata,
                    sonar_template_metadata=new_sonar_metadata
                )

    # REMOVED (Audit 2025112508): Legacy preprocessing methods replaced with shared utilities
    # Old _preprocess_image and _letterbox_resize removed in favor of
    # aquasot3d.utils.preprocessing.preprocess_for_inference
    # This ensures training-inference consistency with correct padding (RGB=114, Sonar=127)
    # and metadata generation for geometry-gated fusion

    def reset(self):
        """Reset tracker state."""
        self.template_buffer = None
        self.frame_count = 0
        self.is_initialized = False

    def get_template_info(self) -> Dict:
        """Get current template information for both RGB and Sonar modalities."""
        if self.template_buffer is None:
            return {}

        return {
            'frame_id_rgb': self.template_buffer.frame_id_rgb,
            'frame_id_sonar': self.template_buffer.frame_id_sonar,
            'quality_score_rgb': self.template_buffer.quality_score_rgb,
            'quality_score_sonar': self.template_buffer.quality_score_sonar,
            'bbox_rgb': self.template_buffer.template_bbox_rgb,
            'bbox_sonar': self.template_buffer.template_bbox_sonar
        }
