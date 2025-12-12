"""
Template-Search Dataset for Single Object Tracking

This module implements the mainstream SOT paradigm: sample (template, search) pairs
from continuous sequences for training class-agnostic trackers.

CRITICAL FIX (Audit 2025112602): Implements OSTrack-style target-centered cropping
Template: Target-centered crop with template_area_factor (default 2.0)
Search: Target-centered crop with search_area_factor (default 4.0)

Key Features:
- Class-agnostic: Template from any object in frame 1
- Curriculum learning: Configurable frame gap [1, 30]
- Template policies: first_frame | recent | keyframe | mixformer
- MixFormer strategy: Independent RGB/Sonar template updates based on quality
- 0000 annotation handling: Validity propagation
- Multi-sequence support: Respects sequence boundaries
- Multi-modal: RGB + Sonar
- Target-centered cropping: Following OSTrack/MixFormer methodology
- Boundary padding: cv2.BORDER_CONSTANT with image mean
- Attention masks: Track padded regions for transformer

Author: AquaSOT-3D Team
Date: 2025-11-21
Updated: 2025-11-26 (Audit 2025112602 - Target-centered search cropping)
"""

import math
import random
import cv2
import numpy as np
import torch
import logging
from typing import Dict, Tuple, Optional, List, Union
from torch.utils.data import Dataset

from .rgb_sonar_dataset import RGBSonarDataset, MultiSequenceRGBSonarDataset
from aquasot3d.utils.bbox_utils import convert_bbox_format, normalize_bbox

logger = logging.getLogger(__name__)


class TemplateSearchDataset(Dataset):
    """
    Template-Search paired dataset for SOT training.

    CRITICAL FIX (Audit 2025112602): Implements OSTrack-style target-centered cropping
    instead of full-frame letterbox. This is essential for the model to learn
    target localization rather than predicting image center by default.

    CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering for validation.
    When enable_template_quality_filter=True, scans first N frames to select
    the best quality template (not edge-clipped, sufficient size).

    Samples (template, search) pairs from continuous sequences.
    - Template: Target-centered crop with template_area_factor (default 2.0)
    - Search: Target-centered crop with search_area_factor (default 4.0)

    This follows mainstream SOT methods (OSTrack, TransT, MixFormer) where:
    1. Template carries target appearance from reference frame
    2. Search is a target-centered crop where we locate the target
    3. GT bbox center is always near the search region center (with jitter)
    4. Model learns to predict relative offset, not absolute position

    Args:
        base_dataset: RGBSonarDataset or MultiSequenceRGBSonarDataset instance
        template_size: Template output size (default: 128, OSTrack-256 standard)
        search_size: Search output size (default: 256, OSTrack-256 standard)
        template_search_gap: Frame gap range (min, max) for curriculum learning
        template_update_policy: 'first_frame' | 'recent' | 'keyframe' | 'mixformer'
        search_area_factor: Search crop = sqrt(w*h) * factor (default: 4.0)
        template_area_factor: Template crop = sqrt(w*h) * factor (default: 2.0)
        center_jitter_factor: Center jitter for training (default: 0.0)
        scale_jitter_factor: Scale jitter for training (default: 0.0)
        use_attention_mask: Enable attention mask for padded regions (default: True)
        augmentation: Optional augmentation module

    Example:
        >>> base = RGBSonarDataset(data_root='...', sequence_name='seq1')
        >>> dataset = TemplateSearchDataset(
        ...     base_dataset=base,
        ...     template_search_gap=(1, 30),
        ...     search_area_factor=4.0,
        ...     center_jitter_factor=0.25
        ... )
        >>> sample = dataset[100]
        >>> print(sample['rgb_template'].shape)  # [3, 128, 128] OSTrack-256
        >>> print(sample['rgb_search'].shape)    # [3, 256, 256] OSTrack-256
        >>> print(sample['search_att_mask'].shape)  # [256, 256]
    """

    # CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering defaults
    # These match OnlineTrackingEvaluator and EvaluationSequenceDataset for consistency
    TEMPLATE_MIN_SIZE = 5.0       # Minimum bbox dimension (pixels)
    TEMPLATE_EDGE_MARGIN = 2      # Minimum distance from image edge (pixels)
    TEMPLATE_SCAN_FRAMES = 20     # Number of frames to scan for best template

    def __init__(
        self,
        base_dataset,
        template_size: int = 128,  # OSTrack-256 standard
        search_size: int = 256,    # OSTrack-256 standard
        template_search_gap: Tuple[int, int] = (1, 30),
        template_update_policy: str = 'first_frame',
        quality_threshold_rgb: float = 0.7,
        quality_threshold_sonar: float = 0.7,
        # CRITICAL FIX (Audit 2025112602): OSTrack-style target-centered cropping
        search_area_factor: float = 4.0,      # Search crop = sqrt(w*h) * 4.0
        template_area_factor: float = 2.0,    # Template crop = sqrt(w*h) * 2.0
        center_jitter_factor: float = 0.0,    # Center jitter for training
        scale_jitter_factor: float = 0.0,     # Scale jitter for training
        use_attention_mask: bool = True,      # Enable attention mask for padding
        context_factor: float = 2.0,          # Deprecated, use template_area_factor
        augmentation=None,
        normalization_config: Optional[Dict] = None,
        # CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering for validation
        enable_template_quality_filter: bool = False,  # Disabled by default for training
        template_min_size: float = None,        # Min bbox dimension (pixels)
        template_edge_margin: int = None,       # Min distance from image edge (pixels)
        template_scan_frames: int = None,       # Frames to scan for best template
        # CRITICAL FIX (Audit 2025121201 S3): Configurable negative sampling
        # Training: 0.1 (10%) for has_object calibration
        # Validation/Test: 0.0 (disabled) to match online deployment
        negative_sample_prob: float = 0.1       # Negative sample probability
    ):
        self.base_dataset = base_dataset
        self.template_size = template_size
        self.search_size = search_size
        self.gap_range = template_search_gap
        self.update_policy = template_update_policy
        self.quality_threshold_rgb = quality_threshold_rgb
        self.quality_threshold_sonar = quality_threshold_sonar

        # CRITICAL FIX (Audit 2025112602): Target-centered cropping parameters
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.use_attention_mask = use_attention_mask
        # Legacy support: context_factor maps to template_area_factor
        self.context_factor = template_area_factor if template_area_factor != 2.0 else context_factor

        # CRITICAL FIX (Audit 2025112504 Finding 3):
        # Log visibility_policy from base_dataset for transparency
        # TemplateSearchDataset inherits filtering behavior from base_dataset.valid_indices
        if hasattr(base_dataset, 'visibility_policy'):
            visibility_policy = base_dataset.visibility_policy
            logger.info(
                f"TemplateSearchDataset using visibility_policy='{visibility_policy}' "
                f"from base_dataset ({len(base_dataset)} valid frames)"
            )
            # Warn if using 'all' policy (includes zero-modality frames)
            if visibility_policy == 'all':
                logger.warning(
                    "base_dataset.visibility_policy='all' includes zero-modality frames (both 0000). "
                    "This may dilute training signal. Consider using 'any' or 'both' for cleaner supervision."
                )

        # CRITICAL FIX (Audit 2025112401 Finding 3):
        # Enable augmentation pipeline to improve robustness
        # Apply augmentation in __getitem__ before letterbox transform
        # to ensure bbox-aware transforms work with original image coordinates
        self.augmentation = augmentation

        # CRITICAL FIX (Audit 2025112402 Finding 3):
        # Load normalization parameters from config instead of hardcoding
        # Fallback to ImageNet/0.5 defaults if not provided
        if normalization_config is None:
            normalization_config = {
                'rgb_mean': [0.485, 0.456, 0.406],
                'rgb_std': [0.229, 0.224, 0.225],
                'sonar_mean': [0.5],
                'sonar_std': [0.5]
            }

        self.rgb_mean = torch.tensor(normalization_config.get('rgb_mean', [0.485, 0.456, 0.406])).view(3, 1, 1)
        self.rgb_std = torch.tensor(normalization_config.get('rgb_std', [0.229, 0.224, 0.225])).view(3, 1, 1)
        self.sonar_mean = torch.tensor(normalization_config.get('sonar_mean', [0.5])).view(1, 1, 1)
        self.sonar_std = torch.tensor(normalization_config.get('sonar_std', [0.5])).view(1, 1, 1)

        # CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering
        # When enabled, selects best quality template from first N frames
        # instead of always using the first valid frame
        self.enable_template_quality_filter = enable_template_quality_filter
        self.template_min_size = template_min_size or self.TEMPLATE_MIN_SIZE
        self.template_edge_margin = template_edge_margin or self.TEMPLATE_EDGE_MARGIN
        self.template_scan_frames = template_scan_frames or self.TEMPLATE_SCAN_FRAMES

        if enable_template_quality_filter:
            logger.info(
                f"TemplateSearchDataset template quality filter ENABLED: "
                f"min_size={self.template_min_size}, edge_margin={self.template_edge_margin}, "
                f"scan_frames={self.template_scan_frames}"
            )

        # CRITICAL FIX (Audit 2025120902 Issue 5 + Audit 2025121201 S3):
        # Negative sampling for sonar objectness calibration
        # Training data has sparse sonar occlusion, causing objectness to always predict 1.0
        # Negative samples have search crop centered away from target, simulating "no object" scenarios
        # This is essential for calibrating objectness head to distinguish presence/absence
        # CRITICAL: Val/Test MUST set negative_sample_prob=0.0 to match online deployment
        self.negative_sample_prob = negative_sample_prob  # Configurable, default 10% for training
        self.negative_offset_range = (2.0, 4.0)  # Offset center by 2-4x bbox size

        if negative_sample_prob > 0:
            logger.info(
                f"TemplateSearchDataset negative_sample_prob={negative_sample_prob:.1%} "
                f"(negative samples for has_object calibration)"
            )
        else:
            logger.info("TemplateSearchDataset negative_sample_prob=0 (disabled for validation/test)")

        # DUAL-TEMPLATE SUPPORT (MixFormer-style):
        # When enabled, returns two templates per modality:
        # - template_init: First frame template (anchor, prevents drift)
        # - template_dyn: Dynamic template (sampled from recent high-quality frames)
        self.use_dual_template = (template_update_policy == 'mixformer')

        # Multi-sequence support (PRESERVED from FrameSequenceDataset)
        self.is_multi_sequence = isinstance(
            base_dataset, MultiSequenceRGBSonarDataset
        )

        if self.is_multi_sequence:
            self._build_sequence_index_mapping()
        else:
            self.num_frames = len(base_dataset)
            self.sequence_ranges = None

    def _build_sequence_index_mapping(self):
        """
        Build sequence boundary mapping.

        PRESERVED from sequence_dataset.py (lines 79-95)
        Prevents template-search pairs from crossing sequence boundaries.
        """
        self.sequence_ranges = {}
        self.num_frames = len(self.base_dataset)

        for seq_metadata in self.base_dataset.sequences:
            seq_name = seq_metadata.sequence_name
            self.sequence_ranges[seq_name] = {
                'start': seq_metadata.start_idx,
                'end': seq_metadata.end_idx,
                'num_frames': seq_metadata.end_idx - seq_metadata.start_idx,
            }

    def __len__(self):
        return self.num_frames

    def get_sequence_name(self, index: int) -> str:
        """
        Get sequence name for given index.

        PRESERVED from sequence_dataset.py (lines 100-110)
        Used by SequenceGroupedBatchSampler for sequence-aware batching.
        """
        if self.is_multi_sequence:
            return self.base_dataset.get_frame_sequence(index)
        else:
            return 'default'

    def __getitem__(self, index: int) -> Dict:
        """
        Sample template-search pair.

        Returns:
            Dictionary with keys:
                rgb_template: [3, 128, 128] - RGB template crop (OSTrack-256)
                rgb_search: [3, 256, 256] - RGB search crop (OSTrack-256)
                sonar_template: [1, 128, 128] - Sonar template crop
                sonar_search: [1, 256, 256] - Sonar search crop
                template_bbox: [4] - GT bbox in template (XYWH pixel, for reference)
                search_bbox: [4] - GT bbox in search (XYWH pixel, TARGET)
                is_valid_rgb: bool - RGB validity (0000 handling)
                is_valid_sonar: bool - Sonar validity
                gap: int - Actual frame gap used
                sequence_name: str
                template_frame_id: int - 1-based frame ID
                search_frame_id: int - 1-based frame ID
                rgb_metadata: Dict - Transformation metadata for search
                sonar_metadata: Dict
        """
        # 1. Sample search frame (current frame = index)
        search_sample = self.base_dataset[index]

        # 2. Sample template frame(s) based on policy
        # DUAL-TEMPLATE SUPPORT (MixFormer-style):
        if self.use_dual_template:
            # Init template: first frame (anchor)
            template_init_idx = self._sample_init_template_index(index)
            template_init_sample = self.base_dataset[template_init_idx]
            # Dyn template: recent frame (adaptive)
            template_dyn_idx = self._sample_dyn_template_index(index)
            template_dyn_sample = self.base_dataset[template_dyn_idx]
            # For backward compatibility, use init as the "primary" template
            template_idx = template_init_idx
            template_sample = template_init_sample
        else:
            # Single template mode
            template_idx = self._sample_template_index(index)
            template_sample = self.base_dataset[template_idx]
            template_init_idx = template_idx
            template_init_sample = template_sample
            template_dyn_idx = template_idx
            template_dyn_sample = template_sample

        # CRITICAL FIX (Audit 2025112401 Finding 3):
        # Apply augmentation BEFORE letterbox to ensure bbox-aware transforms
        # work with original image coordinates
        if self.augmentation is not None:
            # Apply to search sample (current frame)
            search_sample = self.augmentation(search_sample)
            # Note: Template is typically not augmented to preserve reference appearance
            # If needed, can add template augmentation with different policy

        # 3. Process template (crop around target bbox)
        # Always process both sensors (target visibility handled by None checks)
        # CRITICAL FIX (Audit 2025112507 Finding 3): Capture crop metadata for bbox transformation
        # CRITICAL FIX (Audit 2025120909): Capture template attention masks
        if template_sample['rgb_img'] is not None:
            rgb_template, rgb_template_att_mask, rgb_crop_meta = self._crop_template(
                template_sample['rgb_img'],
                template_sample['rgb_bbox'],
                self.template_size
            )
        else:
            rgb_template = None
            rgb_template_att_mask = np.zeros((self.template_size, self.template_size), dtype=np.float32)
            rgb_crop_meta = None

        if template_sample['sonar_img'] is not None:
            sonar_template, sonar_template_att_mask, sonar_crop_meta = self._crop_template(
                template_sample['sonar_img'],
                template_sample['sonar_bbox'],
                self.template_size
            )
        else:
            sonar_template = None
            sonar_template_att_mask = np.zeros((self.template_size, self.template_size), dtype=np.float32)
            sonar_crop_meta = None

        # 3b. Process DYNAMIC template for dual-template mode
        # These are the templates that will adapt to appearance changes
        if self.use_dual_template:
            # RGB dynamic template
            if template_dyn_sample['rgb_img'] is not None:
                rgb_template_dyn, rgb_template_dyn_att_mask, _ = self._crop_template(
                    template_dyn_sample['rgb_img'],
                    template_dyn_sample['rgb_bbox'],
                    self.template_size
                )
            else:
                rgb_template_dyn = None
                rgb_template_dyn_att_mask = np.zeros((self.template_size, self.template_size), dtype=np.float32)

            # Sonar dynamic template
            if template_dyn_sample['sonar_img'] is not None:
                sonar_template_dyn, sonar_template_dyn_att_mask, _ = self._crop_template(
                    template_dyn_sample['sonar_img'],
                    template_dyn_sample['sonar_bbox'],
                    self.template_size
                )
            else:
                sonar_template_dyn = None
                sonar_template_dyn_att_mask = np.zeros((self.template_size, self.template_size), dtype=np.float32)
        else:
            # Single template mode: dyn = init
            rgb_template_dyn = rgb_template
            rgb_template_dyn_att_mask = rgb_template_att_mask
            sonar_template_dyn = sonar_template
            sonar_template_dyn_att_mask = sonar_template_att_mask

        # 4. Process search using TEMPLATE-CENTERED CROP
        # CRITICAL FIX (Audit 2025120102): Use Template bbox center to crop Search region
        # This is essential for learning tracking ability:
        # - Template crop: centered on Template GT bbox (target at center)
        # - Search crop: centered on TEMPLATE GT bbox position (NOT Search GT bbox!)
        # - Natural frame-to-frame motion causes target to be offset from Search center
        # - Model learns to predict this offset, which is the core tracking task
        #
        # Previous approach (Search centered on Search GT) caused target to always be
        # at center, preventing the model from learning spatial offset prediction.

        # Extract Template bbox centers for crop positioning
        # RGB: use RGB Template bbox center
        rgb_template_bbox = template_sample.get('rgb_bbox', None)
        if rgb_template_bbox is not None:
            if isinstance(rgb_template_bbox, torch.Tensor):
                rgb_template_bbox = rgb_template_bbox.cpu().numpy()
            rgb_template_cx = rgb_template_bbox[0] + rgb_template_bbox[2] / 2
            rgb_template_cy = rgb_template_bbox[1] + rgb_template_bbox[3] / 2
        else:
            rgb_template_cx, rgb_template_cy = None, None

        # Sonar: use Sonar Template bbox center
        sonar_template_bbox = template_sample.get('sonar_bbox', None)
        if sonar_template_bbox is not None:
            if isinstance(sonar_template_bbox, torch.Tensor):
                sonar_template_bbox = sonar_template_bbox.cpu().numpy()
            sonar_template_cx = sonar_template_bbox[0] + sonar_template_bbox[2] / 2
            sonar_template_cy = sonar_template_bbox[1] + sonar_template_bbox[3] / 2
        else:
            sonar_template_cx, sonar_template_cy = None, None

        # CRITICAL FIX (Audit 2025120902 Issue 5): Negative sampling for objectness calibration
        # Decide if this sample should be a negative sample (target not in search crop)
        # This is essential for training objectness head to output low values when no target
        is_negative_sample = np.random.rand() < self.negative_sample_prob

        # Compute negative offset if this is a negative sample
        # Offset the search crop center by 2-4x bbox size in random direction
        negative_offset_rgb = None
        negative_offset_sonar = None

        if is_negative_sample:
            # For RGB
            rgb_bbox_search = search_sample.get('rgb_bbox', None)
            if rgb_bbox_search is not None:
                if isinstance(rgb_bbox_search, torch.Tensor):
                    rgb_bbox_np = rgb_bbox_search.cpu().numpy()
                else:
                    rgb_bbox_np = rgb_bbox_search
                # Offset by 2-4x bbox diagonal in random direction
                rgb_diag = np.sqrt(rgb_bbox_np[2]**2 + rgb_bbox_np[3]**2)
                offset_mag = np.random.uniform(
                    self.negative_offset_range[0],
                    self.negative_offset_range[1]
                ) * rgb_diag
                offset_angle = np.random.uniform(0, 2 * np.pi)
                negative_offset_rgb = (
                    offset_mag * np.cos(offset_angle),
                    offset_mag * np.sin(offset_angle)
                )

            # For Sonar - critical for objectness calibration
            sonar_bbox_search = search_sample.get('sonar_bbox', None)
            if sonar_bbox_search is not None:
                if isinstance(sonar_bbox_search, torch.Tensor):
                    sonar_bbox_np = sonar_bbox_search.cpu().numpy()
                else:
                    sonar_bbox_np = sonar_bbox_search
                # Offset by 2-4x bbox diagonal in random direction
                sonar_diag = np.sqrt(sonar_bbox_np[2]**2 + sonar_bbox_np[3]**2)
                offset_mag = np.random.uniform(
                    self.negative_offset_range[0],
                    self.negative_offset_range[1]
                ) * sonar_diag
                offset_angle = np.random.uniform(0, 2 * np.pi)
                negative_offset_sonar = (
                    offset_mag * np.cos(offset_angle),
                    offset_mag * np.sin(offset_angle)
                )

        # Process RGB Search
        if search_sample['rgb_img'] is not None:
            rgb_search_raw = search_sample['rgb_img']  # Raw [H, W, 3] numpy array (after augmentation)
            # CRITICAL FIX (Audit 2025112505 Issue 7): Convert Tensor to numpy if needed
            if isinstance(rgb_search_raw, torch.Tensor):
                rgb_search_raw = rgb_search_raw.cpu().numpy()
                # CRITICAL FIX (Audit 2025112510): Augmentation returns [C, H, W], convert to [H, W, C]
                if rgb_search_raw.ndim == 3 and rgb_search_raw.shape[0] in [1, 3]:
                    rgb_search_raw = rgb_search_raw.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            # CRITICAL FIX (Audit 2025112505 Issue 6): Handle both numpy and tensor types
            rgb_bbox = search_sample['rgb_bbox']
            if isinstance(rgb_bbox, torch.Tensor):
                rgb_bbox_original = rgb_bbox.clone().cpu().numpy()  # XYWH format
            else:
                rgb_bbox_original = rgb_bbox.copy()  # XYWH format

            # CRITICAL FIX (Audit 2025120301 Finding 4): Use CURRENT frame GT center + jitter
            # Previous bug (Audit 2025120102): Used template frame center, causing:
            #   - Training-evaluation distribution mismatch
            #   - Target may be truncated or outside crop
            #   - Model learns wrong localization patterns
            #
            # Correct behavior (OSTrack/MixFormer):
            #   - Training: search center = current frame GT center + jitter
            #   - Evaluation: search center = previous frame prediction
            #   - Template: template frame GT center (unchanged)
            #
            # Now: crop_center_override=None means _crop_search_region uses
            # the search bbox (rgb_bbox_original) center, which IS the current frame GT.
            # The jitter is applied inside _crop_search_region via center_jitter_factor.

            # CRITICAL FIX (Audit 2025120902 Issue 5): Apply negative sample offset
            # For negative samples, offset the crop center so target is outside
            rgb_crop_center_override = None
            if is_negative_sample and negative_offset_rgb is not None:
                # Compute original center and apply offset
                orig_cx = rgb_bbox_original[0] + rgb_bbox_original[2] / 2
                orig_cy = rgb_bbox_original[1] + rgb_bbox_original[3] / 2
                rgb_crop_center_override = (
                    orig_cx + negative_offset_rgb[0],
                    orig_cy + negative_offset_rgb[1]
                )

            rgb_search, rgb_att_mask, rgb_search_meta = self._crop_search_region(
                rgb_search_raw,
                rgb_bbox_original,
                output_size=self.search_size,
                area_factor=self.search_area_factor,
                center_jitter_factor=self.center_jitter_factor if not is_negative_sample else 0.0,
                scale_jitter_factor=self.scale_jitter_factor,
                crop_center_override=rgb_crop_center_override  # Offset for negative samples
            )
        else:
            rgb_search_raw = None
            rgb_bbox_original = None
            rgb_search = None
            # CRITICAL FIX (Audit 2025112603): None image should have all-ZERO mask
            # (all invalid), not all-ones. This prevents model from attending to fake pixels.
            rgb_att_mask = np.zeros((self.search_size, self.search_size), dtype=np.float32)
            rgb_search_meta = {
                'crop_center': (0, 0), 'crop_sz': self.search_size,
                'resize_factor': 1.0, 'crop_x1': 0, 'crop_y1': 0,
                'pad_left': 0, 'pad_top': 0, 'pad_right': 0, 'pad_bottom': 0,
                'original_size': (self.search_size, self.search_size),
                'output_size': self.search_size
            }

        # Process Sonar Search
        if search_sample['sonar_img'] is not None:
            sonar_search_raw = search_sample['sonar_img']  # Raw [H, W] numpy array (after augmentation)
            # CRITICAL FIX (Audit 2025112505 Issue 7): Convert Tensor to numpy if needed
            if isinstance(sonar_search_raw, torch.Tensor):
                sonar_search_raw = sonar_search_raw.cpu().numpy()
                # CRITICAL FIX (Audit 2025112510): Augmentation returns [C, H, W], convert to [H, W, C] or [H, W]
                if sonar_search_raw.ndim == 3 and sonar_search_raw.shape[0] == 1:
                    sonar_search_raw = sonar_search_raw.transpose(1, 2, 0)  # [1, H, W] -> [H, W, 1]
            # CRITICAL FIX (Audit 2025112505 Issue 6): Handle both numpy and tensor types
            sonar_bbox = search_sample['sonar_bbox']
            if isinstance(sonar_bbox, torch.Tensor):
                sonar_bbox_original = sonar_bbox.clone().cpu().numpy()  # XYWH format
            else:
                sonar_bbox_original = sonar_bbox.copy()  # XYWH format

            # CRITICAL FIX (Audit 2025120301 Finding 4): Use CURRENT frame GT center + jitter
            # Same fix as RGB branch - see comments above.
            # crop_center_override=None uses sonar_bbox_original (current frame GT) center.

            # CRITICAL FIX (Audit 2025120902 Issue 5): Apply negative sample offset for sonar
            # This is especially important for sonar objectness calibration
            sonar_crop_center_override = None
            if is_negative_sample and negative_offset_sonar is not None:
                # Compute original center and apply offset
                orig_cx = sonar_bbox_original[0] + sonar_bbox_original[2] / 2
                orig_cy = sonar_bbox_original[1] + sonar_bbox_original[3] / 2
                sonar_crop_center_override = (
                    orig_cx + negative_offset_sonar[0],
                    orig_cy + negative_offset_sonar[1]
                )

            sonar_search, sonar_att_mask, sonar_search_meta = self._crop_search_region(
                sonar_search_raw,
                sonar_bbox_original,
                output_size=self.search_size,
                area_factor=self.search_area_factor,
                center_jitter_factor=self.center_jitter_factor if not is_negative_sample else 0.0,
                scale_jitter_factor=self.scale_jitter_factor,
                crop_center_override=sonar_crop_center_override  # Offset for negative samples
            )
        else:
            sonar_search_raw = None
            sonar_bbox_original = None
            sonar_search = None
            # CRITICAL FIX (Audit 2025112603): None image should have all-ZERO mask
            sonar_att_mask = np.zeros((self.search_size, self.search_size), dtype=np.float32)
            sonar_search_meta = {
                'crop_center': (0, 0), 'crop_sz': self.search_size,
                'resize_factor': 1.0, 'crop_x1': 0, 'crop_y1': 0,
                'pad_left': 0, 'pad_top': 0, 'pad_right': 0, 'pad_bottom': 0,
                'original_size': (self.search_size, self.search_size),
                'output_size': self.search_size
            }

        # Convert to tensors and apply normalization
        # CRITICAL FIX (Audit 2025112401 Finding 4):
        # Apply ImageNet normalization for pretrained ViT compatibility
        # CRITICAL FIX (Audit 2025112402 Finding 3):
        # Use config-loaded normalization parameters instead of hardcoding
        if rgb_search is not None and isinstance(rgb_search, np.ndarray):
            rgb_search = torch.from_numpy(rgb_search).float()
            if rgb_search.ndim == 3 and rgb_search.shape[2] == 3:  # [H, W, 3]
                rgb_search = rgb_search.permute(2, 0, 1)  # → [3, H, W]
            # Convert to [0, 1] then apply normalization from config
            rgb_search = rgb_search / 255.0
            rgb_search = (rgb_search - self.rgb_mean) / self.rgb_std

        if sonar_search is not None and isinstance(sonar_search, np.ndarray):
            sonar_search = torch.from_numpy(sonar_search).float()
            if sonar_search.ndim == 2:  # [H, W]
                sonar_search = sonar_search.unsqueeze(0)  # → [1, H, W]
            # Convert to [0, 1] then apply normalization from config
            sonar_search = sonar_search / 255.0
            sonar_search = (sonar_search - self.sonar_mean) / self.sonar_std

        # 5. Extract and merge metadata with crop metadata
        # CRITICAL FIX (Audit 2025112602): Update to use crop metadata instead of letterbox
        # CRITICAL FIX (Audit 2025112403 Finding 3): Add original_height/width for radial binning
        rgb_metadata = search_sample.get('rgb_metadata', {})
        rgb_metadata.update(rgb_search_meta)
        # Add explicit original_height/width fields for fusion module
        if 'original_size' in rgb_search_meta:
            rgb_metadata['original_height'] = rgb_search_meta['original_size'][0]
            rgb_metadata['original_width'] = rgb_search_meta['original_size'][1]

        sonar_metadata = search_sample.get('sonar_metadata', {})
        sonar_metadata.update(sonar_search_meta)
        # Add explicit original_height/width fields for fusion module
        if 'original_size' in sonar_search_meta:
            sonar_metadata['original_height'] = sonar_search_meta['original_size'][0]
            sonar_metadata['original_width'] = sonar_search_meta['original_size'][1]

        # Pass transform metadata (scale/padding) and rectification status for fusion.
        sonar_metadata['is_rectified'] = getattr(self.base_dataset, 'sonar_is_rectified', True)

        # CRITICAL FIX (Audit 2025112602): Transform bboxes to crop coordinates
        # Use _transform_bbox_to_crop for target-centered cropping coordinate system

        # CRITICAL FIX (Template-Search Migration 2025112105):
        # Convert GT boxes from XYWH pixels to CXCYWH normalized [0, 1]
        # Trainer expects normalized CXCYWH format to match model predictions

        # CRITICAL FIX (Audit 2025112507 Finding 3): Transform template bbox into crop coordinates
        # Helper function for template (uses different metadata structure)
        def transform_template_bbox(bbox_xywh, crop_meta):
            """Transform bbox from original image to template crop coordinates."""
            if crop_meta is None:
                return bbox_xywh

            x, y, w, h = bbox_xywh
            # Step 1: Translate to crop-relative coordinates
            x_crop = x - crop_meta['crop_x1']
            y_crop = y - crop_meta['crop_y1']

            # Step 2: Apply letterbox transform (scale + padding)
            scale = crop_meta['letterbox_scale']
            pad_left = crop_meta['letterbox_pad_left']
            pad_top = crop_meta['letterbox_pad_top']

            x_letterbox = x_crop * scale + pad_left
            y_letterbox = y_crop * scale + pad_top
            w_letterbox = w * scale
            h_letterbox = h * scale

            return np.array([x_letterbox, y_letterbox, w_letterbox, h_letterbox], dtype=np.float32)

        # Template bbox (in template crop) - Use RGB if available, else Sonar
        if template_sample['rgb_bbox'] is not None and rgb_crop_meta is not None:
            template_bbox_crop = transform_template_bbox(
                template_sample['rgb_bbox'],
                rgb_crop_meta
            )
            template_bbox_cxcywh = convert_bbox_format(
                template_bbox_crop.reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            template_bbox_normalized = normalize_bbox(
                template_bbox_cxcywh,
                image_size=(self.template_size, self.template_size),
                format='cxcywh'
            )[0]
        elif template_sample['sonar_bbox'] is not None and sonar_crop_meta is not None:
            template_bbox_crop = transform_template_bbox(
                template_sample['sonar_bbox'],
                sonar_crop_meta
            )
            template_bbox_cxcywh = convert_bbox_format(
                template_bbox_crop.reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            template_bbox_normalized = normalize_bbox(
                template_bbox_cxcywh,
                image_size=(self.template_size, self.template_size),
                format='cxcywh'
            )[0]
        else:
            template_bbox_normalized = np.zeros(4, dtype=np.float32)

        # CRITICAL FIX (Audit 2025112602 + Audit 2025112312):
        # Process RGB and Sonar search bboxes using TARGET-CENTERED crop transformation
        # This is the KEY change: bbox is now relative to the target-centered crop

        # RGB search bbox processing
        if rgb_bbox_original is not None:
            # CRITICAL FIX (Audit 2025112602): Use _transform_bbox_to_crop
            rgb_bbox_transformed = self._transform_bbox_to_crop(rgb_bbox_original, rgb_search_meta)

            # RGB search bbox (in search_size x search_size crop, target-centered)
            search_bbox_rgb_cxcywh = convert_bbox_format(
                rgb_bbox_transformed.reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            search_bbox_rgb_normalized = normalize_bbox(
                search_bbox_rgb_cxcywh,
                image_size=(self.search_size, self.search_size),
                format='cxcywh'
            )[0]
            # Keep pixel-space boxes for evaluation (in crop coordinates)
            search_bbox_rgb_pixels = rgb_bbox_transformed
        else:
            search_bbox_rgb_normalized = np.zeros(4, dtype=np.float32)
            search_bbox_rgb_pixels = np.zeros(4, dtype=np.float32)

        # Sonar search bbox processing
        if sonar_bbox_original is not None:
            # CRITICAL FIX (Audit 2025112602): Use _transform_bbox_to_crop
            sonar_bbox_transformed = self._transform_bbox_to_crop(sonar_bbox_original, sonar_search_meta)

            # Sonar search bbox (in search_size x search_size crop, target-centered)
            search_bbox_sonar_cxcywh = convert_bbox_format(
                sonar_bbox_transformed.reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            search_bbox_sonar_normalized = normalize_bbox(
                search_bbox_sonar_cxcywh,
                image_size=(self.search_size, self.search_size),
                format='cxcywh'
            )[0]
            # Keep pixel-space boxes for evaluation (in crop coordinates)
            search_bbox_sonar_pixels = sonar_bbox_transformed
        else:
            search_bbox_sonar_normalized = np.zeros(4, dtype=np.float32)
            search_bbox_sonar_pixels = np.zeros(4, dtype=np.float32)

        # Extract has_object flags from base_dataset
        has_object_rgb = search_sample.get('has_object_rgb', None)
        has_object_sonar = search_sample.get('has_object_sonar', None)
        has_object = search_sample.get('has_object', None)

        # CRITICAL FIX (Audit 2025120902 Issue 5): Override has_object for negative samples
        # For negative samples, target is NOT in the search crop (by design)
        # This is essential for training objectness head to output low values
        if is_negative_sample:
            has_object_rgb = False
            has_object_sonar = False
            has_object = False
            # Also set GT boxes to zero for negative samples
            # (target not in crop = no valid GT)
            search_bbox_rgb_normalized = np.zeros(4, dtype=np.float32)
            search_bbox_sonar_normalized = np.zeros(4, dtype=np.float32)
            search_bbox_rgb_pixels = np.zeros(4, dtype=np.float32)
            search_bbox_sonar_pixels = np.zeros(4, dtype=np.float32)

        # Use base dataset's is_valid flag (target visibility)
        # base_dataset already computes is_valid correctly based on target visibility:
        # - zero-modality: target not visible in either sensor (is_valid=False)
        # - single-RGB/sonar: target visible in one sensor (is_valid depends on return_occluded)
        # - dual-modality: target visible in both sensors (is_valid=True)
        # Do not recompute here to preserve base_dataset's visibility logic
        is_valid = search_sample.get('is_valid', True)
        is_valid_rgb = search_sample['is_rgb_valid']
        is_valid_sonar = search_sample['is_sonar_valid']

        # CRITICAL FIX (Audit 2025121001 Issue 6): Negative samples must set is_valid flags to False
        # This ensures ConditionalDCMA routes correctly (bypass/zero-modality branch) and
        # fusion/consistency losses are zeroed out for negative samples.
        # The has_object flags are already set to False above (line 722-728).
        if is_negative_sample:
            is_valid = False
            is_valid_rgb = False
            is_valid_sonar = False

        return {
            # Init templates (anchor, from first frame)
            'rgb_template': rgb_template,              # Backward compatible: init template
            'rgb_template_init': rgb_template,         # Explicit init template
            'sonar_template': sonar_template,          # Backward compatible: init template
            'sonar_template_init': sonar_template,     # Explicit init template
            # Dynamic templates (from recent high-quality frame)
            'rgb_template_dyn': rgb_template_dyn,      # Dynamic template for appearance adaptation
            'sonar_template_dyn': sonar_template_dyn,  # Dynamic template for appearance adaptation
            # Search regions
            'rgb_search': rgb_search,
            'sonar_search': sonar_search,
            # Bounding boxes
            'template_bbox': template_bbox_normalized,  # Normalized CXCYWH [0,1] (RGB)
            'search_bbox_rgb': search_bbox_rgb_normalized,      # RGB GT in target-centered crop
            'search_bbox_sonar': search_bbox_sonar_normalized,  # Sonar GT in target-centered crop
            # CRITICAL FIX (Audit 2025112401 Finding 9): Add pixel boxes for evaluation
            'search_bbox_rgb_pixels': search_bbox_rgb_pixels,    # RGB GT in pixels (crop space)
            'search_bbox_sonar_pixels': search_bbox_sonar_pixels,  # Sonar GT in pixels (crop space)
            # CRITICAL FIX (Audit 2025112602): Add attention masks for padded regions
            'rgb_att_mask': rgb_att_mask,               # [H, W] binary mask (1=valid, 0=padded)
            'sonar_att_mask': sonar_att_mask,           # [H, W] binary mask (1=valid, 0=padded)
            # Template attention masks (init)
            'rgb_template_att_mask': rgb_template_att_mask,     # [H, W] init template mask
            'sonar_template_att_mask': sonar_template_att_mask, # [H, W] init template mask
            'rgb_template_init_att_mask': rgb_template_att_mask,     # Explicit init mask
            'sonar_template_init_att_mask': sonar_template_att_mask, # Explicit init mask
            # Template attention masks (dynamic)
            'rgb_template_dyn_att_mask': rgb_template_dyn_att_mask,     # [H, W] dyn template mask
            'sonar_template_dyn_att_mask': sonar_template_dyn_att_mask, # [H, W] dyn template mask
            # Validity flags
            # CRITICAL FIX (Audit 2025121001 Issue 6): Use pre-computed validity flags
            # that are set to False for negative samples
            'is_valid': is_valid,                       # General validity flag
            'is_valid_rgb': is_valid_rgb,               # RGB validity (False for negative samples)
            'is_valid_sonar': is_valid_sonar,           # Sonar validity (False for negative samples)
            'has_object_rgb': has_object_rgb,           # RGB objectness flag
            'has_object_sonar': has_object_sonar,       # Sonar objectness flag
            'has_object': has_object,                   # Combined objectness flag
            # Frame info
            'gap': index - template_idx,
            'gap_init': index - template_init_idx,      # Gap from init template
            'gap_dyn': index - template_dyn_idx,        # Gap from dyn template
            'sequence_name': search_sample.get('sequence_name', 'default'),
            'template_frame_id': template_idx + 1,      # 1-based (backward compat)
            'template_init_frame_id': template_init_idx + 1,  # 1-based
            'template_dyn_frame_id': template_dyn_idx + 1,    # 1-based
            'search_frame_id': index + 1,
            'rgb_metadata': rgb_metadata,
            'sonar_metadata': sonar_metadata,
            # Dual template mode flag
            'use_dual_template': self.use_dual_template,
        }

    def _sample_template_index(self, search_idx: int) -> int:
        """
        Sample template frame index based on policy.

        CRITICAL FIX (Audit 2025112401 Finding 6):
        Now checks validity flags to avoid selecting occluded templates (0,0,0,0).

        Policies:
            - 'first_frame': Always use first frame of sequence
            - 'recent': Sample recent frame within gap_range
            - 'keyframe': Use keyframe from sequence (future work)
            - 'mixformer': MixFormer-style quality-driven updates
                * Training: Behaves like 'recent' (random gap sampling) to learn from diverse template gaps
                * Inference: Quality-driven updates (implemented in OnlineTracker)
                * Note: quality_threshold params stored but not used during training (no prior predictions)

        Args:
            search_idx: Search frame index

        Returns:
            Template frame index (0-based), guaranteed to be valid if possible
        """
        def is_frame_valid(idx):
            """Check if frame has valid (non-occluded) annotations."""
            try:
                sample = self.base_dataset[idx]
                # A frame is valid if EITHER modality is valid
                # (allows template from one modality when the other is occluded)
                return sample.get('is_rgb_valid', True) or sample.get('is_sonar_valid', True)
            except:
                return False

        def find_nearest_valid_frame(target_idx, start_idx, end_idx):
            """
            Find nearest valid frame to target_idx within [start_idx, end_idx].

            Search strategy:
                1. Check target_idx first
                2. Search backwards from target_idx to start_idx
                3. If no valid frame found, return target_idx (fallback)

            Returns:
                Index of nearest valid frame, or target_idx if none found
            """
            # Check target first
            if is_frame_valid(target_idx):
                return target_idx

            # Search backwards from target to start
            for offset in range(1, target_idx - start_idx + 1):
                candidate_idx = target_idx - offset
                if candidate_idx < start_idx:
                    break
                if is_frame_valid(candidate_idx):
                    return candidate_idx

            # No valid frame found - log warning and use target anyway
            logger.warning(
                f"No valid template frame found in range [{start_idx}, {target_idx}]. "
                f"Using frame {target_idx} which may be occluded (0,0,0,0)."
            )
            return target_idx

        # Determine sequence boundaries
        if self.is_multi_sequence:
            seq_name = self.base_dataset.get_frame_sequence(search_idx)
            seq_info = self.sequence_ranges[seq_name]
            seq_start = seq_info['start']
            seq_end = seq_info['end'] - 1  # Exclusive to inclusive
        else:
            seq_start = 0
            seq_end = len(self.base_dataset) - 1

        if self.update_policy == 'first_frame':
            # CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering
            # When enabled, select best quality template from first N frames
            # instead of always using the first valid frame
            if self.enable_template_quality_filter:
                template_idx, reason = self._select_best_template_frame_for_sequence(
                    seq_start, seq_end + 1  # seq_end is inclusive, method expects exclusive
                )
                return template_idx
            else:
                # Original behavior: use first valid frame of sequence as template
                template_idx = find_nearest_valid_frame(seq_start, seq_start, search_idx)
                return template_idx

        elif self.update_policy in ('recent', 'mixformer'):
            # CRITICAL FIX (Audit 2025112505 Issue 2):
            # Handle first frame case to prevent negative template index
            if search_idx == seq_start:
                # First frame: use itself as template (find nearest valid)
                return find_nearest_valid_frame(seq_start, seq_start, seq_start)

            # Sample from [search_idx - gap_max, search_idx - gap_min]
            # MixFormer training: learn to track from variable temporal distances
            gap = random.randint(self.gap_range[0], self.gap_range[1])
            template_idx = search_idx - gap

            # Clamp to sequence boundaries
            template_idx = max(template_idx, seq_start)
            template_idx = min(template_idx, search_idx - 1)  # Template must be before search

            # Find nearest valid frame
            template_idx = find_nearest_valid_frame(template_idx, seq_start, search_idx - 1)
            return template_idx

        elif self.update_policy == 'keyframe':
            # Future work: Use detected keyframes
            # For now, fallback to first_frame
            import warnings
            warnings.warn(
                "Keyframe policy not implemented yet, using first_frame fallback.",
                UserWarning
            )
            # Recursively call with first_frame policy
            original_policy = self.update_policy
            self.update_policy = 'first_frame'
            result = self._sample_template_index(search_idx)
            self.update_policy = original_policy
            return result

        else:
            raise ValueError(f"Unknown template update policy: {self.update_policy}")

    def _sample_init_template_index(self, search_idx: int) -> int:
        """
        Sample INIT template index (always first frame of sequence).

        For MixFormer dual-template strategy, the init template is always
        from the first frame and serves as an anchor to prevent drift.

        Args:
            search_idx: Search frame index

        Returns:
            Init template frame index (first frame of sequence)
        """
        # Determine sequence boundaries
        if self.is_multi_sequence:
            seq_name = self.base_dataset.get_frame_sequence(search_idx)
            seq_info = self.sequence_ranges[seq_name]
            seq_start = seq_info['start']
        else:
            seq_start = 0

        # Use quality filtering if enabled
        if self.enable_template_quality_filter:
            template_idx, _ = self._select_best_template_frame_for_sequence(
                seq_start, search_idx
            )
            return template_idx

        # Otherwise just use first frame
        return seq_start

    def _sample_dyn_template_index(self, search_idx: int) -> int:
        """
        Sample DYNAMIC template index (recent high-quality frame).

        For MixFormer dual-template strategy, the dynamic template is
        sampled from recent frames with gap sampling to learn diverse
        template-search relationships.

        Args:
            search_idx: Search frame index

        Returns:
            Dynamic template frame index
        """
        # Determine sequence boundaries
        if self.is_multi_sequence:
            seq_name = self.base_dataset.get_frame_sequence(search_idx)
            seq_info = self.sequence_ranges[seq_name]
            seq_start = seq_info['start']
            seq_end = seq_info['end'] - 1
        else:
            seq_start = 0
            seq_end = len(self.base_dataset) - 1

        # Handle first frame case
        if search_idx == seq_start:
            return seq_start

        # Sample gap within range
        min_gap, max_gap = self.gap_range
        available_gap = min(search_idx - seq_start, max_gap)
        min_gap = min(min_gap, available_gap)
        if min_gap > available_gap:
            min_gap = 1 if available_gap >= 1 else 0

        if available_gap > 0:
            gap = random.randint(min_gap, available_gap)
        else:
            gap = 0

        template_idx = search_idx - gap

        # Validate template index is valid
        template_idx = max(seq_start, min(template_idx, search_idx - 1))

        return template_idx

    def _evaluate_template_quality(
        self,
        bbox: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[bool, float]:
        """
        Evaluate template quality based on bbox completeness.

        CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering
        for validation mode. Checks if bbox is edge-clipped or too small.

        Args:
            bbox: Bounding box in XYWH format
            image_size: Image dimensions (width, height)

        Returns:
            Tuple of (is_complete, quality_score)
            - is_complete: True if bbox passes all quality checks
            - quality_score: Combined score [0, 1] for ranking templates
        """
        if bbox is None or np.sum(np.abs(bbox)) == 0:
            return False, 0.0

        x, y, w, h = bbox
        img_w, img_h = image_size

        # Check minimum size
        if w < self.template_min_size or h < self.template_min_size:
            return False, 0.1  # Too small

        # Check edge proximity
        margin = self.template_edge_margin
        left_margin = x
        top_margin = y
        right_margin = img_w - (x + w)
        bottom_margin = img_h - (y + h)

        min_margin = min(left_margin, top_margin, right_margin, bottom_margin)
        is_edge_clipped = min_margin < margin

        # Compute quality score
        # 1. Size score: larger bbox = better (up to a point)
        area = w * h
        max_area = img_w * img_h * 0.5  # Cap at 50% of image
        size_score = min(1.0, area / max_area)

        # 2. Edge score: further from edge = better
        max_margin = min(img_w, img_h) / 4
        edge_score = min(1.0, min_margin / max_margin)

        # 3. Aspect ratio score: prefer reasonable aspect ratios
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        aspect_score = 1.0 / (1.0 + max(0, aspect_ratio - 3))  # Penalize extreme ratios

        # Combined score
        quality_score = 0.4 * size_score + 0.4 * edge_score + 0.2 * aspect_score

        is_complete = not is_edge_clipped and w >= self.template_min_size and h >= self.template_min_size

        return is_complete, quality_score

    def _select_best_template_frame_for_sequence(
        self,
        seq_start: int,
        seq_end: int
    ) -> Tuple[int, str]:
        """
        Select the best template frame from first N frames of a sequence.

        CRITICAL FIX (Audit 2025120402 Finding 5): Template quality filtering
        for validation mode. Scans first N frames and selects the one with
        best combined quality score.

        Args:
            seq_start: Start index of sequence
            seq_end: End index of sequence (exclusive)

        Returns:
            Tuple of (frame_index, selection_reason)
        """
        scan_limit = min(seq_start + self.template_scan_frames, seq_end)

        best_frame_idx = -1
        best_quality = -1.0
        best_reason = "no_valid_frame"

        # Track candidates
        candidates = []

        for frame_idx in range(seq_start, scan_limit):
            try:
                # Get frame data directly from base dataset
                if hasattr(self.base_dataset, '_get_frame_data'):
                    frame_data = self.base_dataset._get_frame_data(frame_idx)
                    rgb_bbox = frame_data.get('rgb_bbox')
                    sonar_bbox = frame_data.get('sonar_bbox')
                    rgb_img = frame_data.get('rgb_img')
                    sonar_img = frame_data.get('sonar_img')
                else:
                    sample = self.base_dataset[frame_idx]
                    rgb_bbox = sample.get('rgb_bbox')
                    sonar_bbox = sample.get('sonar_bbox')
                    rgb_img = sample.get('rgb_img')
                    sonar_img = sample.get('sonar_img')

                # Get image sizes
                if rgb_img is not None:
                    rgb_size = (rgb_img.shape[1], rgb_img.shape[0])  # (W, H)
                else:
                    rgb_size = (1920, 1080)  # Default fallback

                if sonar_img is not None:
                    if sonar_img.ndim == 3:
                        sonar_size = (sonar_img.shape[1], sonar_img.shape[0])
                    else:
                        sonar_size = (sonar_img.shape[1], sonar_img.shape[0])
                else:
                    sonar_size = (682, 376)  # Default fallback

                # Convert bbox to numpy if needed
                if rgb_bbox is not None and isinstance(rgb_bbox, torch.Tensor):
                    rgb_bbox = rgb_bbox.cpu().numpy()
                if sonar_bbox is not None and isinstance(sonar_bbox, torch.Tensor):
                    sonar_bbox = sonar_bbox.cpu().numpy()

                # Evaluate quality
                rgb_complete, rgb_quality = self._evaluate_template_quality(rgb_bbox, rgb_size)
                sonar_complete, sonar_quality = self._evaluate_template_quality(sonar_bbox, sonar_size)

                # Check validity
                rgb_valid = rgb_bbox is not None and np.sum(np.abs(rgb_bbox)) > 0
                sonar_valid = sonar_bbox is not None and np.sum(np.abs(sonar_bbox)) > 0

                if not rgb_valid and not sonar_valid:
                    continue  # Skip occluded frames

                # Determine combined quality and reason
                if rgb_complete and sonar_complete:
                    combined_quality = (rgb_quality + sonar_quality) / 2 + 0.2  # Bonus for dual complete
                    reason = "dual_complete"
                elif rgb_complete:
                    combined_quality = rgb_quality + 0.1
                    reason = "rgb_complete"
                elif sonar_complete:
                    combined_quality = sonar_quality + 0.1
                    reason = "sonar_complete"
                elif rgb_valid or sonar_valid:
                    combined_quality = max(rgb_quality, sonar_quality)
                    reason = "partial_valid"
                else:
                    continue

                candidates.append((frame_idx, combined_quality, reason))

                if combined_quality > best_quality:
                    best_quality = combined_quality
                    best_frame_idx = frame_idx
                    best_reason = reason

            except Exception as e:
                logger.warning(f"Error evaluating frame {frame_idx}: {e}")
                continue

        # Fallback: if no good template found, use first valid frame
        if best_frame_idx < 0:
            for frame_idx in range(seq_start, seq_end):
                try:
                    if hasattr(self.base_dataset, '_get_frame_data'):
                        frame_data = self.base_dataset._get_frame_data(frame_idx)
                        rgb_bbox = frame_data.get('rgb_bbox')
                        sonar_bbox = frame_data.get('sonar_bbox')
                    else:
                        sample = self.base_dataset[frame_idx]
                        rgb_bbox = sample.get('rgb_bbox')
                        sonar_bbox = sample.get('sonar_bbox')

                    if rgb_bbox is not None and isinstance(rgb_bbox, torch.Tensor):
                        rgb_bbox = rgb_bbox.cpu().numpy()
                    if sonar_bbox is not None and isinstance(sonar_bbox, torch.Tensor):
                        sonar_bbox = sonar_bbox.cpu().numpy()

                    rgb_valid = rgb_bbox is not None and np.sum(np.abs(rgb_bbox)) > 0
                    sonar_valid = sonar_bbox is not None and np.sum(np.abs(sonar_bbox)) > 0

                    if rgb_valid or sonar_valid:
                        return frame_idx, "first_valid_fallback"
                except Exception:
                    continue

            return seq_start, "fallback_to_start"

        return best_frame_idx, best_reason

    def _denormalize_image_if_needed(
        self,
        image: Union[torch.Tensor, np.ndarray],
        modality: str = 'rgb'
    ) -> np.ndarray:
        """
        Detect and denormalize image if it has been ImageNet/Sonar normalized.

        CRITICAL FIX (Audit 2025120301 Finding 1, 2, 8):
        Augmentation pipeline may return normalized tensors (range ~[-2, 2] for RGB,
        ~[-1, 1] for Sonar). Cropping functions expect uint8 [0, 255].
        This function detects normalized data and reverses the normalization.

        Detection logic:
        - Float dtype with values < 0 or values in [-3, 3] range: ImageNet normalized
        - Float dtype with max <= 1.0 and min >= 0: [0,1] normalized
        - uint8: already in [0, 255], no change needed

        Args:
            image: Input image (tensor or ndarray), may be normalized
            modality: 'rgb' or 'sonar' to select correct denormalization params

        Returns:
            numpy array in uint8 [0, 255] range, shape [H, W, C] or [H, W]
        """
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # [C, H, W]
                image_np = image.permute(1, 2, 0).cpu().numpy()  # -> [H, W, C]
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image.copy() if isinstance(image, np.ndarray) else np.array(image)

        # Already uint8 [0, 255]: no processing needed
        if image_np.dtype == np.uint8:
            return image_np

        # Detect normalization type based on value range
        img_min, img_max = image_np.min(), image_np.max()

        # Case 1: ImageNet/Sonar normalized (range approximately [-3, 3])
        # Typical ImageNet range: [-2.1, 2.6] after (x - mean) / std
        if img_min < -0.5 or (img_max > 1.5 and img_max < 10):
            # Denormalize using stored parameters
            if modality == 'rgb':
                mean = self.rgb_mean.squeeze().numpy()  # [3]
                std = self.rgb_std.squeeze().numpy()    # [3]
            else:
                mean = self.sonar_mean.squeeze().numpy()  # [1]
                std = self.sonar_std.squeeze().numpy()    # [1]

            # Reverse normalization: x_original = x_normalized * std + mean
            # Then scale to [0, 255]
            image_np = image_np * std + mean  # Back to [0, 1]
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            return image_np

        # Case 2: [0, 1] normalized (no mean/std applied)
        if img_max <= 1.0 and img_min >= 0:
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            return image_np

        # Case 3: Already in ~[0, 255] range (float)
        if img_max > 1.0 and img_max <= 255:
            return np.clip(image_np, 0, 255).astype(np.uint8)

        # Fallback: clamp and convert
        logger.warning(
            f"Unexpected image range [{img_min:.2f}, {img_max:.2f}], "
            f"clamping to [0, 255]"
        )
        return np.clip(image_np, 0, 255).astype(np.uint8)

    def _crop_template(
        self,
        image: torch.Tensor,
        bbox: np.ndarray,
        crop_size: int
    ) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        """
        Crop template region around target bbox using OSTrack methodology.

        CRITICAL FIX (Audit 2025112602 BUG 1-3):
        - Uses sqrt(w*h) * template_area_factor (same formula as search)
        - Adds boundary padding instead of clamping
        - Returns attention mask for padded regions

        CRITICAL FIX (Audit 2025120301 Finding 1, 2, 8):
        - Detects and handles already-normalized input from augmentation pipeline
        - Prevents double normalization by using _denormalize_image_if_needed()

        CRITICAL FIX (Audit 2025120909): Generate template attention mask
        - Templates can have padding when target is near image boundary
        - Mask marks valid (non-padded) regions for ViT attention masking

        Args:
            image: [3, H, W] or [1, H, W] tensor (from base_dataset)
            bbox: [x, y, w, h] in XYWH pixel format (numpy array)
            crop_size: Target crop size (e.g., 256)

        Returns:
            Tuple of (crop_tensor, attention_mask, crop_metadata):
                - crop_tensor: Cropped template [C, crop_size, crop_size]
                - attention_mask: [crop_size, crop_size] binary mask (1=valid, 0=padded)
                - crop_metadata: Dict with transformation parameters
        """
        # CRITICAL FIX (Audit 2025120301): Detect modality from channel count
        if isinstance(image, torch.Tensor):
            n_channels = image.shape[0] if image.dim() == 3 else 1
        else:
            n_channels = image.shape[-1] if image.ndim == 3 else 1
        modality = 'rgb' if n_channels == 3 else 'sonar'

        # CRITICAL FIX (Audit 2025120301 Finding 1, 2, 8):
        # Use unified denormalization to handle all input types
        image_np = self._denormalize_image_if_needed(image, modality)

        H, W = image_np.shape[:2]
        x, y, w, h = bbox

        # Handle degenerate bbox (0,0,0,0 or very small)
        if w <= 0 or h <= 0:
            w = max(w, 10)
            h = max(h, 10)

        # Compute bbox center
        cx = x + w / 2
        cy = y + h / 2

        # CRITICAL FIX (Audit 2025112602 BUG 1):
        # Use OSTrack formula: sqrt(w*h) * template_area_factor
        # NOT: max(w,h) * context_factor (inconsistent with search)
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * self.template_area_factor))
        crop_sz = max(crop_sz, 1)

        # Compute crop boundaries (may extend beyond image)
        x1 = int(round(cx - crop_sz / 2))
        y1 = int(round(cy - crop_sz / 2))
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        # CRITICAL FIX (Audit 2025112602 BUG 2):
        # Compute padding amounts for boundary handling (NOT clamping!)
        x1_pad = max(0, -x1)
        x2_pad = max(0, x2 - W)
        y1_pad = max(0, -y1)
        y2_pad = max(0, y2 - H)

        # Clamp to valid image region
        x1_valid = max(0, x1)
        x2_valid = min(W, x2)
        y1_valid = max(0, y1)
        y2_valid = min(H, y2)

        # Extract valid region and pad
        if x2_valid <= x1_valid or y2_valid <= y1_valid:
            # Completely out of bounds
            if image_np.ndim == 2:
                crop_padded = np.full((crop_sz, crop_sz), 127, dtype=np.uint8)
            else:
                avg_color = image_np.mean(axis=(0, 1)).astype(np.uint8)
                crop_padded = np.full((crop_sz, crop_sz, image_np.shape[2]), avg_color, dtype=np.uint8)
        else:
            crop = image_np[y1_valid:y2_valid, x1_valid:x2_valid]

            # Compute padding color (image mean)
            if image_np.ndim == 2:
                pad_value = int(image_np.mean())
            else:
                pad_value = tuple(int(c) for c in image_np.mean(axis=(0, 1)))

            # Pad to full crop size
            crop_padded = cv2.copyMakeBorder(
                crop,
                y1_pad, y2_pad, x1_pad, x2_pad,
                cv2.BORDER_CONSTANT,
                value=pad_value
            )

        # Resize to crop_size
        resize_factor = crop_size / crop_sz
        crop_resized = cv2.resize(
            crop_padded,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Store metadata for bbox transformation (consistent with search)
        crop_metadata = {
            'crop_x1': x1,
            'crop_y1': y1,
            'crop_sz': crop_sz,
            'resize_factor': resize_factor,
            'pad_left': x1_pad,
            'pad_top': y1_pad,
            'pad_right': x2_pad,
            'pad_bottom': y2_pad,
            'original_size': (H, W),
            'output_size': crop_size,
            # Legacy keys for backward compatibility
            'crop_width': crop_sz,
            'crop_height': crop_sz,
            'letterbox_scale': resize_factor,
            'letterbox_pad_left': int(x1_pad * resize_factor),
            'letterbox_pad_top': int(y1_pad * resize_factor),
        }

        # CRITICAL FIX (Audit 2025120909): Generate attention mask for template
        # Mask marks valid (non-padded) regions: 1=valid, 0=padded
        # This is used by ViT to exclude padded tokens from attention
        att_mask_full = np.ones((crop_sz, crop_sz), dtype=np.float32)
        if y1_pad > 0:
            att_mask_full[:y1_pad, :] = 0
        if y2_pad > 0:
            att_mask_full[-y2_pad:, :] = 0
        if x1_pad > 0:
            att_mask_full[:, :x1_pad] = 0
        if x2_pad > 0:
            att_mask_full[:, -x2_pad:] = 0

        # Resize mask to match crop_size
        att_mask_resized = cv2.resize(
            att_mask_full,
            (crop_size, crop_size),
            interpolation=cv2.INTER_NEAREST  # Use nearest to preserve binary values
        )

        # Convert to tensor and apply normalization
        if crop_resized.ndim == 2:  # Grayscale (sonar)
            crop_tensor = torch.from_numpy(crop_resized).unsqueeze(0).float() / 255.0
            crop_tensor = (crop_tensor - self.sonar_mean) / self.sonar_std
        else:  # RGB
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = (crop_tensor - self.rgb_mean) / self.rgb_std

        return crop_tensor, att_mask_resized, crop_metadata

    def _letterbox(self, img: np.ndarray, target_size: int) -> Tuple[np.ndarray, Dict]:
        """
        Letterbox preprocessing: resize to fit within target_size and pad to square.

        CRITICAL FIX (Audit 2025112401 Finding 1):
        Now returns transformation metadata for bbox coordinate transformation.

        CRITICAL: Preserves aspect ratio to avoid distortion.
        Following YOLOv5/OSTrack letterbox implementation.

        Args:
            img: Input image [H, W] or [H, W, C]
            target_size: Target square size (e.g., 256)

        Returns:
            Tuple of (letterboxed_image, metadata):
                - letterboxed_image: [target_size, target_size, C] or [target_size, target_size]
                - metadata: Dict with keys:
                    - 'scale': float, resize scale factor
                    - 'pad_left': int, left padding in pixels
                    - 'pad_top': int, top padding in pixels
                    - 'original_size': Tuple[int, int], (h, w) before transform
                    - 'new_size': Tuple[int, int], (new_h, new_w) after resize
        """
        h, w = img.shape[:2]

        # CRITICAL FIX: Track if input is single-channel (H, W, 1)
        # cv2.resize removes the last dimension for (H, W, 1) inputs
        is_single_channel = img.ndim == 3 and img.shape[2] == 1

        # Compute scale factor to fit within target_size
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with preserved aspect ratio
        # CRITICAL FIX: Squeeze (H, W, 1) to (H, W) before resize to avoid issues
        if is_single_channel:
            img_to_resize = img.squeeze(-1)
        else:
            img_to_resize = img

        img_resized = cv2.resize(img_to_resize, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Pad with gray (114 for RGB, 127 for grayscale/single-channel)
        if img_resized.ndim == 2:  # Grayscale or squeezed single-channel
            pad_value = 127
        else:  # RGB
            pad_value = (114, 114, 114)

        img_letterboxed = cv2.copyMakeBorder(
            img_resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        # CRITICAL FIX (Audit 2025112401 Finding 1):
        # Return transformation metadata for bbox coordinate transformation
        metadata = {
            'scale': scale,
            'pad_left': left,
            'pad_top': top,
            'original_size': (h, w),
            'new_size': (new_h, new_w)
        }

        return img_letterboxed, metadata

    def _crop_search_region(
        self,
        image: Union[np.ndarray, torch.Tensor],
        bbox: np.ndarray,
        output_size: int,
        area_factor: float,
        center_jitter_factor: float = 0.0,
        scale_jitter_factor: float = 0.0,
        crop_center_override: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Crop search region centered on target bbox following OSTrack methodology.

        CRITICAL FIX (Audit 2025112602): This implements the standard SOT search
        region cropping that ensures target is always near the center of the crop.
        This is essential for the model to learn relative offset prediction.

        CRITICAL FIX (Audit 2025120102): Added crop_center_override parameter to
        support multi-modal training where RGB and Sonar must use the same crop
        center (from Template bbox) to maintain spatial correspondence.

        CRITICAL FIX (Audit 2025120301 Finding 1, 2, 8):
        - Detects and handles already-normalized input from augmentation pipeline
        - Prevents double normalization by using _denormalize_image_if_needed()

        Algorithm:
        1. Compute crop size: sqrt(w*h) * area_factor
        2. Use crop_center_override if provided, otherwise compute from bbox
        3. Compute crop boundaries (may extend beyond image)
        4. Pad out-of-bounds regions with image mean
        5. Resize to output_size
        6. Create attention mask (1=valid, 0=padded)
        7. Transform bbox to crop coordinates

        Args:
            image: Input image [H, W, C] or [H, W] numpy array, or tensor
            bbox: Target bbox [x, y, w, h] in XYWH pixel format
            output_size: Output crop size (e.g., 256)
            area_factor: Crop size multiplier (e.g., 4.0 for search, 2.0 for template)
            center_jitter_factor: Random center jitter (0.0 = no jitter) - DEPRECATED for multi-modal
            scale_jitter_factor: Random scale jitter (0.0 = no jitter) - DEPRECATED for multi-modal
            crop_center_override: (cx, cy) in pixel coordinates to use instead of bbox center.
                                  Used for multi-modal training to ensure RGB/Sonar alignment.

        Returns:
            Tuple of (crop, attention_mask, metadata):
                - crop: [output_size, output_size, C] cropped and resized image
                - attention_mask: [output_size, output_size] binary mask (1=valid, 0=padded)
                - metadata: Dict with transformation parameters for bbox conversion
        """
        if image is None:
            # Return black image and all-ZERO mask for None images
            # CRITICAL FIX (Audit 2025112603): None image should have all-ZERO mask
            # (all invalid), not all-ones. This prevents model from attending to fake pixels.
            crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
            att_mask = np.zeros((output_size, output_size), dtype=np.float32)
            metadata = {
                'crop_center': (0, 0),
                'crop_sz': output_size,
                'resize_factor': 1.0,
                'crop_x1': 0,
                'crop_y1': 0,
                'pad_left': 0,
                'pad_top': 0,
                'pad_right': 0,
                'pad_bottom': 0,
                'original_size': (output_size, output_size),
                'output_size': output_size,
            }
            return crop, att_mask, metadata

        # CRITICAL FIX (Audit 2025120301 Finding 1, 2, 8):
        # Detect modality and denormalize if input was already normalized
        if isinstance(image, torch.Tensor):
            n_channels = image.shape[0] if image.dim() == 3 else 1
        elif isinstance(image, np.ndarray):
            n_channels = image.shape[-1] if image.ndim == 3 else 1
        else:
            n_channels = 3  # Default to RGB
        modality = 'rgb' if n_channels == 3 else 'sonar'

        # Denormalize to uint8 [0, 255] for cropping operations
        image = self._denormalize_image_if_needed(image, modality)

        H, W = image.shape[:2]
        x, y, w, h = bbox

        # Handle degenerate bbox (0,0,0,0 or very small)
        if w <= 0 or h <= 0:
            w = max(w, 10)
            h = max(h, 10)

        # Compute crop size following OSTrack: sqrt(w*h) * area_factor
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * area_factor))
        crop_sz = max(crop_sz, 1)  # Ensure positive

        # CRITICAL FIX (Audit 2025120102): Use crop_center_override if provided
        # This enables Template-centered cropping for multi-modal training
        if crop_center_override is not None:
            cx, cy = crop_center_override
        else:
            # Default: compute center from Search bbox
            cx = x + w / 2
            cy = y + h / 2

            # Apply jitter for training (center and scale) - only when not using override
            # CRITICAL FIX (Audit 2025112704): Align EXACTLY with OSTrack jitter implementation
            # OSTrack: jittered_center = center + max_offset * (rand - 0.5)
            # where max_offset = sqrt(w*h) * center_jitter_factor
            # NOTE: OSTrack uses (rand - 0.5) with range [-0.5, 0.5], NOT (rand - 0.5) * 2!
            # With CENTER_JITTER=3: max_offset_range = [-1.5, 1.5] * target_sz
            # Since crop_half = 2 * target_sz, target center stays within crop
            # Reference: https://github.com/botaoye/OSTrack/blob/main/lib/train/data/processing.py
            if center_jitter_factor > 0:
                # Jitter center proportional to target size (OSTrack style)
                # max_offset = target_sz * center_jitter_factor
                # offset = max_offset * (rand - 0.5), range = [-0.5, 0.5] * max_offset
                max_offset = target_sz * center_jitter_factor
                cx += max_offset * (random.random() - 0.5)
                cy += max_offset * (random.random() - 0.5)

        if scale_jitter_factor > 0:
            # Jitter scale using exponential (OSTrack style: exp(randn * factor))
            # OSTrack uses torch.randn which has std=1, so range is roughly [-3, 3]
            # We use (rand - 0.5) * 2 with range [-1, 1] as approximation
            scale_jitter = math.exp((random.random() - 0.5) * 2 * scale_jitter_factor)
            crop_sz = int(crop_sz * scale_jitter)
            crop_sz = max(crop_sz, 1)

        # Compute crop boundaries (may be negative or exceed image bounds)
        x1 = int(round(cx - crop_sz / 2))
        y1 = int(round(cy - crop_sz / 2))
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        # Compute padding amounts for out-of-bounds regions
        x1_pad = max(0, -x1)
        x2_pad = max(0, x2 - W)
        y1_pad = max(0, -y1)
        y2_pad = max(0, y2 - H)

        # Clamp to valid image region
        x1_valid = max(0, x1)
        x2_valid = min(W, x2)
        y1_valid = max(0, y1)
        y2_valid = min(H, y2)

        # Extract valid region from image
        if x2_valid <= x1_valid or y2_valid <= y1_valid:
            # Completely out of bounds - return mean-filled image
            if image.ndim == 2:
                crop_padded = np.full((crop_sz, crop_sz), 127, dtype=np.uint8)
            else:
                avg_color = image.mean(axis=(0, 1)).astype(np.uint8)
                crop_padded = np.full((crop_sz, crop_sz, image.shape[2]), avg_color, dtype=np.uint8)
        else:
            crop = image[y1_valid:y2_valid, x1_valid:x2_valid]

            # Compute padding color (image mean)
            if image.ndim == 2:
                pad_value = int(image.mean())
            else:
                pad_value = tuple(int(c) for c in image.mean(axis=(0, 1)))

            # Pad to full crop size using cv2.copyMakeBorder
            crop_padded = cv2.copyMakeBorder(
                crop,
                y1_pad, y2_pad, x1_pad, x2_pad,
                cv2.BORDER_CONSTANT,
                value=pad_value
            )

        # Create attention mask before resize (1=valid, 0=padded)
        att_mask_full = np.ones((crop_sz, crop_sz), dtype=np.float32)
        if y1_pad > 0:
            att_mask_full[:y1_pad, :] = 0
        if y2_pad > 0:
            att_mask_full[-y2_pad:, :] = 0
        if x1_pad > 0:
            att_mask_full[:, :x1_pad] = 0
        if x2_pad > 0:
            att_mask_full[:, -x2_pad:] = 0

        # Resize crop to output_size
        resize_factor = output_size / crop_sz
        crop_resized = cv2.resize(
            crop_padded,
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Resize attention mask (use nearest to preserve binary values)
        att_mask_resized = cv2.resize(
            att_mask_full,
            (output_size, output_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Store metadata for bbox transformation
        metadata = {
            'crop_center': (cx, cy),       # Center of crop in original image
            'crop_sz': crop_sz,            # Crop size before resize
            'resize_factor': resize_factor, # output_size / crop_sz
            'crop_x1': x1,                 # Crop x1 (may be negative)
            'crop_y1': y1,                 # Crop y1 (may be negative)
            'pad_left': x1_pad,
            'pad_top': y1_pad,
            'pad_right': x2_pad,
            'pad_bottom': y2_pad,
            'original_size': (H, W),
            'output_size': output_size,
        }

        return crop_resized, att_mask_resized, metadata

    def _transform_bbox_to_crop(
        self,
        bbox_xywh: np.ndarray,
        crop_metadata: Dict
    ) -> np.ndarray:
        """
        Transform bbox from original image coordinates to crop coordinates.

        Following OSTrack's transform_image_to_crop methodology:
        1. Translate bbox relative to crop origin
        2. Account for padding
        3. Scale by resize factor

        Args:
            bbox_xywh: [x, y, w, h] in original image pixel coordinates
            crop_metadata: Dict from _crop_search_region with transformation params

        Returns:
            bbox_xywh_crop: [x, y, w, h] in crop coordinate space (output_size)
        """
        x, y, w, h = bbox_xywh

        # Get crop parameters
        crop_x1 = crop_metadata['crop_x1']
        crop_y1 = crop_metadata['crop_y1']
        resize_factor = crop_metadata['resize_factor']
        pad_left = crop_metadata['pad_left']
        pad_top = crop_metadata['pad_top']

        # Step 1: Translate to crop-relative coordinates
        # Account for the crop origin (which may be negative for padding)
        x_rel = x - crop_x1
        y_rel = y - crop_y1

        # Step 2: Scale to output size
        x_out = x_rel * resize_factor
        y_out = y_rel * resize_factor
        w_out = w * resize_factor
        h_out = h * resize_factor

        return np.array([x_out, y_out, w_out, h_out], dtype=np.float32)

    def _get_jittered_bbox(
        self,
        bbox: np.ndarray,
        center_jitter: float,
        scale_jitter: float
    ) -> np.ndarray:
        """
        Apply random jitter to bbox for training diversity.

        Following OSTrack's jitter strategy to prevent center bias.

        Args:
            bbox: [x, y, w, h] original bbox
            center_jitter: Max center offset as fraction of bbox size
            scale_jitter: Max scale change as fraction (e.g., 0.25 = +/-25%)

        Returns:
            jittered_bbox: [x, y, w, h] with jitter applied
        """
        x, y, w, h = bbox

        # Center jitter
        cx = x + w / 2
        cy = y + h / 2
        if center_jitter > 0:
            cx += (random.random() - 0.5) * 2 * center_jitter * w
            cy += (random.random() - 0.5) * 2 * center_jitter * h

        # Scale jitter
        if scale_jitter > 0:
            scale = 1.0 + (random.random() - 0.5) * 2 * scale_jitter
            w = w * scale
            h = h * scale

        # Convert back to XYWH
        x_new = cx - w / 2
        y_new = cy - h / 2

        return np.array([x_new, y_new, w, h], dtype=np.float32)


def template_search_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for template-search pairs.

    CRITICAL FIX (Audit 2025112602): Updated for target-centered cropping.
    - search_size is now 256 (OSTrack standard) instead of 512
    - Added attention masks for padded regions

    Input batch[i]:
        {
            'rgb_template': [3, 128, 128],  # OSTrack-256 standard
            'rgb_search': [3, 256, 256],  # Target-centered crop
            'sonar_template': [1, 128, 128],  # OSTrack-256 standard
            'sonar_search': [1, 256, 256],  # Target-centered crop
            'template_bbox': [4] CXCYWH normalized [0,1],
            'search_bbox_rgb': [4] CXCYWH normalized [0,1] - RGB GT in crop,
            'search_bbox_sonar': [4] CXCYWH normalized [0,1] - Sonar GT in crop,
            'search_bbox_rgb_pixels': [4] XYWH pixels (crop space),
            'search_bbox_sonar_pixels': [4] XYWH pixels (crop space),
            'rgb_att_mask': [H, W] attention mask (1=valid, 0=padded),
            'sonar_att_mask': [H, W] attention mask (1=valid, 0=padded),
            'is_valid': bool,
            'is_valid_rgb': bool,
            'is_valid_sonar': bool,
            'has_object_rgb': bool,
            'has_object_sonar': bool,
            'has_object': bool,
            'gap': int,
            'sequence_name': str,
            'template_frame_id': int,
            'search_frame_id': int,
            'rgb_metadata': dict,
            'sonar_metadata': dict,
        }

    Output:
        {
            'rgb_template': [B, 3, 128, 128],  # OSTrack-256 standard
            'rgb_search': [B, 3, 256, 256],
            'sonar_template': [B, 1, 128, 128],  # OSTrack-256 standard
            'sonar_search': [B, 1, 256, 256],
            'template_bbox': [B, 4],
            'search_bbox_rgb': [B, 4],
            'search_bbox_sonar': [B, 4],
            'search_bbox_rgb_pixels': [B, 4],
            'search_bbox_sonar_pixels': [B, 4],
            'rgb_att_mask': [B, H, W],
            'sonar_att_mask': [B, H, W],
            'is_valid_rgb': [B],
            'is_valid_sonar': [B],
            'has_object_rgb': [B],
            'has_object_sonar': [B],
            'has_object': [B],
            'gap': [B],
            'sequence_name': List[str],
            'template_frame_id': [B],
            'search_frame_id': [B],
            'rgb_metadata': List[Dict],
            'sonar_metadata': List[Dict],
        }
    """
    # CRITICAL FIX (Audit 2025112312 + Audit 2025112602):
    # Handle separate RGB and Sonar GT boxes + has_object flags + attention masks

    return {
        'rgb_template': torch.stack([b['rgb_template'] for b in batch]),
        'rgb_search': torch.stack([b['rgb_search'] for b in batch]),
        'sonar_template': torch.stack([b['sonar_template'] for b in batch]),
        'sonar_search': torch.stack([b['sonar_search'] for b in batch]),
        'template_bbox': torch.stack([
            torch.from_numpy(b['template_bbox']) if isinstance(b['template_bbox'], np.ndarray)
            else b['template_bbox']
            for b in batch
        ]),
        # Separate RGB and Sonar GT boxes (normalized)
        'search_bbox_rgb': torch.stack([
            torch.from_numpy(b['search_bbox_rgb']) if isinstance(b['search_bbox_rgb'], np.ndarray)
            else b['search_bbox_rgb']
            for b in batch
        ]),
        'search_bbox_sonar': torch.stack([
            torch.from_numpy(b['search_bbox_sonar']) if isinstance(b['search_bbox_sonar'], np.ndarray)
            else b['search_bbox_sonar']
            for b in batch
        ]),
        # CRITICAL FIX (Audit 2025112401 Finding 9):
        # Pass pixel-space boxes for evaluation
        'search_bbox_rgb_pixels': torch.stack([
            torch.from_numpy(b['search_bbox_rgb_pixels']) if isinstance(b['search_bbox_rgb_pixels'], np.ndarray)
            else b['search_bbox_rgb_pixels']
            for b in batch
        ]),
        'search_bbox_sonar_pixels': torch.stack([
            torch.from_numpy(b['search_bbox_sonar_pixels']) if isinstance(b['search_bbox_sonar_pixels'], np.ndarray)
            else b['search_bbox_sonar_pixels']
            for b in batch
        ]),
        # CRITICAL FIX (Audit 2025112602): Add attention masks for padded regions
        'rgb_att_mask': torch.stack([
            torch.from_numpy(b['rgb_att_mask']) if isinstance(b['rgb_att_mask'], np.ndarray)
            else b['rgb_att_mask']
            for b in batch
        ]),
        'sonar_att_mask': torch.stack([
            torch.from_numpy(b['sonar_att_mask']) if isinstance(b['sonar_att_mask'], np.ndarray)
            else b['sonar_att_mask']
            for b in batch
        ]),
        # CRITICAL FIX (Audit 2025120909): Add template attention masks
        'rgb_template_att_mask': torch.stack([
            torch.from_numpy(b['rgb_template_att_mask']) if isinstance(b['rgb_template_att_mask'], np.ndarray)
            else b['rgb_template_att_mask']
            for b in batch
        ]),
        'sonar_template_att_mask': torch.stack([
            torch.from_numpy(b['sonar_template_att_mask']) if isinstance(b['sonar_template_att_mask'], np.ndarray)
            else b['sonar_template_att_mask']
            for b in batch
        ]),
        'is_valid': torch.tensor([b['is_valid'] for b in batch], dtype=torch.bool),
        'is_valid_rgb': torch.tensor([b['is_valid_rgb'] for b in batch], dtype=torch.bool),
        'is_valid_sonar': torch.tensor([b['is_valid_sonar'] for b in batch], dtype=torch.bool),
        # has_object flags for continuous flow design
        'has_object_rgb': torch.tensor([
            b['has_object_rgb'] if b['has_object_rgb'] is not None else True
            for b in batch
        ], dtype=torch.float32),
        'has_object_sonar': torch.tensor([
            b['has_object_sonar'] if b['has_object_sonar'] is not None else True
            for b in batch
        ], dtype=torch.float32),
        'has_object': torch.tensor([
            b['has_object'] if b['has_object'] is not None else True
            for b in batch
        ], dtype=torch.float32),
        'gap': torch.tensor([b['gap'] for b in batch], dtype=torch.long),
        'sequence_name': [b['sequence_name'] for b in batch],
        'template_frame_id': torch.tensor([b['template_frame_id'] for b in batch], dtype=torch.long),
        'search_frame_id': torch.tensor([b['search_frame_id'] for b in batch], dtype=torch.long),
        'rgb_metadata': [b['rgb_metadata'] for b in batch],
        'sonar_metadata': [b['sonar_metadata'] for b in batch],
    }
