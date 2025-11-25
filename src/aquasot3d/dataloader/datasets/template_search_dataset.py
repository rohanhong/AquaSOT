"""
Template-Search Dataset for Single Object Tracking

This module implements the mainstream SOT paradigm: sample (template, search) pairs
from continuous sequences for training class-agnostic trackers.

Template: Initial target appearance (256x256 crop around bbox)
Search: Current frame (512x512 full-image, no cropping)

Key Features:
- Class-agnostic: Template from any object in frame 1
- Curriculum learning: Configurable frame gap [1, 30]
- Template policies: first_frame | recent | keyframe | mixformer
- MixFormer strategy: Independent RGB/Sonar template updates based on quality
- 0000 annotation handling: Validity propagation
- Multi-sequence support: Respects sequence boundaries
- Multi-modal: RGB + Sonar

Author: AquaSOT-3D Team
Date: 2025-11-21
"""

import random
import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from torch.utils.data import Dataset

from .rgb_sonar_dataset import RGBSonarDataset, MultiSequenceRGBSonarDataset
from aquasot3d.utils.bbox_utils import convert_bbox_format, normalize_bbox


class TemplateSearchDataset(Dataset):
    """
    Template-Search paired dataset for SOT training.

    Samples (template, search) pairs from continuous sequences.
    - Template: First frame or recent keyframe (256x256 crop around target)
    - Search: Current frame (512x512 full-image)

    This follows mainstream SOT methods (OSTrack, TransT, ARTrack) where:
    1. Template carries target appearance from reference frame
    2. Search is the current frame where we locate the target

    Args:
        base_dataset: RGBSonarDataset or MultiSequenceRGBSonarDataset instance
                      IMPORTANT (Audit 2025112504 Finding 3): Set base_dataset.visibility_policy
                      to control frame filtering:
                        - 'any': At least one modality visible (recommended for robust training)
                        - 'both': Both modalities visible (pure dual-modality training)
                        - 'all': Include all frames including zero-modality (not recommended)
        template_size: Template crop size (default: 256)
        search_size: Search region size (default: 512, full-frame)
        template_search_gap: Frame gap range (min, max) for curriculum learning
        template_update_policy: 'first_frame' | 'recent' | 'keyframe'
        context_factor: Template crop context factor (default: 2.0)
        augmentation: Optional augmentation module (deprecated, use base_dataset)

    Example:
        >>> base = RGBSonarDataset(data_root='...', sequence_name='seq1')
        >>> dataset = TemplateSearchDataset(
        ...     base_dataset=base,
        ...     template_search_gap=(1, 30),
        ...     template_update_policy='recent'
        ... )
        >>> sample = dataset[100]
        >>> print(sample['rgb_template'].shape)  # [3, 256, 256]
        >>> print(sample['rgb_search'].shape)    # [3, 512, 512]
        >>> print(sample['gap'])                 # 15 (random in [1, 30])
    """

    def __init__(
        self,
        base_dataset,
        template_size: int = 256,
        search_size: int = 512,
        template_search_gap: Tuple[int, int] = (1, 30),
        template_update_policy: str = 'first_frame',
        quality_threshold_rgb: float = 0.7,
        quality_threshold_sonar: float = 0.7,
        context_factor: float = 2.0,
        augmentation=None,  # Deprecated
        normalization_config: Optional[Dict] = None
    ):
        self.base_dataset = base_dataset
        self.template_size = template_size
        self.search_size = search_size
        self.gap_range = template_search_gap
        self.update_policy = template_update_policy
        self.quality_threshold_rgb = quality_threshold_rgb
        self.quality_threshold_sonar = quality_threshold_sonar
        self.context_factor = context_factor

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
                rgb_template: [3, 256, 256] - RGB template crop
                rgb_search: [3, 512, 512] - RGB search full-frame
                sonar_template: [1, 256, 256] - Sonar template crop
                sonar_search: [1, 512, 512] - Sonar search full-frame
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

        # 2. Sample template frame based on policy
        template_idx = self._sample_template_index(index)
        template_sample = self.base_dataset[template_idx]

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
        if template_sample['rgb_img'] is not None:
            rgb_template = self._crop_template(
                template_sample['rgb_img'],
                template_sample['rgb_bbox'],
                self.template_size
            )
        else:
            rgb_template = None

        if template_sample['sonar_img'] is not None:
            sonar_template = self._crop_template(
                template_sample['sonar_img'],
                template_sample['sonar_bbox'],
                self.template_size
            )
        else:
            sonar_template = None

        # 4. Process search (letterbox resize to search_size)
        # CRITICAL FIX (Template-Search Migration 2025112104):
        # base_dataset returns raw images, TemplateSearchDataset must resize them
        # Always process both sensors (target visibility handled by None checks)
        if search_sample['rgb_img'] is not None:
            rgb_search_raw = search_sample['rgb_img']  # Raw [H, W, 3] numpy array (after augmentation)
            # CRITICAL FIX (Audit 2025112401 Finding 1): Store original bbox before letterbox
            # Note: bbox may have been transformed by augmentation (e.g., horizontal flip)
            rgb_bbox_original = search_sample['rgb_bbox'].copy()  # XYWH format
            # Apply letterbox resize to search size (preserve aspect ratio)
            # CRITICAL FIX (Audit 2025112401 Finding 1): Capture metadata for bbox transform
            rgb_search, rgb_letterbox_meta = self._letterbox(rgb_search_raw, self.search_size)
        else:
            rgb_search_raw = None
            rgb_bbox_original = None
            rgb_search = None
            rgb_letterbox_meta = {'scale': 1.0, 'pad_left': 0, 'pad_top': 0}

        if search_sample['sonar_img'] is not None:
            sonar_search_raw = search_sample['sonar_img']  # Raw [H, W] numpy array (after augmentation)
            # CRITICAL FIX (Audit 2025112401 Finding 1): Store original bbox before letterbox
            sonar_bbox_original = search_sample['sonar_bbox'].copy()  # XYWH format
            # Apply letterbox resize to search size (preserve aspect ratio)
            sonar_search, sonar_letterbox_meta = self._letterbox(sonar_search_raw, self.search_size)
        else:
            sonar_search_raw = None
            sonar_bbox_original = None
            sonar_search = None
            sonar_letterbox_meta = {'scale': 1.0, 'pad_left': 0, 'pad_top': 0}

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

        # 5. Extract and merge metadata with letterbox metadata
        # CRITICAL FIX (Audit 2025112401 Finding 1): Add letterbox metadata
        # CRITICAL FIX (Audit 2025112403 Finding 3): Add original_height/width for radial binning
        rgb_metadata = search_sample.get('rgb_metadata', {})
        rgb_metadata.update(rgb_letterbox_meta)
        # Add explicit original_height/width fields required by GeometryGatedAttention
        if 'original_size' in rgb_letterbox_meta:
            rgb_metadata['original_height'] = rgb_letterbox_meta['original_size'][0]
            rgb_metadata['original_width'] = rgb_letterbox_meta['original_size'][1]

        sonar_metadata = search_sample.get('sonar_metadata', {})
        sonar_metadata.update(sonar_letterbox_meta)
        # Add explicit original_height/width fields required by GeometryGatedAttention
        if 'original_size' in sonar_letterbox_meta:
            sonar_metadata['original_height'] = sonar_letterbox_meta['original_size'][0]
            sonar_metadata['original_width'] = sonar_letterbox_meta['original_size'][1]

        # CRITICAL FIX (Audit 2025112501 P3 - FINAL DESIGN):
        # G²-DCMA uses HEURISTIC geometry gating based on letterbox metadata (scale/padding).
        # External sonar geometry priors (FOV, range) are NOT used - this is the final implementation.
        # GeometryGatedAttention computes radial bins from letterbox-compensated sensor origins,
        # which provides sufficient geometric regularization without external calibration.
        #
        # Design rationale:
        #   - Letterbox metadata (scale/pad) captures spatial transformations reliably
        #   - Heuristic radial estimation from feature maps provides geometry-aware gating
        #   - Avoids dependency on potentially inaccurate external geometry parameters
        #   - Maintains calibration-free fusion capability (core system requirement)
        #
        # Only pass transform metadata (scale/padding) and rectification status.
        sonar_metadata['is_rectified'] = getattr(self.base_dataset, 'sonar_is_rectified', True)

        # CRITICAL FIX (Audit 2025112401 Finding 1):
        # Apply letterbox transform to bboxes BEFORE normalization
        # This ensures bbox coordinates align with the letterboxed image

        # Helper function to apply letterbox transform to XYWH bbox
        def apply_letterbox_to_bbox(bbox_xywh, metadata):
            """
            Apply letterbox transform to bbox coordinates.

            Args:
                bbox_xywh: [x, y, w, h] in original image coordinates
                metadata: Dict with 'scale', 'pad_left', 'pad_top'

            Returns:
                bbox_xywh_transformed: [x, y, w, h] in letterboxed image coordinates
            """
            x, y, w, h = bbox_xywh
            scale = metadata['scale']
            pad_left = metadata['pad_left']
            pad_top = metadata['pad_top']

            # Apply scale then offset
            x_new = x * scale + pad_left
            y_new = y * scale + pad_top
            w_new = w * scale
            h_new = h * scale

            return np.array([x_new, y_new, w_new, h_new], dtype=np.float32)

        # CRITICAL FIX (Template-Search Migration 2025112105):
        # Convert GT boxes from XYWH pixels to CXCYWH normalized [0, 1]
        # Trainer expects normalized CXCYWH format to match model predictions
        # Use existing bbox_utils functions per code review

        # Template bbox (in 256x256 template crop) - Use RGB if available, else Sonar
        if template_sample['rgb_bbox'] is not None:
            template_bbox_cxcywh = convert_bbox_format(
                template_sample['rgb_bbox'].reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            template_bbox_normalized = normalize_bbox(
                template_bbox_cxcywh,
                image_size=(self.template_size, self.template_size),
                format='cxcywh'
            )[0]  # Extract single box from batch
        elif template_sample['sonar_bbox'] is not None:
            template_bbox_cxcywh = convert_bbox_format(
                template_sample['sonar_bbox'].reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            template_bbox_normalized = normalize_bbox(
                template_bbox_cxcywh,
                image_size=(self.template_size, self.template_size),
                format='cxcywh'
            )[0]
        else:
            # Fallback: zero bbox
            template_bbox_normalized = np.zeros(4, dtype=np.float32)

        # CRITICAL FIX (Audit 2025112312 - Single-Modality Visibility):
        # Process RGB and Sonar search bboxes separately to support
        # single-modality visibility scenarios (e.g., target out of RGB FOV but in Sonar FOV)

        # RGB search bbox processing (always process both sensors)
        if rgb_bbox_original is not None:
            # CRITICAL FIX (Audit 2025112401 Finding 1):
            # Apply letterbox transform to RGB search bbox
            rgb_bbox_transformed = apply_letterbox_to_bbox(rgb_bbox_original, rgb_letterbox_meta)

            # RGB search bbox (in 512x512 search image, RGB coordinate system)
            search_bbox_rgb_cxcywh = convert_bbox_format(
                rgb_bbox_transformed.reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            search_bbox_rgb_normalized = normalize_bbox(
                search_bbox_rgb_cxcywh,
                image_size=(self.search_size, self.search_size),
                format='cxcywh'
            )[0]  # Extract single box from batch
            # CRITICAL FIX (Audit 2025112401 Finding 9):
            # Keep pixel-space boxes for evaluation (after letterbox transform)
            search_bbox_rgb_pixels = rgb_bbox_transformed
        else:
            search_bbox_rgb_normalized = np.zeros(4, dtype=np.float32)
            search_bbox_rgb_pixels = np.zeros(4, dtype=np.float32)

        # Sonar search bbox processing (always process both sensors)
        if sonar_bbox_original is not None:
            # CRITICAL FIX (Audit 2025112401 Finding 1):
            # Apply letterbox transform to Sonar search bbox
            sonar_bbox_transformed = apply_letterbox_to_bbox(sonar_bbox_original, sonar_letterbox_meta)

            # Sonar search bbox (in 512x512 search image, Sonar coordinate system)
            search_bbox_sonar_cxcywh = convert_bbox_format(
                sonar_bbox_transformed.reshape(1, 4),
                src_format='xywh',
                dst_format='cxcywh'
            )
            search_bbox_sonar_normalized = normalize_bbox(
                search_bbox_sonar_cxcywh,
                image_size=(self.search_size, self.search_size),
                format='cxcywh'
            )[0]  # Extract single box from batch
            # CRITICAL FIX (Audit 2025112401 Finding 9):
            # Keep pixel-space boxes for evaluation (after letterbox transform)
            search_bbox_sonar_pixels = sonar_bbox_transformed
        else:
            search_bbox_sonar_normalized = np.zeros(4, dtype=np.float32)
            search_bbox_sonar_pixels = np.zeros(4, dtype=np.float32)

        # Extract has_object flags from base_dataset
        has_object_rgb = search_sample.get('has_object_rgb', None)
        has_object_sonar = search_sample.get('has_object_sonar', None)
        has_object = search_sample.get('has_object', None)

        # Use base dataset's is_valid flag (target visibility)
        # base_dataset already computes is_valid correctly based on target visibility:
        # - zero-modality: target not visible in either sensor (is_valid=False)
        # - single-RGB/sonar: target visible in one sensor (is_valid depends on return_occluded)
        # - dual-modality: target visible in both sensors (is_valid=True)
        # Do not recompute here to preserve base_dataset's visibility logic
        is_valid = search_sample.get('is_valid', True)

        return {
            'rgb_template': rgb_template,
            'rgb_search': rgb_search,
            'sonar_template': sonar_template,
            'sonar_search': sonar_search,
            'template_bbox': template_bbox_normalized,  # Normalized CXCYWH [0,1] (RGB)
            'search_bbox_rgb': search_bbox_rgb_normalized,      # RGB GT in RGB coordinate system
            'search_bbox_sonar': search_bbox_sonar_normalized,  # Sonar GT in Sonar coordinate system
            # CRITICAL FIX (Audit 2025112401 Finding 9): Add pixel boxes for evaluation
            'search_bbox_rgb_pixels': search_bbox_rgb_pixels,    # RGB GT in pixels (letterboxed space)
            'search_bbox_sonar_pixels': search_bbox_sonar_pixels,  # Sonar GT in pixels (letterboxed space)
            'is_valid': is_valid,                       # General validity flag
            'is_valid_rgb': search_sample['is_rgb_valid'],
            'is_valid_sonar': search_sample['is_sonar_valid'],
            'has_object_rgb': has_object_rgb,           # RGB objectness flag
            'has_object_sonar': has_object_sonar,       # Sonar objectness flag
            'has_object': has_object,                   # Combined objectness flag
            'gap': index - template_idx,
            'sequence_name': search_sample.get('sequence_name', 'default'),
            'template_frame_id': template_idx + 1,  # 1-based
            'search_frame_id': index + 1,
            'rgb_metadata': rgb_metadata,
            'sonar_metadata': sonar_metadata,
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
            import logging
            logger = logging.getLogger(__name__)
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
            # Use first valid frame of sequence as template
            template_idx = find_nearest_valid_frame(seq_start, seq_start, search_idx)
            return template_idx

        elif self.update_policy in ('recent', 'mixformer'):
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

    def _crop_template(
        self,
        image: torch.Tensor,
        bbox: np.ndarray,
        crop_size: int
    ) -> torch.Tensor:
        """
        Crop template region around target bbox using letterbox (resize+pad).

        CRITICAL: Uses letterbox preprocessing to preserve aspect ratio and avoid
        distortion, following mainstream SOT methods (OSTrack/TransT).

        Args:
            image: [3, H, W] or [1, H, W] tensor (from base_dataset)
            bbox: [x, y, w, h] in XYWH pixel format (numpy array)
            crop_size: Target crop size (e.g., 256)

        Returns:
            Letterboxed template [C, crop_size, crop_size] with preserved aspect ratio
        """
        # Convert tensor to numpy for cropping
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # [C, H, W]
                image_np = image.permute(1, 2, 0).numpy()  # → [H, W, C]
            else:
                image_np = image.numpy()
        else:
            image_np = image

        # Denormalize if needed (base_dataset may normalize to [0, 1])
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        x, y, w, h = bbox

        # Compute crop region (centered on bbox with context padding)
        cx, cy = x + w / 2, y + h / 2
        crop_sz = max(w, h) * self.context_factor  # 2x context factor

        # Clamp to image boundaries
        H, W = image_np.shape[:2]
        x1 = max(0, int(cx - crop_sz / 2))
        y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(W, int(cx + crop_sz / 2))
        y2 = min(H, int(cy + crop_sz / 2))

        # Crop
        crop = image_np[y1:y2, x1:x2]

        # Handle empty crops (degenerate bbox)
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            # Return black template
            # CRITICAL FIX: Check channel dimension, not ndim
            # Sonar images are (H, W, 1) with ndim=3, not ndim=2
            if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[2] == 1):
                # Grayscale/single-channel (sonar)
                crop = np.zeros((crop_size, crop_size), dtype=np.uint8)
            else:
                # RGB (3-channel)
                crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop_letterboxed = crop  # Already correct size
        else:
            # Apply letterbox preprocessing (resize + pad to preserve aspect ratio)
            # CRITICAL FIX (Audit 2025112401 Finding 1): _letterbox now returns metadata
            crop_letterboxed, _ = self._letterbox(crop, crop_size)

        # Convert to tensor [C, H, W] and apply normalization
        # CRITICAL FIX (Audit 2025112401 Finding 4):
        # Apply ImageNet normalization for pretrained ViT compatibility
        # CRITICAL FIX (Audit 2025112402 Finding 3):
        # Use config-loaded normalization parameters instead of hardcoding
        if crop_letterboxed.ndim == 2:  # Grayscale (sonar)
            crop_tensor = torch.from_numpy(crop_letterboxed).unsqueeze(0).float() / 255.0
            crop_tensor = (crop_tensor - self.sonar_mean) / self.sonar_std
        else:  # RGB
            crop_tensor = torch.from_numpy(crop_letterboxed).permute(2, 0, 1).float() / 255.0
            crop_tensor = (crop_tensor - self.rgb_mean) / self.rgb_std

        return crop_tensor

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


def template_search_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for template-search pairs.

    Input batch[i]:
        {
            'rgb_template': [3, 256, 256],
            'rgb_search': [3, 512, 512],
            'sonar_template': [1, 256, 256],
            'sonar_search': [1, 512, 512],
            'template_bbox': [4] CXCYWH normalized [0,1],
            'search_bbox_rgb': [4] CXCYWH normalized [0,1] - RGB GT,
            'search_bbox_sonar': [4] CXCYWH normalized [0,1] - Sonar GT,
            'search_bbox_rgb_pixels': [4] XYWH pixels (letterboxed space),
            'search_bbox_sonar_pixels': [4] XYWH pixels (letterboxed space),
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
            'rgb_template': [B, 3, 256, 256],
            'rgb_search': [B, 3, 512, 512],
            'sonar_template': [B, 1, 256, 256],
            'sonar_search': [B, 1, 512, 512],
            'template_bbox': [B, 4],
            'search_bbox_rgb': [B, 4],
            'search_bbox_sonar': [B, 4],
            'search_bbox_rgb_pixels': [B, 4],
            'search_bbox_sonar_pixels': [B, 4],
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
    # CRITICAL FIX (Audit 2025112312 - Single-Modality Visibility):
    # Handle separate RGB and Sonar GT boxes + has_object flags

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
