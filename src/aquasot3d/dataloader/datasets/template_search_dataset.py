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
        augmentation=None  # Deprecated
    ):
        self.base_dataset = base_dataset
        self.template_size = template_size
        self.search_size = search_size
        self.gap_range = template_search_gap
        self.update_policy = template_update_policy
        self.quality_threshold_rgb = quality_threshold_rgb
        self.quality_threshold_sonar = quality_threshold_sonar
        self.context_factor = context_factor

        # Augmentation Order Optimization (Audit 2025-11-20):
        # Augmentation should be applied in base_dataset (before letterbox)
        if augmentation is not None:
            import warnings
            warnings.warn(
                "Passing augmentation to TemplateSearchDataset is deprecated. "
                "Please pass augmentation to base_dataset instead for better quality.",
                DeprecationWarning
            )
        self.augmentation = None  # Not used

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

        # 3. Process template (crop around target bbox)
        rgb_template = self._crop_template(
            template_sample['rgb_img'],
            template_sample['rgb_bbox'],
            self.template_size
        )
        sonar_template = self._crop_template(
            template_sample['sonar_img'],
            template_sample['sonar_bbox'],
            self.template_size
        )

        # 4. Process search (letterbox resize to search_size)
        # CRITICAL FIX (Template-Search Migration 2025112104):
        # base_dataset returns raw images, TemplateSearchDataset must resize them
        rgb_search_raw = search_sample['rgb_img']  # Raw [H, W, 3] numpy array
        sonar_search_raw = search_sample['sonar_img']  # Raw [H, W] numpy array

        # Apply letterbox resize to search size (preserve aspect ratio)
        rgb_search = self._letterbox(rgb_search_raw, self.search_size)
        sonar_search = self._letterbox(sonar_search_raw, self.search_size)

        # Convert to tensors and normalize
        if isinstance(rgb_search, np.ndarray):
            rgb_search = torch.from_numpy(rgb_search).float()
            if rgb_search.ndim == 3 and rgb_search.shape[2] == 3:  # [H, W, 3]
                rgb_search = rgb_search.permute(2, 0, 1)  # → [3, H, W]
            rgb_search = rgb_search / 255.0

        if isinstance(sonar_search, np.ndarray):
            sonar_search = torch.from_numpy(sonar_search).float()
            if sonar_search.ndim == 2:  # [H, W]
                sonar_search = sonar_search.unsqueeze(0)  # → [1, H, W]
            sonar_search = sonar_search / 255.0

        # 5. Extract metadata (if available)
        rgb_metadata = search_sample.get('rgb_metadata', {})
        sonar_metadata = search_sample.get('sonar_metadata', {})

        # CRITICAL FIX (Template-Search Migration 2025112105):
        # Convert GT boxes from XYWH pixels to CXCYWH normalized [0, 1]
        # Trainer expects normalized CXCYWH format to match model predictions
        # Use existing bbox_utils functions per code review

        # Template bbox (in 256x256 template crop)
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

        # Search bbox (in 512x512 search image)
        search_bbox_cxcywh = convert_bbox_format(
            search_sample['rgb_bbox'].reshape(1, 4),
            src_format='xywh',
            dst_format='cxcywh'
        )
        search_bbox_normalized = normalize_bbox(
            search_bbox_cxcywh,
            image_size=(self.search_size, self.search_size),
            format='cxcywh'
        )[0]  # Extract single box from batch

        # General validity flag (required by trainer)
        # Valid if BOTH modalities are valid (conservative approach)
        is_valid = search_sample['is_rgb_valid'] and search_sample['is_sonar_valid']

        return {
            'rgb_template': rgb_template,
            'rgb_search': rgb_search,
            'sonar_template': sonar_template,
            'sonar_search': sonar_search,
            'template_bbox': template_bbox_normalized,  # Normalized CXCYWH [0,1]
            'search_bbox': search_bbox_normalized,      # Normalized CXCYWH [0,1] TARGET
            'is_valid': is_valid,                       # General validity flag
            'is_valid_rgb': search_sample['is_rgb_valid'],
            'is_valid_sonar': search_sample['is_sonar_valid'],
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
            Template frame index (0-based)
        """
        if self.update_policy == 'first_frame':
            # Use first frame of sequence as template
            if self.is_multi_sequence:
                seq_name = self.base_dataset.get_frame_sequence(search_idx)
                seq_info = self.sequence_ranges[seq_name]
                return seq_info['start']  # First frame of sequence
            else:
                return 0  # First frame globally

        elif self.update_policy in ('recent', 'mixformer'):
            # Sample from [search_idx - gap_max, search_idx - gap_min]
            # MixFormer training: learn to track from variable temporal distances
            gap = random.randint(self.gap_range[0], self.gap_range[1])
            template_idx = search_idx - gap

            # Clamp to sequence boundaries
            if self.is_multi_sequence:
                seq_name = self.base_dataset.get_frame_sequence(search_idx)
                seq_info = self.sequence_ranges[seq_name]
                template_idx = max(template_idx, seq_info['start'])
            else:
                template_idx = max(template_idx, 0)

            return template_idx

        elif self.update_policy == 'keyframe':
            # Future work: Use detected keyframes
            # For now, fallback to first_frame
            import warnings
            warnings.warn(
                "Keyframe policy not implemented yet, using first_frame fallback.",
                UserWarning
            )
            return self._sample_template_index.__func__(self, search_idx)  # Recursive call with first_frame

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
            if image_np.ndim == 2:  # Grayscale
                crop = np.zeros((crop_size, crop_size), dtype=np.uint8)
            else:
                crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop_letterboxed = crop  # Already correct size
        else:
            # Apply letterbox preprocessing (resize + pad to preserve aspect ratio)
            crop_letterboxed = self._letterbox(crop, crop_size)

        # Convert to tensor [C, H, W]
        if crop_letterboxed.ndim == 2:  # Grayscale (sonar)
            crop_tensor = torch.from_numpy(crop_letterboxed).unsqueeze(0).float() / 255.0
        else:  # RGB
            crop_tensor = torch.from_numpy(crop_letterboxed).permute(2, 0, 1).float() / 255.0

        return crop_tensor

    def _letterbox(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """
        Letterbox preprocessing: resize to fit within target_size and pad to square.

        CRITICAL: Preserves aspect ratio to avoid distortion.
        Following YOLOv5/OSTrack letterbox implementation.

        Args:
            img: Input image [H, W] or [H, W, C]
            target_size: Target square size (e.g., 256)

        Returns:
            Letterboxed image [target_size, target_size, C] or [target_size, target_size]
        """
        h, w = img.shape[:2]

        # Compute scale factor to fit within target_size
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with preserved aspect ratio
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Pad with gray (114 for RGB, 127 for grayscale)
        if img.ndim == 2:  # Grayscale
            pad_value = 127
        else:  # RGB
            pad_value = (114, 114, 114)

        img_letterboxed = cv2.copyMakeBorder(
            img_resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        return img_letterboxed


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
            'search_bbox': [4] CXCYWH normalized [0,1],
            'is_valid': bool,
            'is_valid_rgb': bool,
            'is_valid_sonar': bool,
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
            'search_bbox': [B, 4],
            'is_valid_rgb': [B],
            'is_valid_sonar': [B],
            'gap': [B],
            'sequence_name': List[str],
            'template_frame_id': [B],
            'search_frame_id': [B],
            'rgb_metadata': List[Dict],
            'sonar_metadata': List[Dict],
        }
    """
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
        'search_bbox': torch.stack([
            torch.from_numpy(b['search_bbox']) if isinstance(b['search_bbox'], np.ndarray)
            else b['search_bbox']
            for b in batch
        ]),
        'is_valid': torch.tensor([b['is_valid'] for b in batch], dtype=torch.bool),
        'is_valid_rgb': torch.tensor([b['is_valid_rgb'] for b in batch], dtype=torch.bool),
        'is_valid_sonar': torch.tensor([b['is_valid_sonar'] for b in batch], dtype=torch.bool),
        'gap': torch.tensor([b['gap'] for b in batch], dtype=torch.long),
        'sequence_name': [b['sequence_name'] for b in batch],
        'template_frame_id': torch.tensor([b['template_frame_id'] for b in batch], dtype=torch.long),
        'search_frame_id': torch.tensor([b['search_frame_id'] for b in batch], dtype=torch.long),
        'rgb_metadata': [b['rgb_metadata'] for b in batch],
        'sonar_metadata': [b['sonar_metadata'] for b in batch],
    }
