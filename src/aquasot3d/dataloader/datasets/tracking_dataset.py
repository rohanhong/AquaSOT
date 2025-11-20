"""
Tracking Dataset for Template-Search Pair Sampling

This module wraps RGBSonarDataset to create template-search pairs for
tracking training following Siamese network paradigm.

Author: AquaSOT-3D Team
Date: 2025-11-11
"""

import random
from typing import Dict, Optional
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from .rgb_sonar_dataset import RGBSonarDataset, MultiSequenceRGBSonarDataset
# Note: letterbox_resize removed - unified with RGB strategy (2025-11-19)


class TrackingDataset(Dataset):
    """
    Dataset wrapper that samples template-search pairs from sequences.

    For tracking, we need pairs of frames:
    - Template: Initial target appearance (frame t)
    - Search: Frame to find target in (frame t+Δt)

    This dataset supports both single-sequence (RGBSonarDataset) and multi-sequence
    (MultiSequenceRGBSonarDataset) base datasets. For multi-sequence datasets, it
    enforces intra-sequence temporal constraints to prevent template/search pairs
    from crossing sequence boundaries.

    Args:
        base_dataset: RGBSonarDataset or MultiSequenceRGBSonarDataset instance
        template_search_gap: Maximum frame gap between template and search (default: 30)
        min_gap: Minimum frame gap (default: 1)
        search_first_prob: Probability of using first frame as template (default: 0.2)
        template_size: Template image size (default: 127)
        search_size: Search image size (default: 255)
    """

    def __init__(
        self,
        base_dataset,
        template_search_gap: int = 30,
        min_gap: int = 1,
        search_first_prob: float = 0.2,
        template_size: int = 127,
        search_size: int = 255,
        augmentation=None,
        search_sequence_length: int = 1,  # CRITICAL FIX (Audit 2025111703 Finding 1): Support T>1 training
    ):
        self.base_dataset = base_dataset
        self.template_search_gap = template_search_gap
        self.min_gap = min_gap
        self.search_first_prob = search_first_prob
        self.template_size = template_size
        self.search_size = search_size
        self.augmentation = augmentation

        # CRITICAL FIX (Audit 2025111703 Finding 1): Sequence length for temporal training
        # search_sequence_length=1: traditional pair sampling (backward compatible)
        # search_sequence_length>1: sample T consecutive search frames for Vision-Mamba
        self.search_sequence_length = search_sequence_length

        # Check if base_dataset is multi-sequence
        self.is_multi_sequence = isinstance(base_dataset, MultiSequenceRGBSonarDataset)

        # Build sequence-aware index mapping
        if self.is_multi_sequence:
            self._build_sequence_index_mapping()
        else:
            # Single sequence case
            self.valid_indices = base_dataset.valid_indices
            self.num_frames = len(self.valid_indices)
            self.sequence_ranges = None

    def _build_sequence_index_mapping(self):
        """
        Build mapping for multi-sequence dataset to ensure temporal constraints.

        Creates a mapping of global indices to their sequence boundaries,
        preventing template/search pairs from crossing sequence boundaries.
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
        Get the sequence name for a given dataset index.

        This method is used by SequenceGroupedBatchSampler to group samples
        from the same sequence into consecutive batches.

        Args:
            index: Dataset index (0 to num_frames-1)

        Returns:
            Sequence name string. For multi-sequence datasets, returns the actual
            sequence name. For single-sequence datasets, returns 'default'.

        CRITICAL FIX (Audit 2025111703 Finding 3):
        Enables sequence-aware batching for EMA aggregator training.
        """
        if self.is_multi_sequence:
            # Multi-sequence: use base_dataset's get_frame_sequence
            # Note: index here is already a valid frame index
            return self.base_dataset.get_frame_sequence(index)
        else:
            # Single sequence: all frames belong to 'default'
            return 'default'

    def __getitem__(self, index: int) -> Dict:
        """
        Sample a template-search pair or sequence.

        For multi-sequence datasets, ensures template and search frames
        come from the same sequence to maintain temporal consistency.

        CRITICAL FIX (Audit 2025111703 Finding 1):
        When search_sequence_length > 1, samples T consecutive search frames.

        Returns:
            Dictionary with keys:
                - rgb_template: Template RGB image, shape (C, H, W)
                - rgb_search: Search RGB image(s), shape (C, H, W) or (T, C, H, W)
                - sonar_template: Template sonar image, shape (C, H, W)
                - sonar_search: Search sonar image(s), shape (C, H, W) or (T, C, H, W)
                - template_bbox_rgb: Template RGB bbox [x, y, w, h]
                - template_bbox_sonar: Template sonar bbox [x, y, w, h]
                - search_bbox_rgb: Search RGB bbox(es) [x, y, w, h] or (T, 4)
                - search_bbox_sonar: Search sonar bbox(es) [x, y, w, h] or (T, 4)
                - template_idx: Template frame index
                - search_idx: Search frame index or list of indices
                - sequence_name: Name of the sequence
                - template_frame_id: 1-based template frame ID
                - search_frame_id: 1-based search frame ID or list
        """
        if self.is_multi_sequence:
            # Multi-sequence case: enforce intra-sequence temporal constraints
            template_idx, search_idx = self._sample_intra_sequence_pair(index)
        else:
            # Single-sequence case: use original logic
            template_idx, search_idx = self._sample_single_sequence_pair(index)

        # CRITICAL FIX (Audit 2025111703 Finding 1): Full sequence sampling implementation
        if isinstance(search_idx, list):
            # T > 1: Load all T search frames and stack them
            return self._get_sequence_sample(template_idx, search_idx)
        else:
            # T = 1: Traditional pair sampling (backward compatible)
            return self._get_pair_sample(template_idx, search_idx)

    def _get_pair_sample(self, template_idx: int, search_idx: int) -> Dict:
        """
        Get template-search pair (T=1, traditional mode).

        Args:
            template_idx: Template frame index
            search_idx: Search frame index

        Returns:
            Dictionary with template and search data
        """
        # Get frame data
        template_data = self.base_dataset[template_idx]
        search_data = self.base_dataset[search_idx]

        # Verify sequence consistency for multi-sequence
        if self.is_multi_sequence:
            template_seq = template_data.get('sequence_name')
            search_seq = search_data.get('sequence_name')
            if template_seq != search_seq:
                raise RuntimeError(
                    f"Template-search sequence mismatch: {template_seq} != {search_seq}. "
                    "This indicates a bug in _sample_intra_sequence_pair."
                )

        # Apply augmentation if provided
        if self.augmentation is not None:
            # Prepare data for augmentation (expects dict with rgb_img, sonar_img, rgb_bbox, sonar_bbox)
            template_aug_input = {
                'rgb_img': template_data['rgb_img'],
                'sonar_img': template_data['sonar_img'],
                'rgb_bbox': template_data['rgb_bbox'],
                'sonar_bbox': template_data['sonar_bbox'],
            }
            search_aug_input = {
                'rgb_img': search_data['rgb_img'],
                'sonar_img': search_data['sonar_img'],
                'rgb_bbox': search_data['rgb_bbox'],
                'sonar_bbox': search_data['sonar_bbox'],
            }

            # Apply augmentation
            template_aug_output = self.augmentation(template_aug_input)
            search_aug_output = self.augmentation(search_aug_input)

            # Update template and search data with augmented results
            template_data['rgb_img'] = template_aug_output.get('rgb_img', template_data['rgb_img'])
            template_data['sonar_img'] = template_aug_output.get('sonar_img', template_data['sonar_img'])
            template_data['rgb_bbox'] = template_aug_output.get('rgb_bbox', template_data['rgb_bbox'])
            template_data['sonar_bbox'] = template_aug_output.get('sonar_bbox', template_data['sonar_bbox'])

            search_data['rgb_img'] = search_aug_output.get('rgb_img', search_data['rgb_img'])
            search_data['sonar_img'] = search_aug_output.get('sonar_img', search_data['sonar_img'])
            search_data['rgb_bbox'] = search_aug_output.get('rgb_bbox', search_data['rgb_bbox'])
            search_data['sonar_bbox'] = search_aug_output.get('sonar_bbox', search_data['sonar_bbox'])

        # Get sequence metadata from base dataset
        if self.is_multi_sequence:
            sequence_name = template_data.get('sequence_name')
        else:
            seq_info = self.base_dataset.get_sequence_info()
            sequence_name = seq_info['sequence_name']

        # Prepare images and get per-axis scale factors
        rgb_template, rgb_template_scale_x, rgb_template_scale_y, rgb_t_pad_x, rgb_t_pad_y = self._prepare_image(template_data['rgb_img'], self.template_size, is_sonar=False)
        rgb_search, rgb_search_scale_x, rgb_search_scale_y, rgb_s_pad_x, rgb_s_pad_y = self._prepare_image(search_data['rgb_img'], self.search_size, is_sonar=False)
        sonar_template, sonar_template_scale_x, sonar_template_scale_y, sonar_t_pad_x, sonar_t_pad_y = self._prepare_image(template_data['sonar_img'], self.template_size, is_sonar=True)
        sonar_search, sonar_search_scale_x, sonar_search_scale_y, sonar_s_pad_x, sonar_s_pad_y = self._prepare_image(search_data['sonar_img'], self.search_size, is_sonar=True)

        # Scale bounding boxes to match resized images with per-axis scaling
        # CRITICAL FIX (Audit 2025111405 Requirement #1): Use separate scale_x and scale_y
        # CRITICAL FIX (Audit 2025111408 Finding 2): Add padding offsets for letterbox sonar
        template_bbox_rgb_scaled = self._scale_bbox(template_data['rgb_bbox'], rgb_template_scale_x, rgb_template_scale_y)
        search_bbox_rgb_scaled = self._scale_bbox(search_data['rgb_bbox'], rgb_search_scale_x, rgb_search_scale_y)

        # Sonar bboxes need letterbox padding adjustments
        template_bbox_sonar_scaled = self._scale_bbox_with_padding(
            template_data['sonar_bbox'], sonar_template_scale_x, sonar_template_scale_y,
            sonar_t_pad_x, sonar_t_pad_y
        )
        search_bbox_sonar_scaled = self._scale_bbox_with_padding(
            search_data['sonar_bbox'], sonar_search_scale_x, sonar_search_scale_y,
            sonar_s_pad_x, sonar_s_pad_y
        )

        # CRITICAL FIX (Audit 2025111411 Finding 1):
        # Propagate transformation metadata for training pipeline
        # Quality estimators and weak-3D projectors need physical coordinate context

        # Get original image dimensions
        template_rgb_h, template_rgb_w = template_data['rgb_img'].shape[:2] if template_data['rgb_img'] is not None else (self.template_size, self.template_size)
        search_rgb_h, search_rgb_w = search_data['rgb_img'].shape[:2] if search_data['rgb_img'] is not None else (self.search_size, self.search_size)
        template_sonar_h, template_sonar_w = template_data['sonar_img'].shape[:2] if template_data['sonar_img'] is not None else (self.template_size, self.template_size)
        search_sonar_h, search_sonar_w = search_data['sonar_img'].shape[:2] if search_data['sonar_img'] is not None else (self.search_size, self.search_size)

        # Build tracking pair
        pair = {
            # Images (resized tensors)
            'rgb_template': rgb_template,
            'rgb_search': rgb_search,
            'sonar_template': sonar_template,
            'sonar_search': sonar_search,

            # Bounding boxes (scaled to match resized images)
            'template_bbox_rgb': template_bbox_rgb_scaled,
            'template_bbox_sonar': template_bbox_sonar_scaled,
            'search_bbox_rgb': search_bbox_rgb_scaled,
            'search_bbox_sonar': search_bbox_sonar_scaled,

            # Bounding boxes (trainer-expected keys for loss computation)
            'rgb_boxes': search_bbox_rgb_scaled,  # Alias for search_bbox_rgb
            'sonar_boxes': search_bbox_sonar_scaled,  # Alias for search_bbox_sonar

            # Sequence metadata (required for evaluation)
            'sequence_name': sequence_name,
            'template_frame_id': template_data['frame_id'],
            'search_frame_id': search_data['frame_id'],
            'frame_id': search_data['frame_id'],  # Alias for search_frame_id

            # Metadata
            'template_idx': template_idx,
            'search_idx': search_idx,
            # CRITICAL FIX (Audit 2025112001): Pass fine-grained validity flags for loss masking
            # Use search validity for loss computation (template only used for feature extraction)
            'is_valid': search_data['is_valid'],
            'is_valid_rgb': search_data.get('is_valid_rgb', search_data['is_valid']),
            'is_valid_sonar': search_data.get('is_valid_sonar', search_data['is_valid']),

            # CRITICAL FIX (Audit 2025111411 Finding 1): Transformation metadata for training
            # These dicts enable quality estimators and weak-3D projectors to operate in physical coordinates
            'template_rgb_metadata': {
                'original_height': template_rgb_h,
                'original_width': template_rgb_w,
                'scale_x': rgb_template_scale_x,
                'scale_y': rgb_template_scale_y,
                'pad_x': rgb_t_pad_x,
                'pad_y': rgb_t_pad_y,
            },
            'rgb_search_metadata': {
                'original_height': search_rgb_h,
                'original_width': search_rgb_w,
                'scale_x': rgb_search_scale_x,
                'scale_y': rgb_search_scale_y,
                'pad_x': rgb_s_pad_x,
                'pad_y': rgb_s_pad_y,
            },
            'template_sonar_metadata': {
                'original_height': template_sonar_h,
                'original_width': template_sonar_w,
                'scale_x': sonar_template_scale_x,
                'scale_y': sonar_template_scale_y,
                'pad_x': sonar_t_pad_x,
                'pad_y': sonar_t_pad_y,
            },
            'sonar_search_metadata': {
                'original_height': search_sonar_h,
                'original_width': search_sonar_w,
                'scale_x': sonar_search_scale_x,
                'scale_y': sonar_search_scale_y,
                'pad_x': sonar_s_pad_x,
                'pad_y': sonar_s_pad_y,
            },
        }

        return pair

    def _get_sequence_sample(self, template_idx: int, search_indices: list) -> Dict:
        """
        Get template and T consecutive search frames (T>1, sequence mode).

        CRITICAL FIX (Audit 2025111703 Finding 1):
        Full implementation of sequence sampling for Vision-Mamba temporal training.

        Args:
            template_idx: Template frame index
            search_indices: List of T consecutive search frame indices

        Returns:
            Dictionary with template and stacked search sequences
                - rgb_search: [T, C, H, W]
                - sonar_search: [T, C, H, W]
                - search_bbox_rgb: [T, 4]
                - search_bbox_sonar: [T, 4]
                - rgb_search_metadata: List[Dict] of length T
                - sonar_search_metadata: List[Dict] of length T
        """
        T = len(search_indices)

        # Get template data (single frame)
        template_data = self.base_dataset[template_idx]

        # Get sequence metadata
        if self.is_multi_sequence:
            sequence_name = template_data.get('sequence_name')
        else:
            seq_info = self.base_dataset.get_sequence_info()
            sequence_name = seq_info['sequence_name']

        # Prepare template (same as pair mode)
        if self.augmentation is not None:
            template_aug_input = {
                'rgb_img': template_data['rgb_img'],
                'sonar_img': template_data['sonar_img'],
                'rgb_bbox': template_data['rgb_bbox'],
                'sonar_bbox': template_data['sonar_bbox'],
            }
            template_aug_output = self.augmentation(template_aug_input)
            template_data['rgb_img'] = template_aug_output.get('rgb_img', template_data['rgb_img'])
            template_data['sonar_img'] = template_aug_output.get('sonar_img', template_data['sonar_img'])
            template_data['rgb_bbox'] = template_aug_output.get('rgb_bbox', template_data['rgb_bbox'])
            template_data['sonar_bbox'] = template_aug_output.get('sonar_bbox', template_data['sonar_bbox'])

        rgb_template, rgb_template_scale_x, rgb_template_scale_y, rgb_t_pad_x, rgb_t_pad_y = self._prepare_image(template_data['rgb_img'], self.template_size, is_sonar=False)
        sonar_template, sonar_template_scale_x, sonar_template_scale_y, sonar_t_pad_x, sonar_t_pad_y = self._prepare_image(template_data['sonar_img'], self.template_size, is_sonar=True)

        template_bbox_rgb_scaled = self._scale_bbox(template_data['rgb_bbox'], rgb_template_scale_x, rgb_template_scale_y)
        template_bbox_sonar_scaled = self._scale_bbox_with_padding(
            template_data['sonar_bbox'], sonar_template_scale_x, sonar_template_scale_y,
            sonar_t_pad_x, sonar_t_pad_y
        )

        template_rgb_h, template_rgb_w = template_data['rgb_img'].shape[:2] if template_data['rgb_img'] is not None else (self.template_size, self.template_size)
        template_sonar_h, template_sonar_w = template_data['sonar_img'].shape[:2] if template_data['sonar_img'] is not None else (self.template_size, self.template_size)

        # Process each search frame
        rgb_search_list = []
        sonar_search_list = []
        rgb_bbox_list = []
        sonar_bbox_list = []
        rgb_metadata_list = []
        sonar_metadata_list = []

        for search_idx in search_indices:
            search_data = self.base_dataset[search_idx]

            # Verify sequence consistency
            if self.is_multi_sequence:
                search_seq = search_data.get('sequence_name')
                if search_seq != sequence_name:
                    raise RuntimeError(
                        f"Sequence mismatch in temporal sampling: template={sequence_name}, search={search_seq}"
                    )

            # Apply augmentation
            if self.augmentation is not None:
                search_aug_input = {
                    'rgb_img': search_data['rgb_img'],
                    'sonar_img': search_data['sonar_img'],
                    'rgb_bbox': search_data['rgb_bbox'],
                    'sonar_bbox': search_data['sonar_bbox'],
                }
                search_aug_output = self.augmentation(search_aug_input)
                search_data['rgb_img'] = search_aug_output.get('rgb_img', search_data['rgb_img'])
                search_data['sonar_img'] = search_aug_output.get('sonar_img', search_data['sonar_img'])
                search_data['rgb_bbox'] = search_aug_output.get('rgb_bbox', search_data['rgb_bbox'])
                search_data['sonar_bbox'] = search_aug_output.get('sonar_bbox', search_data['sonar_bbox'])

            # Prepare images
            rgb_search, rgb_search_scale_x, rgb_search_scale_y, rgb_s_pad_x, rgb_s_pad_y = self._prepare_image(search_data['rgb_img'], self.search_size, is_sonar=False)
            sonar_search, sonar_search_scale_x, sonar_search_scale_y, sonar_s_pad_x, sonar_s_pad_y = self._prepare_image(search_data['sonar_img'], self.search_size, is_sonar=True)

            # Scale bboxes
            search_bbox_rgb_scaled = self._scale_bbox(search_data['rgb_bbox'], rgb_search_scale_x, rgb_search_scale_y)
            search_bbox_sonar_scaled = self._scale_bbox_with_padding(
                search_data['sonar_bbox'], sonar_search_scale_x, sonar_search_scale_y,
                sonar_s_pad_x, sonar_s_pad_y
            )

            # Get original dimensions
            search_rgb_h, search_rgb_w = search_data['rgb_img'].shape[:2] if search_data['rgb_img'] is not None else (self.search_size, self.search_size)
            search_sonar_h, search_sonar_w = search_data['sonar_img'].shape[:2] if search_data['sonar_img'] is not None else (self.search_size, self.search_size)

            # Collect frame data
            rgb_search_list.append(rgb_search)
            sonar_search_list.append(sonar_search)
            rgb_bbox_list.append(search_bbox_rgb_scaled)
            sonar_bbox_list.append(search_bbox_sonar_scaled)

            rgb_metadata_list.append({
                'original_height': search_rgb_h,
                'original_width': search_rgb_w,
                'scale_x': rgb_search_scale_x,
                'scale_y': rgb_search_scale_y,
                'pad_x': rgb_s_pad_x,
                'pad_y': rgb_s_pad_y,
            })
            sonar_metadata_list.append({
                'original_height': search_sonar_h,
                'original_width': search_sonar_w,
                'scale_x': sonar_search_scale_x,
                'scale_y': sonar_search_scale_y,
                'pad_x': sonar_s_pad_x,
                'pad_y': sonar_s_pad_y,
            })

        # Stack sequences: [T, C, H, W]
        import torch
        rgb_search_stacked = torch.stack(rgb_search_list, dim=0)  # [T, C, H, W]
        sonar_search_stacked = torch.stack(sonar_search_list, dim=0)  # [T, C, H, W]

        # Convert bbox lists from numpy to tensor before stacking
        rgb_bbox_list = [torch.from_numpy(bbox).float() if isinstance(bbox, np.ndarray) else bbox for bbox in rgb_bbox_list]
        sonar_bbox_list = [torch.from_numpy(bbox).float() if isinstance(bbox, np.ndarray) else bbox for bbox in sonar_bbox_list]

        rgb_bbox_stacked = torch.stack(rgb_bbox_list, dim=0)  # [T, 4]
        sonar_bbox_stacked = torch.stack(sonar_bbox_list, dim=0)  # [T, 4]

        # Build sequence sample
        sequence = {
            # Template (single frame)
            'rgb_template': rgb_template,
            'sonar_template': sonar_template,
            'template_bbox_rgb': template_bbox_rgb_scaled,
            'template_bbox_sonar': template_bbox_sonar_scaled,

            # Search sequence (T frames)
            'rgb_search': rgb_search_stacked,  # [T, C, H, W]
            'sonar_search': sonar_search_stacked,  # [T, C, H, W]
            'search_bbox_rgb': rgb_bbox_stacked,  # [T, 4]
            'search_bbox_sonar': sonar_bbox_stacked,  # [T, 4]

            # CRITICAL FIX: Return ALL frames' bboxes for quality estimation
            # In sequence mode, quality estimator processes [B*T, C, H, W] images, so needs [B*T, 4] bboxes
            'rgb_boxes': rgb_bbox_stacked,  # [T, 4] - will be collated to [B, T, 4] then flattened
            'sonar_boxes': sonar_bbox_stacked,  # [T, 4]

            # Metadata
            'sequence_name': sequence_name,
            'template_frame_id': template_data['frame_id'],
            'search_frame_id': [self.base_dataset[idx]['frame_id'] for idx in search_indices],
            'frame_id': search_indices[-1],  # Use last frame ID for compatibility
            'template_idx': template_idx,
            'search_idx': search_indices,
            # CRITICAL FIX (Audit 2025112001): Pass fine-grained validity flags for sequence mode
            # Use aggregated validity from all search frames for loss computation
            'is_valid': all([self.base_dataset[idx]['is_valid'] for idx in search_indices]),
            'is_valid_rgb': all([self.base_dataset[idx].get('is_valid_rgb', self.base_dataset[idx]['is_valid']) for idx in search_indices]),
            'is_valid_sonar': all([self.base_dataset[idx].get('is_valid_sonar', self.base_dataset[idx]['is_valid']) for idx in search_indices]),

            # Transformation metadata
            'template_rgb_metadata': {
                'original_height': template_rgb_h,
                'original_width': template_rgb_w,
                'scale_x': rgb_template_scale_x,
                'scale_y': rgb_template_scale_y,
                'pad_x': rgb_t_pad_x,
                'pad_y': rgb_t_pad_y,
            },
            'template_sonar_metadata': {
                'original_height': template_sonar_h,
                'original_width': template_sonar_w,
                'scale_x': sonar_template_scale_x,
                'scale_y': sonar_template_scale_y,
                'pad_x': sonar_t_pad_x,
                'pad_y': sonar_t_pad_y,
            },
            'rgb_search_metadata': rgb_metadata_list,  # List[Dict] of length T
            'sonar_search_metadata': sonar_metadata_list,  # List[Dict] of length T

            # Sequence-specific flag
            'is_sequence': True,
            'sequence_length': T,
        }

        return sequence

    def _sample_single_sequence_pair(self, index: int):
        """
        Sample template-search pair for single-sequence dataset.

        Args:
            index: Current sample index

        Returns:
            Tuple of (template_idx, search_idx) for T=1
            OR Tuple of (template_idx, list[search_indices]) for T>1

        CRITICAL FIX (Audit 2025111703 Finding 1):
        Now supports sampling T>1 consecutive search frames for temporal training.
        """
        # Decide template frame
        if random.random() < self.search_first_prob:
            # Use first valid frame as template
            template_idx = 0
        else:
            # Random template from current index vicinity
            template_idx = index

        # Sample search frame(s) with gap
        max_search_idx = min(
            template_idx + self.template_search_gap,
            self.num_frames - 1
        )
        min_search_idx = min(
            template_idx + self.min_gap,
            self.num_frames - 1
        )

        # Ensure valid range
        if min_search_idx > max_search_idx:
            if self.search_sequence_length == 1:
                return template_idx, template_idx
            else:
                # Not enough frames, repeat the template
                return template_idx, [template_idx] * self.search_sequence_length

        if self.search_sequence_length == 1:
            # Traditional pair sampling
            search_idx = random.randint(min_search_idx, max_search_idx)
            return template_idx, search_idx
        else:
            # CRITICAL FIX (Audit 2025111703 Finding 1): Sample T consecutive frames
            # Ensure we have enough frames for the sequence
            start_idx = random.randint(
                min_search_idx,
                min(max_search_idx, self.num_frames - self.search_sequence_length)
            )
            search_indices = list(range(start_idx, start_idx + self.search_sequence_length))
            return template_idx, search_indices

    def _sample_intra_sequence_pair(self, index: int):
        """
        Sample template-search pair for multi-sequence dataset with intra-sequence constraints.

        Ensures both template and search frames come from the same sequence.

        Args:
            index: Global frame index

        Returns:
            Tuple of (template_idx, search_idx) for T=1
            OR Tuple of (template_idx, list[search_indices]) for T>1

        CRITICAL FIX (Audit 2025111703 Finding 1):
        Now supports sampling T>1 consecutive search frames for temporal training.
        """
        # Get sequence name for this index
        sequence_name = self.base_dataset.get_frame_sequence(index)
        seq_range = self.sequence_ranges[sequence_name]

        # Convert to local index within sequence
        local_index = index - seq_range['start']

        # Decide template frame (local)
        if random.random() < self.search_first_prob:
            # Use first frame of this sequence as template
            local_template_idx = 0
        else:
            # Use current frame as template
            local_template_idx = local_index

        # Sample search frame(s) with gap (ensuring it stays within sequence)
        local_max_search_idx = min(
            local_template_idx + self.template_search_gap,
            seq_range['num_frames'] - 1
        )
        local_min_search_idx = min(
            local_template_idx + self.min_gap,
            seq_range['num_frames'] - 1
        )

        # Ensure valid range
        if local_min_search_idx > local_max_search_idx:
            local_search_idx = local_template_idx
            if self.search_sequence_length > 1:
                # Not enough frames, repeat the template
                template_idx = seq_range['start'] + local_template_idx
                return template_idx, [template_idx] * self.search_sequence_length
            else:
                template_idx = seq_range['start'] + local_template_idx
                search_idx = seq_range['start'] + local_search_idx
                return template_idx, search_idx

        # Convert template to global
        template_idx = seq_range['start'] + local_template_idx

        if self.search_sequence_length == 1:
            # Traditional pair sampling
            local_search_idx = random.randint(local_min_search_idx, local_max_search_idx)
            search_idx = seq_range['start'] + local_search_idx

            # Sanity check: ensure indices are within sequence boundaries
            assert seq_range['start'] <= template_idx < seq_range['end'], \
                f"Template index {template_idx} out of sequence range [{seq_range['start']}, {seq_range['end']})"
            assert seq_range['start'] <= search_idx < seq_range['end'], \
                f"Search index {search_idx} out of sequence range [{seq_range['start']}, {seq_range['end']})"

            return template_idx, search_idx
        else:
            # CRITICAL FIX (Audit 2025111703 Finding 1): Sample T consecutive frames
            # Ensure we have enough frames for the sequence within this sequence boundary
            max_local_start = min(local_max_search_idx, seq_range['num_frames'] - self.search_sequence_length)

            # Check if we have enough frames
            if max_local_start < local_min_search_idx:
                # Not enough frames, fall back to repeating template frame
                template_idx = seq_range['start'] + local_template_idx
                return template_idx, [template_idx] * self.search_sequence_length

            local_start_idx = random.randint(local_min_search_idx, max_local_start)
            # Generate consecutive local indices
            local_search_indices = list(range(local_start_idx, local_start_idx + self.search_sequence_length))

            # Convert to global indices
            search_indices = [seq_range['start'] + local_idx for local_idx in local_search_indices]

            # Sanity check: ensure all indices are within sequence boundaries
            assert seq_range['start'] <= template_idx < seq_range['end'], \
                f"Template index {template_idx} out of sequence range [{seq_range['start']}, {seq_range['end']})"
            for search_idx in search_indices:
                assert seq_range['start'] <= search_idx < seq_range['end'], \
                    f"Search index {search_idx} out of sequence range [{seq_range['start']}, {seq_range['end']})"

            return template_idx, search_indices

    def _prepare_image(self, img, target_size, is_sonar=False):
        """
        Prepare image for training.

        Converts HWC numpy array to CHW tensor and resizes to target_size.

        Args:
            img: Input image (numpy array or tensor)
            target_size: Target image size (square)
            is_sonar: If True, prepare as 1-channel grayscale; if False, as 3-channel RGB

        Returns:
            Tuple of (resized_tensor, scale_x, scale_y, pad_x, pad_y):
                - resized_tensor: Tensor of shape (C, target_size, target_size) where C=1 for sonar, C=3 for RGB
                - scale_x: float, x-axis scale factor
                - scale_y: float, y-axis scale factor
                - pad_x: int, left padding offset (0 for RGB direct resize)
                - pad_y: int, top padding offset (0 for RGB direct resize)

        Note:
            CRITICAL FIX (Audit 2025111408 Finding 2): For sonar images, use letterbox padding
            to preserve aspect ratio and prevent log-polar transform distortion. RGB images
            use direct resize as before (no geometric constraints).
        """
        if img is None:
            # Return zeros for None images (occluded frames)
            num_channels = 1 if is_sonar else 3
            return torch.zeros(num_channels, target_size, target_size), 1.0, 1.0, 0, 0

        # Convert to numpy array if tensor
        if isinstance(img, torch.Tensor):
            # Convert CHW to HWC for letterbox_resize
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0).cpu().numpy()
            elif img.ndim == 2:
                img = img.cpu().numpy()
            # Denormalize if in [0, 1] range
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

        # Ensure numpy array
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Expected numpy array or tensor, got {type(img)}")

        original_h, original_w = img.shape[:2]

        # Unified resize strategy for both RGB and Sonar
        # Note: Sonar now uses rectangular coordinates (682×376), no special handling needed
        import cv2
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        scale_x = target_size / original_w
        scale_y = target_size / original_h
        pad_x = 0
        pad_y = 0

        # Convert to tensor and normalize
        # Handle different channel formats
        if img_resized.ndim == 2:
            # Grayscale (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        elif img_resized.ndim == 3:
            # Color (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected image shape: {img_resized.shape}")

        # Normalize to [0, 1] if in [0, 255] range
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        return img_tensor, scale_x, scale_y, pad_x, pad_y

    def _scale_bbox(self, bbox, scale_x, scale_y):
        """
        Scale bounding box coordinates after image resize with per-axis scaling.

        CRITICAL FIX (Audit 2025111405 Requirement #1): Uses separate scale factors
        for x and y axes to maintain correct geometry when resizing non-square images
        to square targets. Previous implementation used averaged scale_factor which
        caused ~28% distortion in both axes.

        Args:
            bbox: Bounding box in [x, y, w, h] format (pixel coordinates)
            scale_x: X-axis scale factor (target_width / original_width)
            scale_y: Y-axis scale factor (target_height / original_height)

        Returns:
            Scaled bounding box in [x, y, w, h] format (resized pixel coordinates)

        Example:
            >>> # Original 1920x1080 image with bbox [1000, 500, 200, 100]
            >>> # Resized to 255x255
            >>> scale_x = 255 / 1920  # 0.1328
            >>> scale_y = 255 / 1080  # 0.2361
            >>> scaled = _scale_bbox([1000, 500, 200, 100], scale_x, scale_y)
            >>> # Result: [132.8, 118.0, 26.6, 23.6] (correct geometry)
            >>> # vs. averaged scale (0.1845): [184.5, 92.3, 36.9, 18.5] (distorted)
        """
        if bbox is None or len(bbox) != 4:
            return bbox

        # Apply per-axis scaling to maintain correct geometry
        x, y, w, h = bbox
        return np.array([
            x * scale_x,  # Scale x-coordinate by x-axis factor
            y * scale_y,  # Scale y-coordinate by y-axis factor
            w * scale_x,  # Scale width by x-axis factor
            h * scale_y   # Scale height by y-axis factor
        ], dtype=np.float32)

    def _scale_bbox_with_padding(self, bbox, scale_x, scale_y, pad_x, pad_y):
        """
        Scale bounding box coordinates and add padding offsets for letterbox resizing.

        CRITICAL FIX (Audit 2025111408 Finding 2): For sonar images resized with letterbox
        padding, bounding boxes must be both scaled and offset by the padding amount.

        Args:
            bbox: Bounding box in [x, y, w, h] format (pixel coordinates)
            scale_x: X-axis scale factor
            scale_y: Y-axis scale factor
            pad_x: Left padding offset (pixels)
            pad_y: Top padding offset (pixels)

        Returns:
            Scaled and padded bounding box in [x, y, w, h] format

        Example:
            >>> # Original 682x376 sonar with bbox [300, 150, 100, 80]
            >>> # Letterbox to 256x256: scale=0.3733, pad_x=126, pad_y=58
            >>> scaled = _scale_bbox_with_padding([300, 150, 100, 80], 0.3733, 0.3733, 126, 58)
            >>> # Result: x=300*0.3733+126=238, y=150*0.3733+58=114, w=37, h=30
        """
        if bbox is None or len(bbox) != 4:
            return bbox

        # Apply scaling and padding
        x, y, w, h = bbox
        return np.array([
            x * scale_x + pad_x,  # Scale and offset x-coordinate
            y * scale_y + pad_y,  # Scale and offset y-coordinate
            w * scale_x,          # Scale width (no padding)
            h * scale_y           # Scale height (no padding)
        ], dtype=np.float32)