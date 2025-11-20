"""
Frame Sequence Dataset for Temporal Tracking

This module implements frame-level temporal tracking dataset that samples
continuous frame sequences for Vision-Mamba temporal modeling.

ARCHITECTURE CHANGE (Audit 2025112006):
Replaces Template-Search paired paradigm with pure sequence-based tracking.

Author: AquaSOT-3D Team
Date: 2025-11-20
"""

import random
import cv2
from typing import Dict, Optional, List
import torch
import numpy as np
from torch.utils.data import Dataset

from .rgb_sonar_dataset import RGBSonarDataset, MultiSequenceRGBSonarDataset


class FrameSequenceDataset(Dataset):
    """
    Dataset for frame-level temporal tracking with continuous sequences.

    Samples continuous frame sequences for temporal modeling instead of
    template-search pairs. Each sample contains T consecutive frames.

    CRITICAL LOGIC PRESERVED (from TrackingDataset):
    - 0000 annotation handling (is_valid_* propagation)
    - Single-modality support (modalities parameter via base_dataset)
    - Multi-sequence support (sequence boundary respect)
    - Full-image resize strategy (no search region cropping)

    Args:
        base_dataset: RGBSonarDataset or MultiSequenceRGBSonarDataset instance
        sequence_length: Number of consecutive frames per sample (T)
        frame_size: Unified frame size (H=W) for all frames
        frame_stride: Stride for frame sampling (1=consecutive, 2=skip 1, etc.)
        augmentation: Optional augmentation module (should support temporal consistency)
    """

    def __init__(
        self,
        base_dataset,
        sequence_length: int = 5,
        frame_size: int = 512,  # Updated default: 384 -> 512
        frame_stride: int = 1,
        augmentation=None  # Deprecated: augmentation now applied in base_dataset
    ):
        self.base_dataset = base_dataset
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.frame_stride = frame_stride

        # Augmentation Order Optimization (2025-11-20):
        # Augmentation is now applied in base_dataset (before letterbox)
        # This parameter is kept for backward compatibility but will be ignored
        if augmentation is not None:
            import warnings
            warnings.warn(
                "Passing augmentation to FrameSequenceDataset is deprecated. "
                "Please pass augmentation to base_dataset instead for better quality.",
                DeprecationWarning
            )
        self.augmentation = None  # Not used anymore

        # PRESERVED: Multi-sequence support (from tracking_dataset.py:68-78)
        self.is_multi_sequence = isinstance(base_dataset, MultiSequenceRGBSonarDataset)

        if self.is_multi_sequence:
            self._build_sequence_index_mapping()
        else:
            self.num_frames = len(base_dataset)
            self.sequence_ranges = None

    def _build_sequence_index_mapping(self):
        """
        Build sequence boundary mapping.

        PRESERVED from tracking_dataset.py:80-96
        Prevents frame sequences from crossing sequence boundaries.
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

        PRESERVED from tracking_dataset.py:101-124
        Used by SequenceGroupedBatchSampler for sequence-aware batching.
        """
        if self.is_multi_sequence:
            return self.base_dataset.get_frame_sequence(index)
        else:
            return 'default'

    def __getitem__(self, index: int) -> Dict:
        """
        Sample continuous frame sequence centered at index.

        Returns:
            Dictionary with keys:
                - rgb_frames: [T, 3, H, W] or None
                - sonar_frames: [T, 1, H, W] or None
                - rgb_bboxes: [T, 4]
                - sonar_bboxes: [T, 4]
                - is_valid_rgb: [T] (CRITICAL: 0000 handling)
                - is_valid_sonar: [T]
                - target_idx: int (index of prediction target, usually T//2)
                - sequence_name: str
                - frame_ids: List[int] (1-based frame IDs)
                - rgb_metadata: List[Dict] (per-frame transformation metadata)
                - sonar_metadata: List[Dict]
        """
        # Determine frame range (centered sampling)
        half_len = self.sequence_length // 2

        if self.is_multi_sequence:
            frame_indices, target_idx = self._sample_intra_sequence_frames(index, half_len)
        else:
            frame_indices, target_idx = self._sample_single_sequence_frames(index, half_len)

        # Load and process frames
        return self._load_sequence(frame_indices, target_idx)

    def _sample_single_sequence_frames(self, index: int, half_len: int):
        """
        Sample frame sequence for single-sequence dataset.

        Args:
            index: Current sample index (center)
            half_len: Half of sequence length

        Returns:
            Tuple of (frame_indices, target_idx)
        """
        # Compute frame range with stride
        start_idx = max(0, index - half_len * self.frame_stride)
        end_idx = min(self.num_frames - 1, index + half_len * self.frame_stride)

        # Generate frame indices
        frame_indices = list(range(start_idx, end_idx + 1, self.frame_stride))

        # Pad if needed (boundary handling)
        while len(frame_indices) < self.sequence_length:
            if start_idx == 0:
                # At sequence start: repeat first frame
                frame_indices.append(frame_indices[-1])
            else:
                # At sequence end: repeat last frame
                frame_indices.insert(0, frame_indices[0])

        # Trim to exact length if overshot
        if len(frame_indices) > self.sequence_length:
            # Center crop
            excess = len(frame_indices) - self.sequence_length
            start_trim = excess // 2
            frame_indices = frame_indices[start_trim:start_trim + self.sequence_length]

        # Target is center frame
        target_idx = half_len

        return frame_indices, target_idx

    def _sample_intra_sequence_frames(self, index: int, half_len: int):
        """
        Sample frame sequence for multi-sequence dataset with boundary respect.

        PRESERVED from tracking_dataset.py:587-680
        Ensures frames don't cross sequence boundaries.

        Args:
            index: Global frame index
            half_len: Half of sequence length

        Returns:
            Tuple of (frame_indices, target_idx)
        """
        # Get sequence info
        sequence_name = self.base_dataset.get_frame_sequence(index)
        seq_range = self.sequence_ranges[sequence_name]

        # Convert to local index
        local_index = index - seq_range['start']

        # Compute local frame range
        local_start = max(0, local_index - half_len * self.frame_stride)
        local_end = min(
            seq_range['num_frames'] - 1,
            local_index + half_len * self.frame_stride
        )

        # Generate local indices
        local_indices = list(range(local_start, local_end + 1, self.frame_stride))

        # Pad if needed
        while len(local_indices) < self.sequence_length:
            if local_start == 0:
                local_indices.append(local_indices[-1])
            else:
                local_indices.insert(0, local_indices[0])

        # Trim if overshot
        if len(local_indices) > self.sequence_length:
            excess = len(local_indices) - self.sequence_length
            start_trim = excess // 2
            local_indices = local_indices[start_trim:start_trim + self.sequence_length]

        # Convert to global indices
        frame_indices = [seq_range['start'] + local_idx for local_idx in local_indices]

        # Target is center frame
        target_idx = half_len

        return frame_indices, target_idx

    def _load_sequence(self, frame_indices: List[int], target_idx: int) -> Dict:
        """
        Load and process frame sequence.

        CRITICAL: Preserves 0000 handling and validity propagation.

        Args:
            frame_indices: List of frame indices to load
            target_idx: Index of prediction target within sequence

        Returns:
            Dictionary with processed sequence data
        """
        T = len(frame_indices)

        # Load frames from base dataset
        frame_data_list = [
            self.base_dataset[frame_idx]
            for frame_idx in frame_indices
        ]

        # Get sequence name
        if self.is_multi_sequence:
            sequence_name = frame_data_list[0].get('sequence_name', 'unknown')
        else:
            seq_info = self.base_dataset.get_sequence_info()
            sequence_name = seq_info['sequence_name']

        # Initialize lists
        rgb_frames = []
        sonar_frames = []
        rgb_bboxes = []
        sonar_bboxes = []
        is_valid_rgb_list = []
        is_valid_sonar_list = []
        rgb_metadata_list = []
        sonar_metadata_list = []
        frame_ids = []

        # Process each frame
        for frame_data in frame_data_list:
            # CRITICAL: Preserve 0000 handling (validity flags)
            # From tracking_dataset.py:292-293, 502-504
            is_valid_rgb_list.append(
                frame_data.get('is_valid_rgb', frame_data.get('is_valid', True))
            )
            is_valid_sonar_list.append(
                frame_data.get('is_valid_sonar', frame_data.get('is_valid', True))
            )

            # Collect frame IDs
            frame_ids.append(frame_data.get('frame_id', 0))

            # Process RGB frame (if modality enabled)
            if frame_data.get('rgb_img') is not None:
                rgb_tensor, scale_x, scale_y, pad_x, pad_y = self._prepare_image(
                    frame_data['rgb_img'],
                    self.frame_size,
                    is_sonar=False
                )
                rgb_frames.append(rgb_tensor)

                # Scale bbox with letterbox padding (Letterbox Migration 2025-11-20)
                # All images now use automatic letterbox with padding
                bbox = self._scale_bbox_with_padding(
                    frame_data['rgb_bbox'],
                    scale_x,
                    scale_y,
                    pad_x,
                    pad_y
                )
                rgb_bboxes.append(bbox)

                # Collect metadata
                original_h, original_w = frame_data['rgb_img'].shape[:2]
                rgb_metadata_list.append({
                    'original_height': original_h,
                    'original_width': original_w,
                    'scale_x': scale_x,
                    'scale_y': scale_y,
                    'pad_x': pad_x,
                    'pad_y': pad_y,
                })

            # Process Sonar frame (if modality enabled)
            if frame_data.get('sonar_img') is not None:
                sonar_tensor, scale_x, scale_y, pad_x, pad_y = self._prepare_image(
                    frame_data['sonar_img'],
                    self.frame_size,
                    is_sonar=True
                )
                sonar_frames.append(sonar_tensor)

                # Scale bbox with padding (for letterbox)
                bbox = self._scale_bbox_with_padding(
                    frame_data['sonar_bbox'],
                    scale_x,
                    scale_y,
                    pad_x,
                    pad_y
                )
                sonar_bboxes.append(bbox)

                # Collect metadata
                original_h, original_w = frame_data['sonar_img'].shape[:2]
                sonar_metadata_list.append({
                    'original_height': original_h,
                    'original_width': original_w,
                    'scale_x': scale_x,
                    'scale_y': scale_y,
                    'pad_x': pad_x,
                    'pad_y': pad_y,
                })

        # Stack to tensors
        sample = {
            # Frame sequences [T, C, H, W]
            'rgb_frames': torch.stack(rgb_frames, dim=0) if rgb_frames else None,
            'sonar_frames': torch.stack(sonar_frames, dim=0) if sonar_frames else None,

            # Bounding boxes [T, 4]
            'rgb_bboxes': torch.from_numpy(np.stack(rgb_bboxes)).float() if rgb_bboxes else torch.zeros(T, 4),
            'sonar_bboxes': torch.from_numpy(np.stack(sonar_bboxes)).float() if sonar_bboxes else torch.zeros(T, 4),

            # CRITICAL: Validity flags [T] (0000 handling)
            'is_valid_rgb': torch.tensor(is_valid_rgb_list, dtype=torch.float32),
            'is_valid_sonar': torch.tensor(is_valid_sonar_list, dtype=torch.float32),

            # Metadata
            'target_idx': target_idx,
            'sequence_name': sequence_name,
            'frame_ids': frame_ids,
            'frame_indices': frame_indices,
            'sequence_length': T,

            # Transformation metadata (List[Dict])
            'rgb_metadata': rgb_metadata_list if rgb_metadata_list else None,
            'sonar_metadata': sonar_metadata_list if sonar_metadata_list else None,
        }

        # Augmentation Order Optimization (2025-11-20):
        # Augmentation is now applied in base_dataset (before letterbox)
        # No augmentation applied here anymore

        return sample

    def _apply_augmentation(self, sample: Dict) -> Dict:
        """
        Apply temporal-consistent augmentation to sequence.

        TODO: Implement temporal-consistent augmentation wrapper.
        For now, apply per-frame augmentation (less ideal but functional).
        """
        # Placeholder: Per-frame augmentation
        # In full implementation, should apply same random transform to all T frames
        return sample

    def _prepare_image(self, img, target_size, is_sonar=False):
        """
        Prepare image using automatic letterbox strategy.

        Letterbox Migration (2025-11-20):
        Replaces distorting resize with aspect-ratio-preserving letterbox:
        - Automatically detects image aspect ratio
        - Scales uniformly to fit within target_size (preserves aspect)
        - Adds black padding to reach target_size x target_size
        - No image distortion

        Args:
            img: Input image (numpy array or None)
            target_size: Target size (square, default 640)
            is_sonar: If True, 1-channel; if False, 3-channel

        Returns:
            Tuple of (tensor, scale, scale, pad_left, pad_top)
            Note: scale_x = scale_y = scale (uniform scaling)
        """
        if img is None:
            # Return zeros for None images (CRITICAL: 0000 handling)
            num_channels = 1 if is_sonar else 3
            return torch.zeros(num_channels, target_size, target_size), 1.0, 1.0, 0, 0

        # Convert to numpy if needed
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0).cpu().numpy()
            elif img.ndim == 2:
                img = img.cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

        if not isinstance(img, np.ndarray):
            raise ValueError(f"Expected numpy array or tensor, got {type(img)}")

        original_h, original_w = img.shape[:2]

        # AUTOMATIC LETTERBOX ALGORITHM
        # Step 1: Calculate uniform scale factor (preserves aspect ratio)
        scale = min(target_size / original_w, target_size / original_h)

        # Step 2: Calculate new dimensions after scaling
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        # Step 3: Resize with uniform scale (no distortion)
        img_resized = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Step 4: Calculate padding (automatic based on aspect ratio)
        pad_w = target_size - new_w
        pad_h = target_size - new_h

        # Distribute padding equally on both sides
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Step 5: Add black padding to reach target_size x target_size
        if img_resized.ndim == 2:
            # Grayscale sonar
            img_padded = cv2.copyMakeBorder(
                img_resized,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=0  # Black padding
            )
        else:
            # Color RGB
            img_padded = cv2.copyMakeBorder(
                img_resized,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # Black padding
            )

        # Convert to tensor
        if img_padded.ndim == 2:
            # Grayscale (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_padded).unsqueeze(0).float()
        elif img_padded.ndim == 3:
            # Color (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected image shape: {img_padded.shape}")

        # Normalize to [0, 1]
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        # Return uniform scale for both axes (scale_x = scale_y = scale)
        return img_tensor, scale, scale, pad_left, pad_top

    def _scale_bbox(self, bbox, scale_x, scale_y):
        """
        Scale bounding box with uniform letterbox scaling and padding.

        Letterbox Migration (2025-11-20):
        Updated to use uniform scale factor from letterbox preprocessing.
        Since scale_x = scale_y = scale (aspect ratio preserved),
        this is now equivalent to _scale_bbox_with_padding with padding.

        Args:
            bbox: [x, y, w, h] in original coordinates
            scale_x: X-axis scale factor (uniform scale from letterbox)
            scale_y: Y-axis scale factor (same as scale_x)

        Returns:
            Scaled bbox [x, y, w, h] in letterboxed coordinates
            Note: This method assumes no padding (pad_x=0, pad_y=0)
            For images with padding, use _scale_bbox_with_padding
        """
        if bbox is None or len(bbox) != 4:
            return bbox

        x, y, w, h = bbox
        # Uniform scaling (scale_x = scale_y for letterbox)
        return np.array([
            x * scale_x,
            y * scale_y,
            w * scale_x,
            h * scale_y
        ], dtype=np.float32)

    def _scale_bbox_with_padding(self, bbox, scale_x, scale_y, pad_x, pad_y):
        """
        Scale bounding box with uniform letterbox scaling and add padding offsets.

        Letterbox Migration (2025-11-20):
        Updated for automatic letterbox preprocessing:
        - scale_x = scale_y = scale (uniform scaling preserves aspect ratio)
        - pad_x, pad_y automatically calculated based on image aspect ratio
        - Landscape images: pad_y > 0, pad_x = 0
        - Portrait images: pad_x > 0, pad_y = 0

        Args:
            bbox: [x, y, w, h] in original coordinates
            scale_x, scale_y: Uniform scale factors (scale_x = scale_y)
            pad_x, pad_y: Padding offsets in pixels (left, top)

        Returns:
            Scaled and padded bbox [x, y, w, h] in letterboxed coordinates
        """
        if bbox is None or len(bbox) != 4:
            return bbox

        x, y, w, h = bbox

        # Apply uniform scaling
        x_scaled = x * scale_x
        y_scaled = y * scale_y
        w_scaled = w * scale_x
        h_scaled = h * scale_y

        # Add automatic padding offset
        x_padded = x_scaled + pad_x
        y_padded = y_scaled + pad_y

        return np.array([
            x_padded,
            y_padded,
            w_scaled,
            h_scaled
        ], dtype=np.float32)
