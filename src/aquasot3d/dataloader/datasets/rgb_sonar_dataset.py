"""
RGB-Sonar Multi-Modal Dataset Loader

This module implements the dataset loader for paired RGB and Sonar sequences
following the RGBS50-style annotation format. It handles temporal synchronization,
occlusion detection, and multi-modal data loading.

Dataset Structure:
    data/
    ├── uuv2/                    # Training set (2,740 frames)
    │   ├── light/               # RGB camera modality
    │   │   ├── img/             # Image sequences (1.jpg - 2740.jpg)
    │   │   └── groundtruth.txt  # Bounding box annotations
    │   └── sonar/               # FLS sonar modality
    │       ├── img/             # Image sequences (1.jpg - 2740.jpg)
    │       └── groundtruth.txt  # Bounding box annotations

Annotation Format:
    Each line in groundtruth.txt: x,y,w,h
    Special case: 0,0,0,0 indicates occlusion/invisibility

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ...core.registry import DATASET_REGISTRY


class SequenceMetadata:
    """
    Metadata for a single sequence in multi-sequence dataset.

    Attributes:
        sequence_name: Name of the sequence (e.g., 'uuv2')
        sequence_path: Full path to sequence directory
        num_frames: Total number of frames in sequence
        valid_indices: List of valid frame indices (0-based within sequence)
        start_idx: Starting index in global dataset
        end_idx: Ending index in global dataset (exclusive)
        rgb_resolution: RGB image resolution (width, height)
        sonar_resolution: Sonar image resolution (width, height)
    """

    def __init__(
        self,
        sequence_name: str,
        sequence_path: Path,
        num_frames: int,
        valid_indices: List[int],
        start_idx: int,
        end_idx: int,
        rgb_resolution: Tuple[int, int],
        sonar_resolution: Tuple[int, int],
    ):
        self.sequence_name = sequence_name
        self.sequence_path = sequence_path
        self.num_frames = num_frames
        self.valid_indices = valid_indices
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.rgb_resolution = rgb_resolution
        self.sonar_resolution = sonar_resolution

    def __repr__(self):
        return (
            f"SequenceMetadata(name={self.sequence_name}, "
            f"frames={self.num_frames}, valid={len(self.valid_indices)}, "
            f"global_idx=[{self.start_idx}:{self.end_idx}], "
            f"rgb_res={self.rgb_resolution}, sonar_res={self.sonar_resolution})"
        )


def is_occluded(bbox: np.ndarray) -> bool:
    """
    Check if bounding box indicates occlusion.

    Args:
        bbox: Bounding box array [x, y, w, h]

    Returns:
        True if bbox is (0,0,0,0), indicating occlusion
    """
    return bool(np.all(bbox == 0))


def parse_groundtruth(gt_path: Union[str, Path]) -> np.ndarray:
    """
    Parse groundtruth.txt file into bounding box array.

    Args:
        gt_path: Path to groundtruth.txt file

    Returns:
        Array of shape (num_frames, 4) with format [x, y, w, h]

    Raises:
        FileNotFoundError: If groundtruth file does not exist
        ValueError: If file format is invalid
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Groundtruth file not found: {gt_path}")

    bboxes = []
    with open(gt_path, 'r') as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split(',')
                if len(parts) != 4:
                    raise ValueError(
                        f"Line {line_idx}: Expected 4 values, got {len(parts)}"
                    )
                # Parse as float to preserve sub-pixel precision
                bbox = [float(x) for x in parts]
                bboxes.append(bbox)
            except ValueError as e:
                raise ValueError(
                    f"Invalid bbox format at line {line_idx} in {gt_path}: {e}"
                )

    return np.array(bboxes, dtype=np.float32)


@DATASET_REGISTRY.register("RGBSonarDataset")
class RGBSonarDataset(Dataset):
    """
    Multi-modal RGB-Sonar dataset for underwater object tracking.

    This dataset loads paired RGB camera and Forward-Looking Sonar (FLS) images
    with synchronized annotations. It supports occlusion handling, flexible
    frame sampling, and optional data augmentation.

    Args:
        data_root: Root directory containing sequence folders (e.g., 'data/')
        sequence_name: Name of sequence folder (e.g., 'uuv2', 'uuv3')
        rgb_transform: Optional transform for RGB images
        sonar_transform: Optional transform for Sonar images
        return_occluded: If False, skip occluded frames during iteration
        start_frame: Starting frame index (1-based, default=1)
        end_frame: Ending frame index (1-based, None=all frames)

    Attributes:
        num_frames: Total number of frames in sequence
        rgb_img_dir: Path to RGB image directory
        sonar_img_dir: Path to Sonar image directory
        rgb_bboxes: RGB bounding boxes array (num_frames, 4)
        sonar_bboxes: Sonar bounding boxes array (num_frames, 4)
        valid_indices: List of non-occluded frame indices

    Example:
        >>> dataset = RGBSonarDataset(
        ...     data_root='data/',
        ...     sequence_name='uuv2',
        ...     return_occluded=False
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['rgb_img', 'sonar_img', 'rgb_bbox', 'sonar_bbox',
                   'frame_id', 'is_valid'])
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        sequence_name: str,
        modalities: List[str] = None,
        rgb_transform: Optional[callable] = None,
        sonar_transform: Optional[callable] = None,
        augmentation: Optional[callable] = None,
        return_occluded: bool = True,
        start_frame: int = 1,
        end_frame: Optional[int] = None,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.sequence_name = sequence_name

        # CRITICAL FIX (Audit 2025112003): Support single-modality training mode
        # Default to both modalities for backward compatibility
        if modalities is None:
            modalities = ['rgb', 'sonar']
        self.modalities = modalities

        # Validate modalities parameter
        valid_modalities = {'rgb', 'sonar'}
        for modality in self.modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality '{modality}'. Must be one of {valid_modalities}")

        if not self.modalities:
            raise ValueError("At least one modality must be specified")

        self.rgb_transform = rgb_transform
        self.sonar_transform = sonar_transform

        # Augmentation Order Optimization (2025-11-20):
        # Apply augmentation BEFORE letterbox (on original resolution)
        # for better augmentation quality and avoid augmenting padding
        self.augmentation = augmentation

        self.return_occluded = return_occluded

        # Construct directory paths
        self.sequence_dir = self.data_root / sequence_name

        # CRITICAL FIX (Audit 2025112003): Conditionally set up paths based on enabled modalities
        if 'rgb' in self.modalities:
            self.rgb_dir = self.sequence_dir / "light"
            self.rgb_img_dir = self.rgb_dir / "img"
        else:
            self.rgb_dir = None
            self.rgb_img_dir = None

        if 'sonar' in self.modalities:
            self.sonar_dir = self.sequence_dir / "sonar"
            self.sonar_img_dir = self.sonar_dir / "img_rect"
        else:
            self.sonar_dir = None
            self.sonar_img_dir = None

        # Validate directory structure (only for enabled modalities)
        self._validate_structure()

        # CRITICAL FIX (Audit 2025112003): Load annotations only for enabled modalities
        if 'rgb' in self.modalities:
            self.rgb_bboxes = parse_groundtruth(self.rgb_dir / "groundtruth.txt")
        else:
            self.rgb_bboxes = None

        if 'sonar' in self.modalities:
            self.sonar_bboxes = parse_groundtruth(self.sonar_dir / "groundtruth_rect.txt")
        else:
            self.sonar_bboxes = None

        # Validate frame consistency (only if both modalities enabled)
        if 'rgb' in self.modalities and 'sonar' in self.modalities:
            if len(self.rgb_bboxes) != len(self.sonar_bboxes):
                raise ValueError(
                    f"Frame count mismatch: RGB has {len(self.rgb_bboxes)} frames, "
                    f"Sonar has {len(self.sonar_bboxes)} frames"
                )
            self.num_frames = len(self.rgb_bboxes)
        elif 'rgb' in self.modalities:
            self.num_frames = len(self.rgb_bboxes)
        else:  # sonar only
            self.num_frames = len(self.sonar_bboxes)

        # Apply frame range filter
        self.start_frame = max(1, start_frame)
        self.end_frame = min(self.num_frames, end_frame) if end_frame else self.num_frames

        # Build valid frame indices (excluding occlusions if requested)
        self.valid_indices = self._build_valid_indices()

    def _validate_structure(self):
        """Validate dataset directory structure (only for enabled modalities)."""
        # CRITICAL FIX (Audit 2025112003): Only validate enabled modalities
        required_dirs = []
        required_files = []

        if 'rgb' in self.modalities:
            required_dirs.extend([self.rgb_dir, self.rgb_img_dir])
            required_files.append(self.rgb_dir / "groundtruth.txt")

        if 'sonar' in self.modalities:
            required_dirs.extend([self.sonar_dir, self.sonar_img_dir])
            required_files.append(self.sonar_dir / "groundtruth_rect.txt")

        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {dir_path}"
                )

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {file_path}"
                )

    def _build_valid_indices(self) -> List[int]:
        """
        Build list of valid frame indices.

        CRITICAL FIX (Audit 2025112003): Validity logic adapted for single-modality mode.
        In single-modality training, only check the enabled modality.

        Returns:
            List of frame indices (0-based) that are valid for training
        """
        valid_indices = []

        for idx in range(self.start_frame - 1, self.end_frame):
            # CRITICAL FIX (Audit 2025112003): Check validity based on enabled modalities
            if 'rgb' in self.modalities:
                rgb_bbox = self.rgb_bboxes[idx]
                rgb_valid = not is_occluded(rgb_bbox)
            else:
                rgb_valid = True  # Not used, assume valid

            if 'sonar' in self.modalities:
                sonar_bbox = self.sonar_bboxes[idx]
                sonar_valid = not is_occluded(sonar_bbox)
            else:
                sonar_valid = True  # Not used, assume valid

            if self.return_occluded:
                # Return all frames
                valid_indices.append(idx)
            else:
                # Only return frames where all enabled modalities are visible
                if self.modalities == ['rgb']:
                    is_valid = rgb_valid
                elif self.modalities == ['sonar']:
                    is_valid = sonar_valid
                else:  # Both modalities
                    is_valid = rgb_valid and sonar_valid

                if is_valid:
                    valid_indices.append(idx)

        return valid_indices

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.valid_indices)

    def _load_image(self, img_path: Path, is_sonar: bool = False) -> Optional[np.ndarray]:
        """
        Load image from disk.

        Args:
            img_path: Path to image file
            is_sonar: If True, load as grayscale (1-channel); if False, load as RGB (3-channel)

        Returns:
            Image array in RGB format (H, W, 3) for RGB images,
            or grayscale format (H, W, 1) for sonar images,
            or None if loading fails
        """
        if not img_path.exists():
            return None

        if is_sonar:
            # Load sonar as grayscale (Forward-Looking Sonar is single-channel intensity)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                return None

            # Add channel dimension: (H, W) -> (H, W, 1)
            img = img[:, :, np.newaxis]
        else:
            # Load RGB image
            img = cv2.imread(str(img_path))

            if img is None:
                return None

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_frame_by_id(self, frame_id: int) -> Dict:
        """
        Get frame data by 1-based frame ID.

        Args:
            frame_id: Frame ID (1-based indexing)

        Returns:
            Dictionary containing frame data

        Raises:
            IndexError: If frame_id is out of range
        """
        if frame_id < 1 or frame_id > self.num_frames:
            raise IndexError(
                f"Frame ID {frame_id} out of range [1, {self.num_frames}]"
            )

        # Convert to 0-based index
        idx = frame_id - 1
        return self._get_frame_data(idx)

    def _get_frame_data(self, idx: int) -> Dict:
        """
        Internal method to get frame data by 0-based index.

        Args:
            idx: Frame index (0-based)

        Returns:
            Dictionary with keys:
                - rgb_img: RGB image (H, W, 3) or None
                - sonar_img: Sonar image (H, W, 3) or None
                - rgb_bbox: RGB bounding box [x, y, w, h]
                - sonar_bbox: Sonar bounding box [x, y, w, h]
                - frame_id: 1-based frame ID
                - is_rgb_valid: Boolean indicating RGB validity
                - is_sonar_valid: Boolean indicating Sonar validity
                - is_valid: Boolean indicating both modalities valid
        """
        # Frame ID (1-based)
        frame_id = idx + 1

        # CRITICAL FIX (Audit 2025112003): Only load and process enabled modalities
        # This reduces I/O and memory usage in single-modality training

        # Load RGB modality if enabled
        if 'rgb' in self.modalities:
            rgb_img_path = self.rgb_img_dir / f"{frame_id}.jpg"
            rgb_img = self._load_image(rgb_img_path, is_sonar=False)
            rgb_bbox = self.rgb_bboxes[idx].copy()

            # CRITICAL FIX (Audit 2025112306 Phase 2): Continuous flow design
            # Preserve 0000 annotation (no-object), do NOT replace with dummy bbox
            # This allows network to learn "no object" concept via confidence head
            has_rgb_object = not is_occluded(rgb_bbox)
            is_rgb_valid = has_rgb_object and rgb_img is not None

            # Apply transforms (pass None bbox for no-object cases)
            if self.rgb_transform is not None and rgb_img is not None:
                rgb_img = self.rgb_transform(rgb_img, rgb_bbox if has_rgb_object else None)
        else:
            rgb_img = None
            rgb_bbox = None
            is_rgb_valid = None  # Not applicable

        # Load Sonar modality if enabled
        if 'sonar' in self.modalities:
            sonar_img_path = self.sonar_img_dir / f"{frame_id}.jpg"
            sonar_img = self._load_image(sonar_img_path, is_sonar=True)
            sonar_bbox = self.sonar_bboxes[idx].copy()

            # CRITICAL FIX (Audit 2025112306 Phase 2): Continuous flow design
            # Preserve 0000 annotation (no-object), do NOT replace with dummy bbox
            has_sonar_object = not is_occluded(sonar_bbox)
            is_sonar_valid = has_sonar_object and sonar_img is not None

            # Apply transforms (pass None bbox for no-object cases)
            if self.sonar_transform is not None and sonar_img is not None:
                sonar_img = self.sonar_transform(
                    sonar_img, sonar_bbox if has_sonar_object else None
                )
        else:
            sonar_img = None
            sonar_bbox = None
            is_sonar_valid = None  # Not applicable

        # CRITICAL FIX (Audit 2025112003): Compute combined validity based on enabled modalities
        if self.modalities == ['rgb']:
            is_valid = is_rgb_valid
        elif self.modalities == ['sonar']:
            is_valid = is_sonar_valid
        else:  # Both modalities
            is_valid = is_rgb_valid and is_sonar_valid

        # CRITICAL FIX (Audit 2025112306 Phase 3): Add has_object flags
        # Compute has_object for each modality (explicit no-object detection)
        if 'rgb' in self.modalities:
            has_object_rgb = has_rgb_object
        else:
            has_object_rgb = None  # Not applicable

        if 'sonar' in self.modalities:
            has_object_sonar = has_sonar_object
        else:
            has_object_sonar = None  # Not applicable

        # Combined has_object flag
        if self.modalities == ['rgb']:
            has_object = has_object_rgb
        elif self.modalities == ['sonar']:
            has_object = has_object_sonar
        else:  # Both modalities
            has_object = has_object_rgb and has_object_sonar

        return {
            "rgb_img": rgb_img,
            "sonar_img": sonar_img,
            "rgb_bbox": rgb_bbox,
            "sonar_bbox": sonar_bbox,
            "frame_id": frame_id,
            "is_rgb_valid": is_rgb_valid,
            "is_sonar_valid": is_sonar_valid,
            "is_valid": is_valid,
            "has_object_rgb": has_object_rgb,  # NEW: Explicit no-object detection
            "has_object_sonar": has_object_sonar,
            "has_object": has_object,
        }

    def __getitem__(self, index: int) -> Dict:
        """
        Get sample by index.

        Augmentation Order Optimization (2025-11-20):
        Applies augmentation BEFORE letterbox preprocessing (which happens in
        FrameSequenceDataset) for better quality and to avoid augmenting padding.

        Args:
            index: Index in valid_indices list (0-based)

        Returns:
            Dictionary containing frame data

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.valid_indices):
            raise IndexError(
                f"Index {index} out of range [0, {len(self.valid_indices)})"
            )

        # Map to actual frame index
        frame_idx = self.valid_indices[index]
        data = self._get_frame_data(frame_idx)

        # Apply augmentation on original resolution images (before letterbox)
        if self.augmentation is not None:
            data = self.augmentation(data)

        return data

    def get_sequence_info(self) -> Dict:
        """
        Get sequence metadata with fixed resolutions.

        Returns:
            Dictionary with sequence information including resolutions
        """
        # Fixed resolutions (preprocessed data)
        rgb_resolution = (1920, 1080)    # Standard RGB resolution
        sonar_resolution = (682, 376)    # Fixed preprocessed sonar resolution

        return {
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "num_valid_frames": len(self.valid_indices),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "rgb_resolution": rgb_resolution,
            "sonar_resolution": sonar_resolution,
        }

    def find_last_valid_bbox(self, frame_idx: int, modality: str = "both") -> Optional[np.ndarray]:
        """
        Find the last valid (non-occluded) bounding box before given frame.

        This is useful for search region generation when current frame is occluded.

        Args:
            frame_idx: Current frame index (0-based)
            modality: Which modality to check ("rgb", "sonar", or "both")

        Returns:
            Last valid bounding box [x, y, w, h], or None if not found
        """
        if modality not in ["rgb", "sonar", "both"]:
            raise ValueError(f"Invalid modality: {modality}")

        # Search backwards from frame_idx
        for idx in range(frame_idx - 1, -1, -1):
            if modality == "rgb":
                bbox = self.rgb_bboxes[idx]
                if not is_occluded(bbox):
                    return bbox.copy()
            elif modality == "sonar":
                bbox = self.sonar_bboxes[idx]
                if not is_occluded(bbox):
                    return bbox.copy()
            else:  # both
                rgb_bbox = self.rgb_bboxes[idx]
                sonar_bbox = self.sonar_bboxes[idx]
                if not is_occluded(rgb_bbox) and not is_occluded(sonar_bbox):
                    return {
                        "rgb_bbox": rgb_bbox.copy(),
                        "sonar_bbox": sonar_bbox.copy(),
                    }

        return None


@DATASET_REGISTRY.register("MultiSequenceRGBSonarDataset")
class MultiSequenceRGBSonarDataset(Dataset):
    """
    Multi-sequence RGB-Sonar dataset for underwater object tracking.

    This dataset loads multiple sequences with per-sequence metadata and resolution,
    providing a unified interface for training across all sequences while preserving
    temporal structure within each sequence.

    Args:
        data_root: Root directory containing sequence folders (e.g., 'data/')
        sequence_names: List of sequence names (e.g., ['uuv1', 'uuv2', 'uuv4'])
        rgb_transform: Optional transform for RGB images
        sonar_transform: Optional transform for Sonar images
        return_occluded: If False, skip occluded frames during iteration
        sampling_policy: Sampling strategy ('frame_balanced', 'sequence_balanced', 'weighted')
        sequence_weights: Optional weights for weighted sampling (must match sequence count)

    Attributes:
        sequences: List of SequenceMetadata objects
        sequence_datasets: Dict mapping sequence_name to RGBSonarDataset
        total_frames: Total number of valid frames across all sequences

    Example:
        >>> dataset = MultiSequenceRGBSonarDataset(
        ...     data_root='data/',
        ...     sequence_names=['uuv1', 'uuv2', 'uuv4'],
        ...     return_occluded=False,
        ...     sampling_policy='frame_balanced'
        ... )
        >>> sample = dataset[0]
        >>> print(sample['sequence_name'])
        'uuv1'
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        sequence_names: List[str],
        modalities: List[str] = None,
        rgb_transform: Optional[callable] = None,
        sonar_transform: Optional[callable] = None,
        return_occluded: bool = True,
        sampling_policy: str = "frame_balanced",
        sequence_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.sequence_names = sequence_names

        # CRITICAL FIX (Audit 2025112003): Support single-modality training mode
        if modalities is None:
            modalities = ['rgb', 'sonar']
        self.modalities = modalities

        self.rgb_transform = rgb_transform
        self.sonar_transform = sonar_transform
        self.return_occluded = return_occluded
        self.sampling_policy = sampling_policy

        if sampling_policy not in ["frame_balanced", "sequence_balanced", "weighted"]:
            raise ValueError(
                f"Invalid sampling_policy: {sampling_policy}. "
                "Must be one of: frame_balanced, sequence_balanced, weighted"
            )

        if sampling_policy == "weighted":
            if sequence_weights is None:
                raise ValueError("sequence_weights must be provided for weighted sampling")
            if len(sequence_weights) != len(sequence_names):
                raise ValueError(
                    f"sequence_weights length ({len(sequence_weights)}) must match "
                    f"sequence_names length ({len(sequence_names)})"
                )
            self.sequence_weights = sequence_weights
        else:
            self.sequence_weights = None

        # Initialize sequence datasets and metadata
        self.sequence_datasets = {}
        self.sequences = []
        self._build_sequence_metadata()

        # Build global index mapping
        self._build_global_index()

        self.total_frames = sum(len(seq.valid_indices) for seq in self.sequences)

    def _build_sequence_metadata(self):
        """Build metadata for all sequences."""
        global_idx = 0

        # Fixed resolutions (preprocessed data)
        RGB_RESOLUTION = (1920, 1080)    # Standard RGB resolution
        SONAR_RESOLUTION = (682, 376)    # Fixed preprocessed sonar resolution

        for seq_name in self.sequence_names:
            # CRITICAL FIX (Audit 2025112003): Pass modalities parameter to child datasets
            # Create single-sequence dataset
            seq_dataset = RGBSonarDataset(
                data_root=self.data_root,
                sequence_name=seq_name,
                modalities=self.modalities,
                rgb_transform=self.rgb_transform,
                sonar_transform=self.sonar_transform,
                return_occluded=self.return_occluded,
            )

            # Store dataset
            self.sequence_datasets[seq_name] = seq_dataset

            # Create metadata with fixed resolutions
            num_valid = len(seq_dataset.valid_indices)
            metadata = SequenceMetadata(
                sequence_name=seq_name,
                sequence_path=self.data_root / seq_name,
                num_frames=seq_dataset.num_frames,
                valid_indices=seq_dataset.valid_indices,
                start_idx=global_idx,
                end_idx=global_idx + num_valid,
                rgb_resolution=RGB_RESOLUTION,
                sonar_resolution=SONAR_RESOLUTION,
            )

            self.sequences.append(metadata)
            global_idx += num_valid

    def _build_global_index(self):
        """Build global index to sequence mapping."""
        self.index_to_sequence = []

        for seq_metadata in self.sequences:
            for local_idx in range(len(seq_metadata.valid_indices)):
                self.index_to_sequence.append((seq_metadata.sequence_name, local_idx))

    def __len__(self) -> int:
        """Return total number of valid samples across all sequences."""
        return self.total_frames

    def __getitem__(self, index: int) -> Dict:
        """
        Get sample by global index.

        Args:
            index: Global index across all sequences (0-based)

        Returns:
            Dictionary containing frame data with additional 'sequence_name' field

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= self.total_frames:
            raise IndexError(
                f"Index {index} out of range [0, {self.total_frames})"
            )

        # Map global index to sequence and local index
        sequence_name, local_idx = self.index_to_sequence[index]

        # Get sample from sequence dataset
        seq_dataset = self.sequence_datasets[sequence_name]
        sample = seq_dataset[local_idx]

        # Add sequence name to sample
        sample["sequence_name"] = sequence_name

        return sample

    def get_sequence_info(self, sequence_name: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Get sequence metadata.

        Args:
            sequence_name: If provided, return info for that sequence only.
                          If None, return info for all sequences.

        Returns:
            Dictionary with sequence information, or list of dictionaries for all sequences
        """
        if sequence_name is not None:
            # Return info for specific sequence
            for seq_metadata in self.sequences:
                if seq_metadata.sequence_name == sequence_name:
                    return {
                        "sequence_name": seq_metadata.sequence_name,
                        "num_frames": seq_metadata.num_frames,
                        "num_valid_frames": len(seq_metadata.valid_indices),
                        "rgb_resolution": seq_metadata.rgb_resolution,
                        "sonar_resolution": seq_metadata.sonar_resolution,
                        "global_range": (seq_metadata.start_idx, seq_metadata.end_idx),
                    }
            raise ValueError(f"Sequence not found: {sequence_name}")
        else:
            # Return info for all sequences
            return [
                {
                    "sequence_name": seq.sequence_name,
                    "num_frames": seq.num_frames,
                    "num_valid_frames": len(seq.valid_indices),
                    "rgb_resolution": seq.rgb_resolution,
                    "sonar_resolution": seq.sonar_resolution,
                    "global_range": (seq.start_idx, seq.end_idx),
                }
                for seq in self.sequences
            ]

    def get_frame_sequence(self, frame_index: int) -> str:
        """
        Get sequence name for a given global frame index.

        Args:
            frame_index: Global frame index

        Returns:
            Sequence name
        """
        sequence_name, _ = self.index_to_sequence[frame_index]
        return sequence_name

    def get_sequence_frames(self, sequence_name: str) -> List[int]:
        """
        Get global indices of all frames in a specific sequence.

        Args:
            sequence_name: Name of the sequence

        Returns:
            List of global frame indices belonging to that sequence
        """
        for seq_metadata in self.sequences:
            if seq_metadata.sequence_name == sequence_name:
                return list(range(seq_metadata.start_idx, seq_metadata.end_idx))

        raise ValueError(f"Sequence not found: {sequence_name}")