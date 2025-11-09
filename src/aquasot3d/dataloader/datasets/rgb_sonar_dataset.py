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
                bbox = [int(x) for x in parts]
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
        rgb_transform: Optional[callable] = None,
        sonar_transform: Optional[callable] = None,
        return_occluded: bool = True,
        start_frame: int = 1,
        end_frame: Optional[int] = None,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.sequence_name = sequence_name
        self.rgb_transform = rgb_transform
        self.sonar_transform = sonar_transform
        self.return_occluded = return_occluded

        # Construct directory paths
        self.sequence_dir = self.data_root / sequence_name
        self.rgb_dir = self.sequence_dir / "light"
        self.sonar_dir = self.sequence_dir / "sonar"
        self.rgb_img_dir = self.rgb_dir / "img"
        self.sonar_img_dir = self.sonar_dir / "img"

        # Validate directory structure
        self._validate_structure()

        # Load annotations
        self.rgb_bboxes = parse_groundtruth(self.rgb_dir / "groundtruth.txt")
        self.sonar_bboxes = parse_groundtruth(self.sonar_dir / "groundtruth.txt")

        # Validate frame consistency
        if len(self.rgb_bboxes) != len(self.sonar_bboxes):
            raise ValueError(
                f"Frame count mismatch: RGB has {len(self.rgb_bboxes)} frames, "
                f"Sonar has {len(self.sonar_bboxes)} frames"
            )

        self.num_frames = len(self.rgb_bboxes)

        # Apply frame range filter
        self.start_frame = max(1, start_frame)
        self.end_frame = min(self.num_frames, end_frame) if end_frame else self.num_frames

        # Build valid frame indices (excluding occlusions if requested)
        self.valid_indices = self._build_valid_indices()

    def _validate_structure(self):
        """Validate dataset directory structure."""
        required_dirs = [
            self.rgb_dir,
            self.sonar_dir,
            self.rgb_img_dir,
            self.sonar_img_dir,
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {dir_path}"
                )

        required_files = [
            self.rgb_dir / "groundtruth.txt",
            self.sonar_dir / "groundtruth.txt",
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {file_path}"
                )

    def _build_valid_indices(self) -> List[int]:
        """
        Build list of valid frame indices.

        Returns:
            List of frame indices (0-based) that are valid for training
        """
        valid_indices = []

        for idx in range(self.start_frame - 1, self.end_frame):
            rgb_bbox = self.rgb_bboxes[idx]
            sonar_bbox = self.sonar_bboxes[idx]

            # Check if both modalities are valid
            rgb_valid = not is_occluded(rgb_bbox)
            sonar_valid = not is_occluded(sonar_bbox)

            if self.return_occluded:
                # Return all frames
                valid_indices.append(idx)
            else:
                # Only return frames where both modalities are visible
                if rgb_valid and sonar_valid:
                    valid_indices.append(idx)

        return valid_indices

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.valid_indices)

    def _load_image(self, img_path: Path) -> Optional[np.ndarray]:
        """
        Load image from disk.

        Args:
            img_path: Path to image file

        Returns:
            Image array in RGB format (H, W, 3), or None if loading fails
        """
        if not img_path.exists():
            return None

        # Load image in BGR format (OpenCV default)
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

        # Load images
        rgb_img_path = self.rgb_img_dir / f"{frame_id}.jpg"
        sonar_img_path = self.sonar_img_dir / f"{frame_id}.jpg"

        rgb_img = self._load_image(rgb_img_path)
        sonar_img = self._load_image(sonar_img_path)

        # Get bounding boxes
        rgb_bbox = self.rgb_bboxes[idx].copy()
        sonar_bbox = self.sonar_bboxes[idx].copy()

        # Check validity
        is_rgb_valid = not is_occluded(rgb_bbox) and rgb_img is not None
        is_sonar_valid = not is_occluded(sonar_bbox) and sonar_img is not None
        is_valid = is_rgb_valid and is_sonar_valid

        # Apply transforms
        if self.rgb_transform is not None and rgb_img is not None:
            rgb_img = self.rgb_transform(rgb_img, rgb_bbox if is_rgb_valid else None)

        if self.sonar_transform is not None and sonar_img is not None:
            sonar_img = self.sonar_transform(
                sonar_img, sonar_bbox if is_sonar_valid else None
            )

        return {
            "rgb_img": rgb_img,
            "sonar_img": sonar_img,
            "rgb_bbox": rgb_bbox,
            "sonar_bbox": sonar_bbox,
            "frame_id": frame_id,
            "is_rgb_valid": is_rgb_valid,
            "is_sonar_valid": is_sonar_valid,
            "is_valid": is_valid,
        }

    def __getitem__(self, index: int) -> Dict:
        """
        Get sample by index.

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
        return self._get_frame_data(frame_idx)

    def get_sequence_info(self) -> Dict:
        """
        Get sequence metadata.

        Returns:
            Dictionary with sequence information
        """
        return {
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "num_valid_frames": len(self.valid_indices),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "rgb_resolution": (1920, 1080),  # Standard RGB resolution
            "sonar_resolution": (682, 376),   # Standard Sonar resolution
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