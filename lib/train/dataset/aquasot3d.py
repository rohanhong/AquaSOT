"""
AquaSOT-3D Dataset for RGB-Sonar paired video tracking.

Following OSTrack's BaseVideoDataset pattern.

Actual data structure:
    root/
        uuv1/
            light/
                groundtruth.txt
                img/
                    1000.jpg, 1001.jpg, ...
            sonar/
                groundtruth.txt       # original polar coordinates
                groundtruth_rect.txt  # rect-converted coordinates (used)
                img/                  # original polar images
                img_rect/             # rect-converted images (used)
                    1000.jpg, 1001.jpg, ...
"""

import os
import json
import torch
import numpy as np
import cv2

from .base_video_dataset import BaseVideoDataset, default_image_loader


def sonar_image_loader(path):
    """Load sonar image as grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load sonar image: {path}")
    return img


class AquaSOT3DDataset(BaseVideoDataset):
    """RGB-Sonar paired video dataset for AquaSOT-3D.

    Supports actual AquaSOT-3D data structure:
        - RGB: light/img/ + light/groundtruth.txt
        - Sonar: sonar/img_rect/ + sonar/groundtruth_rect.txt (rect-converted)
    """

    def __init__(self, root, sequence_names=None, image_loader=default_image_loader,
                 sonar_loader=sonar_image_loader, use_rect_sonar=True):
        """
        Args:
            root: Root directory containing sequence folders
            sequence_names: List of sequence names to use. If None, use all.
            image_loader: Function to load RGB images
            sonar_loader: Function to load sonar images
            use_rect_sonar: If True, use rect-converted sonar (img_rect/groundtruth_rect.txt)
        """
        super().__init__('aquasot3d', root, image_loader)
        self.sonar_loader = sonar_loader
        self.use_rect_sonar = use_rect_sonar

        # Build sequence list
        if sequence_names is not None:
            self.sequence_list = sequence_names
        else:
            self.sequence_list = self._get_all_sequences()

        # Cache sequence info
        self._seq_info_cache = {}

    def _get_all_sequences(self):
        """Get all sequence names from root directory."""
        sequences = []
        if os.path.exists(self.root):
            for name in sorted(os.listdir(self.root)):
                seq_path = os.path.join(self.root, name)
                if os.path.isdir(seq_path):
                    # Check if it has both light and sonar folders
                    rgb_path = os.path.join(seq_path, 'light', 'img')
                    sonar_path = os.path.join(seq_path, 'sonar', 'img_rect' if self.use_rect_sonar else 'img')
                    if os.path.exists(rgb_path) and os.path.exists(sonar_path):
                        sequences.append(name)
        return sequences

    def _get_sequence_path(self, seq_id):
        """Get path to sequence folder."""
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def _load_groundtruth(self, seq_path, modality):
        """Load groundtruth bounding boxes.

        Args:
            seq_path: Path to sequence folder
            modality: 'rgb' or 'sonar'

        Returns:
            numpy array of shape [N, 4] with [x, y, w, h] format
        """
        if modality == 'rgb':
            # RGB: light/groundtruth.txt
            gt_file = os.path.join(seq_path, 'light', 'groundtruth.txt')
        else:
            # Sonar: sonar/groundtruth_rect.txt or sonar/groundtruth.txt
            if self.use_rect_sonar:
                gt_file = os.path.join(seq_path, 'sonar', 'groundtruth_rect.txt')
            else:
                gt_file = os.path.join(seq_path, 'sonar', 'groundtruth.txt')

        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"Groundtruth file not found: {gt_file}")

        # Load groundtruth (comma or space separated)
        gt = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Try comma separator first, then space
                if ',' in line:
                    values = [float(x) for x in line.split(',')]
                else:
                    values = [float(x) for x in line.split()]
                gt.append(values[:4])  # Take first 4 values [x, y, w, h]

        return np.array(gt, dtype=np.float32)

    def _get_frame_list(self, seq_path, modality):
        """Get numerically sorted list of frame paths.

        IMPORTANT: Files are named 1.jpg, 2.jpg, ... without leading zeros.
        Must use numerical sorting to match groundtruth line order.
        """
        if modality == 'rgb':
            # RGB: light/img/
            folder = os.path.join(seq_path, 'light', 'img')
        else:
            # Sonar: sonar/img_rect/ or sonar/img/
            if self.use_rect_sonar:
                folder = os.path.join(seq_path, 'sonar', 'img_rect')
            else:
                folder = os.path.join(seq_path, 'sonar', 'img')

        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        # Collect valid image files
        valid_files = []
        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                valid_files.append(fname)

        # Sort numerically by extracting the number from filename
        # e.g., "123.jpg" -> 123
        def extract_frame_number(fname):
            try:
                return int(os.path.splitext(fname)[0])
            except ValueError:
                return float('inf')  # Put non-numeric files at the end

        valid_files.sort(key=extract_frame_number)

        # Build full paths
        frames = [os.path.join(folder, fname) for fname in valid_files]

        return frames

    def get_sequence_info(self, seq_id):
        """Get sequence information.

        Args:
            seq_id: Sequence index

        Returns:
            dict with:
                - rgb_bbox: RGB bounding boxes [N, 4]
                - sonar_bbox: Sonar bounding boxes [N, 4]
                - valid: Valid frame mask [N]
                - visible: Visible frame mask [N] (alias for valid in this dataset)
                - rgb_frames: List of RGB frame paths
                - sonar_frames: List of sonar frame paths
        """
        # Check cache
        if seq_id in self._seq_info_cache:
            return self._seq_info_cache[seq_id]

        seq_path = self._get_sequence_path(seq_id)

        # Load groundtruth
        rgb_bbox = self._load_groundtruth(seq_path, 'rgb')
        sonar_bbox = self._load_groundtruth(seq_path, 'sonar')

        # Get frame lists
        rgb_frames = self._get_frame_list(seq_path, 'rgb')
        sonar_frames = self._get_frame_list(seq_path, 'sonar')

        # Ensure matching frame counts
        n_frames = min(len(rgb_bbox), len(sonar_bbox), len(rgb_frames), len(sonar_frames))
        rgb_bbox = rgb_bbox[:n_frames]
        sonar_bbox = sonar_bbox[:n_frames]
        rgb_frames = rgb_frames[:n_frames]
        sonar_frames = sonar_frames[:n_frames]

        # Compute valid mask (bbox with positive area)
        rgb_valid = (rgb_bbox[:, 2] > 0) & (rgb_bbox[:, 3] > 0)
        sonar_valid = (sonar_bbox[:, 2] > 0) & (sonar_bbox[:, 3] > 0)
        valid = rgb_valid & sonar_valid

        seq_info = {
            'rgb_bbox': torch.from_numpy(rgb_bbox),
            'sonar_bbox': torch.from_numpy(sonar_bbox),
            'valid': torch.from_numpy(valid),
            'visible': torch.from_numpy(valid),  # Alias
            'rgb_frames': rgb_frames,
            'sonar_frames': sonar_frames,
        }

        self._seq_info_cache[seq_id] = seq_info
        return seq_info

    def get_frames(self, seq_id, frame_ids, anno=None):
        """Get RGB and Sonar frames from a sequence.

        Args:
            seq_id: Sequence index
            frame_ids: List of frame indices to load
            anno: Pre-loaded annotation (optional)

        Returns:
            rgb_frames: List of RGB images
            sonar_frames: List of sonar images
            rgb_anno: dict with 'bbox' key containing RGB boxes
            sonar_anno: dict with 'bbox' key containing Sonar boxes
            meta: Metadata dict
        """
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Load RGB frames
        rgb_frames = []
        for fid in frame_ids:
            frame_path = anno['rgb_frames'][fid]
            img = self.image_loader(frame_path)
            rgb_frames.append(img)

        # Load Sonar frames
        sonar_frames = []
        for fid in frame_ids:
            frame_path = anno['sonar_frames'][fid]
            img = self.sonar_loader(frame_path)
            sonar_frames.append(img)

        # Get annotations
        rgb_bbox = [anno['rgb_bbox'][fid] for fid in frame_ids]
        sonar_bbox = [anno['sonar_bbox'][fid] for fid in frame_ids]

        rgb_anno = {'bbox': rgb_bbox}
        sonar_anno = {'bbox': sonar_bbox}

        meta = {
            'sequence_name': self.sequence_list[seq_id],
            'object_class_name': 'underwater_object',
        }

        return rgb_frames, sonar_frames, rgb_anno, sonar_anno, meta


class AquaSOT3DDatasetWrapper(BaseVideoDataset):
    """Wrapper that returns combined RGB-Sonar data in OSTrack-compatible format.

    This wrapper adapts the dual-modal AquaSOT3DDataset to work with
    OSTrack-style samplers that expect single-modal input.
    """

    def __init__(self, aquasot3d_dataset):
        """
        Args:
            aquasot3d_dataset: AquaSOT3DDataset instance
        """
        self._dataset = aquasot3d_dataset
        super().__init__(
            name=aquasot3d_dataset.name,
            root=aquasot3d_dataset.root,
            image_loader=aquasot3d_dataset.image_loader
        )
        self.sequence_list = aquasot3d_dataset.sequence_list

    def get_sequence_info(self, seq_id):
        """Get sequence info in OSTrack-compatible format."""
        info = self._dataset.get_sequence_info(seq_id)
        # Combine RGB and Sonar validity
        return {
            'bbox': info['rgb_bbox'],  # Use RGB as primary
            'valid': info['valid'],
            'visible': info['visible'],
        }

    def get_frames(self, seq_id, frame_ids, anno=None):
        """Get frames in format compatible with dual-modal processing."""
        rgb_frames, sonar_frames, rgb_anno, sonar_anno, meta = \
            self._dataset.get_frames(seq_id, frame_ids, anno)

        # Return in a format that can be processed by MultiModalProcessing
        return {
            'rgb_frames': rgb_frames,
            'sonar_frames': sonar_frames,
            'rgb_anno': rgb_anno,
            'sonar_anno': sonar_anno,
            'meta': meta,
        }
