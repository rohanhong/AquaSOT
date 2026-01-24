"""
Base video dataset class.

Copied from OSTrack with minimal modifications for AquaSOT.
"""

import torch.utils.data
import cv2


def default_image_loader(path):
    """Default image loader using OpenCV."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class BaseVideoDataset(torch.utils.data.Dataset):
    """Base class for video datasets.

    Provides the interface for loading sequences and frames.
    Following OSTrack's BaseVideoDataset pattern.
    """

    def __init__(self, name, root, image_loader=default_image_loader):
        """
        Args:
            name: Dataset name
            root: Root path to the dataset
            image_loader: Function to load images
        """
        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.sequence_list = []  # List of sequence names/ids
        self.class_list = []  # List of object classes (if applicable)

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return self.get_num_sequences()

    def __getitem__(self, index):
        """Not to be used directly. Use get_frames() instead."""
        return None

    def is_video_sequence(self):
        """Returns whether this is a video dataset (not image dataset)."""
        return True

    def get_name(self):
        """Returns dataset name."""
        return self.name

    def get_num_sequences(self):
        """Returns number of sequences."""
        return len(self.sequence_list)

    def has_class_info(self):
        """Returns whether class information is available."""
        return len(self.class_list) > 0

    def get_num_classes(self):
        """Returns number of object classes."""
        return len(self.class_list)

    def get_class_list(self):
        """Returns list of object classes."""
        return self.class_list

    def get_sequence_info(self, seq_id):
        """Returns information about a sequence.

        Args:
            seq_id: Sequence index

        Returns:
            dict with:
                - bbox: Bounding boxes for all frames [N, 4]
                - valid: Boolean mask for valid frames [N]
                - visible: Boolean mask for visible frames [N]
        """
        raise NotImplementedError

    def get_frames(self, seq_id, frame_ids, anno=None):
        """Get frames from a sequence.

        Args:
            seq_id: Sequence index
            frame_ids: List of frame indices to load
            anno: Pre-loaded annotation (optional)

        Returns:
            list: List of images
            dict: Annotations for the frames
            dict: Metadata about the sequence
        """
        raise NotImplementedError
