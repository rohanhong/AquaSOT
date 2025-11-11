"""
DataLoader Utilities

This module provides DataLoader utilities and collate functions for
multi-modal RGB-Sonar training, handling:
- Batch collation for variable-length sequences
- Multi-worker data loading
- Distributed sampler support
- Pin memory optimization

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np


def collate_fn_tracking(batch: List[Dict]) -> Dict:
    """
    Collate function for tracking samples.

    Handles batching of multi-modal tracking data with potential None values
    for occluded frames.

    Args:
        batch: List of sample dictionaries from dataset

    Returns:
        Batched dictionary with stacked tensors

    Sample structure:
        batch[i] = {
            'rgb_img': Tensor or None,
            'sonar_img': Tensor or None,
            'rgb_bbox': ndarray,
            'sonar_bbox': ndarray,
            'frame_id': int,
            'is_valid': bool,
            ...
        }
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    # Initialize batched data
    batched = {}

    # Handle image tensors (may have None values)
    for key in ['rgb_img', 'sonar_img']:
        if key in batch[0]:
            imgs = [sample[key] for sample in batch if sample[key] is not None]

            if len(imgs) > 0:
                # Check if already tensors
                if isinstance(imgs[0], torch.Tensor):
                    batched[key] = torch.stack(imgs, dim=0)
                else:
                    # Convert numpy arrays to tensors
                    imgs_tensor = [torch.from_numpy(img) for img in imgs]
                    batched[key] = torch.stack(imgs_tensor, dim=0)
            else:
                batched[key] = None

    # Handle bounding boxes
    for key in ['rgb_bbox', 'sonar_bbox']:
        if key in batch[0]:
            bboxes = [sample[key] for sample in batch]

            # Convert to tensor
            if isinstance(bboxes[0], np.ndarray):
                bboxes_tensor = torch.from_numpy(np.stack(bboxes, axis=0))
            else:
                bboxes_tensor = torch.stack([torch.tensor(bbox) for bbox in bboxes], dim=0)

            batched[key] = bboxes_tensor

    # Handle scalar values
    for key in ['frame_id', 'is_valid', 'is_rgb_valid', 'is_sonar_valid']:
        if key in batch[0]:
            values = [sample[key] for sample in batch]
            batched[key] = torch.tensor(values)

    return batched


def collate_fn_template_search(batch: List[Dict]) -> Dict:
    """
    Collate function for template-search pairs.

    Args:
        batch: List of template-search sample dictionaries

    Returns:
        Batched dictionary with stacked tensors

    Sample structure (TrackingDataset format):
        batch[i] = {
            'rgb_template': Tensor (C, H_t, W_t),
            'rgb_search': Tensor (C, H_s, W_s),
            'sonar_template': Tensor (C, H_t, W_t),
            'sonar_search': Tensor (C, H_s, W_s),
            'template_bbox_rgb': ndarray [x, y, w, h],
            'template_bbox_sonar': ndarray [x, y, w, h],
            'search_bbox_rgb': ndarray [x, y, w, h],
            'search_bbox_sonar': ndarray [x, y, w, h],
            'is_valid': bool,
            ...
        }
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    batched = {}

    # Handle image tensors (trainer expects: rgb_template, rgb_search, sonar_template, sonar_search)
    image_keys = ['rgb_template', 'rgb_search', 'sonar_template', 'sonar_search']
    for key in image_keys:
        if key in batch[0]:
            imgs = [sample[key] for sample in batch]

            # Stack tensors (already in CHW format from TrackingDataset)
            if isinstance(imgs[0], torch.Tensor):
                batched[key] = torch.stack(imgs, dim=0)
            elif isinstance(imgs[0], np.ndarray):
                # Convert numpy arrays to tensors
                imgs_tensor = [torch.from_numpy(img) for img in imgs]
                batched[key] = torch.stack(imgs_tensor, dim=0)

    # Handle bounding boxes with normalization
    # Critical fix: Normalize pixel coordinates to [0, 1] range for numerical stability
    # This prevents gradient explosion in L1/GIoU losses
    bbox_keys = [
        'template_bbox_rgb', 'template_bbox_sonar',
        'search_bbox_rgb', 'search_bbox_sonar',
        'search_region_rgb', 'search_region_sonar',
        'rgb_boxes', 'sonar_boxes'  # Trainer-expected keys for loss computation
    ]

    # Get image dimensions for normalization
    # RGB search images are typically 255x255 or 256x256
    # Template images are typically 127x127
    rgb_search_h, rgb_search_w = batch[0]['rgb_search'].shape[-2:] if 'rgb_search' in batch[0] else (256, 256)
    sonar_search_h, sonar_search_w = batch[0]['sonar_search'].shape[-2:] if 'sonar_search' in batch[0] else (256, 256)

    for key in bbox_keys:
        if key in batch[0]:
            bboxes = [sample[key] for sample in batch]

            if isinstance(bboxes[0], np.ndarray):
                batched[key] = torch.from_numpy(np.stack(bboxes, axis=0).astype(np.float32))
            else:
                batched[key] = torch.stack([torch.tensor(bbox, dtype=torch.float32) for bbox in bboxes], dim=0)

            # Normalize bounding boxes to [0, 1] based on image dimensions
            # Format: [x, y, w, h] in pixels → [cx, cy, w, h] in [0, 1]
            if 'rgb' in key and not key.endswith('_normalized'):
                # RGB bounding boxes
                img_w, img_h = rgb_search_w, rgb_search_h
            elif 'sonar' in key and not key.endswith('_normalized'):
                # Sonar bounding boxes
                img_w, img_h = sonar_search_w, sonar_search_h
            else:
                # Generic fallback (256x256)
                img_w, img_h = 256.0, 256.0

            # Convert [x, y, w, h] to [cx, cy, w, h] and normalize
            # cx = (x + w/2) / img_w, cy = (y + h/2) / img_h
            # w_norm = w / img_w, h_norm = h / img_h
            bbox_tensor = batched[key].clone()
            x, y, w, h = bbox_tensor[:, 0], bbox_tensor[:, 1], bbox_tensor[:, 2], bbox_tensor[:, 3]

            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Stack normalized coordinates [cx, cy, w, h] in [0, 1]
            batched[key] = torch.stack([cx, cy, w_norm, h_norm], dim=-1)

            # Clamp to [0, 1] for safety (handles edge cases)
            batched[key] = torch.clamp(batched[key], min=0.0, max=1.0)

    # Handle scalar values
    scalar_keys = ['template_idx', 'search_idx', 'is_valid', 'is_rgb_valid', 'is_sonar_valid']
    for key in scalar_keys:
        if key in batch[0]:
            values = [sample[key] for sample in batch]
            batched[key] = torch.tensor(values)

    return batched


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[callable] = None,
    distributed: bool = False,
) -> DataLoader:
    """
    Create DataLoader with standard configurations.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data (ignored if distributed=True)
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        collate_fn: Custom collate function (default: collate_fn_tracking)
        distributed: Use distributed sampler for multi-GPU training

    Returns:
        DataLoader instance
    """
    if collate_fn is None:
        collate_fn = collate_fn_tracking

    # Create sampler for distributed training
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,  # Keep workers alive
    )

    return loader


class InfiniteDataLoader:
    """
    Wrapper for DataLoader that cycles indefinitely.

    Useful for training loops that iterate by steps rather than epochs.

    Args:
        dataloader: DataLoader to wrap

    Example:
        >>> loader = create_dataloader(dataset, batch_size=32)
        >>> infinite_loader = InfiniteDataLoader(loader)
        >>> for i, batch in enumerate(infinite_loader):
        ...     if i >= max_steps:
        ...         break
        ...     # Training step
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.iterator is None:
                self.iterator = iter(self.dataloader)
            return next(self.iterator)
        except StopIteration:
            # Reset iterator and continue
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def __len__(self):
        return len(self.dataloader)


class PrefetchLoader:
    """
    DataLoader wrapper with GPU prefetching for faster training.

    Pre-loads next batch to GPU while current batch is being processed.

    Args:
        dataloader: DataLoader to wrap
        device: Target device for prefetching

    Example:
        >>> loader = create_dataloader(dataset, batch_size=32)
        >>> prefetch_loader = PrefetchLoader(loader, device='cuda:0')
        >>> for batch in prefetch_loader:
        ...     # Batch is already on GPU
        ...     output = model(batch['rgb_img'])
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        self.loader_iter = iter(self.dataloader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration

        # Preload next batch
        self.preload()
        return batch

    def __len__(self):
        return len(self.dataloader)

    def preload(self):
        """Preload next batch to GPU."""
        try:
            self.next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self._move_to_device(self.next_batch)

    def _move_to_device(self, data):
        """
        Recursively move data to device.

        Args:
            data: Data to move (dict, list, tensor, or other)

        Returns:
            Data on target device
        """
        if isinstance(data, dict):
            return {key: self._move_to_device(val) for key, val in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        else:
            return data


def worker_init_fn(worker_id: int):
    """
    Worker initialization function for reproducibility.

    Sets different random seeds for each worker to ensure diversity
    while maintaining reproducibility.

    Args:
        worker_id: Worker process ID
    """
    import random
    import numpy as np

    # Get base seed from PyTorch
    worker_seed = torch.initial_seed() % 2**32

    # Set seeds for this worker
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def create_training_loaders(
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
) -> tuple:
    """
    Create training and validation DataLoaders with standard configurations.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        distributed: Use distributed sampling

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_template_search,
        distributed=distributed,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_template_search,
        distributed=False,  # No distributed for validation
    )

    return train_loader, val_loader