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
from torch.utils.data import DataLoader, DistributedSampler, Sampler
import numpy as np
import random
from collections import defaultdict


class SequenceGroupedBatchSampler(Sampler):
    """
    Sampler that groups samples by sequence to enable temporal state persistence.

    CRITICAL FIX (Audit 2025111703 Finding 3):
    Groups samples from the same sequence into consecutive batches, so the trainer's
    smart reset strategy can maintain temporal state across multiple batches from
    the same sequence.

    Without this sampler, random shuffling mixes sequences in every batch, causing
    the trainer to reset temporal state before each optimizer step, preventing
    the EMA aggregator from learning.

    Args:
        dataset: Dataset with sequence_name metadata
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle sequences (default: True)
        drop_last: Drop last incomplete batch (default: False)
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Group indices by sequence
        self.sequence_to_indices = defaultdict(list)

        # Check if dataset has sequence information
        if hasattr(dataset, 'get_sequence_name'):
            # Dataset with get_sequence_name method (e.g., FrameSequenceDataset)
            for idx in range(len(dataset)):
                try:
                    seq_name = dataset.get_sequence_name(idx)
                    self.sequence_to_indices[seq_name].append(idx)
                except (AttributeError, IndexError, KeyError) as e:
                    # Fallback: put in default sequence
                    self.sequence_to_indices['default'].append(idx)
        else:
            # Single sequence or unknown: treat all as one sequence
            self.sequence_to_indices['default'] = list(range(len(dataset)))

    def __iter__(self):
        # Get sequence names
        sequences = list(self.sequence_to_indices.keys())

        if self.shuffle:
            # Shuffle sequence order
            random.shuffle(sequences)

        # Generate batches by iterating through sequences
        for seq_name in sequences:
            indices = self.sequence_to_indices[seq_name].copy()

            if self.shuffle:
                # Shuffle within sequence
                random.shuffle(indices)

            # Generate batches from this sequence
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                # Skip incomplete batches if drop_last
                if len(batch) < self.batch_size and self.drop_last:
                    continue

                yield batch

    def __len__(self):
        total_batches = 0
        for indices in self.sequence_to_indices.values():
            num_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                num_batches += 1
            total_batches += num_batches
        return total_batches


def collate_fn_sequence(batch: List[Dict]) -> Dict:
    """
    Collate function for frame sequence datasets.

    ARCHITECTURE CHANGE (Audit 2025112006):
    Designed for FrameSequenceDataset which returns continuous frame sequences
    instead of template-search pairs.

    Args:
        batch: List of samples from FrameSequenceDataset
            Each sample: {
                'rgb_frames': [T, 3, H, W] or None,
                'sonar_frames': [T, 1, H, W] or None,
                'rgb_bboxes': [T, 4],
                'sonar_bboxes': [T, 4],
                'is_valid_rgb': [T],
                'is_valid_sonar': [T],
                'target_idx': int,
                'sequence_name': str,
                ...
            }

    Returns:
        Batched dictionary: {
            'rgb_frames': [B, T, 3, H, W] or None,
            'sonar_frames': [B, T, 1, H, W] or None,
            'rgb_bboxes': [B, T, 4],
            'sonar_bboxes': [B, T, 4],
            'is_valid_rgb': [B, T],
            'is_valid_sonar': [B, T],
            'target_idx': [B],
            ...
        }

    CRITICAL: Preserves validity flag batching for 0000 handling (Audit 2025112001)
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    batched = {}

    # Stack frame sequences
    # CRITICAL FIX (Audit 2025112003): Handle single-modality mode (None frames)
    if 'rgb_frames' in batch[0]:
        rgb_frames_list = [sample['rgb_frames'] for sample in batch if sample['rgb_frames'] is not None]
        if len(rgb_frames_list) > 0:
            batched['rgb_frames'] = torch.stack(rgb_frames_list, dim=0)  # [B, T, 3, H, W]
        else:
            batched['rgb_frames'] = None

    if 'sonar_frames' in batch[0]:
        sonar_frames_list = [sample['sonar_frames'] for sample in batch if sample['sonar_frames'] is not None]
        if len(sonar_frames_list) > 0:
            batched['sonar_frames'] = torch.stack(sonar_frames_list, dim=0)  # [B, T, 1, H, W]
        else:
            batched['sonar_frames'] = None

    # Stack bounding boxes [B, T, 4]
    # CRITICAL FIX (Bbox Format Audit 2025-11-20): Convert and normalize bboxes
    # Old collate_fn_template_search did this, but collate_fn_sequence was missing it.
    # Without this, loss receives XYWH pixel instead of CXCYWH normalized → training broken
    bbox_keys = ['rgb_bboxes', 'sonar_bboxes']
    for key in bbox_keys:
        if key in batch[0]:
            bboxes = [sample[key] for sample in batch]
            bbox_tensor = torch.stack(bboxes, dim=0)  # [B, T, 4] in XYWH pixel format

            # Get frame size for normalization
            # Letterbox Migration (2025-11-20): All frames letterboxed to frame_size x frame_size (default 512x512)
            # Normalization uses full frame size (including padding) - matches YOLO/DETR practice
            if 'rgb' in key and 'rgb_frames' in batched and batched['rgb_frames'] is not None:
                frame_size = float(batched['rgb_frames'].shape[-1])  # H or W (square)
            elif 'sonar' in key and 'sonar_frames' in batched and batched['sonar_frames'] is not None:
                frame_size = float(batched['sonar_frames'].shape[-1])
            else:
                # Fallback to default frame_size from config (default 512)
                frame_size = 512.0

            # Convert XYWH pixel to CXCYWH normalized
            # Input:  [x, y, w, h] in [0, frame_size] (pixel coordinates, scaled)
            # Output: [cx, cy, w, h] in [0, 1] (normalized, center format)
            x = bbox_tensor[..., 0]  # [B, T]
            y = bbox_tensor[..., 1]
            w = bbox_tensor[..., 2]
            h = bbox_tensor[..., 3]

            # Convert to center coordinates and normalize
            cx = (x + w / 2.0) / frame_size       # Center x, normalized to [0, 1]
            cy = (y + h / 2.0) / frame_size       # Center y, normalized to [0, 1]
            w_norm = w / frame_size               # Width, normalized to [0, 1]
            h_norm = h / frame_size               # Height, normalized to [0, 1]

            # Stack normalized CXCYWH [B, T, 4]
            batched[key] = torch.stack([cx, cy, w_norm, h_norm], dim=-1)

            # CRITICAL (Audit 2025111408 Finding 3): Store pixel XYWH for quality estimators
            # Quality estimators need pixel coordinates to crop ROIs from feature maps
            batched[key + '_pixels'] = bbox_tensor.clone()  # [B, T, 4] in XYWH pixel

    # CRITICAL: Stack validity flags [B, T] (0000 handling)
    validity_keys = ['is_valid_rgb', 'is_valid_sonar']
    for key in validity_keys:
        if key in batch[0]:
            validity = [sample[key] for sample in batch]
            batched[key] = torch.stack(validity, dim=0)  # [B, T]

    # Stack target indices [B]
    if 'target_idx' in batch[0]:
        batched['target_idx'] = torch.tensor([
            sample['target_idx'] for sample in batch
        ], dtype=torch.long)

    # Stack sequence lengths [B]
    if 'sequence_length' in batch[0]:
        batched['sequence_length'] = torch.tensor([
            sample['sequence_length'] for sample in batch
        ], dtype=torch.long)

    # Collect metadata lists
    metadata_keys = ['rgb_metadata', 'sonar_metadata', 'sequence_name', 'frame_ids', 'frame_indices']
    for key in metadata_keys:
        if key in batch[0]:
            batched[key] = [sample[key] for sample in batch]

    return batched


def collate_fn_template_search(batch: List[Dict]) -> Dict:
    """
    Collate function for template-search evaluation datasets.

    Used exclusively by EvaluationSequenceDataset for sequential tracking evaluation.
    This maintains the template-search paradigm required by evaluation protocols.

    Args:
        batch: List of samples from EvaluationSequenceDataset
            Each sample: {
                'rgb_template': [3, H, W],
                'rgb_search': [3, H, W],
                'sonar_template': [1, H, W],
                'sonar_search': [1, H, W],
                'template_bbox_rgb': [4],
                'template_bbox_sonar': [4],
                'search_bbox_rgb': [4],
                'search_bbox_sonar': [4],
                'sequence_name': str,
                'frame_id': int,
                ...
            }

    Returns:
        Batched dictionary: {
            'rgb_template': [B, 3, H, W],
            'rgb_search': [B, 3, H, W],
            'sonar_template': [B, 1, H, W],
            'sonar_search': [B, 1, H, W],
            'template_bbox_rgb': [B, 4],
            'template_bbox_sonar': [B, 4],
            'search_bbox_rgb': [B, 4],
            'search_bbox_sonar': [B, 4],
            ...
        }
    """
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    batched = {}

    # Stack image tensors
    image_keys = ['rgb_template', 'rgb_search', 'sonar_template', 'sonar_search']
    for key in image_keys:
        if key in batch[0] and batch[0][key] is not None:
            images = [sample[key] for sample in batch if sample.get(key) is not None]
            if len(images) > 0:
                batched[key] = torch.stack(images, dim=0)
            else:
                batched[key] = None
        else:
            batched[key] = None

    # Stack bounding boxes
    bbox_keys = ['template_bbox_rgb', 'template_bbox_sonar', 'search_bbox_rgb', 'search_bbox_sonar']
    for key in bbox_keys:
        if key in batch[0] and batch[0][key] is not None:
            bboxes = [sample[key] for sample in batch if sample.get(key) is not None]
            if len(bboxes) > 0:
                batched[key] = torch.stack(bboxes, dim=0)  # [B, 4]
            else:
                batched[key] = None

    # Collect metadata (keep as lists)
    metadata_keys = [
        'sequence_name', 'frame_id', 'template_frame_id',
        'rgb_boxes', 'sonar_boxes', 'is_valid',
        'template_rgb_metadata', 'template_sonar_metadata',
        'search_rgb_metadata', 'search_sonar_metadata'
    ]
    for key in metadata_keys:
        if key in batch[0]:
            batched[key] = [sample[key] for sample in batch]

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
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    use_sequence_grouping: bool = False,
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
        collate_fn: Custom collate function (default: collate_fn_sequence)
        distributed: Use distributed sampler for multi-GPU training
        prefetch_factor: Number of batches to prefetch (default: 2 if num_workers > 0, else None)
        persistent_workers: Keep workers alive between epochs (default: True if num_workers > 0, else False)
        use_sequence_grouping: Use SequenceGroupedBatchSampler for temporal training (default: False)

    Returns:
        DataLoader instance

    CRITICAL FIX (Audit 2025111703 Finding 3):
    When use_sequence_grouping=True, uses SequenceGroupedBatchSampler to ensure
    samples from the same sequence are grouped into consecutive batches. This
    enables the trainer's smart reset strategy to work correctly, allowing
    the EMA aggregator to learn from consecutive frames.
    """
    if collate_fn is None:
        collate_fn = collate_fn_sequence

    # Set defaults for prefetch_factor and persistent_workers
    if prefetch_factor is None:
        prefetch_factor = 2 if num_workers > 0 else None
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # CRITICAL FIX (Audit 2025111703 Finding 3):
    # Use sequence-grouped batch sampler for temporal training
    if use_sequence_grouping and not distributed:
        batch_sampler = SequenceGroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

        # Build DataLoader kwargs for batch_sampler mode
        # Note: batch_size, shuffle, sampler, drop_last must NOT be specified with batch_sampler
        loader_kwargs = {
            'dataset': dataset,
            'batch_sampler': batch_sampler,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
        }
    else:
        # Standard mode: use sampler for distributed training
        if distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
            )
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None

        # Build DataLoader kwargs (standard mode)
        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'sampler': sampler,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'pin_memory': pin_memory,
            'drop_last': drop_last,
            'persistent_workers': persistent_workers,
        }

    # Only add prefetch_factor if num_workers > 0 (required by PyTorch)
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = DataLoader(**loader_kwargs)

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
        collate_fn=collate_fn_sequence,
        distributed=distributed,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_sequence,
        distributed=False,  # No distributed for validation
    )

    return train_loader, val_loader