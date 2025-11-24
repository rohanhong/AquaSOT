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
    Sampler that groups samples by sequence for consistent batch processing.

    CRITICAL FIX (Audit 2025111703 Finding 3):
    Groups samples from the same sequence into consecutive batches to maintain
    sequence coherence during training.

    Without this sampler, random shuffling mixes sequences in every batch, which
    can break sequence-based processing and data consistency.

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
            # Dataset with get_sequence_name method (e.g., TemplateSearchDataset, EvaluationSequenceDataset)
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


# DELETED (Audit 2025112105): collate_fn_sequence removed
# This function was for the OLD temporal/frame-sequence architecture.
# The current template-search architecture uses template_search_collate_fn instead.
# See: src/aquasot3d/dataloader/datasets/template_search_dataset.py



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
    # CRITICAL FIX (Audit 2025112401 Finding 9): Include pixel-space boxes
    bbox_keys = [
        'template_bbox_rgb', 'template_bbox_sonar',
        'search_bbox_rgb', 'search_bbox_sonar',
        'search_bbox_rgb_pixels', 'search_bbox_sonar_pixels',
        'rgb_boxes_pixels', 'sonar_boxes_pixels'
    ]
    for key in bbox_keys:
        if key in batch[0] and batch[0][key] is not None:
            bboxes = [sample[key] for sample in batch if sample.get(key) is not None]
            if len(bboxes) > 0:
                # Handle both tensor and numpy array inputs
                if isinstance(bboxes[0], np.ndarray):
                    bboxes = [torch.from_numpy(bbox) for bbox in bboxes]
                batched[key] = torch.stack(bboxes, dim=0)  # [B, 4]
            else:
                batched[key] = None

    # CRITICAL FIX (Audit 2025112306 P0-2): Stack validity flags as tensors
    # Validity flags must be float32 tensors for loss masking, NOT metadata lists
    validity_keys = ['is_valid', 'is_valid_rgb', 'is_valid_sonar', 'has_object', 'has_object_rgb', 'has_object_sonar']
    for key in validity_keys:
        if key in batch[0]:
            # Convert bool/int to float32 tensor [B]
            validity_list = [sample.get(key, True) for sample in batch]
            batched[key] = torch.tensor(validity_list, dtype=torch.float32)

    # Collect metadata (keep as lists)
    # CRITICAL FIX (Audit 2025112403 Finding 2): Align metadata key names
    # EvaluationSequenceDataset emits 'rgb_search_metadata'/'sonar_search_metadata'
    # TemplateSearchDataset emits 'rgb_metadata'/'sonar_metadata' (for training)
    # CRITICAL: Must collect ALL variants to prevent metadata loss
    metadata_keys = [
        'sequence_name', 'frame_id', 'template_frame_id',
        'rgb_boxes', 'sonar_boxes',  # Removed 'is_valid' - now handled above
        'template_rgb_metadata', 'template_sonar_metadata',
        # Training datasets (TemplateSearchDataset) emit these
        'rgb_metadata', 'sonar_metadata',
        # Evaluation datasets emit these
        'rgb_search_metadata', 'sonar_search_metadata',
        'search_rgb_metadata', 'search_sonar_metadata'
    ]
    for key in metadata_keys:
        if key in batch[0]:
            batched[key] = [sample[key] for sample in batch]

    # CRITICAL FIX (Audit 2025112403 Finding 2): Alias for backward compatibility
    # Training: rgb_metadata → also expose as rgb_search_metadata and search_rgb_metadata
    if 'rgb_metadata' in batched:
        if 'rgb_search_metadata' not in batched:
            batched['rgb_search_metadata'] = batched['rgb_metadata']
        if 'search_rgb_metadata' not in batched:
            batched['search_rgb_metadata'] = batched['rgb_metadata']

    if 'sonar_metadata' in batched:
        if 'sonar_search_metadata' not in batched:
            batched['sonar_search_metadata'] = batched['sonar_metadata']
        if 'search_sonar_metadata' not in batched:
            batched['search_sonar_metadata'] = batched['sonar_metadata']

    # Evaluation: rgb_search_metadata → also expose as search_rgb_metadata
    if 'rgb_search_metadata' in batched and 'search_rgb_metadata' not in batched:
        batched['search_rgb_metadata'] = batched['rgb_search_metadata']
    if 'sonar_search_metadata' in batched and 'search_sonar_metadata' not in batched:
        batched['search_sonar_metadata'] = batched['sonar_search_metadata']

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
        use_sequence_grouping: Use SequenceGroupedBatchSampler for sequence-based training (default: False)

    Returns:
        DataLoader instance

    CRITICAL FIX (Audit 2025111703 Finding 3):
    When use_sequence_grouping=True, uses SequenceGroupedBatchSampler to ensure
    samples from the same sequence are grouped into consecutive batches. This
    enables the trainer's smart reset strategy to work correctly, allowing
    the EMA aggregator to learn from consecutive frames.
    """
    if collate_fn is None:
        # CRITICAL FIX (Audit 2025112105): Use template-search collate function
        # Changed from collate_fn_sequence (legacy temporal architecture) to
        # template_search_collate_fn (current template-search architecture)
        from aquasot3d.dataloader.datasets import template_search_collate_fn
        collate_fn = template_search_collate_fn

    # Set defaults for prefetch_factor and persistent_workers
    if prefetch_factor is None:
        prefetch_factor = 2 if num_workers > 0 else None
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # CRITICAL FIX (Audit 2025111703 Finding 3):
    # Use sequence-grouped batch sampler for sequence-based training
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
    # CRITICAL FIX (Audit 2025112105): Use template-search collate function
    # Changed from collate_fn_sequence (legacy) to template_search_collate_fn
    from aquasot3d.dataloader.datasets import template_search_collate_fn

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=template_search_collate_fn,
        distributed=distributed,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=template_search_collate_fn,
        distributed=False,  # No distributed for validation
    )

    return train_loader, val_loader