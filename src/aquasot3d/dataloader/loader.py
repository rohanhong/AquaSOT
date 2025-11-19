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
            # TrackingDataset with get_sequence_name method
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

    # Handle string values (sequence names, etc.)
    string_keys = ['sequence_name', 'template_sequence', 'search_sequence']
    for key in string_keys:
        if key in batch[0]:
            # Keep as list of strings (cannot convert to tensor)
            batched[key] = [sample[key] for sample in batch]

    return batched


def collate_fn_template_search(batch: List[Dict]) -> Dict:
    """
    Collate function for template-search pairs and sequences.

    CRITICAL FIX (Audit 2025111703 Finding 1):
    Now supports sequence mode where rgb_search/sonar_search are [T, C, H, W].

    Args:
        batch: List of template-search sample dictionaries

    Returns:
        Batched dictionary with stacked tensors

    Sample structure (TrackingDataset format):
        Pair mode (T=1):
            batch[i] = {
                'rgb_search': Tensor (C, H_s, W_s),
                ...
            }

        Sequence mode (T>1):
            batch[i] = {
                'rgb_search': Tensor (T, C, H_s, W_s),
                'is_sequence': True,
                ...
            }
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    batched = {}

    # Detect if this is a sequence batch
    is_sequence_batch = batch[0].get('is_sequence', False)

    # Handle image tensors (trainer expects: rgb_template, rgb_search, sonar_template, sonar_search)
    image_keys = ['rgb_template', 'rgb_search', 'sonar_template', 'sonar_search']
    for key in image_keys:
        if key in batch[0]:
            imgs = [sample[key] for sample in batch]

            # Stack tensors (already in CHW format from TrackingDataset)
            if isinstance(imgs[0], torch.Tensor):
                # CRITICAL FIX (Audit 2025111703 Finding 1): Handle sequence stacking
                if is_sequence_batch and 'search' in key:
                    # Sequence mode: imgs[i] has shape [T, C, H, W]
                    # Stack to [B, T, C, H, W]
                    batched[key] = torch.stack(imgs, dim=0)  # [B, T, C, H, W]
                else:
                    # Pair mode or template: imgs[i] has shape [C, H, W]
                    # Stack to [B, C, H, W]
                    batched[key] = torch.stack(imgs, dim=0)  # [B, C, H, W]
            elif isinstance(imgs[0], np.ndarray):
                # Convert numpy arrays to tensors (HWC → CHW)
                imgs_tensor = [torch.from_numpy(img).permute(2, 0, 1) for img in imgs]
                if is_sequence_batch and 'search' in key:
                    batched[key] = torch.stack(imgs_tensor, dim=0)  # [B, T, C, H, W]
                else:
                    batched[key] = torch.stack(imgs_tensor, dim=0)  # [B, C, H, W]

    # Handle bounding boxes with normalization
    # Critical fix: Normalize pixel coordinates to [0, 1] range for numerical stability
    # This prevents gradient explosion in L1/GIoU losses
    bbox_keys = [
        'template_bbox_rgb', 'template_bbox_sonar',
        'search_bbox_rgb', 'search_bbox_sonar',
        'search_region_rgb',  # Only RGB search region (sonar uses full frame per Audit 2025111408)
        'rgb_boxes', 'sonar_boxes'  # Trainer-expected keys for loss computation
    ]

    # Get image dimensions for normalization
    # Search images are typically 255x255 or 256x256
    # Template images are typically 127x127 or 128x128
    rgb_search_h, rgb_search_w = batch[0]['rgb_search'].shape[-2:] if 'rgb_search' in batch[0] else (256, 256)
    sonar_search_h, sonar_search_w = batch[0]['sonar_search'].shape[-2:] if 'sonar_search' in batch[0] else (256, 256)
    rgb_template_h, rgb_template_w = batch[0]['rgb_template'].shape[-2:] if 'rgb_template' in batch[0] else (128, 128)
    sonar_template_h, sonar_template_w = batch[0]['sonar_template'].shape[-2:] if 'sonar_template' in batch[0] else (128, 128)

    for key in bbox_keys:
        if key in batch[0]:
            bboxes = [sample[key] for sample in batch]

            if isinstance(bboxes[0], np.ndarray):
                batched[key] = torch.from_numpy(np.stack(bboxes, axis=0).astype(np.float32))
            else:
                batched[key] = torch.stack([torch.tensor(bbox, dtype=torch.float32) for bbox in bboxes], dim=0)

            # Normalize bounding boxes to [0, 1] based on image dimensions
            # Format: [x, y, w, h] in pixels → [cx, cy, w, h] in [0, 1]
            # Priority 2.2: Use correct image dimensions for template vs search bboxes
            if 'template' in key and 'rgb' in key and not key.endswith('_normalized'):
                # RGB template bounding boxes - use template dimensions
                img_w, img_h = rgb_template_w, rgb_template_h
            elif 'template' in key and 'sonar' in key and not key.endswith('_normalized'):
                # Sonar template bounding boxes - use template dimensions
                img_w, img_h = sonar_template_w, sonar_template_h
            elif 'rgb' in key and not key.endswith('_normalized'):
                # RGB search/other bounding boxes - use search dimensions
                img_w, img_h = rgb_search_w, rgb_search_h
            elif 'sonar' in key and not key.endswith('_normalized'):
                # Sonar search/other bounding boxes - use search dimensions
                img_w, img_h = sonar_search_w, sonar_search_h
            else:
                # Generic fallback (256x256)
                img_w, img_h = 256.0, 256.0

            # CRITICAL FIX (Audit 2025111408 Finding 3):
            # Store BOTH pixel coordinates (for quality estimators) and normalized (for losses)
            # Quality estimators need pixel xywh to crop ROIs correctly

            bbox_tensor = batched[key].clone()

            # CRITICAL FIX: Handle sequence mode where bbox shape is [B, T, 4] instead of [B, 4]
            # Check dimensionality directly instead of relying on key name
            if bbox_tensor.ndim == 3 and bbox_tensor.shape[1] > 1:
                # Sequence mode: bbox_tensor shape is [B, T, 4]
                # For quality estimation, flatten to [B*T, 4] to match reshaped images [B*T, C, H, W]
                pixel_key = key + '_pixels'
                B, T = bbox_tensor.shape[0], bbox_tensor.shape[1]
                batched[pixel_key] = bbox_tensor.reshape(B * T, 4).clone()  # [B*T, 4]

                # Extract coordinates for normalization
                x, y, w, h = bbox_tensor[..., 0], bbox_tensor[..., 1], bbox_tensor[..., 2], bbox_tensor[..., 3]
            else:
                # Pair mode or template: bbox_tensor shape is [B, 4] (or [B, 1, 4] squeezed)
                if bbox_tensor.ndim == 3:
                    bbox_tensor = bbox_tensor.squeeze(1)  # [B, 1, 4] -> [B, 4]
                    batched[key] = bbox_tensor  # Update batched with squeezed version

                x, y, w, h = bbox_tensor[:, 0], bbox_tensor[:, 1], bbox_tensor[:, 2], bbox_tensor[:, 3]

                # Store original pixel coordinates in xywh format for quality estimation
                pixel_key = key + '_pixels'
                batched[pixel_key] = bbox_tensor.clone()  # Keep pixel xywh

            # Convert [x, y, w, h] to [cx, cy, w, h] and normalize for losses
            # cx = (x + w/2) / img_w, cy = (y + h/2) / img_h
            # w_norm = w / img_w, h_norm = h / img_h
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Stack normalized coordinates [cx, cy, w, h] in [0, 1]
            # CRITICAL FIX: For sequence mode, only use last frame for loss computation
            # This matches temporal aggregation output shape [B, 4]
            if bbox_tensor.ndim == 3 and bbox_tensor.shape[1] > 1:
                # Sequence mode: extract only last frame for gt_boxes
                batched[key] = torch.stack([cx[:, -1], cy[:, -1], w_norm[:, -1], h_norm[:, -1]], dim=-1)
            else:
                # Pair/template mode: use full tensor
                batched[key] = torch.stack([cx, cy, w_norm, h_norm], dim=-1)

            # Clamp to [0, 1] for safety (handles edge cases)
            batched[key] = torch.clamp(batched[key], min=0.0, max=1.0)

    # CRITICAL FIX (Audit 2025111903 Phase 3): Handle validity flags
    # Validity tracking system to mask invalid samples (occlusions, augmentation filtering)
    # These flags propagate from dataset → augmentation → collate → loss computation
    validity_keys = ['is_valid', 'is_valid_rgb', 'is_valid_sonar']
    for key in validity_keys:
        if key in batch[0]:
            values = [sample[key] for sample in batch]
            # Convert boolean/int values to float tensor for masking operations
            batched[key] = torch.tensor(values, dtype=torch.float32)

    # Handle other scalar values
    scalar_keys = ['template_idx', 'search_idx',
                   'template_frame_id', 'search_frame_id', 'frame_id']
    for key in scalar_keys:
        if key in batch[0]:
            values = [sample[key] for sample in batch]
            batched[key] = torch.tensor(values)

    # Handle string values (sequence names, etc.)
    string_keys = ['sequence_name', 'template_sequence', 'search_sequence']
    for key in string_keys:
        if key in batch[0]:
            # Keep as list of strings (cannot convert to tensor)
            batched[key] = [sample[key] for sample in batch]

    # CRITICAL FIX (Audit 2025111409 Finding 1 & Audit 2025111411 Finding 2):
    # Handle transformation metadata for physical coordinate reconstruction
    # Metadata dicts contain: original_height, original_width, scale_x, scale_y, pad_x, pad_y
    #
    # IMPORTANT (Audit 2025111411 Finding 2): Store per-sample metadata to support batch_size > 1
    # Previously assumed batch_size=1 and only used batch[0], which breaks for multi-sample batches
    metadata_keys = [
        'rgb_search_metadata', 'sonar_search_metadata',
        'template_rgb_metadata', 'template_sonar_metadata'
    ]
    for key in metadata_keys:
        if key in batch[0]:
            # Store as list of dicts (one per sample in batch)
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
        collate_fn: Custom collate function (default: collate_fn_tracking)
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
        collate_fn = collate_fn_tracking

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