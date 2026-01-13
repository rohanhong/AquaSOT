"""
Data loader for AquaSOT-3D training.

Based on OSTrack's LTRLoader with simplifications.
"""

import collections.abc
import torch
import numpy as np


def ltr_collate(batch):
    """Collate function that stacks tensors along dim=0.

    Args:
        batch: List of samples to collate

    Returns:
        Collated batch
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return torch.stack([torch.from_numpy(b) for b in batch], 0)

    elif isinstance(batch[0], (int, float)):
        if isinstance(batch[0], int):
            return torch.LongTensor(batch)
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], str):
        return batch

    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}

    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]

    elif batch[0] is None:
        return batch

    raise TypeError(error_msg.format(type(batch[0])))


def ltr_collate_stack1(batch):
    """Collate function that stacks tensors along dim=1.

    Args:
        batch: List of samples to collate

    Returns:
        Collated batch with tensors stacked at dim=1
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 1)

    elif isinstance(batch[0], np.ndarray):
        return torch.stack([torch.from_numpy(b) for b in batch], 1)

    elif isinstance(batch[0], (int, float)):
        if isinstance(batch[0], int):
            return torch.LongTensor(batch)
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], str):
        return batch

    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}

    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]

    elif batch[0] is None:
        return batch

    raise TypeError(error_msg.format(type(batch[0])))


class LTRLoader(torch.utils.data.DataLoader):
    """Custom DataLoader for LTR-style training.

    Extends PyTorch DataLoader with:
    - Named loaders (for train/val distinction)
    - Configurable stack dimension
    - Epoch interval for validation

    Args:
        name: Name of the loader (e.g., 'train', 'val')
        dataset: Dataset to load from
        training: Whether this is a training loader
        batch_size: Batch size
        shuffle: Whether to shuffle data
        sampler: Custom sampler
        batch_sampler: Custom batch sampler
        num_workers: Number of worker processes
        epoch_interval: Run every N epochs (for validation)
        collate_fn: Custom collate function
        stack_dim: Dimension to stack batches (0 or 1)
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop last incomplete batch
        timeout: Worker timeout
        worker_init_fn: Worker initialization function
    """

    def __init__(
        self,
        name,
        dataset,
        training=True,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        epoch_interval=1,
        collate_fn=None,
        stack_dim=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        # Select collate function based on stack_dim
        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError(f'Invalid stack_dim: {stack_dim}. Must be 0 or 1.')

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim
