"""
Base training functions for AquaSOT-3D.

Based on OSTrack's base_functions.py.
Contains utilities for settings, dataloaders, and optimizers.
"""

import torch
from torch.utils.data.distributed import DistributedSampler


def is_main_process():
    """Check if current process is the main process."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def update_settings(settings, cfg):
    """Update settings based on config.

    Args:
        settings: Settings object to update
        cfg: Configuration dict
    """
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {
        'rgb_template': cfg.DATA.RGB.TEMPLATE.FACTOR,
        'rgb_search': cfg.DATA.RGB.SEARCH.FACTOR,
        'sonar_template': cfg.DATA.SONAR.TEMPLATE.FACTOR,
        'sonar_search': cfg.DATA.SONAR.SEARCH.FACTOR,
    }
    settings.output_sz = {
        'rgb_template': cfg.DATA.RGB.TEMPLATE.SIZE,
        'rgb_search': cfg.DATA.RGB.SEARCH.SIZE,
        'sonar_template': cfg.DATA.SONAR.TEMPLATE.SIZE,
        'sonar_search': cfg.DATA.SONAR.SEARCH.SIZE,
    }
    settings.center_jitter_factor = {
        'rgb_template': cfg.DATA.RGB.TEMPLATE.CENTER_JITTER,
        'rgb_search': cfg.DATA.RGB.SEARCH.CENTER_JITTER,
        'sonar_template': cfg.DATA.SONAR.TEMPLATE.CENTER_JITTER,
        'sonar_search': cfg.DATA.SONAR.SEARCH.CENTER_JITTER,
    }
    settings.scale_jitter_factor = {
        'rgb_template': cfg.DATA.RGB.TEMPLATE.SCALE_JITTER,
        'rgb_search': cfg.DATA.RGB.SEARCH.SCALE_JITTER,
        'sonar_template': cfg.DATA.SONAR.TEMPLATE.SCALE_JITTER,
        'sonar_search': cfg.DATA.SONAR.SEARCH.SCALE_JITTER,
    }
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.checkpoint_interval = cfg.TRAIN.CHECKPOINT_INTERVAL


def build_dataloaders(cfg, settings):
    """Build training and validation dataloaders.

    Args:
        cfg: Configuration dict
        settings: Settings object

    Returns:
        Tuple of (train_loader, val_loader)

    Note:
        This is a placeholder implementation.
        Full implementation will be added in Phase 4 (Data Loading).
    """
    # Import here to avoid circular imports
    from lib.train.data import LTRLoader

    # Placeholder: Create dummy datasets for smoke testing
    # Real implementation will use AquaSOT dataset
    class DummyDataset(torch.utils.data.Dataset):
        """Dummy dataset for smoke testing."""

        def __init__(self, size, samples_per_epoch=1000):
            self.size = size
            self.samples_per_epoch = samples_per_epoch

        def __len__(self):
            return self.samples_per_epoch

        def __getitem__(self, idx):
            rgb_template_sz = settings.output_sz['rgb_template']
            rgb_search_sz = settings.output_sz['rgb_search']
            sonar_template_sz = settings.output_sz['sonar_template']
            sonar_search_sz = settings.output_sz['sonar_search']

            return {
                'rgb_template': torch.randn(3, rgb_template_sz, rgb_template_sz),
                'rgb_search': torch.randn(3, rgb_search_sz, rgb_search_sz),
                'sonar_template': torch.randn(1, sonar_template_sz, sonar_template_sz),
                'sonar_search': torch.randn(1, sonar_search_sz, sonar_search_sz),
                'rgb_template_bbox': torch.tensor([0.25, 0.25, 0.5, 0.5]),  # [cx, cy, w, h]
                'rgb_search_bbox': torch.tensor([0.5, 0.5, 0.3, 0.3]),
                'sonar_template_bbox': torch.tensor([0.25, 0.25, 0.5, 0.5]),
                'sonar_search_bbox': torch.tensor([0.5, 0.5, 0.3, 0.3]),
            }

    # Create dummy datasets
    dataset_train = DummyDataset(
        size=cfg.DATA.RGB.SEARCH.SIZE,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH
    )
    dataset_val = DummyDataset(
        size=cfg.DATA.RGB.SEARCH.SIZE,
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH
    )

    # Create samplers for distributed training
    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None

    shuffle = False if settings.local_rank != -1 else True

    # Create data loaders
    loader_train = LTRLoader(
        'train',
        dataset_train,
        training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=train_sampler
    )

    loader_val = LTRLoader(
        'val',
        dataset_val,
        training=False,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=val_sampler,
        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL
    )

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    """Create optimizer and learning rate scheduler.

    Args:
        net: Network to optimize
        cfg: Configuration dict

    Returns:
        Tuple of (optimizer, lr_scheduler)
    """
    # Separate backbone and non-backbone parameters
    backbone_params = []
    non_backbone_params = []

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    # Create parameter groups with different learning rates
    param_dicts = [
        {"params": non_backbone_params},
        {
            "params": backbone_params,
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        },
    ]

    if is_main_process():
        print("Learnable parameters:")
        total_params = 0
        for name, param in net.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                print(f"  {name}: {param.numel()}")
        print(f"Total learnable parameters: {total_params:,}")

    # Create optimizer
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    elif cfg.TRAIN.OPTIMIZER == "ADAM":
        optimizer = torch.optim.Adam(
            param_dicts,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    elif cfg.TRAIN.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            param_dicts,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.TRAIN.OPTIMIZER}")

    # Create learning rate scheduler
    scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    if scheduler_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.TRAIN.LR_DROP_EPOCH,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA
        )
    elif scheduler_type == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA
        )
    elif scheduler_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.EPOCH,
            eta_min=cfg.TRAIN.LR * 0.01
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    return optimizer, lr_scheduler
