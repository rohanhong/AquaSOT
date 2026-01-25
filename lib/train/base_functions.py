"""
Base training functions for AquaSOT.

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
    """
    # Import here to avoid circular imports
    from lib.train.data import LTRLoader
    from lib.train.dataset.aquasot import AquaSOTDataset
    from lib.train.data.sampler import MultiModalSampler
    from lib.train.data.processing import DualModalProcessing
    from lib.train.data.transforms import (
        Transform, ToTensorAndJitter, Normalize, ToGrayscale,
        ColorJitter, GaussianBlur, RandomErasing, RandomHorizontalFlipNorm,
        SonarToTensor, SonarNormalize, SonarSpeckleNoise, SonarRandomErasing
    )

    aug_cfg = cfg.DATA.AUGMENTATION
    aug_enabled = aug_cfg.ENABLED

    # Build RGB transforms
    rgb_transforms = []

    # Brightness jitter (via ToTensorAndJitter)
    brightness = aug_cfg.RGB.BRIGHTNESS if aug_enabled else 0.0
    rgb_transforms.append(ToTensorAndJitter(brightness_jitter=brightness))

    # Grayscale (before ColorJitter to avoid wasting saturation/hue adjustments)
    if aug_enabled and aug_cfg.RGB.GRAYSCALE:
        rgb_transforms.append(ToGrayscale(probability=aug_cfg.RGB.GRAYSCALE_PROB))

    # Color jitter (contrast, saturation, hue)
    # Note: saturation/hue have no effect on grayscale images, only contrast applies
    if aug_enabled and aug_cfg.RGB.COLOR_JITTER:
        rgb_transforms.append(ColorJitter(
            brightness=0.0,  # Already handled by ToTensorAndJitter
            contrast=aug_cfg.RGB.CONTRAST,
            saturation=aug_cfg.RGB.SATURATION,
            hue=aug_cfg.RGB.HUE
        ))

    # Gaussian blur
    if aug_enabled and aug_cfg.RGB.BLUR:
        rgb_transforms.append(GaussianBlur(probability=aug_cfg.RGB.BLUR_PROB))

    # Random erasing (occlusion)
    if aug_enabled and aug_cfg.RGB.OCCLUSION:
        rgb_transforms.append(RandomErasing(probability=aug_cfg.RGB.OCCLUSION_PROB))

    # Normalization (always applied)
    rgb_transforms.append(Normalize(mean=cfg.DATA.RGB_MEAN, std=cfg.DATA.RGB_STD))
    rgb_transform = Transform(*rgb_transforms)

    # Build Sonar transforms
    sonar_transforms = []

    # Speckle noise (before ToTensor)
    if aug_enabled and aug_cfg.SONAR.SPECKLE:
        sonar_transforms.append(SonarSpeckleNoise(
            probability=aug_cfg.SONAR.SPECKLE_PROB,
            noise_level=aug_cfg.SONAR.SPECKLE_LEVEL
        ))

    # To tensor
    sonar_transforms.append(SonarToTensor())

    # Random erasing (occlusion)
    if aug_enabled and aug_cfg.SONAR.OCCLUSION:
        sonar_transforms.append(SonarRandomErasing(
            probability=aug_cfg.SONAR.OCCLUSION_PROB
        ))

    # Normalization
    sonar_transforms.append(SonarNormalize(
        mean=cfg.DATA.SONAR_MEAN[0],
        std=cfg.DATA.SONAR_STD[0]
    ))
    sonar_transform = Transform(*sonar_transforms)

    # Synchronized horizontal flip (applied in processing)
    # Use RandomHorizontalFlipNorm because bboxes are normalized to [0,1]
    joint_flip = None
    if aug_enabled and aug_cfg.HORIZONTAL_FLIP:
        joint_flip = RandomHorizontalFlipNorm(probability=aug_cfg.HORIZONTAL_FLIP_PROB)

    if is_main_process():
        print("Data augmentation enabled:", aug_enabled)
        if aug_enabled:
            print(f"  RGB: brightness={brightness}, color_jitter={aug_cfg.RGB.COLOR_JITTER}, "
                  f"grayscale={aug_cfg.RGB.GRAYSCALE}, blur={aug_cfg.RGB.BLUR}, "
                  f"occlusion={aug_cfg.RGB.OCCLUSION}")
            print(f"  Sonar: speckle={aug_cfg.SONAR.SPECKLE}, "
                  f"occlusion={aug_cfg.SONAR.OCCLUSION}")
            print(f"  Joint: horizontal_flip={aug_cfg.HORIZONTAL_FLIP}")

    # Build processing
    search_area_factor = {
        'template': cfg.DATA.TEMPLATE.FACTOR,
        'search': cfg.DATA.SEARCH.FACTOR
    }
    output_sz = {
        'template': cfg.DATA.TEMPLATE.SIZE,
        'search': cfg.DATA.SEARCH.SIZE
    }
    center_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
        'search': cfg.DATA.SEARCH.CENTER_JITTER
    }
    scale_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
        'search': cfg.DATA.SEARCH.SCALE_JITTER
    }

    processing = DualModalProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=center_jitter_factor,
        scale_jitter_factor=scale_jitter_factor,
        mode='sequence',
        rgb_transform=rgb_transform,
        sonar_transform=sonar_transform,
        joint_flip=joint_flip,
        settings=settings,
        stride=cfg.MODEL.BACKBONE.RGB.STRIDE
    )

    # Build training dataset
    train_dataset = AquaSOTDataset(
        root=cfg.DATA.ROOT,
        sequence_names=cfg.DATA.TRAIN.SEQUENCES,
        use_rect_sonar=True
    )

    # Build validation dataset
    val_dataset = AquaSOTDataset(
        root=cfg.DATA.ROOT,
        sequence_names=cfg.DATA.VAL.SEQUENCES,
        use_rect_sonar=True
    )

    # Build samplers
    # Get online template number (0 for validation to match original behavior)
    num_online_template = getattr(cfg.DATA.TEMPLATE, 'ONLINE_NUMBER', 0)

    train_sampler = MultiModalSampler(
        datasets=[train_dataset],
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=cfg.DATA.SEARCH.NUMBER,
        num_template_frames=cfg.DATA.TEMPLATE.NUMBER,
        num_online_template_frames=num_online_template,
        processing=processing,
        frame_sample_mode=cfg.DATA.SAMPLER_MODE
    )

    val_sampler = MultiModalSampler(
        datasets=[val_dataset],
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=cfg.DATA.SEARCH.NUMBER,
        num_template_frames=cfg.DATA.TEMPLATE.NUMBER,
        num_online_template_frames=0,  # No online templates for validation
        processing=processing,
        frame_sample_mode=cfg.DATA.SAMPLER_MODE
    )

    # Create distributed samplers if needed
    dist_train_sampler = DistributedSampler(train_sampler) if settings.local_rank != -1 else None
    dist_val_sampler = DistributedSampler(val_sampler) if settings.local_rank != -1 else None

    shuffle = False if settings.local_rank != -1 else True

    # Create data loaders
    loader_train = LTRLoader(
        'train',
        train_sampler,
        training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=dist_train_sampler
    )

    loader_val = LTRLoader(
        'val',
        val_sampler,
        training=False,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=dist_val_sampler,
        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL
    )

    if is_main_process():
        print(f"Training sequences: {cfg.DATA.TRAIN.SEQUENCES}")
        print(f"Validation sequences: {cfg.DATA.VAL.SEQUENCES}")
        print(f"Training samples per epoch: {cfg.DATA.TRAIN.SAMPLE_PER_EPOCH}")
        print(f"Validation samples per epoch: {cfg.DATA.VAL.SAMPLE_PER_EPOCH}")

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
            eta_min=cfg.TRAIN.SCHEDULER.MIN_LR
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    return optimizer, lr_scheduler
