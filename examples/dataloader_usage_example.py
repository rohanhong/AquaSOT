"""
AquaSOT-3D DataLoader Usage Example

This script demonstrates how to use the data loading pipeline for
multi-modal RGB-Sonar tracking.

Usage:
    python examples/dataloader_usage_example.py

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import sys
sys.path.insert(0, 'src')

import yaml
from pathlib import Path
import torch

from aquasot3d.dataloader.datasets import (
    RGBSonarDataset,
    TemplateSearchSampler,
    build_augmentation,
)
from aquasot3d.dataloader import create_dataloader, collate_fn_template_search


def example_1_basic_dataset_loading():
    """Example 1: Basic dataset loading."""
    print("\n" + "="*60)
    print("Example 1: Basic Dataset Loading")
    print("="*60)

    # Create dataset
    dataset = RGBSonarDataset(
        data_root='data/',
        sequence_name='uuv2',
        return_occluded=True,
    )

    print(f"Dataset: {dataset.sequence_name}")
    print(f"Total frames: {dataset.num_frames}")
    print(f"Valid frames: {len(dataset)}")

    # Get first frame
    sample = dataset[0]
    print(f"\nFirst frame (ID={sample['frame_id']}):")
    print(f"  RGB image shape: {sample['rgb_img'].shape if sample['rgb_img'] is not None else None}")
    print(f"  Sonar image shape: {sample['sonar_img'].shape if sample['sonar_img'] is not None else None}")
    print(f"  RGB bbox: {sample['rgb_bbox']}")
    print(f"  Sonar bbox: {sample['sonar_bbox']}")
    print(f"  Is valid: {sample['is_valid']}")


def example_2_template_search_sampling():
    """Example 2: Template-search pair generation."""
    print("\n" + "="*60)
    print("Example 2: Template-Search Sampling")
    print("="*60)

    # Create dataset
    dataset = RGBSonarDataset(
        data_root='data/',
        sequence_name='uuv2',
        return_occluded=False,
    )

    # Create template-search sampler
    sampler = TemplateSearchSampler(
        dataset=dataset,
        template_mode='first',
        search_scale=2.0,
        search_size=(256, 256),
        template_size=(128, 128),
    )

    # Sample a pair
    search_idx = 10
    pair = sampler.sample(search_idx)

    if pair is not None:
        print(f"\nTemplate-Search Pair:")
        print(f"  Template frame: {pair['template_idx'] + 1}")
        print(f"  Search frame: {pair['search_idx'] + 1}")
        print(f"  Template RGB shape: {pair['template_rgb'].shape}")
        print(f"  Search RGB shape: {pair['search_rgb'].shape}")
        print(f"  Search bbox (normalized): {pair['search_bbox_rgb']}")
        print(f"  Is valid: {pair['is_valid']}")
    else:
        print("Failed to generate pair (frame occluded)")


def example_3_augmentation():
    """Example 3: Data augmentation pipeline."""
    print("\n" + "="*60)
    print("Example 3: Data Augmentation")
    print("="*60)

    # Load augmentation config
    config = {
        'type': 'TemplateSearchAugmentation',
        'enable_flip': True,
        'template_config': {
            'rgb': {'enable_color_jitter': False},
            'sonar': {'enable_speckle': False},
        },
        'search_config': {
            'rgb': {'enable_color_jitter': True, 'color_jitter_prob': 1.0},
            'sonar': {'enable_speckle': True, 'speckle_prob': 1.0},
        },
    }

    # Build augmentation
    aug = build_augmentation(config, is_train=True)

    print(f"Augmentation type: {type(aug).__name__}")
    print(f"Training mode: {aug.is_train}")

    # Create dummy data
    import numpy as np
    data = {
        'template_rgb': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        'template_sonar': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        'search_rgb': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        'search_sonar': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        'template_bbox_rgb': np.array([10, 20, 50, 60], dtype=np.float32),
        'template_bbox_sonar': np.array([5, 10, 25, 30], dtype=np.float32),
        'search_bbox_rgb': np.array([15, 25, 60, 70], dtype=np.float32),
        'search_bbox_sonar': np.array([8, 12, 30, 35], dtype=np.float32),
    }

    # Apply augmentation
    augmented = aug(data)

    print(f"\nAugmented data:")
    print(f"  Template RGB shape: {augmented['template_rgb'].shape}")
    print(f"  Search RGB shape: {augmented['search_rgb'].shape}")
    print(f"  Template RGB dtype: {augmented['template_rgb'].dtype}")
    print(f"  Template RGB range: [{augmented['template_rgb'].min():.2f}, {augmented['template_rgb'].max():.2f}]")


def example_4_dataloader():
    """Example 4: Creating DataLoader with batching."""
    print("\n" + "="*60)
    print("Example 4: DataLoader with Batching")
    print("="*60)

    # For this example, we'll create a wrapper dataset
    class TemplateSearchDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, sampler, augmentation=None):
            self.base_dataset = base_dataset
            self.sampler = sampler
            self.augmentation = augmentation

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            # Get search frame index from valid indices
            search_idx = self.base_dataset.valid_indices[idx]

            # Sample template-search pair
            pair = self.sampler.sample(search_idx)

            if pair is None:
                return None

            # Apply augmentation
            if self.augmentation is not None:
                pair = self.augmentation(pair)

            return pair

    # Create dataset
    base_dataset = RGBSonarDataset(
        data_root='data/',
        sequence_name='uuv2',
        return_occluded=False,
    )

    sampler = TemplateSearchSampler(
        dataset=base_dataset,
        template_mode='first',
    )

    aug_config = {'type': 'NoAugmentation'}
    aug = build_augmentation(aug_config, is_train=False)

    ts_dataset = TemplateSearchDataset(base_dataset, sampler, aug)

    # Create DataLoader
    loader = create_dataloader(
        ts_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_template_search,
    )

    print(f"DataLoader created:")
    print(f"  Dataset size: {len(ts_dataset)}")
    print(f"  Batch size: 4")
    print(f"  Number of batches: {len(loader)}")

    # Get first batch
    batch = next(iter(loader))

    if batch is not None:
        print(f"\nFirst batch:")
        print(f"  Template RGB shape: {batch['template_rgb'].shape}")
        print(f"  Search RGB shape: {batch['search_rgb'].shape}")
        print(f"  Template bbox RGB shape: {batch['template_bbox_rgb'].shape}")
        print(f"  Batch size: {batch['template_rgb'].shape[0]}")


def example_5_sequence_info():
    """Example 5: Dataset statistics and metadata."""
    print("\n" + "="*60)
    print("Example 5: Dataset Statistics")
    print("="*60)

    dataset = RGBSonarDataset(
        data_root='data/',
        sequence_name='uuv2',
        return_occluded=True,
    )

    info = dataset.get_sequence_info()

    print("Sequence Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Analyze occlusions
    num_rgb_occluded = 0
    num_sonar_occluded = 0
    num_both_occluded = 0

    for idx in range(dataset.num_frames):
        rgb_occ = (dataset.rgb_bboxes[idx] == 0).all()
        sonar_occ = (dataset.sonar_bboxes[idx] == 0).all()

        if rgb_occ:
            num_rgb_occluded += 1
        if sonar_occ:
            num_sonar_occluded += 1
        if rgb_occ and sonar_occ:
            num_both_occluded += 1

    print(f"\nOcclusion Statistics:")
    print(f"  RGB occluded frames: {num_rgb_occluded} ({num_rgb_occluded/dataset.num_frames*100:.1f}%)")
    print(f"  Sonar occluded frames: {num_sonar_occluded} ({num_sonar_occluded/dataset.num_frames*100:.1f}%)")
    print(f"  Both occluded frames: {num_both_occluded} ({num_both_occluded/dataset.num_frames*100:.1f}%)")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" AquaSOT-3D DataLoader Usage Examples")
    print("="*80)

    # Check if data exists
    if not Path('data/uuv2').exists():
        print("\n⚠️  Warning: Data directory 'data/uuv2' not found.")
        print("   Please download the dataset first or update the data_root path.")
        print("\n   Running with dummy explanations...\n")

        # Run only augmentation example with dummy data
        example_3_augmentation()

    else:
        # Run all examples
        example_1_basic_dataset_loading()
        example_2_template_search_sampling()
        example_3_augmentation()
        example_4_dataloader()
        example_5_sequence_info()

    print("\n" + "="*80)
    print(" Examples completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
