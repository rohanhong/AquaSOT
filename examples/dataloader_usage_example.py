"""
AquaSOT-3D DataLoader Usage Example

This script demonstrates how to use the data loading pipeline for
multi-modal RGB-Sonar tracking with multi-sequence support.

Usage:
    python examples/dataloader_usage_example.py

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import sys
sys.path.insert(0, 'src')

import yaml
from pathlib import Path
import torch
import numpy as np

from aquasot3d.dataloader.datasets import (
    RGBSonarDataset,
    MultiSequenceRGBSonarDataset,
    TrackingDataset,
    build_augmentation,
)
from aquasot3d.dataloader import create_dataloader


def example_1_multi_sequence_dataset_loading():
    """Example 1: Multi-sequence dataset loading (recommended)."""
    print("\n" + "="*60)
    print("Example 1: Multi-Sequence Dataset Loading")
    print("="*60)

    # Create multi-sequence dataset
    dataset = MultiSequenceRGBSonarDataset(
        data_root='data',
        sequence_names=['uuv2', 'uuv3'],
        sampling_policy='frame_balanced',
        return_occluded=True,
    )

    print(f"Multi-sequence dataset:")
    print(f"  Total sequences: {len(dataset.sequence_metadata)}")
    print(f"  Total valid frames: {len(dataset)}")
    print(f"  Sampling policy: {dataset.sampling_policy}")

    # Display per-sequence metadata
    print(f"\nPer-sequence metadata:")
    for seq_name, metadata in dataset.sequence_metadata.items():
        print(f"  {seq_name}:")
        print(f"    Frames: {metadata.num_frames}")
        print(f"    Valid indices: {len(metadata.valid_indices)}")
        print(f"    RGB resolution: {metadata.rgb_resolution}")
        print(f"    Sonar resolution: {metadata.sonar_resolution}")
        print(f"    Global index range: [{metadata.start_idx}, {metadata.end_idx})")

    # Get first frame
    sample = dataset[0]
    print(f"\nFirst frame:")
    print(f"  Sequence: {sample['sequence_name']}")
    print(f"  Frame ID: {sample['frame_id']}")
    print(f"  RGB image shape: {sample['rgb_img'].shape if sample['rgb_img'] is not None else None}")
    print(f"  Sonar image shape: {sample['sonar_img'].shape if sample['sonar_img'] is not None else None}")
    print(f"  RGB bbox: {sample['rgb_bbox']}")
    print(f"  Sonar bbox: {sample['sonar_bbox']}")
    print(f"  Is valid: {sample['is_valid']}")


def example_2_intra_sequence_tracking():
    """Example 2: Multi-sequence tracking with intra-sequence constraint."""
    print("\n" + "="*60)
    print("Example 2: Intra-Sequence Tracking Dataset")
    print("="*60)

    # Create multi-sequence base dataset
    base_dataset = MultiSequenceRGBSonarDataset(
        data_root='data',
        sequence_names=['uuv2', 'uuv3'],
        sampling_policy='frame_balanced',
        return_occluded=False,
    )

    # Create tracking dataset with intra-sequence temporal constraint
    tracking_dataset = TrackingDataset(
        base_dataset=base_dataset,
        template_mode='first',
        max_gap=50,
        search_scale=2.0,
        search_size=(256, 256),
        template_size=(128, 128),
        return_search_only=False,
    )

    print(f"Tracking dataset:")
    print(f"  Base dataset size: {len(base_dataset)}")
    print(f"  Tracking pairs: {len(tracking_dataset)}")
    print(f"  Template mode: {tracking_dataset.template_mode}")
    print(f"  Max gap: {tracking_dataset.max_gap}")

    # Sample a pair
    sample = tracking_dataset[10]

    if sample is not None:
        print(f"\nTemplate-Search Pair:")
        print(f"  Template sequence: {sample['template_sequence']}")
        print(f"  Search sequence: {sample['search_sequence']}")
        print(f"  Template frame ID: {sample['template_frame_id']}")
        print(f"  Search frame ID: {sample['search_frame_id']}")
        print(f"  Template RGB shape: {sample['template_rgb'].shape}")
        print(f"  Search RGB shape: {sample['search_rgb'].shape}")
        print(f"  Intra-sequence pair: {sample['template_sequence'] == sample['search_sequence']}")
    else:
        print("Failed to generate pair (frame occluded)")

    # Verify intra-sequence constraint
    print(f"\nVerifying intra-sequence constraint (sampling 50 pairs):")
    violations = 0
    for i in range(min(50, len(tracking_dataset))):
        sample = tracking_dataset[i]
        if sample is not None:
            if sample['template_sequence'] != sample['search_sequence']:
                violations += 1
    print(f"  Cross-sequence violations: {violations}/50")
    print(f"  Constraint satisfied: {violations == 0}")


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


def example_4_multi_sequence_dataloader():
    """Example 4: Multi-sequence DataLoader with batching."""
    print("\n" + "="*60)
    print("Example 4: Multi-Sequence DataLoader")
    print("="*60)

    # Create multi-sequence base dataset
    base_dataset = MultiSequenceRGBSonarDataset(
        data_root='data',
        sequence_names=['uuv2', 'uuv3'],
        sampling_policy='frame_balanced',
        return_occluded=False,
    )

    # Create tracking dataset
    tracking_dataset = TrackingDataset(
        base_dataset=base_dataset,
        template_mode='first',
        max_gap=50,
        search_scale=2.0,
        search_size=(256, 256),
        template_size=(128, 128),
    )

    # Create DataLoader
    loader = create_dataloader(
        tracking_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print(f"Multi-sequence DataLoader:")
    print(f"  Total sequences: {len(base_dataset.sequence_metadata)}")
    print(f"  Dataset size: {len(tracking_dataset)}")
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

        # Show sequence distribution in batch
        template_seqs = batch['template_sequence']
        search_seqs = batch['search_sequence']
        print(f"\nSequence distribution in batch:")
        for i in range(len(template_seqs)):
            print(f"    Sample {i}: template={template_seqs[i]}, search={search_seqs[i]}")


def example_5_multi_sequence_statistics():
    """Example 5: Multi-sequence dataset statistics and metadata."""
    print("\n" + "="*60)
    print("Example 5: Multi-Sequence Statistics")
    print("="*60)

    dataset = MultiSequenceRGBSonarDataset(
        data_root='data',
        sequence_names=['uuv2', 'uuv3'],
        sampling_policy='frame_balanced',
        return_occluded=True,
    )

    print(f"Multi-sequence dataset overview:")
    print(f"  Total sequences: {len(dataset.sequence_metadata)}")
    print(f"  Total frames: {sum(m.num_frames for m in dataset.sequence_metadata.values())}")
    print(f"  Total valid frames: {len(dataset)}")
    print(f"  Sampling policy: {dataset.sampling_policy}")

    # Per-sequence statistics
    print(f"\nPer-sequence statistics:")
    for seq_name, metadata in dataset.sequence_metadata.items():
        print(f"\n  {seq_name}:")
        print(f"    Total frames: {metadata.num_frames}")
        print(f"    Valid frames: {len(metadata.valid_indices)}")
        print(f"    Validity rate: {len(metadata.valid_indices)/metadata.num_frames*100:.1f}%")
        print(f"    RGB resolution: {metadata.rgb_resolution}")
        print(f"    Sonar resolution: {metadata.sonar_resolution}")
        print(f"    Global index range: [{metadata.start_idx}, {metadata.end_idx})")

        # Analyze occlusions for this sequence
        seq_dataset = dataset.sequence_datasets[seq_name]
        num_rgb_occluded = 0
        num_sonar_occluded = 0
        num_both_occluded = 0

        for idx in range(seq_dataset.num_frames):
            rgb_occ = (seq_dataset.rgb_bboxes[idx] == 0).all()
            sonar_occ = (seq_dataset.sonar_bboxes[idx] == 0).all()

            if rgb_occ:
                num_rgb_occluded += 1
            if sonar_occ:
                num_sonar_occluded += 1
            if rgb_occ and sonar_occ:
                num_both_occluded += 1

        print(f"    Occlusions:")
        print(f"      RGB occluded: {num_rgb_occluded} ({num_rgb_occluded/seq_dataset.num_frames*100:.1f}%)")
        print(f"      Sonar occluded: {num_sonar_occluded} ({num_sonar_occluded/seq_dataset.num_frames*100:.1f}%)")
        print(f"      Both occluded: {num_both_occluded} ({num_both_occluded/seq_dataset.num_frames*100:.1f}%)")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" AquaSOT-3D Multi-Sequence DataLoader Usage Examples")
    print("="*80)

    # Check if data exists
    data_exists = Path('data/uuv2').exists() and Path('data/uuv3').exists()

    if not data_exists:
        print("\n⚠️  Warning: Data directories 'data/uuv2' and/or 'data/uuv3' not found.")
        print("   Please download the dataset first or update the data_root path.")
        print("\n   Running with augmentation example only (uses dummy data)...\n")

        # Run only augmentation example with dummy data
        example_3_augmentation()

    else:
        # Run all examples
        print("\n📊 Running multi-sequence examples with UUV2 and UUV3 sequences...")
        example_1_multi_sequence_dataset_loading()
        example_2_intra_sequence_tracking()
        example_3_augmentation()
        example_4_multi_sequence_dataloader()
        example_5_multi_sequence_statistics()

    print("\n" + "="*80)
    print(" Examples completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
