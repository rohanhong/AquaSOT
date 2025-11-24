#!/usr/bin/env python3
"""
Dataset Analysis Script

Analyze dataset statistics including sequence lengths, frame sizes,
occlusion rates, and bounding box distributions.

CRITICAL FIX (Audit 2025112404 Finding 5): Full implementation of dataset analysis tool.

Usage:
    python scripts/analyze_dataset.py --data_root data/UUV2 --output stats.json
    python scripts/analyze_dataset.py --data_root data/UUV2 --output stats.json --visualize

Author: AquaSOT-3D Team
Date: 2025-11-24
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from aquasot3d.dataloader.datasets.rgb_sonar_dataset import MultiSequenceRGBSonarDataset


def compute_bbox_statistics(bboxes):
    """
    Compute statistics for bounding boxes.

    Args:
        bboxes: List of bboxes in XYWH format

    Returns:
        Dictionary of statistics
    """
    valid_bboxes = [bbox for bbox in bboxes if not np.all(bbox == 0)]

    if len(valid_bboxes) == 0:
        return {
            'count': 0,
            'occlusion_rate': 1.0,
            'mean_width': 0,
            'mean_height': 0,
            'mean_area': 0,
            'std_width': 0,
            'std_height': 0,
            'std_area': 0
        }

    widths = [bbox[2] for bbox in valid_bboxes]
    heights = [bbox[3] for bbox in valid_bboxes]
    areas = [bbox[2] * bbox[3] for bbox in valid_bboxes]

    occlusion_count = len(bboxes) - len(valid_bboxes)
    occlusion_rate = occlusion_count / len(bboxes) if len(bboxes) > 0 else 0

    return {
        'count': len(valid_bboxes),
        'total_frames': len(bboxes),
        'occlusion_count': occlusion_count,
        'occlusion_rate': float(occlusion_rate),
        'mean_width': float(np.mean(widths)),
        'mean_height': float(np.mean(heights)),
        'mean_area': float(np.mean(areas)),
        'std_width': float(np.std(widths)),
        'std_height': float(np.std(heights)),
        'std_area': float(np.std(areas)),
        'min_width': float(np.min(widths)),
        'max_width': float(np.max(widths)),
        'min_height': float(np.min(heights)),
        'max_height': float(np.max(heights))
    }


def analyze_dataset(data_root: Path):
    """
    Analyze multi-sequence dataset.

    Args:
        data_root: Path to dataset root

    Returns:
        Dictionary containing dataset statistics
    """
    print(f"Loading dataset from: {data_root}")
    print("Analyzing both RGB and Sonar sensors (dual-sensor system)")

    # CRITICAL FIX (Audit 2025112404): Discover sequences from data_root
    # MultiSequenceRGBSonarDataset requires explicit sequence_names list
    sequence_names = []
    for item in data_root.iterdir():
        if item.is_dir() and (item / "light").exists() and (item / "sonar").exists():
            sequence_names.append(item.name)

    if not sequence_names:
        raise ValueError(
            f"No valid sequences found in {data_root}. "
            f"Each sequence must have 'light/' and 'sonar/' subdirectories."
        )

    sequence_names.sort()  # Consistent ordering
    print(f"Discovered sequences: {sequence_names}")

    # Load dataset (always loads both RGB and Sonar)
    dataset = MultiSequenceRGBSonarDataset(
        data_root=data_root,
        sequence_names=sequence_names,
        return_occluded=True  # Include occluded frames for statistics
    )

    print(f"Total sequences: {len(dataset.sequences)}")
    print(f"Total frames: {len(dataset)}")

    # Collect statistics per sequence
    sequence_stats = {}

    # CRITICAL FIX: dataset.sequences returns SequenceMetadata objects, not strings
    # Need to extract sequence_name and use it as key
    for seq_meta in tqdm(dataset.sequences, desc="Analyzing sequences"):
        seq_name = seq_meta.sequence_name if hasattr(seq_meta, 'sequence_name') else str(seq_meta)
        seq_dataset = dataset.sequence_datasets[seq_name]

        seq_stats = {
            'sequence_name': seq_name,
            'num_frames': seq_dataset.num_frames,
        }

        # Always analyze both RGB and Sonar (dual-sensor system)
        seq_stats['rgb'] = compute_bbox_statistics(seq_dataset.rgb_bboxes)
        seq_stats['sonar'] = compute_bbox_statistics(seq_dataset.sonar_bboxes)
        seq_stats['sonar']['is_rectified'] = seq_dataset.sonar_is_rectified

        sequence_stats[seq_name] = seq_stats

    # Compute overall statistics
    total_frames = sum(stats['num_frames'] for stats in sequence_stats.values())

    overall_stats = {
        'dataset_root': str(data_root),
        'num_sequences': len(dataset.sequences),
        'total_frames': total_frames,
        'modalities': ['rgb', 'sonar'],  # Always analyze both sensors
        'sequences': [
            seq_meta.sequence_name if hasattr(seq_meta, 'sequence_name') else str(seq_meta)
            for seq_meta in dataset.sequences
        ]
    }

    # Aggregate bbox statistics across sequences (always both RGB and Sonar)
    all_rgb_stats = [stats['rgb'] for stats in sequence_stats.values()]
    overall_stats['rgb'] = {
        'total_valid_frames': sum(s['count'] for s in all_rgb_stats),
        'total_occluded_frames': sum(s['occlusion_count'] for s in all_rgb_stats),
        'overall_occlusion_rate': float(sum(s['occlusion_count'] for s in all_rgb_stats) / total_frames),
        'mean_bbox_width': float(np.mean([s['mean_width'] for s in all_rgb_stats if s['count'] > 0])),
        'mean_bbox_height': float(np.mean([s['mean_height'] for s in all_rgb_stats if s['count'] > 0])),
        'mean_bbox_area': float(np.mean([s['mean_area'] for s in all_rgb_stats if s['count'] > 0]))
    }

    all_sonar_stats = [stats['sonar'] for stats in sequence_stats.values()]
    rectified_count = sum(1 for stats in sequence_stats.values() if stats['sonar'].get('is_rectified', False))
    overall_stats['sonar'] = {
        'total_valid_frames': sum(s['count'] for s in all_sonar_stats),
        'total_occluded_frames': sum(s['occlusion_count'] for s in all_sonar_stats),
        'overall_occlusion_rate': float(sum(s['occlusion_count'] for s in all_sonar_stats) / total_frames),
        'mean_bbox_width': float(np.mean([s['mean_width'] for s in all_sonar_stats if s['count'] > 0])),
        'mean_bbox_height': float(np.mean([s['mean_height'] for s in all_sonar_stats if s['count'] > 0])),
        'mean_bbox_area': float(np.mean([s['mean_area'] for s in all_sonar_stats if s['count'] > 0])),
        'rectified_sequences': rectified_count,
        'polar_sequences': len(dataset.sequences) - rectified_count
    }

    return {
        'overall': overall_stats,
        'per_sequence': sequence_stats
    }


def visualize_statistics(stats, output_dir: Path):
    """
    Generate visualizations for dataset statistics.

    Args:
        stats: Statistics dictionary from analyze_dataset
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot occlusion rates per sequence (always both RGB and Sonar)
    seq_names = list(stats['per_sequence'].keys())

    # RGB occlusion rates
    rgb_occlusion_rates = [stats['per_sequence'][seq]['rgb']['occlusion_rate'] for seq in seq_names]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(seq_names)), rgb_occlusion_rates)
    plt.xlabel('Sequence')
    plt.ylabel('Occlusion Rate')
    plt.title('RGB Occlusion Rates per Sequence')
    plt.xticks(range(len(seq_names)), seq_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'rgb_occlusion_rates.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'rgb_occlusion_rates.png'}")

    # Sonar occlusion rates
    sonar_occlusion_rates = [stats['per_sequence'][seq]['sonar']['occlusion_rate'] for seq in seq_names]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(seq_names)), sonar_occlusion_rates)
    plt.xlabel('Sequence')
    plt.ylabel('Occlusion Rate')
    plt.title('Sonar Occlusion Rates per Sequence')
    plt.xticks(range(len(seq_names)), seq_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'sonar_occlusion_rates.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'sonar_occlusion_rates.png'}")

    # Plot bbox size distributions (both sensors)
    rgb_widths = [stats['per_sequence'][seq]['rgb']['mean_width'] for seq in seq_names]
    sonar_widths = [stats['per_sequence'][seq]['sonar']['mean_width'] for seq in seq_names]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(seq_names))
    width = 0.35
    plt.bar(x - width/2, rgb_widths, width, label='RGB', alpha=0.8)
    plt.bar(x + width/2, sonar_widths, width, label='Sonar', alpha=0.8)
    plt.xlabel('Sequence')
    plt.ylabel('Mean Bbox Width (pixels)')
    plt.title('Mean Bounding Box Width Comparison')
    plt.xticks(x, seq_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_width_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'bbox_width_comparison.png'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset Analysis Tool for AquaSOT-3D (Dual-Sensor System)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze full dataset
    python scripts/analyze_dataset.py --data_root data/UUV2 --output stats.json

    # Analyze with visualization
    python scripts/analyze_dataset.py --data_root data/UUV2 --output stats.json --visualize

Note: AquaSOT-3D always analyzes both RGB and Sonar sensors (dual-sensor system)
        """
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--output', type=str, default='dataset_stats.json',
                        help='Output JSON file for statistics')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plot_dir', type=str, default='dataset_analysis',
                        help='Directory to save plots')
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Dataset root not found: {data_root}")
        return 1

    print("="*60)
    print("AquaSOT-3D Dataset Analysis Tool (Dual-Sensor System)")
    print("="*60)

    # Analyze dataset (always both RGB and Sonar)
    stats = analyze_dataset(data_root)

    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total Sequences: {stats['overall']['num_sequences']}")
    print(f"Total Frames: {stats['overall']['total_frames']}")
    print(f"Sensors: RGB and Sonar (dual-sensor system)")

    print(f"\nRGB Statistics:")
    print(f"  Valid Frames: {stats['overall']['rgb']['total_valid_frames']}")
    print(f"  Occluded Frames: {stats['overall']['rgb']['total_occluded_frames']}")
    print(f"  Occlusion Rate: {stats['overall']['rgb']['overall_occlusion_rate']:.2%}")
    print(f"  Mean BBox Size: {stats['overall']['rgb']['mean_bbox_width']:.1f} x {stats['overall']['rgb']['mean_bbox_height']:.1f}")

    print(f"\nSonar Statistics:")
    print(f"  Valid Frames: {stats['overall']['sonar']['total_valid_frames']}")
    print(f"  Occluded Frames: {stats['overall']['sonar']['total_occluded_frames']}")
    print(f"  Occlusion Rate: {stats['overall']['sonar']['overall_occlusion_rate']:.2%}")
    print(f"  Mean BBox Size: {stats['overall']['sonar']['mean_bbox_width']:.1f} x {stats['overall']['sonar']['mean_bbox_height']:.1f}")
    print(f"  Rectified Sequences: {stats['overall']['sonar']['rectified_sequences']}")
    print(f"  Polar Sequences: {stats['overall']['sonar']['polar_sequences']}")

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {output_path}")

    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_statistics(stats, Path(args.plot_dir))
        print(f"Plots saved to: {args.plot_dir}/")

    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
