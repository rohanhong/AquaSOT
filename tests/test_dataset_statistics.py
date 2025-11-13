"""
Unit Test for Dataset Statistics Collection

Analyzes all sequences in data/ directory to collect:
- Frame counts for light and sonar modalities
- Image resolutions for both modalities
- Frame count inconsistencies between modalities
- Groundtruth file integrity

Author: AquaSOT-3D Team
Date: 2025-11-13
"""

import pytest
from pathlib import Path
from PIL import Image
import json


def collect_dataset_statistics(data_root="data"):
    """
    Collect comprehensive statistics for all sequences in dataset.

    Args:
        data_root: Root directory containing all sequences

    Returns:
        Dictionary with statistics for each sequence
    """
    data_path = Path(data_root)

    if not data_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    sequences = sorted([d for d in data_path.iterdir() if d.is_dir()])

    stats = {}

    for seq_dir in sequences:
        seq_name = seq_dir.name
        light_dir = seq_dir / "light" / "img"
        sonar_dir = seq_dir / "sonar" / "img"
        light_gt = seq_dir / "light" / "groundtruth.txt"
        sonar_gt = seq_dir / "sonar" / "groundtruth.txt"

        # Count frames
        light_frames = sorted(list(light_dir.glob("*.jpg"))) if light_dir.exists() else []
        sonar_frames = sorted(list(sonar_dir.glob("*.jpg"))) if sonar_dir.exists() else []

        light_count = len(light_frames)
        sonar_count = len(sonar_frames)

        # Get image resolutions from first frame
        light_resolution = None
        sonar_resolution = None

        if light_frames:
            with Image.open(light_frames[0]) as img:
                light_resolution = img.size  # (width, height)

        if sonar_frames:
            with Image.open(sonar_frames[0]) as img:
                sonar_resolution = img.size  # (width, height)

        # Count groundtruth lines
        light_gt_lines = 0
        sonar_gt_lines = 0

        if light_gt.exists():
            with open(light_gt, 'r') as f:
                light_gt_lines = len(f.readlines())

        if sonar_gt.exists():
            with open(sonar_gt, 'r') as f:
                sonar_gt_lines = len(f.readlines())

        # Detect inconsistencies
        frame_mismatch = light_count != sonar_count
        gt_mismatch = light_gt_lines != sonar_gt_lines
        frame_gt_mismatch = (light_count != light_gt_lines) or (sonar_count != sonar_gt_lines)

        stats[seq_name] = {
            "light_frames": light_count,
            "sonar_frames": sonar_count,
            "light_resolution": light_resolution,
            "sonar_resolution": sonar_resolution,
            "light_gt_lines": light_gt_lines,
            "sonar_gt_lines": sonar_gt_lines,
            "frame_count_mismatch": frame_mismatch,
            "gt_count_mismatch": gt_mismatch,
            "frame_gt_mismatch": frame_gt_mismatch,
        }

    return stats


def print_statistics_report(stats):
    """
    Print formatted statistics report.

    Args:
        stats: Dictionary from collect_dataset_statistics
    """
    print("\n" + "="*80)
    print("Dataset Statistics Report")
    print("="*80)

    for seq_name in sorted(stats.keys()):
        s = stats[seq_name]
        print(f"\n{seq_name}:")
        print(f"  Light: {s['light_frames']} frames, resolution {s['light_resolution']}, GT lines {s['light_gt_lines']}")
        print(f"  Sonar: {s['sonar_frames']} frames, resolution {s['sonar_resolution']}, GT lines {s['sonar_gt_lines']}")

        if s['frame_count_mismatch']:
            diff = abs(s['light_frames'] - s['sonar_frames'])
            print(f"  WARNING: Frame count mismatch! Difference: {diff}")

        if s['gt_count_mismatch']:
            diff = abs(s['light_gt_lines'] - s['sonar_gt_lines'])
            print(f"  WARNING: Groundtruth line count mismatch! Difference: {diff}")

        if s['frame_gt_mismatch']:
            print(f"  WARNING: Frame count does not match groundtruth lines!")

    print("\n" + "="*80)
    print("Summary:")
    print("="*80)

    total_sequences = len(stats)
    total_light_frames = sum(s['light_frames'] for s in stats.values())
    total_sonar_frames = sum(s['sonar_frames'] for s in stats.values())

    print(f"Total sequences: {total_sequences}")
    print(f"Total light frames: {total_light_frames}")
    print(f"Total sonar frames: {total_sonar_frames}")

    # Check resolution consistency
    light_resolutions = set(s['light_resolution'] for s in stats.values() if s['light_resolution'])
    sonar_resolutions = set(s['sonar_resolution'] for s in stats.values() if s['sonar_resolution'])

    print(f"\nLight resolutions found: {light_resolutions}")
    print(f"Sonar resolutions found: {sonar_resolutions}")

    # Count issues
    frame_mismatches = sum(1 for s in stats.values() if s['frame_count_mismatch'])
    gt_mismatches = sum(1 for s in stats.values() if s['gt_count_mismatch'])

    if frame_mismatches > 0:
        print(f"\nSequences with frame count mismatches: {frame_mismatches}")

    if gt_mismatches > 0:
        print(f"Sequences with GT line count mismatches: {gt_mismatches}")

    print("="*80 + "\n")


class TestDatasetStatistics:
    """Test dataset statistics collection."""

    def test_collect_statistics(self):
        """Test statistics collection for all sequences."""
        stats = collect_dataset_statistics()

        assert len(stats) > 0, "No sequences found in data/"

        # Verify all sequences have required fields
        for seq_name, seq_stats in stats.items():
            assert 'light_frames' in seq_stats
            assert 'sonar_frames' in seq_stats
            assert 'light_resolution' in seq_stats
            assert 'sonar_resolution' in seq_stats

            # Verify reasonable frame counts
            assert seq_stats['light_frames'] > 0, f"{seq_name}: no light frames"
            assert seq_stats['sonar_frames'] > 0, f"{seq_name}: no sonar frames"

    def test_print_report(self):
        """Test printing statistics report."""
        stats = collect_dataset_statistics()
        print_statistics_report(stats)


if __name__ == '__main__':
    # Run statistics collection and print report
    stats = collect_dataset_statistics()
    print_statistics_report(stats)

    # Save to JSON for documentation
    output_file = "dataset_statistics.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {output_file}")