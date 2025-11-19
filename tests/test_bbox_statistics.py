"""
Test script to compute bbox statistics from rectangular sonar dataset.

This script validates whether the current bbox initialization values
(derived from polar coordinates) are still accurate for rectangular
coordinate preprocessing.

Usage:
    python tests/test_bbox_statistics.py
    python tests/test_bbox_statistics.py --data-root data
    python tests/test_bbox_statistics.py --visualize

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Sonar image dimensions after rectangular preprocessing
SONAR_WIDTH = 682
SONAR_HEIGHT = 376

# Current initialization values from box_head.py (derived from polar coords)
CURRENT_INIT_VALUES = {
    'cx': 0.37,
    'cy': 0.92,
    'w': 0.11,
    'h': 0.03
}


def logit(p: float) -> float:
    """
    Compute logit (inverse sigmoid): logit(p) = log(p / (1-p))

    Args:
        p: Probability value in range (0, 1)

    Returns:
        Logit value
    """
    if p <= 0 or p >= 1:
        return float('inf') if p >= 1 else float('-inf')
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    """
    Compute sigmoid: σ(x) = 1 / (1 + exp(-x))

    Args:
        x: Input value

    Returns:
        Sigmoid output in range (0, 1)
    """
    return 1.0 / (1.0 + math.exp(-x))


def load_bbox_annotations(data_root: Path) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Load all bbox annotations from rectangular sonar groundtruth files.

    Args:
        data_root: Path to data directory containing uuvN folders

    Returns:
        Dictionary mapping sequence names to lists of (cx, cy, w, h) tuples
        in normalized coordinates [0, 1]
    """
    annotations = {}

    # Find all sequence directories
    sequence_dirs = sorted([d for d in data_root.iterdir()
                           if d.is_dir() and d.name.startswith('uuv')])

    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence directories found in {data_root}")

    for seq_dir in sequence_dirs:
        seq_name = seq_dir.name
        gt_file = seq_dir / 'sonar' / 'groundtruth_rect.txt'

        if not gt_file.exists():
            print(f"Warning: {gt_file} not found, skipping {seq_name}")
            continue

        seq_bboxes = []
        with open(gt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split(',')
                    if len(parts) < 4:
                        print(f"Warning: {seq_name} line {line_num}: "
                              f"expected 4 values, got {len(parts)}")
                        continue

                    # Parse pixel coordinates
                    cx_px, cy_px, w_px, h_px = map(float, parts[:4])

                    # Normalize to [0, 1]
                    cx = cx_px / SONAR_WIDTH
                    cy = cy_px / SONAR_HEIGHT
                    w = w_px / SONAR_WIDTH
                    h = h_px / SONAR_HEIGHT

                    # Sanity check
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and
                           0 <= w <= 1 and 0 <= h <= 1):
                        print(f"Warning: {seq_name} line {line_num}: "
                              f"bbox out of bounds (cx={cx:.3f}, cy={cy:.3f}, "
                              f"w={w:.3f}, h={h:.3f})")
                        continue

                    seq_bboxes.append((cx, cy, w, h))

                except ValueError as e:
                    print(f"Warning: {seq_name} line {line_num}: "
                          f"failed to parse: {e}")
                    continue

        if seq_bboxes:
            annotations[seq_name] = seq_bboxes
            print(f"Loaded {len(seq_bboxes)} bboxes from {seq_name}")

    return annotations


def compute_statistics(annotations: Dict[str, List[Tuple[float, float, float, float]]]) -> Dict:
    """
    Compute bbox statistics across all sequences.

    Args:
        annotations: Dictionary of sequence annotations

    Returns:
        Dictionary containing statistical measures for each parameter
    """
    # Flatten all bboxes
    all_bboxes = []
    for seq_bboxes in annotations.values():
        all_bboxes.extend(seq_bboxes)

    if not all_bboxes:
        raise ValueError("No valid bbox annotations found")

    # Convert to numpy arrays
    all_bboxes = np.array(all_bboxes)  # Shape: (N, 4)
    cx_array = all_bboxes[:, 0]
    cy_array = all_bboxes[:, 1]
    w_array = all_bboxes[:, 2]
    h_array = all_bboxes[:, 3]

    # Compute statistics
    stats = {}
    for name, data in [('cx', cx_array), ('cy', cy_array),
                       ('w', w_array), ('h', h_array)]:
        stats[name] = {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'p25': float(np.percentile(data, 25)),
            'p75': float(np.percentile(data, 75)),
            'count': len(data)
        }

    return stats


def compare_with_initialization(stats: Dict) -> None:
    """
    Compare computed statistics with current initialization values.

    Args:
        stats: Dictionary of computed statistics
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Current Init vs Actual Distribution")
    print("=" * 80)

    print(f"\n{'Param':<6} {'Init':<8} {'Mean':<8} {'Median':<8} {'Std':<8} "
          f"{'Deviation':<12} {'Status'}")
    print("-" * 80)

    needs_update = []

    for param in ['cx', 'cy', 'w', 'h']:
        init = CURRENT_INIT_VALUES[param]
        mean = stats[param]['mean']
        median = stats[param]['median']
        std = stats[param]['std']
        deviation = abs(init - mean)
        deviation_sigma = deviation / std if std > 0 else 0

        # Assess if initialization is still valid
        if deviation < std:
            status = "✓ GOOD"
        elif deviation < 2 * std:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ OUTDATED"
            needs_update.append(param)

        print(f"{param:<6} {init:<8.3f} {mean:<8.3f} {median:<8.3f} "
              f"{std:<8.3f} {deviation:<12.3f} {status} ({deviation_sigma:.2f}σ)")

    print("\nGuidelines:")
    print("  ✓ GOOD:       deviation < 1σ (within 1 std dev)")
    print("  ⚠️  ACCEPTABLE: deviation < 2σ (within 2 std dev)")
    print("  ❌ OUTDATED:   deviation > 2σ (should recompute)")

    if needs_update:
        print(f"\n⚠️  WARNING: Parameters {needs_update} need updating!")
    else:
        print("\n✅ All parameters are within acceptable range")


def print_recommendations(stats: Dict) -> None:
    """
    Print recommended initialization values based on statistics.

    Args:
        stats: Dictionary of computed statistics
    """
    print("\n" + "=" * 80)
    print("RECOMMENDED INITIALIZATION VALUES")
    print("=" * 80)

    print("\nStrategy 1: Use MEDIAN (robust to outliers, recommended):")
    print("```python")
    print("# In src/aquasot3d/models/heads/box_head.py, lines 799-809:")
    for i, param in enumerate(['cx', 'cy', 'w', 'h']):
        median = stats[param]['median']
        logit_val = logit(median)
        print(f"reg_output_layer.bias.data[{i}] = {logit_val:7.3f}  "
              f"# {param}={median:.3f} (median)")
    print("```")

    print("\nStrategy 2: Use MEAN (maximum likelihood estimate):")
    print("```python")
    for i, param in enumerate(['cx', 'cy', 'w', 'h']):
        mean = stats[param]['mean']
        logit_val = logit(mean)
        print(f"reg_output_layer.bias.data[{i}] = {logit_val:7.3f}  "
              f"# {param}={mean:.3f} (mean)")
    print("```")

    print("\nCurrent values (polar-derived):")
    print("```python")
    for i, param in enumerate(['cx', 'cy', 'w', 'h']):
        current = CURRENT_INIT_VALUES[param]
        logit_val = logit(current)
        print(f"reg_output_layer.bias.data[{i}] = {logit_val:7.3f}  "
              f"# {param}={current:.3f} (CURRENT)")
    print("```")


def visualize_distributions(stats: Dict, annotations: Dict) -> None:
    """
    Print ASCII histogram visualization of distributions.

    Args:
        stats: Dictionary of computed statistics
        annotations: Dictionary of sequence annotations
    """
    print("\n" + "=" * 80)
    print("DISTRIBUTION VISUALIZATION")
    print("=" * 80)

    # Flatten all bboxes
    all_bboxes = []
    for seq_bboxes in annotations.values():
        all_bboxes.extend(seq_bboxes)
    all_bboxes = np.array(all_bboxes)

    for i, param in enumerate(['cx', 'cy', 'w', 'h']):
        data = all_bboxes[:, i]

        print(f"\n{param.upper()} Distribution:")
        print(f"  Range: [{stats[param]['min']:.3f}, {stats[param]['max']:.3f}]")
        print(f"  Mean:  {stats[param]['mean']:.3f}")
        print(f"  Median: {stats[param]['median']:.3f}")
        print(f"  Std:   {stats[param]['std']:.3f}")
        print(f"  Q1-Q3: [{stats[param]['p25']:.3f}, {stats[param]['p75']:.3f}]")

        # Simple ASCII histogram
        hist, bins = np.histogram(data, bins=20)
        max_count = max(hist)
        print("\n  Histogram:")
        for j in range(len(hist)):
            bar_len = int(50 * hist[j] / max_count) if max_count > 0 else 0
            print(f"  {bins[j]:.2f}-{bins[j+1]:.2f} |{'█' * bar_len} {hist[j]}")

        # Mark current init value
        current_val = CURRENT_INIT_VALUES[param]
        current_bin = int((current_val - stats[param]['min']) /
                          (stats[param]['max'] - stats[param]['min']) * 20)
        current_bin = min(max(current_bin, 0), 19)
        print(f"  Current init ({current_val:.3f}) ↑ bin {current_bin}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute bbox statistics for rectangular sonar dataset"
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data'),
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show distribution visualizations'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BBOX STATISTICS ANALYSIS: Rectangular Sonar Dataset")
    print("=" * 80)
    print(f"\nData root: {args.data_root}")
    print(f"Image size: {SONAR_WIDTH}x{SONAR_HEIGHT}")

    # Load annotations
    print("\n" + "=" * 80)
    print("LOADING ANNOTATIONS")
    print("=" * 80)

    try:
        annotations = load_bbox_annotations(args.data_root)
    except Exception as e:
        print(f"\n❌ Error loading annotations: {e}")
        return 1

    total_frames = sum(len(bboxes) for bboxes in annotations.values())
    print(f"\nLoaded {len(annotations)} sequences, {total_frames} total frames")

    # Compute statistics
    print("\n" + "=" * 80)
    print("COMPUTING STATISTICS")
    print("=" * 80)

    try:
        stats = compute_statistics(annotations)
    except Exception as e:
        print(f"\n❌ Error computing statistics: {e}")
        return 1

    # Print detailed statistics
    print("\nDetailed Statistics:")
    for param in ['cx', 'cy', 'w', 'h']:
        print(f"\n{param.upper()}:")
        for key, value in stats[param].items():
            if key == 'count':
                print(f"  {key:<10}: {value}")
            else:
                print(f"  {key:<10}: {value:.4f}")

    # Compare with current initialization
    compare_with_initialization(stats)

    # Print recommendations
    print_recommendations(stats)

    # Visualize if requested
    if args.visualize:
        visualize_distributions(stats, annotations)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())