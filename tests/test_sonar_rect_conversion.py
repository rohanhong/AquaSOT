"""
Test Sonar Rectangular Conversion

Validates the polar-to-rectangular sonar image conversion by:
1. Randomly selecting 5 samples across all sequences
2. Visualizing original polar vs transformed rectangular images
3. Verifying bbox transformations
4. Checking dimensional correctness

Usage:
    python tests/test_sonar_rect_conversion.py
    python tests/test_sonar_rect_conversion.py --num-samples 10
    python tests/test_sonar_rect_conversion.py --output-dir custom_output

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SonarConversionValidator:
    """Validator for sonar polar-to-rectangular conversion."""

    def __init__(self, data_root: str = 'data', output_dir: str = 'outputs/sonar_rect_validation'):
        """
        Initialize validator.

        Args:
            data_root: Dataset root directory
            output_dir: Output directory for visualizations
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sequences = [
            'uuv1', 'uuv2', 'uuv3', 'uuv4', 'uuv5', 'uuv6',
            'uuv7', 'uuv8', 'uuv9', 'uuv10', 'uuv11', 'uuv12'
        ]

    def load_sample(
        self, seq_name: str, frame_id: int
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
        """
        Load a sample (polar and rectangular versions).

        Args:
            seq_name: Sequence name
            frame_id: Frame ID (1-indexed)

        Returns:
            Tuple of (polar_img, rect_img, polar_bbox, rect_bbox)
        """
        seq_path = self.data_root / seq_name

        # Load images
        polar_img_path = seq_path / 'sonar' / 'img' / f'{frame_id}.jpg'
        rect_img_path = seq_path / 'sonar' / 'img_rect' / f'{frame_id}.jpg'

        polar_img = cv2.imread(str(polar_img_path))
        rect_img = cv2.imread(str(rect_img_path))

        if polar_img is None:
            raise FileNotFoundError(f"Polar image not found: {polar_img_path}")
        if rect_img is None:
            raise FileNotFoundError(f"Rectangular image not found: {rect_img_path}")

        # Load bboxes
        polar_gt_path = seq_path / 'sonar' / 'groundtruth.txt'
        rect_gt_path = seq_path / 'sonar' / 'groundtruth_rect.txt'

        with open(polar_gt_path) as f:
            polar_bboxes = [
                [float(x) for x in line.strip().split(',')]
                for line in f
            ]

        with open(rect_gt_path) as f:
            rect_bboxes = [
                [float(x) for x in line.strip().split(',')]
                for line in f
            ]

        # Get bbox for this frame (1-indexed)
        polar_bbox = polar_bboxes[frame_id - 1]
        rect_bbox = rect_bboxes[frame_id - 1]

        return polar_img, rect_img, polar_bbox, rect_bbox

    def find_available_samples(self) -> List[Tuple[str, int]]:
        """
        Find all available samples (sequences with converted data).

        Returns:
            List of (seq_name, frame_count) tuples
        """
        available = []

        for seq_name in self.sequences:
            seq_path = self.data_root / seq_name
            rect_dir = seq_path / 'sonar' / 'img_rect'

            if rect_dir.exists():
                frame_files = list(rect_dir.glob('*.jpg'))
                if frame_files:
                    available.append((seq_name, len(frame_files)))

        return available

    def random_sample_selection(
        self, num_samples: int = 5
    ) -> List[Tuple[str, int]]:
        """
        Randomly select samples from available sequences.

        Args:
            num_samples: Number of samples to select

        Returns:
            List of (seq_name, frame_id) tuples
        """
        available = self.find_available_samples()

        if not available:
            raise RuntimeError(
                "No converted sequences found. Run conversion script first:\n"
                "  python scripts/preprocess_sonar.py --all"
            )

        # Sample from available sequences
        samples = []
        for _ in range(num_samples):
            seq_name, frame_count = random.choice(available)
            frame_id = random.randint(1, frame_count)
            samples.append((seq_name, frame_id))

        return samples

    def verify_dimensions(self, rect_img: np.ndarray) -> bool:
        """
        Verify rectangular image has correct dimensions.

        Args:
            rect_img: Rectangular image

        Returns:
            True if dimensions are correct
        """
        h, w = rect_img.shape[:2]
        expected_h, expected_w = 376, 682

        if (h, w) != (expected_h, expected_w):
            print(f"  ✗ Dimension mismatch: got {w}x{h}, expected {expected_w}x{expected_h}")
            return False

        return True

    def verify_bbox(self, bbox: List[float], img_shape: Tuple[int, int]) -> bool:
        """
        Verify bbox is within image bounds.

        Args:
            bbox: Bounding box [x, y, w, h]
            img_shape: Image shape (height, width)

        Returns:
            True if bbox is valid
        """
        x, y, w, h = bbox
        img_h, img_w = img_shape

        # Handle occlusion annotation
        if w == 0 or h == 0:
            return True

        # Check bounds
        if x < 0 or y < 0:
            print(f"  ✗ Negative coordinates: x={x}, y={y}")
            return False

        if x + w > img_w or y + h > img_h:
            print(f"  ✗ Bbox exceeds bounds: x+w={x+w} > {img_w} or y+h={y+h} > {img_h}")
            return False

        if w <= 0 or h <= 0:
            print(f"  ✗ Invalid size: w={w}, h={h}")
            return False

        return True

    def visualize_sample(
        self,
        seq_name: str,
        frame_id: int,
        polar_img: np.ndarray,
        rect_img: np.ndarray,
        polar_bbox: List[float],
        rect_bbox: List[float],
        ax_polar,
        ax_rect
    ):
        """
        Visualize one sample (polar and rectangular versions).

        Args:
            seq_name: Sequence name
            frame_id: Frame ID
            polar_img: Polar image
            rect_img: Rectangular image
            polar_bbox: Polar bbox [x, y, w, h]
            rect_bbox: Rectangular bbox [x, y, w, h]
            ax_polar: Matplotlib axis for polar visualization
            ax_rect: Matplotlib axis for rectangular visualization
        """
        # Convert BGR to RGB
        polar_rgb = cv2.cvtColor(polar_img, cv2.COLOR_BGR2RGB)
        rect_rgb = cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)

        # Plot polar image
        ax_polar.imshow(polar_rgb)
        ax_polar.set_title(f'{seq_name}/frame_{frame_id} (Polar Original)')
        ax_polar.axis('off')

        # Draw polar bbox
        if polar_bbox[2] > 0 and polar_bbox[3] > 0:
            x, y, w, h = polar_bbox
            rect_patch = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax_polar.add_patch(rect_patch)

        # Plot rectangular image
        ax_rect.imshow(rect_rgb)
        ax_rect.set_title(f'{seq_name}/frame_{frame_id} (Rectangular Converted)')
        ax_rect.axis('off')

        # Draw rectangular bbox
        if rect_bbox[2] > 0 and rect_bbox[3] > 0:
            x, y, w, h = rect_bbox
            rect_patch = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax_rect.add_patch(rect_patch)

    def validate_samples(
        self, samples: List[Tuple[str, int]], visualize: bool = True
    ) -> Dict:
        """
        Validate conversion for selected samples.

        Args:
            samples: List of (seq_name, frame_id) tuples
            visualize: Create visualization

        Returns:
            Validation statistics
        """
        stats = {
            'total': len(samples),
            'dimension_pass': 0,
            'bbox_pass': 0,
            'load_errors': [],
            'validation_errors': []
        }

        if visualize:
            num_samples = len(samples)
            fig, axes = plt.subplots(num_samples, 2, figsize=(14, 5 * num_samples))

            if num_samples == 1:
                axes = axes.reshape(1, -1)

        for idx, (seq_name, frame_id) in enumerate(samples):
            print(f"\nValidating {seq_name}/frame_{frame_id}...")

            try:
                # Load sample
                polar_img, rect_img, polar_bbox, rect_bbox = self.load_sample(
                    seq_name, frame_id
                )

                # Verify dimensions
                if self.verify_dimensions(rect_img):
                    stats['dimension_pass'] += 1
                    print(f"  ✓ Dimensions: {rect_img.shape[1]}x{rect_img.shape[0]}")
                else:
                    stats['validation_errors'].append(
                        (seq_name, frame_id, "Dimension mismatch")
                    )

                # Verify bbox
                if self.verify_bbox(rect_bbox, rect_img.shape[:2]):
                    stats['bbox_pass'] += 1
                    print(f"  ✓ Bbox: [{rect_bbox[0]:.1f}, {rect_bbox[1]:.1f}, "
                          f"{rect_bbox[2]:.1f}, {rect_bbox[3]:.1f}]")
                else:
                    stats['validation_errors'].append(
                        (seq_name, frame_id, "Bbox validation failed")
                    )

                # Visualize
                if visualize:
                    self.visualize_sample(
                        seq_name, frame_id,
                        polar_img, rect_img,
                        polar_bbox, rect_bbox,
                        axes[idx, 0], axes[idx, 1]
                    )

            except Exception as e:
                stats['load_errors'].append((seq_name, frame_id, str(e)))
                print(f"  ✗ Error: {e}")

        # Save visualization
        if visualize:
            plt.tight_layout()
            output_path = self.output_dir / 'conversion_validation.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Visualization saved: {output_path}")

        return stats

    def print_summary(self, stats: Dict):
        """
        Print validation summary.

        Args:
            stats: Validation statistics
        """
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {stats['total']}")
        print(f"Dimension checks passed: {stats['dimension_pass']}/{stats['total']}")
        print(f"Bbox checks passed: {stats['bbox_pass']}/{stats['total']}")

        if stats['load_errors']:
            print(f"\nLoad errors: {len(stats['load_errors'])}")
            for seq, frame, error in stats['load_errors']:
                print(f"  {seq}/frame_{frame}: {error}")

        if stats['validation_errors']:
            print(f"\nValidation errors: {len(stats['validation_errors'])}")
            for seq, frame, error in stats['validation_errors']:
                print(f"  {seq}/frame_{frame}: {error}")

        # Overall status
        all_pass = (
            stats['dimension_pass'] == stats['total'] and
            stats['bbox_pass'] == stats['total'] and
            len(stats['load_errors']) == 0
        )

        print("\n" + "=" * 60)
        if all_pass:
            print("✓ ALL CHECKS PASSED")
            print("  Conversion appears correct. Ready for training.")
        else:
            print("✗ VALIDATION FAILED")
            print("  Please review errors above and re-run conversion if needed.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate sonar polar-to-rectangular conversion'
    )
    parser.add_argument(
        '--num-samples', type=int, default=5,
        help='Number of random samples to validate (default: 5)'
    )
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Dataset root directory (default: data)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/sonar_rect_validation',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize validator
    validator = SonarConversionValidator(
        data_root=args.data_root,
        output_dir=args.output_dir
    )

    print("Sonar Rectangular Conversion Validator")
    print("=" * 60)

    # Find available samples
    available = validator.find_available_samples()
    if not available:
        print("✗ No converted sequences found.")
        print("\nPlease run conversion first:")
        print("  python scripts/preprocess_sonar.py --all")
        return 1

    print(f"Found {len(available)} converted sequences:")
    for seq_name, frame_count in available:
        print(f"  {seq_name}: {frame_count} frames")

    # Select random samples
    print(f"\nRandomly selecting {args.num_samples} samples (seed={args.seed})...")
    samples = validator.random_sample_selection(args.num_samples)
    print("Selected samples:")
    for seq_name, frame_id in samples:
        print(f"  {seq_name}/frame_{frame_id}")

    # Validate
    stats = validator.validate_samples(
        samples,
        visualize=not args.no_viz
    )

    # Print summary
    validator.print_summary(stats)

    # Return exit code
    all_pass = (
        stats['dimension_pass'] == stats['total'] and
        stats['bbox_pass'] == stats['total'] and
        len(stats['load_errors']) == 0
    )

    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())