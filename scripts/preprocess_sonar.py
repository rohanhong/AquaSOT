#!/usr/bin/env python3
"""
Sonar Polar-to-Rectangular Conversion Script

Converts fan-shaped polar sonar images to rectangular coordinates with
synchronized bounding box transformation. Designed for preprocessing
AquaSOT-3D dataset to eliminate runtime coordinate transformations.

Architecture Strategy:
- One-time preprocessing before training
- Unified output resolution: 682x376 pixels
- Preserves original data (outputs to *_rect suffixes)
- Lightweight implementation (~300 LOC)

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm


class SonarPolarToRectConverter:
    """
    Lightweight converter for sonar polar-to-rectangular transformation.

    Based on the reference implementation in tmp/coordinate_transform.py,
    simplified to focus on core conversion functionality without alignment.
    """

    def __init__(self, output_size: Tuple[int, int] = (376, 682)):
        """
        Initialize converter.

        Args:
            output_size: Unified output size (height, width) = (376, 682)
        """
        self.output_h, self.output_w = output_size
        print(f"Sonar Polar-to-Rectangular Converter initialized")
        print(f"Output resolution: {self.output_w}x{self.output_h} (unified)")

    def calculate_fov(self, width: int, height: int) -> float:
        """
        Calculate sonar field-of-view (FOV) dynamically based on image dimensions.

        Sonar sensor is positioned at bottom-center (cx=W/2, cy=H-1).
        FOV is determined by the angular span from left edge to right edge.

        Args:
            width: Sonar image width
            height: Sonar image height

        Returns:
            FOV in radians
        """
        center_x = width / 2
        center_y = height - 1

        # Calculate left boundary angle
        if center_x < height:
            sin_left = center_x / height
            angle_left = -np.arcsin(np.clip(sin_left, -1, 1))
        else:
            angle_left = -np.pi / 2

        # Calculate right boundary angle
        if (width - 1 - center_x) < height:
            sin_right = (width - 1 - center_x) / height
            angle_right = np.arcsin(np.clip(sin_right, -1, 1))
        else:
            angle_right = np.pi / 2

        # Total FOV
        fov_rad = angle_right - angle_left
        return fov_rad

    def create_remap_grids(
        self, input_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create cv2.remap mapping grids for polar-to-rectangular transformation.

        Mathematical transformation:
        For each rectangular output pixel (x_rect, y_rect):
        1. Normalize: theta_norm = x_rect / (W_out - 1), r_norm = (H_out - 1 - y_rect) / (H_out - 1)
        2. Map to polar: theta = -FOV/2 + theta_norm * FOV, r = r_norm * H_input
        3. Convert to Cartesian: x_polar = cx + r*sin(theta), y_polar = cy - r*cos(theta)
        4. Use (x_polar, y_polar) as source coordinates for cv2.remap

        Args:
            input_shape: (height, width) of input polar image

        Returns:
            Tuple of (map_x, map_y) for cv2.remap
        """
        h_in, w_in = input_shape

        # Sensor position (bottom-center)
        cx = w_in / 2
        cy = h_in - 1

        # Calculate FOV
        fov_rad = self.calculate_fov(w_in, h_in)

        # Create output coordinate grid (rectangular space)
        x_out = np.linspace(0, self.output_w - 1, self.output_w)
        y_out = np.linspace(0, self.output_h - 1, self.output_h)
        xx_out, yy_out = np.meshgrid(x_out, y_out)

        # Map rectangular coordinates to polar coordinates
        # Horizontal axis (x_out) -> angle (theta)
        # Vertical axis (y_out) -> radius (r), inverted (top=far, bottom=near)
        theta = -fov_rad / 2 + (xx_out / (self.output_w - 1)) * fov_rad
        r_normalized = np.clip((self.output_h - 1 - yy_out) / (self.output_h - 1), 0, 1)
        r_pixels = r_normalized * h_in

        # Convert polar to Cartesian (input image coordinates)
        x_in = cx + r_pixels * np.sin(theta)
        y_in = cy - r_pixels * np.cos(theta)

        # Clamp to valid range
        x_in = np.clip(x_in, 0, w_in - 1).astype(np.float32)
        y_in = np.clip(y_in, 0, h_in - 1).astype(np.float32)

        return x_in, y_in

    def transform_image(self, polar_img: np.ndarray) -> np.ndarray:
        """
        Transform polar sonar image to rectangular coordinates.

        Args:
            polar_img: Input polar image (H_polar, W_polar, C)

        Returns:
            Rectangular image (376, 682, C)
        """
        h, w = polar_img.shape[:2]

        # Create remap grids
        map_x, map_y = self.create_remap_grids((h, w))

        # Apply remapping (bilinear interpolation)
        rect_img = cv2.remap(
            polar_img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Resize to exact output size (ensure uniformity)
        if rect_img.shape[:2] != (self.output_h, self.output_w):
            rect_img = cv2.resize(rect_img, (self.output_w, self.output_h))

        return rect_img

    def transform_bbox(
        self, bbox: List[float], input_shape: Tuple[int, int]
    ) -> List[float]:
        """
        Transform bounding box from polar to rectangular coordinates.

        Uses 4-corner transformation method:
        1. Transform all 4 corners independently
        2. Compute axis-aligned bbox from transformed corners
        3. Handle edge cases (clipping, zero-area)

        Args:
            bbox: Bounding box [x, y, w, h] in polar coordinates
            input_shape: (height, width) of polar image

        Returns:
            Transformed bbox [x, y, w, h] in rectangular coordinates
        """
        x, y, w, h = bbox
        h_in, w_in = input_shape

        # Handle occlusion annotation (0,0,0,0)
        if w == 0 or h == 0:
            return [0.0, 0.0, 0.0, 0.0]

        # Sensor position
        cx = w_in / 2
        cy = h_in - 1

        # Calculate FOV
        fov_rad = self.calculate_fov(w_in, h_in)

        # Get 4 corners
        corners = [
            [x, y],           # Top-left
            [x + w, y],       # Top-right
            [x + w, y + h],   # Bottom-right
            [x, y + h]        # Bottom-left
        ]

        # Transform each corner
        transformed_corners = []
        for corner_x, corner_y in corners:
            # Clamp to valid range
            corner_x = np.clip(corner_x, 0, w_in - 1)
            corner_y = np.clip(corner_y, 0, h_in - 1)

            # Compute polar coordinates
            dx = corner_x - cx
            dy = cy - corner_y

            # Avoid division by zero
            if dy <= 0:
                dy = 1e-6

            theta = np.arctan2(dx, dy)
            r = np.sqrt(dx ** 2 + dy ** 2)

            # Normalize
            theta_norm = (theta + fov_rad / 2) / fov_rad
            r_norm = r / h_in

            # Clamp normalized values
            theta_norm = np.clip(theta_norm, 0, 1)
            r_norm = np.clip(r_norm, 0, 1)

            # Map to rectangular coordinates
            x_rect = theta_norm * (self.output_w - 1)
            y_rect = (1 - r_norm) * (self.output_h - 1)

            transformed_corners.append([x_rect, y_rect])

        # Compute axis-aligned bbox
        xs = [c[0] for c in transformed_corners]
        ys = [c[1] for c in transformed_corners]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w_rect = x_max - x_min
        h_rect = y_max - y_min

        # Filter invalid bboxes (area < 1 pixel)
        if w_rect < 1.0 or h_rect < 1.0:
            return [0.0, 0.0, 0.0, 0.0]

        return [x_min, y_min, w_rect, h_rect]

    def convert_sequence(self, seq_path: Path, verbose: bool = True) -> dict:
        """
        Convert all sonar images and bboxes in a sequence.

        Args:
            seq_path: Path to sequence directory (e.g., data/uuv1)
            verbose: Print progress information

        Returns:
            Statistics dictionary
        """
        # Paths
        img_dir = seq_path / 'sonar' / 'img'
        gt_file = seq_path / 'sonar' / 'groundtruth.txt'

        out_img_dir = seq_path / 'sonar' / 'img_rect'
        out_gt_file = seq_path / 'sonar' / 'groundtruth_rect.txt'

        # Validation
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not gt_file.exists():
            raise FileNotFoundError(f"Groundtruth file not found: {gt_file}")

        # Create output directory
        out_img_dir.mkdir(exist_ok=True)

        # Load groundtruth
        bboxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                bbox = [float(x.strip()) for x in line.strip().split(',')]
                bboxes.append(bbox)

        # Get image files with NUMERICAL sorting (critical for bbox alignment)
        # Files are named 1.jpg, 2.jpg, ... without leading zeros
        def extract_frame_number(path):
            try:
                return int(path.stem)
            except ValueError:
                return float('inf')

        img_files = sorted(img_dir.glob('*.jpg'), key=extract_frame_number)

        if len(img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")

        if verbose:
            print(f"\nProcessing {seq_path.name}...")
            print(f"  Images: {len(img_files)}, Bboxes: {len(bboxes)}")

        # Process images
        rect_bboxes = []
        skipped_frames = []

        iterator = tqdm(img_files, desc=f"  Converting") if verbose else img_files

        for idx, img_file in enumerate(iterator):
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    skipped_frames.append((idx + 1, "Failed to load image"))
                    continue

                h, w = img.shape[:2]

                # Transform image
                rect_img = self.transform_image(img)

                # Save
                out_path = out_img_dir / img_file.name
                cv2.imwrite(str(out_path), rect_img)

                # Transform bbox
                if idx < len(bboxes):
                    bbox_rect = self.transform_bbox(bboxes[idx], (h, w))
                    rect_bboxes.append(bbox_rect)
                else:
                    skipped_frames.append((idx + 1, "Missing bbox annotation"))

            except Exception as e:
                skipped_frames.append((idx + 1, f"Error: {str(e)}"))

        # Save transformed groundtruth
        with open(out_gt_file, 'w') as f:
            for bbox in rect_bboxes:
                f.write(f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}\n")

        # Statistics
        stats = {
            'sequence': seq_path.name,
            'total_images': len(img_files),
            'converted_images': len(rect_bboxes),
            'skipped_frames': skipped_frames,
            'output_dir': str(out_img_dir),
            'output_gt': str(out_gt_file)
        }

        if verbose:
            print(f"  ✓ Converted: {stats['converted_images']}/{stats['total_images']} frames")
            if skipped_frames:
                print(f"  ⚠ Skipped: {len(skipped_frames)} frames")
                for frame_id, reason in skipped_frames[:5]:
                    print(f"      Frame {frame_id}: {reason}")
                if len(skipped_frames) > 5:
                    print(f"      ... and {len(skipped_frames) - 5} more")
            print(f"  Output: {out_img_dir}")
            print(f"  Groundtruth: {out_gt_file}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert sonar polar images to rectangular coordinates'
    )
    parser.add_argument(
        '--sequence', type=str,
        help='Convert single sequence (e.g., uuv1)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Convert all sequences in dataset'
    )
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Dataset root directory (default: data)'
    )
    parser.add_argument(
        '--output-size', type=int, nargs=2, default=[376, 682],
        metavar=('HEIGHT', 'WIDTH'),
        help='Output image size (default: 376 682)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Initialize converter
    converter = SonarPolarToRectConverter(
        output_size=tuple(args.output_size)
    )

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        return 1

    # Determine sequences to process
    if args.sequence:
        sequences = [args.sequence]
    elif args.all:
        # All AquaSOT-3D sequences
        sequences = [
            'uuv1', 'uuv2', 'uuv3', 'uuv4', 'uuv5', 'uuv6',
            'uuv7', 'uuv8', 'uuv9', 'uuv10', 'uuv11', 'uuv12'
        ]
    else:
        print("Error: Specify --sequence <name> or --all")
        print("\nExamples:")
        print("  python scripts/convert_sonar_polar_to_rect.py --sequence uuv1")
        print("  python scripts/convert_sonar_polar_to_rect.py --all")
        return 1

    # Process sequences
    verbose = not args.quiet
    all_stats = []
    errors = []

    for seq_name in sequences:
        seq_path = data_root / seq_name

        if not seq_path.exists():
            msg = f"Sequence directory not found: {seq_path}"
            errors.append((seq_name, msg))
            if verbose:
                print(f"⚠ {msg}")
            continue

        try:
            stats = converter.convert_sequence(seq_path, verbose=verbose)
            all_stats.append(stats)
        except Exception as e:
            msg = f"Error processing {seq_name}: {str(e)}"
            errors.append((seq_name, msg))
            if verbose:
                print(f"✗ {msg}")

    # Summary
    if verbose and len(all_stats) > 1:
        print("\n" + "=" * 60)
        print("CONVERSION SUMMARY")
        print("=" * 60)
        total_converted = sum(s['converted_images'] for s in all_stats)
        total_images = sum(s['total_images'] for s in all_stats)
        print(f"Total sequences processed: {len(all_stats)}")
        print(f"Total images converted: {total_converted}/{total_images}")

        if errors:
            print(f"\nErrors: {len(errors)}")
            for seq_name, msg in errors:
                print(f"  {seq_name}: {msg}")

        print("\nNext steps:")
        print("  1. Validate conversion: python tests/test_sonar_rect_conversion.py")
        print("  2. Update dataset loaders to use img_rect/ and groundtruth_rect.txt")
        print("  3. Train model with preprocessed data")

    return 0 if len(errors) == 0 else 1


if __name__ == '__main__':
    exit(main())