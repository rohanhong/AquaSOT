"""
Visualize tracking results.

Generate side-by-side RGB and Sonar visualization with bounding boxes.

Usage:
    # Generate videos for all sequences
    python scripts/visualize_results.py \
        --result_dir outputs/test/results/AquaSOT \
        --data_root data \
        --output_dir outputs/visualization

    # Generate images for validation sequence
    python scripts/visualize_results.py \
        --result_dir outputs/test/results/AquaSOT \
        --data_root data \
        --output_dir outputs/visualization \
        --format images \
        --sequences uuv3
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# Colors (BGR format)
COLOR_PRED = (0, 0, 255)    # Red for prediction
COLOR_GT = (0, 255, 0)      # Green for ground truth


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize tracking results')

    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing tracking results (e.g., results/AquaSOT)')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization',
                        help='Output directory for visualization')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequences to visualize (default: all)')
    parser.add_argument('--format', type=str, choices=['video', 'images'], default='video',
                        help='Output format: video or images')
    parser.add_argument('--fps', type=int, default=10,
                        help='Video frame rate')
    parser.add_argument('--height', type=int, default=480,
                        help='Output frame height')

    return parser.parse_args()


def load_boxes(file_path):
    """Load bounding boxes from txt file.

    Args:
        file_path: Path to txt file with comma-separated boxes [x,y,w,h]

    Returns:
        numpy array of shape [N, 4] or None
    """
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path, delimiter=',')
    except Exception:
        return None


def get_image_files(img_dir):
    """Get numerically sorted list of image files in directory.

    Args:
        img_dir: Directory containing images

    Returns:
        List of image file paths sorted by frame number
    """
    if not os.path.exists(img_dir):
        return []

    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    for f in os.listdir(img_dir):
        if Path(f).suffix.lower() in extensions:
            files.append(os.path.join(img_dir, f))

    def extract_frame_number(path):
        """Extract numeric frame number from filename for sorting."""
        try:
            return int(Path(path).stem)
        except ValueError:
            return float('inf')

    return sorted(files, key=extract_frame_number)


def draw_box(image, box, color, thickness=2, label=None):
    """Draw bounding box on image.

    Args:
        image: Image to draw on
        box: Bounding box [x, y, w, h]
        color: BGR color tuple
        thickness: Line thickness
        label: Optional label text
    """
    if box is None:
        return

    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(image, (x, y - text_h - 4), (x + text_w + 2, y), color, -1)
        cv2.putText(image, label, (x + 1, y - 2), font, font_scale, (255, 255, 255), 1)


def resize_to_height(image, target_height):
    """Resize image to target height while preserving aspect ratio.

    Args:
        image: Input image
        target_height: Target height

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height))


def scale_box(box, scale):
    """Scale bounding box coordinates.

    Args:
        box: Bounding box [x, y, w, h]
        scale: Scale factor

    Returns:
        Scaled box
    """
    if box is None:
        return None
    return [v * scale for v in box]


def visualize_sequence(seq_name, result_dir, data_root, output_dir,
                       output_format='video', fps=10, target_height=480):
    """Visualize tracking results for a single sequence.

    Args:
        seq_name: Sequence name (e.g., 'uuv3')
        result_dir: Directory containing tracking results
        data_root: Dataset root directory
        output_dir: Output directory
        output_format: 'video' or 'images'
        fps: Video frame rate
        target_height: Target frame height
    """
    print(f'[Visualize] Processing {seq_name}...')

    # Load RGB prediction boxes
    rgb_pred_file = os.path.join(result_dir, f'{seq_name}_rgb.txt')
    rgb_pred_boxes = load_boxes(rgb_pred_file)
    if rgb_pred_boxes is None:
        print(f'  Warning: No RGB prediction file found: {rgb_pred_file}')
        return

    # Load Sonar prediction boxes (if available)
    sonar_pred_file = os.path.join(result_dir, f'{seq_name}_sonar.txt')
    sonar_pred_boxes = load_boxes(sonar_pred_file)
    if sonar_pred_boxes is None:
        print(f'  Note: No Sonar prediction file, using RGB predictions for both')
        sonar_pred_boxes = rgb_pred_boxes

    seq_path = os.path.join(data_root, seq_name)

    # Load image files
    rgb_img_dir = os.path.join(seq_path, 'light', 'img')
    sonar_img_dir = os.path.join(seq_path, 'sonar', 'img_rect')

    rgb_files = get_image_files(rgb_img_dir)
    sonar_files = get_image_files(sonar_img_dir)

    if not rgb_files:
        print(f'  Warning: No RGB images found in {rgb_img_dir}')
        return
    if not sonar_files:
        print(f'  Warning: No Sonar images found in {sonar_img_dir}')
        return

    # Load ground truth
    rgb_gt_file = os.path.join(seq_path, 'light', 'groundtruth.txt')
    sonar_gt_file = os.path.join(seq_path, 'sonar', 'groundtruth_rect.txt')
    rgb_gt = load_boxes(rgb_gt_file)
    sonar_gt = load_boxes(sonar_gt_file)

    num_frames = min(len(rgb_files), len(sonar_files), len(rgb_pred_boxes))
    print(f'  Frames: {num_frames}')

    # Read first frames to calculate dimensions
    rgb_sample = cv2.imread(rgb_files[0])
    sonar_sample = cv2.imread(sonar_files[0])

    rgb_h, rgb_w = rgb_sample.shape[:2]
    sonar_h, sonar_w = sonar_sample.shape[:2]

    # Calculate scales for target height
    rgb_scale = target_height / rgb_h
    sonar_scale = target_height / sonar_h

    rgb_new_w = int(rgb_w * rgb_scale)
    sonar_new_w = int(sonar_w * sonar_scale)

    total_width = rgb_new_w + sonar_new_w

    print(f'  Output size: {total_width}x{target_height} (RGB: {rgb_new_w}, Sonar: {sonar_new_w})')

    # Create output
    if output_format == 'images':
        out_subdir = os.path.join(output_dir, seq_name)
        os.makedirs(out_subdir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f'{seq_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (total_width, target_height))

    # Process frames
    for i in range(num_frames):
        # Read images
        rgb_img = cv2.imread(rgb_files[i])
        sonar_img = cv2.imread(sonar_files[i])

        if rgb_img is None or sonar_img is None:
            continue

        # Convert sonar to BGR if grayscale
        if len(sonar_img.shape) == 2:
            sonar_img = cv2.cvtColor(sonar_img, cv2.COLOR_GRAY2BGR)

        # Get boxes for this frame
        rgb_pred_box = rgb_pred_boxes[i] if i < len(rgb_pred_boxes) else None
        sonar_pred_box = sonar_pred_boxes[i] if i < len(sonar_pred_boxes) else None
        rgb_gt_box = rgb_gt[i] if rgb_gt is not None and i < len(rgb_gt) else None
        sonar_gt_box = sonar_gt[i] if sonar_gt is not None and i < len(sonar_gt) else None

        # Draw boxes on original images (before resize)
        # RGB: draw both GT and prediction
        draw_box(rgb_img, rgb_gt_box, COLOR_GT, 2, 'GT')
        draw_box(rgb_img, rgb_pred_box, COLOR_PRED, 2, 'Pred')
        # Sonar: draw both GT and Sonar prediction
        draw_box(sonar_img, sonar_gt_box, COLOR_GT, 2, 'GT')
        draw_box(sonar_img, sonar_pred_box, COLOR_PRED, 2, 'Pred')

        # Resize to target height
        rgb_resized = cv2.resize(rgb_img, (rgb_new_w, target_height))
        sonar_resized = cv2.resize(sonar_img, (sonar_new_w, target_height))

        # Concatenate side by side
        combined = np.hstack([rgb_resized, sonar_resized])

        # Add labels
        cv2.putText(combined, 'RGB', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, 'Sonar', (rgb_new_w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add frame number
        cv2.putText(combined, f'Frame: {i}', (10, target_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add legend
        legend_y = 60
        cv2.rectangle(combined, (10, legend_y), (25, legend_y + 15), COLOR_GT, -1)
        cv2.putText(combined, 'GT', (30, legend_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(combined, (70, legend_y), (85, legend_y + 15), COLOR_PRED, -1)
        cv2.putText(combined, 'Pred', (90, legend_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_format == 'video':
            video_writer.write(combined)
        else:
            out_file = os.path.join(out_subdir, f'{i:06d}.jpg')
            cv2.imwrite(out_file, combined)

    if output_format == 'video':
        video_writer.release()
        print(f'  Saved: {video_path}')
    else:
        print(f'  Saved: {out_subdir}/')


def main():
    """Main function."""
    args = parse_args()

    # Get available sequences from result directory
    if not os.path.exists(args.result_dir):
        print(f'Error: Result directory not found: {args.result_dir}')
        sys.exit(1)

    # Find all RGB result files (exclude auxiliary files)
    result_files = [f for f in os.listdir(args.result_dir)
                    if f.endswith('_rgb.txt')]

    if not result_files:
        print(f'Error: No result files found in {args.result_dir}')
        sys.exit(1)

    # Get sequence names (strip _rgb.txt suffix)
    all_sequences = [f.replace('_rgb.txt', '') for f in result_files]

    # Filter sequences if specified
    if args.sequences:
        sequences = [s for s in args.sequences if s in all_sequences]
        if not sequences:
            print(f'Error: None of the specified sequences found in results')
            print(f'Available: {all_sequences}')
            sys.exit(1)
    else:
        sequences = all_sequences

    print(f'[Visualize] Output format: {args.format}')
    print(f'[Visualize] Sequences: {len(sequences)}')
    print(f'[Visualize] Output directory: {args.output_dir}')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each sequence
    for seq_name in sorted(sequences):
        visualize_sequence(
            seq_name=seq_name,
            result_dir=args.result_dir,
            data_root=args.data_root,
            output_dir=args.output_dir,
            output_format=args.format,
            fps=args.fps,
            target_height=args.height,
        )

    print(f'[Visualize] Done!')


if __name__ == '__main__':
    main()