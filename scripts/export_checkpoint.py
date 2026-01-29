#!/usr/bin/env python3
"""
Export Inference Checkpoint from Training Checkpoint.

Extracts model weights from full training checkpoints, removing optimizer state,
learning rate scheduler state, and training statistics. The exported checkpoint
is suitable for inference and evaluation with reduced file size.

Checkpoint Formats:
- Training checkpoint: {epoch, net, optimizer, lr_scheduler, stats}
- Inference checkpoint: {net} or raw state_dict

Usage:
    # Export single checkpoint
    python scripts/export_checkpoint.py \
        --input checkpoints/aquasot/checkpoint_ep0100.pth \
        --output checkpoints/aquasot/model_ep0100.pth

    # Export with raw state_dict format (smallest size)
    python scripts/export_checkpoint.py \
        --input checkpoints/aquasot/checkpoint_ep0100.pth \
        --output checkpoints/aquasot/model_ep0100.pth \
        --format raw

    # Batch export all checkpoints in directory
    python scripts/export_checkpoint.py \
        --input_dir checkpoints/aquasot \
        --output_dir checkpoints/aquasot/inference

    # Export latest checkpoint only
    python scripts/export_checkpoint.py \
        --input_dir checkpoints/aquasot \
        --output_dir checkpoints/aquasot/inference \
        --latest_only

Author: AquaSOT Team
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export inference checkpoint from training checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Single file mode
    parser.add_argument('--input', type=str, default=None,
                        help='Input training checkpoint path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output inference checkpoint path')

    # Batch mode
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory for batch export')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for batch export')
    parser.add_argument('--latest_only', action='store_true',
                        help='Only export checkpoint_latest.pth')

    # Output format
    parser.add_argument('--format', type=str, choices=['dict', 'raw'], default='dict',
                        help='Output format: dict ({"net": state_dict}) or raw (state_dict only)')

    # Options
    parser.add_argument('--strip_ddp', action='store_true', default=True,
                        help='Remove "module." prefix from DDP training (default: True)')
    parser.add_argument('--no_strip_ddp', dest='strip_ddp', action='store_false',
                        help='Keep "module." prefix')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def extract_state_dict(checkpoint, strip_ddp=True):
    """
    Extract model state_dict from checkpoint.

    Args:
        checkpoint: Loaded checkpoint (dict or state_dict)
        strip_ddp: Whether to remove 'module.' prefix from DDP training

    Returns:
        Cleaned state_dict
    """
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume it's already a state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove DDP 'module.' prefix if present
    if strip_ddp:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    return state_dict


def export_checkpoint(input_path, output_path, output_format='dict', strip_ddp=True, verbose=False):
    """
    Export inference-only checkpoint from training checkpoint.

    Args:
        input_path: Path to input training checkpoint
        output_path: Path to output inference checkpoint
        output_format: 'dict' for {"net": state_dict} or 'raw' for state_dict only
        strip_ddp: Whether to remove 'module.' prefix
        verbose: Whether to print detailed information

    Returns:
        Tuple of (input_size_mb, output_size_mb)
    """
    # Load checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')

    # Extract state_dict
    state_dict = extract_state_dict(checkpoint, strip_ddp=strip_ddp)

    # Print checkpoint info if verbose
    if verbose:
        print(f'  Input checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else "raw state_dict"}')
        print(f'  Model parameters: {len(state_dict)}')
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f'  Training epoch: {checkpoint["epoch"]}')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Save in specified format
    if output_format == 'dict':
        torch.save({'net': state_dict}, output_path)
    else:
        torch.save(state_dict, output_path)

    # Calculate sizes
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)

    return input_size, output_size


def find_checkpoints(input_dir, latest_only=False):
    """
    Find checkpoint files in directory.

    Args:
        input_dir: Directory to search
        latest_only: Only return checkpoint_latest.pth

    Returns:
        List of checkpoint paths
    """
    input_path = Path(input_dir)
    checkpoints = []

    if latest_only:
        latest = input_path / 'checkpoint_latest.pth'
        if latest.exists():
            checkpoints.append(latest)
    else:
        # Find all checkpoint files
        for pattern in ['checkpoint_ep*.pth', 'checkpoint_latest.pth']:
            checkpoints.extend(input_path.glob(pattern))

    return sorted(set(checkpoints))


def get_output_name(input_name, output_format):
    """
    Generate output filename from input filename.

    Args:
        input_name: Input filename
        output_format: Output format ('dict' or 'raw')

    Returns:
        Output filename
    """
    name = input_name.replace('checkpoint_', 'model_')
    if output_format == 'raw':
        name = name.replace('.pth', '_raw.pth')
    return name


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if args.input and args.output:
        # Single file mode
        if not os.path.exists(args.input):
            print(f'Error: Input file not found: {args.input}')
            sys.exit(1)

        print(f'Exporting checkpoint (format: {args.format})')
        input_size, output_size = export_checkpoint(
            args.input,
            args.output,
            output_format=args.format,
            strip_ddp=args.strip_ddp,
            verbose=args.verbose
        )
        reduction = (1 - output_size / input_size) * 100
        print(f'  {args.input} ({input_size:.1f}MB) -> {args.output} ({output_size:.1f}MB)')
        print(f'  Size reduction: {reduction:.1f}%')

    elif args.input_dir and args.output_dir:
        # Batch mode
        if not os.path.isdir(args.input_dir):
            print(f'Error: Input directory not found: {args.input_dir}')
            sys.exit(1)

        checkpoints = find_checkpoints(args.input_dir, latest_only=args.latest_only)

        if not checkpoints:
            print(f'No checkpoints found in {args.input_dir}')
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)

        print(f'Exporting {len(checkpoints)} checkpoint(s) (format: {args.format})')
        total_input = 0
        total_output = 0

        for ckpt_path in checkpoints:
            output_name = get_output_name(ckpt_path.name, args.format)
            output_path = os.path.join(args.output_dir, output_name)

            input_size, output_size = export_checkpoint(
                str(ckpt_path),
                output_path,
                output_format=args.format,
                strip_ddp=args.strip_ddp,
                verbose=args.verbose
            )
            total_input += input_size
            total_output += output_size
            print(f'  {ckpt_path.name} ({input_size:.1f}MB) -> {output_name} ({output_size:.1f}MB)')

        if len(checkpoints) > 1:
            reduction = (1 - total_output / total_input) * 100
            print(f'\nTotal: {total_input:.1f}MB -> {total_output:.1f}MB (reduction: {reduction:.1f}%)')

    else:
        print('Error: Provide either --input/--output or --input_dir/--output_dir')
        print('Run with --help for usage information')
        sys.exit(1)

    print('Export completed successfully')


if __name__ == '__main__':
    main()