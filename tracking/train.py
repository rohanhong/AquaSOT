"""
Training entry point for AquaSOT.

Based on OSTrack's run_training.py.
Usage:
    python tracking/train.py --script aquasot --config vitb_256_dual --save_dir outputs/train
"""

import os
import sys
import argparse
import random
import importlib

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist

# Add lib to path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from lib.train.admin import Settings


def init_seeds(seed):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(
    script_name,
    config_name,
    cudnn_benchmark=True,
    local_rank=-1,
    save_dir=None,
    base_seed=None,
    wandb_enabled=False,
    wandb_project=None,
    wandb_name=None,
    wandb_entity=None,
):
    """Run the training script.

    Args:
        script_name: Name of experiment script (e.g., 'aquasot')
        config_name: Name of config file (e.g., 'vitb_256_dual')
        cudnn_benchmark: Use cudnn benchmark
        local_rank: Local rank for distributed training
        save_dir: Directory to save checkpoints and logs
        base_seed: Random seed
        wandb_enabled: Enable wandb logging
        wandb_project: wandb project name
        wandb_name: wandb run name
        wandb_entity: wandb entity (username or team)
    """
    # Setup default save directory
    if save_dir is None:
        save_dir = os.path.join(project_path, 'outputs', 'train', script_name, config_name)
        print(f"save_dir not specified, using: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    # Disable OpenCV threading (avoids crashes)
    try:
        import cv2 as cv
        cv.setNumThreads(0)
    except ImportError:
        pass

    # Set cudnn benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark

    print(f"Script: {script_name}.py  Config: {config_name}.yaml")

    # Set random seed
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    # Create settings
    settings = Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = f'train/{script_name}/{config_name}'
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    settings.cfg_file = os.path.join(
        project_path,
        f'experiments/{script_name}/{config_name}.yaml'
    )

    # wandb settings (command-line overrides)
    settings.wandb_enabled = wandb_enabled
    settings.wandb_project = wandb_project
    settings.wandb_name = wandb_name
    settings.wandb_entity = wandb_entity

    # Import and run training script
    train_module = importlib.import_module('lib.train.train_script')
    train_func = getattr(train_module, 'run')
    train_func(settings)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AquaSOT Training')

    parser.add_argument(
        '--script',
        type=str,
        required=True,
        help='Name of the training script (e.g., aquasot)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Name of the config file (e.g., vitb_256_dual)'
    )
    parser.add_argument(
        '--cudnn_benchmark',
        type=bool,
        default=True,
        help='Use cudnn benchmark (default: True)'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints and logs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    # wandb arguments
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='wandb project name'
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default=None,
        help='wandb run name'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='wandb entity (username or team)'
    )

    args = parser.parse_args()

    # Initialize distributed training
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # Run training
    run_training(
        script_name=args.script,
        config_name=args.config,
        cudnn_benchmark=args.cudnn_benchmark,
        local_rank=args.local_rank,
        save_dir=args.save_dir,
        base_seed=args.seed,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
    )


if __name__ == '__main__':
    main()
