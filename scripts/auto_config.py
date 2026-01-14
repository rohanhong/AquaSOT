"""
Auto-configure training parameters based on hardware.

Detects CPU cores and GPU memory to suggest optimal configuration.

Usage:
    # Show recommended configuration
    python scripts/auto_config.py

    # Generate config override file
    python scripts/auto_config.py --output experiments/aquasot3d/auto_hardware.yaml

    # Show detailed hardware info
    python scripts/auto_config.py --verbose
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HardwareInfo:
    """Hardware information container."""
    cpu_cores: int
    cpu_threads: int
    gpu_count: int
    gpu_name: str
    gpu_memory_mb: int
    cuda_available: bool


@dataclass
class ConfigRecommendation:
    """Configuration recommendation based on hardware."""
    batch_size: int
    num_workers: int
    gradient_accumulation: int
    amp_enabled: bool
    num_gpus: int
    effective_batch_size: int


def get_cpu_info() -> Tuple[int, int]:
    """Get CPU core and thread count.

    Returns:
        Tuple of (physical_cores, logical_threads)
    """
    try:
        # Try to get physical cores
        physical_cores = os.cpu_count() or 1

        # On Linux, try to get more accurate info
        if sys.platform == 'linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    # Count physical cores
                    cores = content.count('processor')
                    if cores > 0:
                        physical_cores = cores
            except Exception:
                pass

        logical_threads = os.cpu_count() or physical_cores

        return physical_cores, logical_threads
    except Exception:
        return 1, 1


def get_gpu_info() -> Tuple[int, str, int, bool]:
    """Get GPU information using PyTorch.

    Returns:
        Tuple of (gpu_count, gpu_name, gpu_memory_mb, cuda_available)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, "N/A", 0, False

        gpu_count = torch.cuda.device_count()

        if gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            # Get memory in MB
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        else:
            gpu_name = "N/A"
            gpu_memory_mb = 0

        return gpu_count, gpu_name, gpu_memory_mb, True

    except ImportError:
        print("[Warning] PyTorch not installed, cannot detect GPU")
        return 0, "N/A", 0, False
    except Exception as e:
        print(f"[Warning] GPU detection failed: {e}")
        return 0, "N/A", 0, False


def detect_hardware() -> HardwareInfo:
    """Detect hardware specifications.

    Returns:
        HardwareInfo object with detected specifications
    """
    cpu_cores, cpu_threads = get_cpu_info()
    gpu_count, gpu_name, gpu_memory_mb, cuda_available = get_gpu_info()

    return HardwareInfo(
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        cuda_available=cuda_available,
    )


def recommend_config(hw: HardwareInfo) -> ConfigRecommendation:
    """Generate configuration recommendation based on hardware.

    Args:
        hw: HardwareInfo object

    Returns:
        ConfigRecommendation object
    """
    # GPU memory based recommendations (in MB)
    # AquaSOT-3D with ViT-Base, 256x256 search, 128x128 template
    # Approximate memory usage per batch:
    # - Model: ~350MB
    # - Per sample (dual-modal): ~800MB with AMP, ~1200MB without AMP

    gpu_mem_gb = hw.gpu_memory_mb / 1024

    if gpu_mem_gb >= 24:
        # 24GB+ (RTX 4090, A5000, etc.)
        batch_size = 4
        amp_enabled = False
        gradient_accumulation = 2
    elif gpu_mem_gb >= 16:
        # 16-24GB (RTX 4080, A4000, etc.)
        batch_size = 2
        amp_enabled = False
        gradient_accumulation = 4
    elif gpu_mem_gb >= 12:
        # 12-16GB (RTX 4070, RTX 3080, etc.)
        batch_size = 2
        amp_enabled = True
        gradient_accumulation = 4
    elif gpu_mem_gb >= 8:
        # 8-12GB (RTX 3070, RTX 4060, etc.)
        batch_size = 1
        amp_enabled = True
        gradient_accumulation = 8
    elif gpu_mem_gb >= 6:
        # 6-8GB (RTX 3060, GTX 1660, etc.)
        batch_size = 1
        amp_enabled = True
        gradient_accumulation = 8
    else:
        # <6GB or no GPU
        batch_size = 1
        amp_enabled = True
        gradient_accumulation = 8

    # CPU based recommendations for num_workers
    # Rule: use ~75% of available threads, minimum 2, maximum 16
    num_workers = max(2, min(16, int(hw.cpu_threads * 0.75)))

    # Adjust for low-end systems
    if hw.cpu_threads <= 4:
        num_workers = max(1, hw.cpu_threads - 1)

    # Multi-GPU: use all available GPUs
    num_gpus = max(1, hw.gpu_count)

    # Calculate effective batch size
    effective_batch_size = batch_size * gradient_accumulation * num_gpus

    return ConfigRecommendation(
        batch_size=batch_size,
        num_workers=num_workers,
        gradient_accumulation=gradient_accumulation,
        amp_enabled=amp_enabled,
        num_gpus=num_gpus,
        effective_batch_size=effective_batch_size,
    )


def format_hardware_info(hw: HardwareInfo, verbose: bool = False) -> str:
    """Format hardware information for display.

    Args:
        hw: HardwareInfo object
        verbose: Show detailed information

    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "Hardware Detection",
        "=" * 60,
        "",
        "CPU:",
        f"  Cores/Threads: {hw.cpu_cores} / {hw.cpu_threads}",
        "",
        "GPU:",
        f"  CUDA Available: {hw.cuda_available}",
    ]

    if hw.cuda_available:
        lines.extend([
            f"  GPU Count: {hw.gpu_count}",
            f"  GPU Name: {hw.gpu_name}",
            f"  GPU Memory: {hw.gpu_memory_mb} MB ({hw.gpu_memory_mb / 1024:.1f} GB)",
        ])

    lines.append("")

    return "\n".join(lines)


def format_recommendation(rec: ConfigRecommendation) -> str:
    """Format configuration recommendation for display.

    Args:
        rec: ConfigRecommendation object

    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "Recommended Configuration",
        "=" * 60,
        "",
        f"  BATCH_SIZE: {rec.batch_size}",
        f"  NUM_WORKER: {rec.num_workers}",
        f"  GRADIENT_ACCUMULATION: {rec.gradient_accumulation}",
        f"  AMP (Mixed Precision): {rec.amp_enabled}",
        f"  NUM_GPUS: {rec.num_gpus}",
        "",
        f"  Effective Batch Size: {rec.effective_batch_size}",
        f"    ({rec.batch_size} x {rec.gradient_accumulation} x {rec.num_gpus})",
        "",
    ]

    return "\n".join(lines)


def generate_yaml_config(rec: ConfigRecommendation) -> str:
    """Generate YAML configuration content.

    Args:
        rec: ConfigRecommendation object

    Returns:
        YAML content string
    """
    lines = [
        "# Auto-generated configuration based on hardware detection",
        "# Generated by: python scripts/auto_config.py",
        "",
        "TRAIN:",
        f"  BATCH_SIZE: {rec.batch_size}",
        f"  NUM_WORKER: {rec.num_workers}",
        f"  GRADIENT_ACCUMULATION: {rec.gradient_accumulation}",
        f"  AMP: {str(rec.amp_enabled).lower()}",
        "",
        "SYSTEM:",
        f"  NUM_GPUS: {rec.num_gpus}",
        "",
    ]

    return "\n".join(lines)


def generate_command(rec: ConfigRecommendation) -> str:
    """Generate training command with recommended parameters.

    Args:
        rec: ConfigRecommendation object

    Returns:
        Command string
    """
    cmd_parts = [
        "python tracking/train.py",
        "--script aquasot3d",
        "--config vitb_256_dual",
        f"--batch_size {rec.batch_size}",
        f"--num_workers {rec.num_workers}",
    ]

    if rec.amp_enabled:
        cmd_parts.append("--amp")

    return " \\\n    ".join(cmd_parts)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Auto-configure training parameters based on hardware'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output YAML config file path (e.g., experiments/aquasot3d/auto_hardware.yaml)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed hardware information'
    )

    parser.add_argument(
        '--command-only',
        action='store_true',
        help='Only output the training command'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Detect hardware
    hw = detect_hardware()

    # Generate recommendation
    rec = recommend_config(hw)

    if args.command_only:
        print(generate_command(rec))
        return

    # Display hardware info
    print(format_hardware_info(hw, args.verbose))

    # Display recommendation
    print(format_recommendation(rec))

    # Display suggested command
    print("=" * 60)
    print("Training Command")
    print("=" * 60)
    print()
    print(generate_command(rec))
    print()

    # Save to file if requested
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        yaml_content = generate_yaml_config(rec)
        with open(output_path, 'w') as f:
            f.write(yaml_content)

        print("=" * 60)
        print(f"Configuration saved to: {output_path}")
        print("=" * 60)
        print()
        print("To use this configuration:")
        print(f"  python tracking/train.py --script aquasot3d --config auto_hardware")
        print()


if __name__ == '__main__':
    main()