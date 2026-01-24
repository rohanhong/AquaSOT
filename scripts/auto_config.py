"""
Auto-configure training parameters based on hardware.

Detects CPU cores and GPU memory to suggest optimal configuration.

Usage:
    # Show recommended configuration
    python scripts/auto_config.py

    # Create new config based on existing YAML (creates vitb_256_dual_auto.yaml)
    python scripts/auto_config.py --base experiments/aquasot/vitb_256_dual.yaml

    # Create with custom suffix (creates vitb_256_dual_custom.yaml)
    python scripts/auto_config.py --base experiments/aquasot/vitb_256_dual.yaml --suffix _custom

    # Show detailed hardware info
    python scripts/auto_config.py --verbose
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


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


def compute_batch_size(gpu_memory_mb: int) -> int:
    """Compute optimal batch size based on GPU memory.

    Memory model for AquaSOT with ViT-Base, dual-modal fusion:
    - Model weights and optimizer states: ~600 MB
    - CUDA context and buffers: ~800 MB
    - Per sample (forward + backward + gradients): ~2500 MB

    Calibrated against empirical tests:
    - 8GB GPU (RTX 3070 7.6GB): batch_size=2 validated

    Args:
        gpu_memory_mb: Available GPU memory in MB

    Returns:
        Recommended batch size (minimum 1)
    """
    MODEL_OVERHEAD_MB = 600
    CUDA_OVERHEAD_MB = 800
    PER_SAMPLE_MB = 2500
    SAFETY_FACTOR = 0.95

    available_mb = (gpu_memory_mb * SAFETY_FACTOR) - MODEL_OVERHEAD_MB - CUDA_OVERHEAD_MB
    batch_size = int(available_mb / PER_SAMPLE_MB)

    return max(1, batch_size)


def compute_gradient_accumulation(batch_size: int, target_effective_batch: int = 8) -> int:
    """Compute gradient accumulation steps to reach target effective batch size.

    Uses ceiling division to ensure effective batch size >= target.

    Args:
        batch_size: Per-GPU batch size
        target_effective_batch: Desired effective batch size

    Returns:
        Gradient accumulation steps (minimum 1)
    """
    # Ceiling division: (a + b - 1) // b
    accumulation = (target_effective_batch + batch_size - 1) // batch_size

    return max(1, accumulation)


def compute_num_workers(cpu_threads: int, batch_size: int) -> int:
    """Compute optimal number of data loading workers.

    Strategy:
    - Base: scale with both batch size and CPU threads
    - Use sqrt-based scaling for CPU to handle high-core systems
    - Allow prefetching beyond immediate batch needs for better GPU utilization

    For high-core systems (e.g., 72 threads):
    - More workers enable aggressive prefetching
    - Reduces GPU idle time waiting for data

    Args:
        cpu_threads: Number of CPU threads
        batch_size: Training batch size

    Returns:
        Recommended number of workers
    """
    import math

    MIN_WORKERS = 2
    MAX_WORKERS = 32

    # Special case for low-core systems
    if cpu_threads <= 4:
        return max(1, cpu_threads - 1)

    # Scale with CPU threads using sqrt for diminishing returns
    # sqrt(72) * 4 = 34, sqrt(16) * 4 = 16, sqrt(8) * 4 = 11
    workers_by_cpu = int(math.sqrt(cpu_threads) * 4)

    # Scale with batch size: more samples need more loading parallelism
    # Minimum 6 workers per sample for dual-modal (RGB + Sonar) prefetching
    workers_by_batch = batch_size * 6

    # Use the larger of the two, capped by max
    num_workers = min(max(workers_by_cpu, workers_by_batch), MAX_WORKERS)
    num_workers = max(MIN_WORKERS, num_workers)

    return num_workers


def recommend_config(hw: HardwareInfo) -> ConfigRecommendation:
    """Generate configuration recommendation based on hardware.

    Uses dynamic computation instead of fixed tiers to adapt to
    various hardware configurations.

    Args:
        hw: HardwareInfo object

    Returns:
        ConfigRecommendation object
    """
    # AMP disabled: causes training errors with dual-modal fusion
    amp_enabled = False

    # Compute batch size based on GPU memory
    batch_size = compute_batch_size(hw.gpu_memory_mb)

    # Compute gradient accumulation to reach target effective batch size
    gradient_accumulation = compute_gradient_accumulation(batch_size)

    # Compute num_workers based on CPU threads and batch size
    num_workers = compute_num_workers(hw.cpu_threads, batch_size)

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


def create_config_from_base(
    base_path: str, output_path: str, rec: ConfigRecommendation
) -> Dict[str, List[str]]:
    """Create new configuration file based on existing YAML.

    Reads base config, modifies hardware parameters, writes to new file.

    Args:
        base_path: Path to the base YAML configuration file
        output_path: Path to the output YAML configuration file
        rec: ConfigRecommendation object

    Returns:
        Dict with 'modified' and 'not_found' lists of parameter names
    """
    with open(base_path, 'r') as f:
        content = f.read()

    # Parameters to update with their patterns and new values
    amp_value = str(rec.amp_enabled).lower()
    params = {
        'BATCH_SIZE': (r'^(\s*)BATCH_SIZE:\s*\d+', rec.batch_size),
        'NUM_WORKER': (r'^(\s*)NUM_WORKER:\s*\d+', rec.num_workers),
        'GRADIENT_ACCUMULATION': (r'^(\s*)GRADIENT_ACCUMULATION:\s*\d+', rec.gradient_accumulation),
        'AMP': (r'^(\s*)AMP:\s*(true|false)', amp_value),
    }

    modified = []
    not_found = []

    for param_name, (pattern, new_value) in params.items():
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            indent = match.group(1)
            new_line = f"{indent}{param_name}: {new_value}"
            content = re.sub(pattern, new_line, content, flags=re.MULTILINE)
            modified.append(param_name)
        else:
            not_found.append(param_name)

    with open(output_path, 'w') as f:
        f.write(content)

    return {'modified': modified, 'not_found': not_found}


def generate_command(rec: ConfigRecommendation) -> str:
    """Generate training command with recommended parameters.

    Args:
        rec: ConfigRecommendation object

    Returns:
        Command string
    """
    cmd_parts = [
        "python tracking/train.py",
        "--script aquasot",
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
        '--base', '-b',
        type=str,
        default=None,
        help='Base YAML config file to create new config from (e.g., experiments/aquasot/vitb_256_dual.yaml)'
    )

    parser.add_argument(
        '--suffix', '-s',
        type=str,
        default='_auto',
        help='Suffix for the output config file (default: _auto)'
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

    # Create new config file based on existing YAML if requested
    if args.base:
        base_path = args.base

        if not os.path.exists(base_path):
            print(f"[Error] File not found: {base_path}")
            sys.exit(1)

        # Generate output path with suffix
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_name = f"{name_without_ext}{args.suffix}.yaml"
        output_path = os.path.join(base_dir, output_name)

        result = create_config_from_base(base_path, output_path, rec)

        print("=" * 60)
        print(f"Configuration created: {output_path}")
        print(f"  (based on: {base_path})")
        print("=" * 60)
        print()

        if result['modified']:
            print("Modified parameters:")
            for param in result['modified']:
                print(f"  - {param}")

        if result['not_found']:
            print()
            print("Parameters not found in base file (skipped):")
            for param in result['not_found']:
                print(f"  - {param}")

        # Extract config name for training command
        config_name = name_without_ext + args.suffix
        print()
        print("To use this configuration:")
        print(f"  python tracking/train.py --script aquasot --config {config_name}")
        print()


if __name__ == '__main__':
    main()