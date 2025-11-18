"""
GPU Memory Profiling Test for Optimal Batch Size

This script tests different batch sizes to find the optimal configuration
that maximizes GPU memory utilization for RTX 3070 Laptop (8GB VRAM).

Test Audit: 2025111803
Hardware: RTX 3070 Laptop (8GB), i7-11800H (16 threads)
Current baseline: batch_size=1 uses 76.2% VRAM (6.1GB)
Target: 90-95% VRAM utilization (7.2-7.6GB)
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.cuda as cuda
import logging
from aquasot3d.utils.config import load_config
from aquasot3d.models.tracker import AquaSOT3DTracker

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_gpu_memory_stats():
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return None

    allocated = cuda.memory_allocated() / 1024**3  # GB
    reserved = cuda.memory_reserved() / 1024**3  # GB
    total = cuda.get_device_properties(0).total_memory / 1024**3  # GB

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'allocated_pct': (allocated / total) * 100,
        'reserved_pct': (reserved / total) * 100
    }


def test_batch_size(batch_size: int, config_path: str = "configs/train/train.yaml"):
    """
    Test a specific batch size and measure GPU memory usage.

    Args:
        batch_size: Batch size to test
        config_path: Path to config file

    Returns:
        Dict with memory statistics, or None if OOM
    """
    logger.info("=" * 70)
    logger.info(f"Testing batch_size={batch_size}")
    logger.info("=" * 70)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        # Load config
        cfg = load_config(config_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build model
        model = AquaSOT3DTracker(
            rgb_backbone_config=cfg.model.backbone.rgb,
            sonar_backbone_config=cfg.model.backbone.sonar,
            fusion_config=cfg.model.fusion,
            temporal_config=cfg.model.temporal,
            weak3d_config=cfg.model.heads.weak3d,
            quality_config=None  # Not in train.yaml
        ).to(device)

        model.train()

        # Create dummy batch (sequence mode)
        T = cfg.data.search_sequence_length  # 5 frames
        dummy_batch = {
            'rgb_template': torch.randn(batch_size, 3, 224, 224, device=device),
            'rgb_search': torch.randn(batch_size * T, 3, 224, 224, device=device),
            'sonar_template': torch.randn(batch_size, 3, 224, 224, device=device),
            'sonar_search': torch.randn(batch_size * T, 3, 224, 224, device=device),
            'gt_boxes': torch.randn(batch_size, 4, device=device),
            'metadata': {'num_template': batch_size, 'num_search': batch_size * T}
        }

        # Memory before forward
        mem_before = get_gpu_memory_stats()
        logger.info(f"  Memory before forward: {mem_before['allocated_gb']:.2f}GB "
                   f"({mem_before['allocated_pct']:.1f}%)")

        # Forward pass
        outputs = model(dummy_batch)

        # Memory after forward
        mem_after_forward = get_gpu_memory_stats()
        logger.info(f"  Memory after forward:  {mem_after_forward['allocated_gb']:.2f}GB "
                   f"({mem_after_forward['allocated_pct']:.1f}%)")

        # Backward pass (simulate)
        if 'fusion_bbox' in outputs:
            loss = outputs['fusion_bbox'].sum()
            loss.backward()

        # Memory after backward (peak)
        mem_peak = get_gpu_memory_stats()
        peak_mem_gb = cuda.max_memory_allocated() / 1024**3
        peak_pct = (peak_mem_gb / mem_peak['total_gb']) * 100

        logger.info(f"  Peak memory (backward): {peak_mem_gb:.2f}GB ({peak_pct:.1f}%)")

        # Success
        result = {
            'batch_size': batch_size,
            'success': True,
            'mem_before_gb': mem_before['allocated_gb'],
            'mem_forward_gb': mem_after_forward['allocated_gb'],
            'mem_peak_gb': peak_mem_gb,
            'mem_peak_pct': peak_pct,
            'total_vram_gb': mem_peak['total_gb']
        }

        logger.info(f"  Result: SUCCESS (peak={peak_mem_gb:.2f}GB, {peak_pct:.1f}%)")

        # Cleanup
        del model, dummy_batch, outputs
        torch.cuda.empty_cache()

        return result

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"  Result: OOM (Out of Memory)")
            torch.cuda.empty_cache()
            return {
                'batch_size': batch_size,
                'success': False,
                'error': 'OOM'
            }
        else:
            raise


def main():
    """Test multiple batch sizes and find optimal configuration."""
    logger.info("\n" + "=" * 70)
    logger.info("GPU Memory Profiling for Optimal Batch Size")
    logger.info("=" * 70)
    logger.info(f"Hardware: RTX 3070 Laptop (8GB VRAM)")
    logger.info(f"Target: 90-95% VRAM utilization\n")

    if not torch.cuda.is_available():
        logger.error("CUDA not available! This test requires GPU.")
        return 1

    # Test batch sizes
    batch_sizes = [1, 2, 3]  # Start conservative, stop at first OOM
    results = []

    for bs in batch_sizes:
        result = test_batch_size(bs)

        if result is None:
            logger.error(f"Unexpected error testing batch_size={bs}")
            break

        results.append(result)

        # Stop if OOM
        if not result['success']:
            logger.info(f"\nStopping tests at batch_size={bs} (OOM)\n")
            break

        # Stop if we exceed 95% (unsafe)
        if result['mem_peak_pct'] > 95:
            logger.warning(f"\nStopping tests at batch_size={bs} (>95% VRAM usage, unsafe)\n")
            break

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)

    successful_results = [r for r in results if r['success']]

    if not successful_results:
        logger.error("No batch sizes succeeded!")
        return 1

    # Print table
    logger.info(f"{'Batch Size':<12} {'Peak VRAM (GB)':<16} {'Utilization (%)':<16} {'Status':<10}")
    logger.info("-" * 70)

    for r in results:
        if r['success']:
            logger.info(f"{r['batch_size']:<12} {r['mem_peak_gb']:<16.2f} "
                       f"{r['mem_peak_pct']:<16.1f} {'SUCCESS':<10}")
        else:
            logger.info(f"{r['batch_size']:<12} {'N/A':<16} {'N/A':<16} {'OOM':<10}")

    # Find optimal batch size (closest to 90% without exceeding 95%)
    optimal = max(
        (r for r in successful_results if r['mem_peak_pct'] <= 95),
        key=lambda x: x['mem_peak_pct'],
        default=successful_results[0]
    )

    logger.info("\n" + "=" * 70)
    logger.info("Optimal Configuration")
    logger.info("=" * 70)
    logger.info(f"Batch size: {optimal['batch_size']}")
    logger.info(f"Peak VRAM:  {optimal['mem_peak_gb']:.2f}GB ({optimal['mem_peak_pct']:.1f}%)")
    logger.info(f"Improvement: {optimal['mem_peak_pct'] / 76.2 * 100:.1f}% of baseline utilization")

    # Calculate gradient accumulation for effective batch=4
    target_effective_batch = 4
    grad_accum = max(1, target_effective_batch // optimal['batch_size'])
    actual_effective_batch = optimal['batch_size'] * grad_accum

    logger.info(f"\nRecommended settings for effective_batch={target_effective_batch}:")
    logger.info(f"  batch_size: {optimal['batch_size']}")
    logger.info(f"  gradient_accumulation_steps: {grad_accum}")
    logger.info(f"  Actual effective batch: {actual_effective_batch}")

    # CPU workers recommendation
    num_threads = os.cpu_count() or 16
    recommended_workers = min(num_threads // 2, 8)  # Use half of threads, cap at 8
    logger.info(f"\nCPU optimization (CPU threads={num_threads}):")
    logger.info(f"  num_workers: {recommended_workers} (currently 4)")
    logger.info(f"  prefetch_factor: 2 (keep current)")

    logger.info("\n" + "=" * 70)
    logger.info("SUCCESS: Optimal batch size determined")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())