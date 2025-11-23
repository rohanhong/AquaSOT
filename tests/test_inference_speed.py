"""
Inference Speed Benchmark Tests for Phase 11

Performance benchmarking and FPS measurement for:
- Single-frame inference speed
- Batch inference throughput
- Temporal sequence processing speed
- Component-wise profiling
- Memory usage during inference

Author: AquaSOT-3D Team
Date: 2025-11-11
"""

import pytest
import torch
import time
import numpy as np
from typing import Dict, List

from aquasot3d.models.tracker import build_tracker


@pytest.fixture
def tracker_config():
    """Tracker configuration for speed testing"""
    return {
        'rgb_backbone_config': {
            'type': 'vit_rgb',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False,
            'output_dim': 192
        },
        'sonar_backbone_config': {
            'type': 'sonar',
            'encoder_type': 'conv',
            'output_dim': 128
        },
        'fusion_config': {
            'type': 'g2_dcma_lite',
            'output_dim': 192,
            'num_heads': 4,
            'num_points': 2
        },
        'temporal_config': {
            'type': 'transformer',
            'num_layers': 2,
            'num_heads': 4
        },
        'use_temporal': True,
    }


def measure_inference_time(
    model: torch.nn.Module,
    rgb_template: torch.Tensor,
    rgb_search: torch.Tensor,
    sonar_template: torch.Tensor,
    sonar_search: torch.Tensor,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    use_cuda: bool = False
) -> Dict[str, float]:
    """
    Measure inference time with proper warmup.

    Args:
        model: Tracker model
        rgb_template: RGB template image
        rgb_search: RGB search image
        sonar_template: Sonar template image
        sonar_search: Sonar search image
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        use_cuda: Whether to use CUDA synchronization

    Returns:
        Dictionary with timing statistics
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()

            _ = model(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    times = np.array(times)
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times)),
        'fps': float(1.0 / np.mean(times))
    }


class TestInferenceSpeed:
    """Inference speed benchmark tests"""

    def test_single_frame_inference_speed(self, tracker_config):
        """Benchmark single-frame inference speed"""
        # Build tracker
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker.eval()

        # Create test data
        batch_size = 1
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        # Measure inference time
        timing_stats = measure_inference_time(
            tracker,
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            warmup_runs=5,
            benchmark_runs=20
        )

        # Print results
        print("\n" + "="*60)
        print("Single-Frame Inference Speed (CPU)")
        print("="*60)
        print(f"Mean time: {timing_stats['mean']*1000:.2f} ms")
        print(f"Std time:  {timing_stats['std']*1000:.2f} ms")
        print(f"Min time:  {timing_stats['min']*1000:.2f} ms")
        print(f"Max time:  {timing_stats['max']*1000:.2f} ms")
        print(f"FPS:       {timing_stats['fps']:.2f}")
        print("="*60)

        # Assertions
        assert timing_stats['mean'] > 0, "Mean time should be positive"
        assert timing_stats['fps'] > 0, "FPS should be positive"

        # For ViT-Tiny on CPU, expect at least 0.5 FPS
        assert timing_stats['fps'] > 0.5, "FPS too low for CPU inference"

    def test_batch_inference_throughput(self, tracker_config):
        """Benchmark batch inference throughput"""
        # Build tracker
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker.eval()

        batch_sizes = [1, 2, 4]
        results = {}

        for batch_size in batch_sizes:
            # Create test data
            rgb_template = torch.randn(batch_size, 3, 128, 128)
            rgb_search = torch.randn(batch_size, 3, 256, 256)
            sonar_template = torch.randn(batch_size, 1, 376, 682)
            sonar_search = torch.randn(batch_size, 1, 376, 682)

            # Measure inference time
            timing_stats = measure_inference_time(
                tracker,
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                warmup_runs=3,
                benchmark_runs=10
            )

            results[batch_size] = timing_stats

        # Print results
        print("\n" + "="*60)
        print("Batch Inference Throughput (CPU)")
        print("="*60)
        for bs, stats in results.items():
            throughput = bs * stats['fps']
            print(f"Batch Size {bs}:")
            print(f"  Time per batch: {stats['mean']*1000:.2f} ms")
            print(f"  Throughput:     {throughput:.2f} samples/sec")
        print("="*60)

        # Verify all batch sizes work
        for bs in batch_sizes:
            assert results[bs]['mean'] > 0
            assert results[bs]['fps'] > 0

    def test_temporal_sequence_speed(self, tracker_config):
        """Benchmark temporal sequence processing speed"""
        # Build tracker
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker.eval()

        batch_size = 1
        sequence_length = 5

        # Reset temporal state
        tracker.reset_temporal_state()

        # Measure time for processing sequence
        total_time = 0.0
        times = []

        with torch.no_grad():
            for frame_idx in range(sequence_length):
                rgb_template = torch.randn(batch_size, 3, 128, 128)
                rgb_search = torch.randn(batch_size, 3, 256, 256)
                sonar_template = torch.randn(batch_size, 1, 376, 682)
                sonar_search = torch.randn(batch_size, 1, 376, 682)

                start_time = time.time()

                _ = tracker(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search,
                    return_losses=False
                )

                end_time = time.time()
                frame_time = end_time - start_time
                times.append(frame_time)
                total_time += frame_time

        avg_time_per_frame = total_time / sequence_length

        # Print results
        print("\n" + "="*60)
        print(f"Temporal Sequence Speed ({sequence_length} frames)")
        print("="*60)
        print(f"Total time:         {total_time*1000:.2f} ms")
        print(f"Avg time per frame: {avg_time_per_frame*1000:.2f} ms")
        print(f"FPS:                {1.0/avg_time_per_frame:.2f}")
        print(f"Frame times:        {[f'{t*1000:.1f}' for t in times]} ms")
        print("="*60)

        # Assertions
        assert avg_time_per_frame > 0
        assert total_time > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference_speed(self, tracker_config):
        """Benchmark GPU inference speed"""
        # Build tracker and move to GPU
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker = tracker.cuda()
        tracker.eval()

        # Create test data on GPU
        batch_size = 1
        rgb_template = torch.randn(batch_size, 3, 128, 128).cuda()
        rgb_search = torch.randn(batch_size, 3, 256, 256).cuda()
        sonar_template = torch.randn(batch_size, 1, 376, 682).cuda()
        sonar_search = torch.randn(batch_size, 1, 376, 682).cuda()

        # Measure inference time with CUDA synchronization
        timing_stats = measure_inference_time(
            tracker,
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            warmup_runs=10,
            benchmark_runs=50,
            use_cuda=True
        )

        # Print results
        print("\n" + "="*60)
        print("Single-Frame Inference Speed (GPU)")
        print("="*60)
        print(f"Mean time: {timing_stats['mean']*1000:.2f} ms")
        print(f"Std time:  {timing_stats['std']*1000:.2f} ms")
        print(f"FPS:       {timing_stats['fps']:.2f}")
        print("="*60)

        # GPU should be faster than 1 FPS
        assert timing_stats['fps'] > 1.0, "GPU FPS too low"

    def test_inference_memory_usage(self, tracker_config):
        """Test memory usage during inference"""
        # Build tracker
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker.eval()

        # Create test data
        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        if torch.cuda.is_available():
            tracker = tracker.cuda()
            rgb_template = rgb_template.cuda()
            rgb_search = rgb_search.cuda()
            sonar_template = sonar_template.cuda()
            sonar_search = sonar_search.cuda()

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            initial_memory = torch.cuda.memory_allocated()

        # Perform inference
        with torch.no_grad():
            _ = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()

            memory_used = (peak_memory - initial_memory) / 1024**2  # MB
            final_used = (final_memory - initial_memory) / 1024**2  # MB

            print("\n" + "="*60)
            print("Inference Memory Usage (GPU)")
            print("="*60)
            print(f"Peak memory used:  {memory_used:.2f} MB")
            print(f"Final memory used: {final_used:.2f} MB")
            print("="*60)

            # Memory usage should be reasonable (< 2GB for batch_size=2)
            assert memory_used < 2000, f"Memory usage too high: {memory_used:.2f} MB"


class TestComponentProfiling:
    """Profile individual components"""

    def test_backbone_speed(self, tracker_config):
        """Profile backbone extraction speed"""
        from aquasot3d.models.backbones.backbone_factory import build_multimodal_backbone

        # Build backbone
        backbone = build_multimodal_backbone(
            rgb_config=tracker_config['rgb_backbone_config'],
            sonar_config=tracker_config['sonar_backbone_config']
        )
        backbone.eval()

        # Create test data
        batch_size = 1
        rgb_template = torch.randn(batch_size, 3, 224, 224)
        rgb_search = torch.randn(batch_size, 3, 224, 224)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = backbone(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search
                )

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = backbone(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search
                )
                times.append(time.time() - start)

        avg_time = np.mean(times)

        print("\n" + "="*60)
        print("Backbone Extraction Speed")
        print("="*60)
        print(f"Average time: {avg_time*1000:.2f} ms")
        print(f"FPS:          {1.0/avg_time:.2f}")
        print("="*60)

        assert avg_time > 0
        assert 1.0/avg_time > 0.5  # At least 0.5 FPS for backbone

    def test_fusion_speed(self, tracker_config):
        """Profile fusion module speed"""
        from aquasot3d.models.fusion.g2_dcma import build_g2_dcma_lite

        # Build fusion module
        fusion = build_g2_dcma_lite({
            'embed_dim': tracker_config['fusion_config']['output_dim'],
            'num_heads': tracker_config['fusion_config']['num_heads']
        })
        fusion.eval()

        # Create dummy features
        batch_size = 1
        H, W = 16, 16
        rgb_features = torch.randn(batch_size, 192, H, W)
        sonar_features = torch.randn(batch_size, 128, H, W)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = fusion(rgb_features, sonar_features)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                _ = fusion(rgb_features, sonar_features)
                times.append(time.time() - start)

        avg_time = np.mean(times)

        print("\n" + "="*60)
        print("Fusion Module Speed")
        print("="*60)
        print(f"Average time: {avg_time*1000:.2f} ms")
        print(f"Throughput:   {1.0/avg_time:.2f} fusions/sec")
        print("="*60)

        assert avg_time > 0
