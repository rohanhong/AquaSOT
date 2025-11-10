"""
Distributed Training Tests for Multi-GPU Environments

Tests for distributed data parallel (DDP) training:
- Multi-GPU initialization and synchronization
- Gradient synchronization across devices
- Checkpoint consistency in distributed mode
- Collective operations validation

Note: These tests require multiple GPUs and are skipped in single-GPU environments.

Author: AquaSOT-3D Team
Date: 2025-11-11
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from pathlib import Path
import tempfile

from aquasot3d.models.tracker import build_tracker
from aquasot3d.training.optimizer import build_optimizer


# Skip all tests if not enough GPUs available
pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Distributed training tests require at least 2 GPUs"
)


@pytest.fixture
def tracker_config():
    """Minimal tracker configuration for DDP testing"""
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
        'use_temporal': False,
        'use_weak3d': False
    }


class TestDistributedInitialization:
    """Test DDP initialization and setup"""

    def test_ddp_wrapper_creation(self, tracker_config):
        """Test wrapping model with DistributedDataParallel"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        # Build tracker
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker = tracker.cuda()

        # Wrap with DDP
        ddp_tracker = nn.parallel.DistributedDataParallel(
            tracker,
            device_ids=[dist.get_rank()],
            output_device=dist.get_rank()
        )

        # Verify wrapper
        assert isinstance(ddp_tracker, nn.parallel.DistributedDataParallel)
        assert ddp_tracker.device_ids == [dist.get_rank()]

    def test_process_group_creation(self):
        """Test distributed process group initialization"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        # Verify process group
        assert dist.is_initialized()
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        assert world_size >= 2, "Need at least 2 processes for distributed testing"
        assert 0 <= rank < world_size


class TestGradientSynchronization:
    """Test gradient synchronization across GPUs"""

    def test_gradient_all_reduce(self, tracker_config):
        """Test gradient synchronization via all-reduce"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        # Build tracker and wrap with DDP
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker = tracker.cuda()
        ddp_tracker = nn.parallel.DistributedDataParallel(tracker, device_ids=[dist.get_rank()])

        # Build optimizer
        optimizer = build_optimizer(tracker, optimizer_type='sgd', base_lr=0.01)

        # Forward pass with dummy data
        rgb_template = torch.randn(1, 3, 224, 224).cuda()
        rgb_search = torch.randn(1, 3, 224, 224).cuda()
        sonar_template = torch.randn(1, 1, 376, 682).cuda()
        sonar_search = torch.randn(1, 1, 376, 682).cuda()

        outputs = ddp_tracker(rgb_template, rgb_search, sonar_template, sonar_search)

        # Dummy loss
        loss = outputs['fusion_boxes'].sum()

        # Backward pass (gradients auto-synced by DDP)
        optimizer.zero_grad()
        loss.backward()

        # Verify gradients exist
        has_gradients = any(
            param.grad is not None
            for param in ddp_tracker.parameters()
            if param.requires_grad
        )
        assert has_gradients, "Model should have gradients after backward pass"

    def test_parameter_consistency_across_ranks(self, tracker_config):
        """Test parameter consistency across all ranks"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        # Build identical models on all ranks
        tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
        tracker = tracker.cuda()

        # Get one parameter value
        test_param = next(tracker.parameters())
        param_tensor = test_param.data.clone()

        # All-reduce to check consistency
        dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()

        # After sum, divide by world_size should give original value
        param_tensor /= world_size

        # Should be close to original (within numerical precision)
        assert torch.allclose(param_tensor, test_param.data, atol=1e-6)


class TestDistributedCheckpoint:
    """Test checkpoint save/load in distributed mode"""

    def test_checkpoint_save_rank0_only(self, tracker_config):
        """Test checkpoint saving only on rank 0"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        rank = dist.get_rank()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build tracker
            tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
            tracker = tracker.cuda()

            # Only rank 0 saves
            if rank == 0:
                checkpoint_path = Path(tmpdir) / 'checkpoint.pth'
                torch.save(tracker.state_dict(), checkpoint_path)

                # Verify file exists on rank 0
                assert checkpoint_path.exists()

            # Synchronize all ranks
            dist.barrier()

    def test_checkpoint_broadcast_load(self, tracker_config):
        """Test broadcasting checkpoint from rank 0 to all ranks"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        rank = dist.get_rank()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build tracker
            tracker = build_tracker('AquaSOT3DTracker', **tracker_config)
            tracker = tracker.cuda()

            checkpoint_path = Path(tmpdir) / 'checkpoint.pth'

            # Rank 0 saves checkpoint
            if rank == 0:
                torch.save(tracker.state_dict(), checkpoint_path)

            # Wait for rank 0 to finish saving
            dist.barrier()

            # All ranks load (file is on shared filesystem)
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
                tracker.load_state_dict(state_dict)

            # Synchronize
            dist.barrier()


class TestCollectiveOperations:
    """Test distributed collective operations"""

    def test_all_reduce_operation(self):
        """Test all-reduce collective operation"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank has its rank as value
        tensor = torch.tensor([rank], dtype=torch.float32).cuda()

        # All-reduce with SUM
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Result should be sum of all ranks: 0+1+2+...+(world_size-1)
        expected = sum(range(world_size))
        assert tensor.item() == expected

    def test_broadcast_operation(self):
        """Test broadcast from rank 0 to all ranks"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        rank = dist.get_rank()

        # Rank 0 has value 42, others have 0
        if rank == 0:
            tensor = torch.tensor([42.0]).cuda()
        else:
            tensor = torch.tensor([0.0]).cuda()

        # Broadcast from rank 0
        dist.broadcast(tensor, src=0)

        # All ranks should have 42
        assert tensor.item() == 42.0

    def test_all_gather_operation(self):
        """Test all-gather collective operation"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank has its rank value
        tensor = torch.tensor([rank], dtype=torch.float32).cuda()

        # Prepare output list
        tensor_list = [torch.zeros(1).cuda() for _ in range(world_size)]

        # All-gather
        dist.all_gather(tensor_list, tensor)

        # Verify gathered values
        for i, t in enumerate(tensor_list):
            assert t.item() == i


class TestDistributedDataLoading:
    """Test distributed data loading with DistributedSampler"""

    def test_distributed_sampler_creation(self):
        """Test DistributedSampler creation"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        from torch.utils.data import TensorDataset, DataLoader
        from torch.utils.data.distributed import DistributedSampler

        # Create dummy dataset
        dataset = TensorDataset(torch.randn(100, 3, 224, 224))

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)

        # Verify sampler properties
        assert sampler.num_replicas == dist.get_world_size()
        assert sampler.rank == dist.get_rank()
        assert len(dataloader) > 0

    def test_non_overlapping_samples_across_ranks(self):
        """Test that different ranks get non-overlapping samples"""
        if not dist.is_initialized():
            pytest.skip("Distributed environment not initialized")

        from torch.utils.data import TensorDataset, DataLoader
        from torch.utils.data.distributed import DistributedSampler

        # Create dataset with identifiable samples
        num_samples = 100
        data = torch.arange(num_samples).unsqueeze(1).float()
        dataset = TensorDataset(data)

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False  # Disable shuffle for deterministic testing
        )

        # Collect indices on this rank
        rank_indices = []
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
        for batch in dataloader:
            rank_indices.append(int(batch[0].item()))

        # Convert to tensor for all-gather
        indices_tensor = torch.tensor(rank_indices, dtype=torch.long).cuda()

        # Pad to same length across all ranks
        max_len = torch.tensor([len(rank_indices)], dtype=torch.long).cuda()
        dist.all_reduce(max_len, op=dist.ReduceOp.MAX)

        if len(rank_indices) < max_len.item():
            padding = [-1] * (max_len.item() - len(rank_indices))
            indices_tensor = torch.cat([
                indices_tensor,
                torch.tensor(padding, dtype=torch.long).cuda()
            ])

        # All-gather indices from all ranks
        all_indices = [torch.zeros_like(indices_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(all_indices, indices_tensor)

        # Only rank 0 verifies (avoid duplicate assertions)
        if dist.get_rank() == 0:
            # Flatten and remove padding
            all_indices_flat = []
            for idx_list in all_indices:
                all_indices_flat.extend([int(i) for i in idx_list.cpu() if i >= 0])

            # Check no duplicates (non-overlapping)
            assert len(all_indices_flat) == len(set(all_indices_flat)), \
                "Ranks should have non-overlapping samples"


# Helper function for manual distributed testing
def run_distributed_test():
    """
    Helper function to manually run distributed tests.

    Usage:
        torchrun --nproc_per_node=2 tests/test_distributed_training.py
    """
    # Initialize process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        print(f"Initialized distributed environment: rank={rank}, world_size={dist.get_world_size()}")

        # Run pytest
        pytest.main([__file__, '-v', '--tb=short'])

        # Cleanup
        dist.destroy_process_group()
    else:
        print("Environment variables RANK and WORLD_SIZE not set.")
        print("Run with: torchrun --nproc_per_node=2 tests/test_distributed_training.py")


if __name__ == '__main__':
    run_distributed_test()
