"""
Pytest configuration and shared fixtures for AquaSOT tests.

This module provides common test fixtures and configuration for all test modules.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def device():
    """Get device for testing (CPU or CUDA if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dummy_rgb_image():
    """Create dummy RGB image tensor."""
    # Shape: (B, C, H, W) = (2, 3, 224, 224)
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def dummy_sonar_image():
    """Create dummy Sonar image tensor."""
    # Shape: (B, C, H, W) = (2, 1, 376, 682)
    return torch.randn(2, 1, 376, 682)


@pytest.fixture
def dummy_bbox():
    """Create dummy bounding box annotations."""
    # Format: (x, y, w, h) for batch of 2
    return torch.tensor([
        [100.0, 150.0, 200.0, 150.0],
        [120.0, 160.0, 180.0, 140.0]
    ])


@pytest.fixture
def dummy_config():
    """Create dummy configuration dictionary with multi-sequence support."""
    return {
        'model': {
            'backbone': {
                'type': 'ViT',
                'depth': 12,
                'embed_dim': 768
            },
            'fusion': {
                'type': 'DCMA',
                'hidden_dim': 256
            },
            'temporal': {
                'type': 'Mamba',
                'num_layers': 4
            }
        },
        'training': {
            'batch_size': 32,
            'lr': 2e-4,
            'epochs': 60
        },
        'data': {
            # New multi-sequence format
            'data_root': 'data',
            'train_sequences': ['uuv1', 'uuv2'],
            'val_sequences': ['uuv3'],
            # Legacy paths (deprecated but still supported)
            'train_path': 'data/uuv2',
            'val_path': 'data/uuv3'
        }
    }


@pytest.fixture
def dummy_config_legacy():
    """Create dummy configuration dictionary with legacy single-sequence format."""
    return {
        'model': {
            'backbone': {
                'type': 'ViT',
                'depth': 12,
                'embed_dim': 768
            },
            'fusion': {
                'type': 'DCMA',
                'hidden_dim': 256
            },
            'temporal': {
                'type': 'Mamba',
                'num_layers': 4
            }
        },
        'training': {
            'batch_size': 32,
            'lr': 2e-4,
            'epochs': 60
        },
        'data': {
            'train_path': 'data/uuv2',
            'val_path': 'data/uuv3'
        }
    }


@pytest.fixture
def temp_config_file(tmp_path, dummy_config):
    """Create temporary YAML config file."""
    import yaml

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(dummy_config, f)

    return config_file


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_features():
    """Create sample feature tensors at different scales."""
    return {
        'scale1': torch.randn(2, 256, 56, 56),   # 1/4 resolution
        'scale2': torch.randn(2, 512, 28, 28),   # 1/8 resolution
        'scale3': torch.randn(2, 1024, 14, 14),  # 1/16 resolution
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
