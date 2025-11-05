"""
Unit tests for core/base_module.py

Tests the base module classes for all AquaSOT-3D components.
"""

import pytest
import torch
import torch.nn as nn
from aquasot3d.core.base_module import (
    BaseModule,
    BaseBackbone,
    BaseFusion,
    BaseTemporalHead,
    BaseHead,
    BaseLoss,
    BaseMetric
)


class DummyModule(BaseModule):
    """Dummy module for testing."""

    def forward(self, x):
        return x * 2


class DummyBackbone(BaseBackbone):
    """Dummy backbone for testing."""

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_dims = [256, 512, 1024]
        self.output_strides = [4, 8, 16]

    def forward(self, x):
        B, C, H, W = x.shape
        return {
            'features': [
                torch.randn(B, 256, H//4, W//4),
                torch.randn(B, 512, H//8, W//8),
                torch.randn(B, 1024, H//16, W//16)
            ],
            'feature_dims': self.feature_dims,
            'output_strides': self.output_strides
        }


class DummyFusion(BaseFusion):
    """Dummy fusion module for testing."""

    def forward(self, rgb_features, sonar_features, **kwargs):
        fused = rgb_features + sonar_features
        return {
            'fused_features': fused,
            'fusion_weights': torch.ones_like(rgb_features)
        }


class DummyTemporalHead(BaseTemporalHead):
    """Dummy temporal head for testing."""

    def __init__(self, config=None):
        super().__init__(config)
        self.state = None

    def forward(self, features, temporal_state=None):
        aggregated = features
        new_state = features.mean()
        return aggregated, new_state

    def reset_state(self):
        self.state = None


class DummyHead(BaseHead):
    """Dummy prediction head for testing."""

    def forward(self, features, **kwargs):
        B = features.shape[0]
        return {
            'bbox': torch.randn(B, 4),
            'score': torch.sigmoid(torch.randn(B, 1))
        }


class DummyLoss(BaseLoss):
    """Dummy loss for testing."""

    def forward(self, predictions, targets, **kwargs):
        return torch.tensor(1.0)


class DummyMetric(BaseMetric):
    """Dummy metric for testing."""

    def __init__(self):
        super().__init__()
        self.values = []

    def update(self, predictions, targets):
        self.values.append(1.0)

    def compute(self):
        return {'metric': sum(self.values) / len(self.values) if self.values else 0.0}

    def reset(self):
        self.values = []


class TestBaseModule:
    """Test cases for BaseModule class."""

    def test_base_module_creation(self):
        """Test BaseModule initialization."""
        config = {'param1': 10, 'param2': 20}
        module = DummyModule(config)

        assert module.config == config
        assert not module._is_initialized

    def test_base_module_forward(self):
        """Test forward pass."""
        module = DummyModule()
        x = torch.tensor([1.0, 2.0, 3.0])
        output = module(x)

        assert torch.allclose(output, x * 2)

    def test_get_config(self):
        """Test getting configuration."""
        config = {'param1': 10}
        module = DummyModule(config)
        retrieved_config = module.get_config()

        assert retrieved_config == config
        assert retrieved_config is not config  # Should be a copy

    def test_init_weights(self):
        """Test weight initialization."""
        module = DummyModule()
        module.init_weights()
        assert module._is_initialized

    def test_extra_repr(self):
        """Test extra representation."""
        config = {'param1': 10, 'param2': 20}
        module = DummyModule(config)
        repr_str = module.extra_repr()

        assert 'param1' in repr_str
        assert 'param2' in repr_str


class TestBaseBackbone:
    """Test cases for BaseBackbone class."""

    def test_backbone_creation(self):
        """Test BaseBackbone initialization."""
        backbone = DummyBackbone()
        assert isinstance(backbone.feature_dims, list)
        assert isinstance(backbone.output_strides, list)

    def test_backbone_forward(self):
        """Test backbone forward pass."""
        backbone = DummyBackbone()
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert 'features' in output
        assert 'feature_dims' in output
        assert 'output_strides' in output
        assert len(output['features']) == 3

    def test_get_feature_dims(self):
        """Test getting feature dimensions."""
        backbone = DummyBackbone()
        dims = backbone.get_feature_dims()
        assert dims == [256, 512, 1024]

    def test_get_output_strides(self):
        """Test getting output strides."""
        backbone = DummyBackbone()
        strides = backbone.get_output_strides()
        assert strides == [4, 8, 16]


class TestBaseFusion:
    """Test cases for BaseFusion class."""

    def test_fusion_creation(self):
        """Test BaseFusion initialization."""
        fusion = DummyFusion()
        assert isinstance(fusion, nn.Module)

    def test_fusion_forward(self):
        """Test fusion forward pass."""
        fusion = DummyFusion()
        rgb_feat = torch.randn(2, 256, 28, 28)
        sonar_feat = torch.randn(2, 256, 28, 28)

        output = fusion(rgb_feat, sonar_feat)

        assert 'fused_features' in output
        assert output['fused_features'].shape == rgb_feat.shape


class TestBaseTemporalHead:
    """Test cases for BaseTemporalHead class."""

    def test_temporal_head_creation(self):
        """Test BaseTemporalHead initialization."""
        head = DummyTemporalHead()
        assert isinstance(head, nn.Module)

    def test_temporal_head_forward(self):
        """Test temporal head forward pass."""
        head = DummyTemporalHead()
        features = torch.randn(2, 256, 28, 28)

        aggregated, state = head(features)

        assert aggregated.shape == features.shape
        assert state is not None

    def test_reset_state(self):
        """Test resetting temporal state."""
        head = DummyTemporalHead()
        head.state = torch.randn(10)
        head.reset_state()
        assert head.state is None


class TestBaseHead:
    """Test cases for BaseHead class."""

    def test_head_creation(self):
        """Test BaseHead initialization."""
        head = DummyHead()
        assert isinstance(head, nn.Module)

    def test_head_forward(self):
        """Test head forward pass."""
        head = DummyHead()
        features = torch.randn(2, 256, 28, 28)

        output = head(features)

        assert 'bbox' in output
        assert 'score' in output
        assert output['bbox'].shape == (2, 4)


class TestBaseLoss:
    """Test cases for BaseLoss class."""

    def test_loss_creation(self):
        """Test BaseLoss initialization."""
        loss = DummyLoss(weight=2.0)
        assert loss.weight == 2.0

    def test_loss_forward(self):
        """Test loss computation."""
        loss = DummyLoss(weight=1.0)
        predictions = {'bbox': torch.randn(2, 4)}
        targets = {'bbox': torch.randn(2, 4)}

        loss_value = loss(predictions, targets)

        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0  # Scalar

    def test_loss_weighting(self):
        """Test loss weight application."""
        loss = DummyLoss(weight=2.0)
        predictions = {'bbox': torch.randn(2, 4)}
        targets = {'bbox': torch.randn(2, 4)}

        loss_value = loss(predictions, targets)

        assert loss_value == 2.0  # weight * 1.0


class TestBaseMetric:
    """Test cases for BaseMetric class."""

    def test_metric_creation(self):
        """Test BaseMetric initialization."""
        metric = DummyMetric()
        assert metric.values == []

    def test_metric_update(self):
        """Test metric update."""
        metric = DummyMetric()
        predictions = {'bbox': torch.randn(2, 4)}
        targets = {'bbox': torch.randn(2, 4)}

        metric.update(predictions, targets)
        assert len(metric.values) == 1

    def test_metric_compute(self):
        """Test metric computation."""
        metric = DummyMetric()
        metric.values = [1.0, 2.0, 3.0]

        result = metric.compute()

        assert 'metric' in result
        assert result['metric'] == 2.0  # Average

    def test_metric_reset(self):
        """Test metric reset."""
        metric = DummyMetric()
        metric.values = [1.0, 2.0, 3.0]
        metric.reset()

        assert metric.values == []

    def test_metric_workflow(self):
        """Test complete metric workflow."""
        metric = DummyMetric()

        # Update multiple times
        for _ in range(5):
            metric.update({}, {})

        # Compute
        result = metric.compute()
        assert result['metric'] == 1.0

        # Reset
        metric.reset()
        result = metric.compute()
        assert result['metric'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
