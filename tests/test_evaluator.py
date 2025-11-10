"""
Unit Tests for Tracking Evaluator

Tests for the evaluation pipeline including sequence tracking,
metric computation, and result reporting.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import Mock, MagicMock, patch

from aquasot3d.evaluation.evaluator import (
    TrackingEvaluator,
    evaluate_model
)


class MockTrackingModel(nn.Module):
    """Mock tracking model for testing"""

    def __init__(self):
        super().__init__()
        self.eval_mode = False

    def eval(self):
        self.eval_mode = True
        return self

    def to(self, device):
        return self

    def forward(self, rgb_template, rgb_search, sonar_template, sonar_search):
        """Return mock predictions"""
        batch_size = rgb_search.shape[0]

        return {
            'rgb_box': torch.randn(batch_size, 4),
            'sonar_box': torch.randn(batch_size, 4),
            'fusion_box': torch.randn(batch_size, 4)
        }


class MockDataLoader:
    """Mock dataloader for testing"""

    def __init__(self, num_sequences=3, sequence_length=5):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length

    def __iter__(self):
        for i in range(self.num_sequences):
            batch = {
                'sequence_name': f'seq_{i}',
                'rgb_template': torch.randn(self.sequence_length, 3, 127, 127),
                'rgb_search': torch.randn(self.sequence_length, 3, 255, 255),
                'sonar_template': torch.randn(self.sequence_length, 1, 127, 127),
                'sonar_search': torch.randn(self.sequence_length, 1, 255, 255),
                'rgb_box': torch.rand(self.sequence_length, 4) * 100 + 50,  # [50-150]
                'sonar_box': torch.rand(self.sequence_length, 4) * 100 + 50
            }
            yield batch

    def __len__(self):
        return self.num_sequences


class TestTrackingEvaluatorInitialization:
    """Test suite for TrackingEvaluator initialization"""

    def test_basic_initialization(self):
        """Test basic evaluator initialization"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()

        evaluator = TrackingEvaluator(
            model=model,
            dataloader=dataloader,
            device='cpu'
        )

        assert evaluator.model is model
        assert evaluator.dataloader is dataloader
        assert evaluator.device.type == 'cpu'
        assert evaluator.output_dir is None
        assert not evaluator.save_predictions
        assert evaluator.verbose

    def test_initialization_with_output_dir(self):
        """Test initialization with output directory"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = TrackingEvaluator(
                model=model,
                dataloader=dataloader,
                device='cpu',
                output_dir=tmpdir
            )

            assert evaluator.output_dir is not None
            assert evaluator.output_dir.exists()

    def test_initialization_cuda_device(self):
        """Test initialization with CUDA device"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()

        evaluator = TrackingEvaluator(
            model=model,
            dataloader=dataloader,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        if torch.cuda.is_available():
            assert evaluator.device.type == 'cuda'
        else:
            assert evaluator.device.type == 'cpu'


class TestSequenceTracking:
    """Test suite for sequence tracking"""

    def test_track_sequence_basic(self):
        """Test basic sequence tracking"""
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=1)
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        batch = next(iter(dataloader))
        predictions = evaluator.track_sequence(batch)

        assert 'rgb_pred' in predictions
        assert 'sonar_pred' in predictions
        assert 'fusion_pred' in predictions
        assert 'rgb_gt' in predictions
        assert 'sonar_gt' in predictions

    def test_track_sequence_prediction_shapes(self):
        """Test prediction shapes from sequence tracking"""
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=1, sequence_length=10)
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        batch = next(iter(dataloader))
        predictions = evaluator.track_sequence(batch)

        # Check predictions are lists of arrays
        assert isinstance(predictions['rgb_pred'], list)
        assert isinstance(predictions['fusion_pred'], list)
        assert len(predictions['rgb_pred']) > 0
        assert len(predictions['fusion_pred']) > 0

    def test_track_sequence_missing_outputs(self):
        """Test tracking with missing model outputs"""
        # Model that doesn't return all outputs
        model = MockTrackingModel()
        original_forward = model.forward

        def partial_forward(*args, **kwargs):
            return {'fusion_box': torch.randn(5, 4)}

        model.forward = partial_forward

        dataloader = MockDataLoader(num_sequences=1)
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        batch = next(iter(dataloader))
        predictions = evaluator.track_sequence(batch)

        # Should handle missing outputs gracefully
        assert isinstance(predictions, dict)


class TestMetricsComputation:
    """Test suite for metrics computation"""

    def test_compute_sequence_metrics(self):
        """Test sequence metrics computation"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        # Create dummy predictions
        predictions = {
            'rgb_pred': [np.array([50, 50, 100, 100]) for _ in range(5)],
            'sonar_pred': [np.array([50, 50, 100, 100]) for _ in range(5)],
            'fusion_pred': [np.array([50, 50, 100, 100]) for _ in range(5)],
            'rgb_gt': [np.array([50, 50, 100, 100]) for _ in range(5)],
            'sonar_gt': [np.array([50, 50, 100, 100]) for _ in range(5)]
        }

        metrics = evaluator.compute_sequence_metrics(predictions)

        assert 'rgb' in metrics
        assert 'sonar' in metrics
        assert 'fusion' in metrics
        assert 'success_auc' in metrics['fusion']

    def test_aggregate_all_metrics(self):
        """Test aggregation of all metrics"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        # Create dummy sequence metrics
        sequence_metrics = [
            {
                'rgb': {'success_auc': 0.8, 'precision_20': 0.7},
                'sonar': {'success_auc': 0.75, 'precision_20': 0.65},
                'fusion': {'success_auc': 0.85, 'precision_20': 0.75}
            },
            {
                'rgb': {'success_auc': 0.82, 'precision_20': 0.72},
                'sonar': {'success_auc': 0.77, 'precision_20': 0.67},
                'fusion': {'success_auc': 0.87, 'precision_20': 0.77}
            }
        ]

        aggregated = evaluator.aggregate_all_metrics(sequence_metrics)

        assert 'rgb' in aggregated
        assert 'sonar' in aggregated
        assert 'fusion' in aggregated
        assert 'success_auc_mean' in aggregated['fusion']
        assert 'precision_20_mean' in aggregated['fusion']


class TestEvaluationPipeline:
    """Test suite for full evaluation pipeline"""

    def test_evaluate_basic(self):
        """Test basic evaluation pipeline"""
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=2, sequence_length=5)
        evaluator = TrackingEvaluator(model, dataloader, device='cpu', verbose=False)

        results = evaluator.evaluate()

        assert 'overall' in results
        assert 'sequences' in results
        assert len(results['sequences']) == 2
        assert 'fusion' in results['overall']

    def test_evaluate_with_saving(self):
        """Test evaluation with result saving"""
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = TrackingEvaluator(
                model, dataloader, device='cpu',
                output_dir=tmpdir,
                save_predictions=True,
                verbose=False
            )

            results = evaluator.evaluate()

            # Check that files were saved
            output_path = Path(tmpdir)
            assert (output_path / 'overall_metrics.json').exists()
            assert (output_path / 'evaluation_report.txt').exists()

    def test_evaluate_empty_dataloader(self):
        """Test evaluation with empty dataloader"""
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=0)
        evaluator = TrackingEvaluator(model, dataloader, device='cpu', verbose=False)

        results = evaluator.evaluate()

        assert 'overall' in results
        assert 'sequences' in results
        assert len(results['sequences']) == 0


class TestResultSaving:
    """Test suite for result saving functionality"""

    def test_save_sequence_results(self):
        """Test saving per-sequence results"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = TrackingEvaluator(
                model, dataloader, device='cpu',
                output_dir=tmpdir,
                save_predictions=True
            )

            predictions = {
                'rgb_pred': [np.array([50, 50, 100, 100])],
                'fusion_pred': [np.array([52, 52, 100, 100])]
            }
            metrics = {
                'rgb': {'success_auc': 0.8},
                'fusion': {'success_auc': 0.85}
            }

            evaluator.save_sequence_results('test_seq', predictions, metrics)

            # Check files exist
            seq_dir = Path(tmpdir) / 'sequences' / 'test_seq'
            assert seq_dir.exists()
            assert (seq_dir / 'predictions.npz').exists()
            assert (seq_dir / 'metrics.json').exists()

    def test_save_overall_results(self):
        """Test saving overall results"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = TrackingEvaluator(
                model, dataloader, device='cpu',
                output_dir=tmpdir
            )

            evaluator.overall_metrics = {
                'fusion': {
                    'success_auc_mean': 0.85,
                    'precision_20_mean': 0.75
                }
            }

            evaluator.save_overall_results()

            # Check files exist
            output_path = Path(tmpdir)
            assert (output_path / 'overall_metrics.json').exists()
            assert (output_path / 'evaluation_report.txt').exists()

            # Check content
            with open(output_path / 'overall_metrics.json', 'r') as f:
                loaded_metrics = json.load(f)
                assert loaded_metrics['fusion']['success_auc_mean'] == 0.85

    def test_save_without_output_dir(self):
        """Test that saving doesn't fail when output_dir is None"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        predictions = {'rgb_pred': [np.array([50, 50, 100, 100])]}
        metrics = {'rgb': {'success_auc': 0.8}}

        # Should not raise error
        evaluator.save_sequence_results('test_seq', predictions, metrics)
        evaluator.save_overall_results()


class TestReportGeneration:
    """Test suite for report generation"""

    def test_generate_report_basic(self):
        """Test basic report generation"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        evaluator.overall_metrics = {
            'rgb': {'success_auc_mean': 0.8, 'precision_20_mean': 0.7},
            'fusion': {'success_auc_mean': 0.85, 'precision_20_mean': 0.75}
        }
        evaluator.sequence_results = {'seq_0': {}, 'seq_1': {}}

        report = evaluator.generate_report()

        assert isinstance(report, str)
        assert len(report) > 0
        assert 'AquaSOT-3D' in report
        assert 'RGB' in report.upper()
        assert 'FUSION' in report.upper()

    def test_generate_report_with_sequences(self):
        """Test report generation with sequence information"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        evaluator.overall_metrics = {
            'fusion': {
                'success_auc_mean': 0.85,
                'total_frames': 100
            }
        }
        evaluator.sequence_results = {f'seq_{i}': {} for i in range(5)}

        report = evaluator.generate_report()

        assert '5' in report  # Number of sequences
        assert '100' in report  # Total frames

    def test_print_summary(self, capsys):
        """Test summary printing"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu', verbose=True)

        evaluator.overall_metrics = {
            'rgb': {
                'success_auc_mean': 0.80,
                'success_auc_std': 0.05,
                'precision_20_mean': 0.70,
                'precision_20_std': 0.03,
                'avg_center_error_mean': 10.5,
                'avg_center_error_std': 2.3,
                'mean_iou_mean': 0.72,
                'mean_iou_std': 0.05
            }
        }

        evaluator.print_summary()

        captured = capsys.readouterr()
        assert 'EVALUATION SUMMARY' in captured.out
        assert 'RGB' in captured.out
        assert '0.80' in captured.out or '0.8' in captured.out


class TestBestWorstSequences:
    """Test suite for best/worst sequence identification"""

    def test_get_best_sequences(self):
        """Test getting best sequences"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        evaluator.sequence_results = {
            'seq_0': {'metrics': {'fusion': {'success_auc': 0.9}}},
            'seq_1': {'metrics': {'fusion': {'success_auc': 0.7}}},
            'seq_2': {'metrics': {'fusion': {'success_auc': 0.85}}},
            'seq_3': {'metrics': {'fusion': {'success_auc': 0.95}}}
        }

        best = evaluator.get_best_sequences(n=2)

        assert len(best) == 2
        assert best[0][0] == 'seq_3'  # Highest AUC
        assert best[0][1] == 0.95
        assert best[1][0] == 'seq_0'
        assert best[1][1] == 0.9

    def test_get_worst_sequences(self):
        """Test getting worst sequences"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        evaluator.sequence_results = {
            'seq_0': {'metrics': {'fusion': {'success_auc': 0.9}}},
            'seq_1': {'metrics': {'fusion': {'success_auc': 0.7}}},
            'seq_2': {'metrics': {'fusion': {'success_auc': 0.85}}}
        }

        worst = evaluator.get_worst_sequences(n=2)

        assert len(worst) == 2
        assert worst[0][0] == 'seq_1'  # Lowest AUC
        assert worst[0][1] == 0.7
        assert worst[1][0] == 'seq_2'
        assert worst[1][1] == 0.85

    def test_get_sequences_limit(self):
        """Test that n limits the number of sequences returned"""
        model = MockTrackingModel()
        dataloader = MockDataLoader()
        evaluator = TrackingEvaluator(model, dataloader, device='cpu')

        evaluator.sequence_results = {
            f'seq_{i}': {'metrics': {'fusion': {'success_auc': 0.5 + i*0.1}}}
            for i in range(10)
        }

        best = evaluator.get_best_sequences(n=3)
        assert len(best) == 3


class TestConvenienceFunction:
    """Test suite for convenience evaluation function"""

    def test_evaluate_model_function(self):
        """Test evaluate_model convenience function"""
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = evaluate_model(
                model=model,
                dataloader=dataloader,
                device='cpu',
                output_dir=tmpdir
            )

            assert 'overall' in results
            assert 'sequences' in results
            assert Path(tmpdir).exists()

    def test_evaluate_model_without_output_dir(self):
        """Test evaluate_model without output directory"""
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=1)

        results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device='cpu'
        )

        assert 'overall' in results
        assert 'sequences' in results


class TestEdgeCases:
    """Test suite for edge cases and error handling"""

    def test_model_returns_none(self):
        """Test handling when model returns None"""
        model = MockTrackingModel()
        model.forward = lambda *args, **kwargs: {}

        dataloader = MockDataLoader(num_sequences=1)
        evaluator = TrackingEvaluator(model, dataloader, device='cpu', verbose=False)

        # Should handle gracefully
        results = evaluator.evaluate()
        assert results is not None

    def test_batch_without_sequence_name(self):
        """Test handling batch without sequence name"""
        model = MockTrackingModel()
        dataloader = Mock()
        dataloader.__iter__ = lambda self: iter([{
            'rgb_template': torch.randn(5, 3, 127, 127),
            'rgb_search': torch.randn(5, 3, 255, 255),
            'sonar_template': torch.randn(5, 1, 127, 127),
            'sonar_search': torch.randn(5, 1, 255, 255)
        }])

        evaluator = TrackingEvaluator(model, dataloader, device='cpu', verbose=False)

        # Should use default sequence name
        results = evaluator.evaluate()
        assert 'seq_0' in results['sequences']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
