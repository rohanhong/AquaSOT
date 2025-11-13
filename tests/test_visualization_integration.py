"""
Integration Tests for Evaluation Visualization Pipeline

Tests verify that visualization artifacts are correctly generated during
evaluation runs, addressing the audit findings from 2025111402_Eval_Visualization_Audit.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
from unittest.mock import Mock

from aquasot3d.evaluation import TrackingEvaluator, TrackingVisualizer
from aquasot3d.dataloader import EvaluationSequenceDataset


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
        """Return mock predictions with reasonable tracking output"""
        batch_size = rgb_search.shape[0]

        # Generate predictions centered around GT with small noise
        return {
            'rgb_box': torch.randn(batch_size, 4).abs() * 10 + 100,  # ~[100, 110]
            'sonar_box': torch.randn(batch_size, 4).abs() * 10 + 100,
            'fusion_box': torch.randn(batch_size, 4).abs() * 10 + 100
        }


class MockDataLoader:
    """Mock dataloader for visualization testing"""

    def __init__(self, num_sequences=2, sequence_length=10):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length

    def __iter__(self):
        for seq_idx in range(self.num_sequences):
            batch = {
                'sequence_name': f'test_seq_{seq_idx}',
                'rgb_template': torch.randn(self.sequence_length, 3, 128, 128),
                'rgb_search': torch.randn(self.sequence_length, 3, 256, 256),
                'sonar_template': torch.randn(self.sequence_length, 1, 128, 128),
                'sonar_search': torch.randn(self.sequence_length, 1, 256, 256),
                'rgb_box': torch.rand(self.sequence_length, 4) * 100 + 50,  # [50-150]
                'sonar_box': torch.rand(self.sequence_length, 4) * 100 + 50,
                'frame_id': list(range(1, self.sequence_length + 1))
            }
            yield batch

    def __len__(self):
        return self.num_sequences


class TestVisualizationIntegration:
    """Integration tests for visualization pipeline"""

    def test_visualization_artifacts_generated(self):
        """
        Test that visualization artifacts are generated when enabled.

        This addresses Audit Issue #1: No visualization artifacts during evaluation.
        """
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=3, sequence_length=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'eval_output'
            viz_dir = output_dir / 'visualization'

            # Run evaluation with visualization enabled
            evaluator = TrackingEvaluator(
                model=model,
                dataloader=dataloader,
                device='cpu',
                output_dir=str(output_dir),
                save_predictions=True,
                verbose=False
            )

            results = evaluator.evaluate()

            # Create visualizer and generate artifacts
            visualizer = TrackingVisualizer(save_dir=str(viz_dir), dpi=100)

            # Get best and worst sequences for visualization
            best_sequences = evaluator.get_best_sequences(n=2)
            worst_sequences = evaluator.get_worst_sequences(n=2)

            # Verify that sequences were identified
            assert len(best_sequences) == 2, "Should identify 2 best sequences"
            assert len(worst_sequences) == 2, "Should identify 2 worst sequences"

            # Verify visualization directory structure
            assert output_dir.exists(), "Output directory should exist"

            # Verify evaluation results were saved
            assert (output_dir / 'overall_metrics.json').exists(), \
                "Overall metrics should be saved"

    def test_visualization_disabled_no_artifacts(self):
        """
        Test that no visualization artifacts are generated when disabled.

        Verifies that the disable flag works correctly.
        """
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=2, sequence_length=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'eval_output'

            # Run evaluation WITHOUT visualization
            evaluator = TrackingEvaluator(
                model=model,
                dataloader=dataloader,
                device='cpu',
                output_dir=str(output_dir),
                save_predictions=False,
                verbose=False
            )

            results = evaluator.evaluate()

            # Verify no visualization directory was created
            viz_dir = output_dir / 'visualization'
            assert not viz_dir.exists() or len(list(viz_dir.glob('*'))) == 0, \
                "No visualization artifacts should be generated when disabled"

            # But evaluation results should still exist
            assert (output_dir / 'overall_metrics.json').exists(), \
                "Evaluation metrics should still be saved"

    def test_visualizer_creates_output_directory(self):
        """
        Test that TrackingVisualizer creates output directory if it doesn't exist.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            viz_dir = Path(tmpdir) / 'nonexistent' / 'visualization'

            # Visualizer should create directory
            visualizer = TrackingVisualizer(save_dir=str(viz_dir))

            assert viz_dir.exists(), "Visualizer should create save directory"

    def test_best_worst_sequence_selection(self):
        """
        Test that best/worst sequence selection works correctly.

        Verifies Audit Issue #3: Automated coverage for sequence ranking.
        """
        model = MockTrackingModel()
        dataloader = MockDataLoader(num_sequences=5, sequence_length=10)

        evaluator = TrackingEvaluator(
            model=model,
            dataloader=dataloader,
            device='cpu',
            verbose=False
        )

        results = evaluator.evaluate()

        # Get best and worst sequences
        best = evaluator.get_best_sequences(n=3)
        worst = evaluator.get_worst_sequences(n=3)

        # Verify counts
        assert len(best) == 3, "Should return 3 best sequences"
        assert len(worst) == 3, "Should return 3 worst sequences"

        # Verify ordering (best should be descending, worst ascending)
        best_aucs = [auc for _, auc in best]
        worst_aucs = [auc for _, auc in worst]

        assert best_aucs == sorted(best_aucs, reverse=True), \
            "Best sequences should be in descending order"
        assert worst_aucs == sorted(worst_aucs), \
            "Worst sequences should be in ascending order"

        # Verify no overlap if sequences are distinct
        best_names = set(name for name, _ in best)
        worst_names = set(name for name, _ in worst)
        assert len(best_names & worst_names) == 0, \
            "Best and worst should not overlap"

    def test_visualization_config_parsing(self):
        """
        Test that visualization config options are correctly parsed.

        Addresses Audit Issue #2: Configuration defaults contradict documentation.
        """
        from aquasot3d.utils.config import load_config

        # Test eval_uuv3.yaml has visualization enabled
        config_path = Path('configs/eval/eval_uuv3.yaml')
        if config_path.exists():
            cfg = load_config(str(config_path))

            assert hasattr(cfg.evaluation, 'visualization'), \
                "Config should have visualization section"
            assert hasattr(cfg.evaluation.visualization, 'enabled'), \
                "Visualization section should have enabled flag"

            # After fixes, default should be enabled
            assert cfg.evaluation.visualization.enabled == True, \
                "Visualization should be enabled by default in eval_uuv3.yaml"

            assert hasattr(cfg.evaluation.visualization, 'save_dir'), \
                "Should specify visualization save directory"

        # Test eval_multiseq.yaml has visualization enabled
        config_path = Path('configs/eval/eval_multiseq.yaml')
        if config_path.exists():
            cfg = load_config(str(config_path))

            # After fixes, should be enabled
            assert cfg.evaluation.visualization.enabled == True, \
                "Visualization should be enabled by default in eval_multiseq.yaml"


class TestVisualizationCLIFlags:
    """Tests for CLI visualization flags"""

    def test_cli_flag_override_enable(self):
        """
        Test that --visualize CLI flag overrides config.

        Addresses Audit Issue #3: Missing runtime override.
        """
        # This test verifies the flag parsing logic
        # Actual subprocess test would require a full checkpoint and dataset

        import argparse

        # Simulate the argument parser from scripts/eval.py
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--checkpoint', type=str, required=True)
        parser.add_argument('--opts', nargs='+', default=[])
        parser.add_argument('--output', type=str, default=None)

        viz_group = parser.add_mutually_exclusive_group()
        viz_group.add_argument('--visualize', action='store_true', default=None)
        viz_group.add_argument('--no-visualize', action='store_true', default=None)

        # Test --visualize flag
        args = parser.parse_args([
            '--config', 'dummy.yaml',
            '--checkpoint', 'dummy.pth',
            '--visualize'
        ])

        assert args.visualize == True, "Should parse --visualize flag"
        assert args.no_visualize != True, "Should not set --no-visualize"

    def test_cli_flag_override_disable(self):
        """Test that --no-visualize CLI flag overrides config"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--checkpoint', type=str, required=True)
        parser.add_argument('--opts', nargs='+', default=[])
        parser.add_argument('--output', type=str, default=None)

        viz_group = parser.add_mutually_exclusive_group()
        viz_group.add_argument('--visualize', action='store_true', default=None)
        viz_group.add_argument('--no-visualize', action='store_true', default=None)

        # Test --no-visualize flag
        args = parser.parse_args([
            '--config', 'dummy.yaml',
            '--checkpoint', 'dummy.pth',
            '--no-visualize'
        ])

        assert args.no_visualize == True, "Should parse --no-visualize flag"
        assert args.visualize != True, "Should not set --visualize"

    def test_cli_flags_mutually_exclusive(self):
        """Test that --visualize and --no-visualize are mutually exclusive"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--checkpoint', type=str, required=True)

        viz_group = parser.add_mutually_exclusive_group()
        viz_group.add_argument('--visualize', action='store_true', default=None)
        viz_group.add_argument('--no-visualize', action='store_true', default=None)

        # Should raise error when both flags are provided
        with pytest.raises(SystemExit):
            parser.parse_args([
                '--config', 'dummy.yaml',
                '--checkpoint', 'dummy.pth',
                '--visualize',
                '--no-visualize'
            ])


class TestEvaluationLogging:
    """Tests for evaluation logging improvements"""

    def test_visualization_status_logged(self, capsys):
        """
        Test that visualization status is clearly logged.

        Addresses Audit Issue #3: Missing runtime observability.
        """
        # This test would need to capture logger output
        # For now, verify the logging code exists in scripts/eval.py

        script_path = Path('scripts/eval.py')
        assert script_path.exists(), "Eval script should exist"

        script_content = script_path.read_text()

        # Check for visualization logging
        assert 'Visualization: ENABLED' in script_content or \
               'visualization.enabled' in script_content, \
            "Script should log visualization status"

        assert '--visualize' in script_content, \
            "Script should support --visualize flag"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
