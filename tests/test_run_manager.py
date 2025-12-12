"""
Test Suite for Run Manager Module

Tests output directory versioning with run_id, metadata tracking,
legacy output detection, and train/eval setup workflows.

Author: AquaSOT-3D Team
Date: 2025-12-13
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest

from src.aquasot3d.utils.run_manager import (
    generate_run_id,
    get_git_hash,
    get_git_status,
    create_run_directory,
    save_run_metadata,
    load_run_metadata,
    check_output_directory,
    detect_legacy_outputs,
    migrate_legacy_outputs,
    find_latest_run,
    setup_train_run,
    setup_eval_run,
)


class TestRunIdGeneration:
    """Test run ID generation functionality."""

    def test_generate_run_id_format(self):
        """Test that run_id follows the expected format."""
        run_id = generate_run_id()

        # Check format: YYYYMMDD_HHMMSS_xxxxxx
        assert len(run_id) >= 22
        assert run_id[8] == '_'
        assert run_id[15] == '_'

        # Check timestamp part is numeric
        timestamp_part = run_id[:15].replace('_', '')
        assert timestamp_part.isdigit()

        # Check hash part is hexadecimal
        hash_part = run_id[16:22]
        assert all(c in '0123456789abcdef' for c in hash_part)

    def test_generate_run_id_uniqueness(self):
        """Test that multiple run_ids are unique."""
        run_ids = [generate_run_id() for _ in range(10)]
        assert len(set(run_ids)) == len(run_ids)


class TestGitIntegration:
    """Test git integration functionality."""

    def test_get_git_hash(self):
        """Test getting git commit hash."""
        git_hash = get_git_hash()

        # Should return hash or None
        if git_hash is not None:
            assert len(git_hash) == 7
            assert all(c in '0123456789abcdef' for c in git_hash)

    def test_get_git_status(self):
        """Test getting git status information."""
        status = get_git_status()

        # Check required fields
        assert 'hash' in status
        assert 'branch' in status
        assert 'is_dirty' in status
        assert 'uncommitted_files' in status

        # Check types
        assert isinstance(status['is_dirty'], bool)
        assert isinstance(status['uncommitted_files'], list)


class TestRunDirectory:
    """Test run directory creation and management."""

    def test_create_run_directory_basic(self):
        """Test basic run directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / 'outputs'
            run_dir, run_id = create_run_directory(str(base_dir))

            # Check directory exists
            assert Path(run_dir).exists()
            assert Path(run_dir).is_dir()

            # Check directory name matches run_id
            assert Path(run_dir).name == run_id

    def test_create_run_directory_with_subdirs(self):
        """Test run directory creation with subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / 'outputs'
            subdirs = ['checkpoints', 'logs', 'visualizations']

            run_dir, run_id = create_run_directory(
                str(base_dir),
                create_subdirs=subdirs
            )

            # Check subdirectories exist
            for subdir in subdirs:
                subdir_path = Path(run_dir) / subdir
                assert subdir_path.exists()
                assert subdir_path.is_dir()

    def test_create_run_directory_custom_run_id(self):
        """Test run directory creation with custom run_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / 'outputs'
            custom_run_id = '20251213_120000_abc123'

            run_dir, run_id = create_run_directory(
                str(base_dir),
                run_id=custom_run_id
            )

            # Check custom run_id was used
            assert run_id == custom_run_id
            assert Path(run_dir).name == custom_run_id


class TestMetadata:
    """Test metadata saving and loading."""

    def test_save_run_metadata_basic(self):
        """Test basic metadata saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'run_20251213_120000_abc123'
            run_dir.mkdir(parents=True)

            metadata_path = save_run_metadata(str(run_dir))

            # Check metadata file exists
            assert Path(metadata_path).exists()
            assert Path(metadata_path).name == 'metadata.json'

            # Check metadata content
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            assert 'run_id' in metadata
            assert 'timestamp' in metadata
            assert 'git' in metadata

    def test_save_run_metadata_with_config(self):
        """Test metadata saving with configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'run_20251213_120000_abc123'
            run_dir.mkdir(parents=True)

            config = {
                'model': {'type': 'AquaSOT3DTracker'},
                'training': {'lr': 0.0001}
            }

            metadata_path = save_run_metadata(str(run_dir), config=config)

            # Check metadata content
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            assert 'config' in metadata
            assert metadata['config']['model']['type'] == 'AquaSOT3DTracker'

    def test_save_run_metadata_with_rect_check(self):
        """Test metadata saving with rect_check result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'run_20251213_120000_abc123'
            run_dir.mkdir(parents=True)

            rect_check = {'valid': True, 'errors': []}

            metadata_path = save_run_metadata(
                str(run_dir),
                rect_check_result=rect_check
            )

            # Check metadata content
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            assert 'rect_check' in metadata
            assert metadata['rect_check']['valid'] is True

    def test_load_run_metadata(self):
        """Test loading metadata from run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'run_20251213_120000_abc123'
            run_dir.mkdir(parents=True)

            # Save metadata
            config = {'model': {'type': 'test'}}
            save_run_metadata(str(run_dir), config=config)

            # Load metadata
            metadata = load_run_metadata(str(run_dir))

            assert metadata is not None
            assert 'run_id' in metadata
            assert 'config' in metadata
            assert metadata['config']['model']['type'] == 'test'

    def test_load_run_metadata_missing(self):
        """Test loading metadata from directory without metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = load_run_metadata(tmpdir)
            assert metadata is None


class TestOutputDirectoryChecks:
    """Test output directory checking functionality."""

    def test_check_output_directory_empty(self):
        """Test checking empty output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            has_existing, items = check_output_directory(tmpdir)
            assert has_existing is False
            assert len(items) == 0

    def test_check_output_directory_nonexistent(self):
        """Test checking non-existent output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = Path(tmpdir) / 'nonexistent'
            has_existing, items = check_output_directory(str(nonexistent_dir))
            assert has_existing is False
            assert len(items) == 0

    def test_check_output_directory_with_versioned_outputs(self):
        """Test checking directory with versioned outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create versioned run directories
            run_dir1 = Path(tmpdir) / '20251213_120000_abc123'
            run_dir1.mkdir()
            run_dir2 = Path(tmpdir) / '20251213_130000_def456'
            run_dir2.mkdir()

            has_existing, items = check_output_directory(tmpdir, force=True)
            assert has_existing is True
            assert len(items) == 2

    def test_check_output_directory_with_legacy_outputs(self):
        """Test checking directory with legacy outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy files
            (Path(tmpdir) / 'model.pth').touch()
            (Path(tmpdir) / 'checkpoint.ckpt').touch()
            (Path(tmpdir) / 'checkpoints').mkdir()

            has_existing, items = check_output_directory(tmpdir, force=True)
            assert has_existing is True
            assert len(items) >= 2  # At least the files

    def test_check_output_directory_force_flag(self):
        """Test check_output_directory with force flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing output
            (Path(tmpdir) / 'model.pth').touch()

            # With force=True, should not raise
            has_existing, items = check_output_directory(tmpdir, force=True)
            assert has_existing is True

            # With force=False, should raise
            with pytest.raises(FileExistsError):
                check_output_directory(tmpdir, force=False)


class TestLegacyDetection:
    """Test legacy output detection."""

    def test_detect_legacy_outputs_empty(self):
        """Test legacy detection in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            has_legacy, items = detect_legacy_outputs(tmpdir)
            assert has_legacy is False
            assert len(items) == 0

    def test_detect_legacy_outputs_with_files(self):
        """Test legacy detection with legacy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy files
            (Path(tmpdir) / 'model.pth').touch()
            (Path(tmpdir) / 'results.json').touch()

            has_legacy, items = detect_legacy_outputs(tmpdir)
            assert has_legacy is True
            assert len(items) == 2

    def test_detect_legacy_outputs_with_dirs(self):
        """Test legacy detection with legacy directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy directories with content
            checkpoints_dir = Path(tmpdir) / 'checkpoints'
            checkpoints_dir.mkdir()
            (checkpoints_dir / 'model.pth').touch()

            has_legacy, items = detect_legacy_outputs(tmpdir)
            assert has_legacy is True
            assert any('checkpoints' in item for item in items)

    def test_detect_legacy_outputs_vs_versioned(self):
        """Test that versioned outputs are not detected as legacy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create versioned run directory
            run_dir = Path(tmpdir) / '20251213_120000_abc123'
            run_dir.mkdir()
            (run_dir / 'metadata.json').touch()

            has_legacy, items = detect_legacy_outputs(tmpdir)
            # The run directory itself should not be detected as legacy
            # (it matches the run_id pattern)
            assert has_legacy is False


class TestLegacyMigration:
    """Test legacy output migration."""

    def test_migrate_legacy_outputs_dry_run(self):
        """Test legacy migration in dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy files
            (Path(tmpdir) / 'model.pth').touch()
            (Path(tmpdir) / 'results.json').touch()

            report = migrate_legacy_outputs(tmpdir, dry_run=True)

            assert report['success'] is True
            assert report['dry_run'] is True
            assert len(report['migrated_items']) == 2
            assert len(report['errors']) == 0

            # Files should still exist (dry run)
            assert (Path(tmpdir) / 'model.pth').exists()
            assert (Path(tmpdir) / 'results.json').exists()

    def test_migrate_legacy_outputs_actual(self):
        """Test actual legacy migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy files
            (Path(tmpdir) / 'model.pth').touch()
            (Path(tmpdir) / 'results.json').touch()

            report = migrate_legacy_outputs(tmpdir, dry_run=False)

            assert report['success'] is True
            assert len(report['migrated_items']) == 2
            assert len(report['errors']) == 0

            # Files should be moved to archive
            assert not (Path(tmpdir) / 'model.pth').exists()
            assert not (Path(tmpdir) / 'results.json').exists()

            archive_dir = Path(tmpdir) / 'legacy_archive'
            assert archive_dir.exists()
            assert (archive_dir / 'model.pth').exists()
            assert (archive_dir / 'results.json').exists()

    def test_migrate_legacy_outputs_custom_archive(self):
        """Test migration with custom archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy file
            (Path(tmpdir) / 'model.pth').touch()

            custom_archive = Path(tmpdir) / 'my_archive'
            report = migrate_legacy_outputs(
                tmpdir,
                archive_dir=str(custom_archive),
                dry_run=False
            )

            assert report['success'] is True
            assert custom_archive.exists()
            assert (custom_archive / 'model.pth').exists()


class TestFindLatestRun:
    """Test finding latest run functionality."""

    def test_find_latest_run_empty(self):
        """Test finding latest run in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_run(tmpdir)
            assert result is None

    def test_find_latest_run_single(self):
        """Test finding latest run with single run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create run directory with metadata
            run_dir, run_id = create_run_directory(tmpdir)
            save_run_metadata(run_dir, run_id=run_id)

            result = find_latest_run(tmpdir)
            assert result is not None

            latest_run_dir, metadata = result
            assert latest_run_dir == run_dir
            assert metadata['run_id'] == run_id

    def test_find_latest_run_multiple(self):
        """Test finding latest run with multiple runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple run directories
            run_dir1, run_id1 = create_run_directory(
                tmpdir,
                run_id='20251213_100000_aaa111'
            )
            save_run_metadata(run_dir1, run_id=run_id1)

            run_dir2, run_id2 = create_run_directory(
                tmpdir,
                run_id='20251213_200000_bbb222'
            )
            save_run_metadata(run_dir2, run_id=run_id2)

            result = find_latest_run(tmpdir)
            assert result is not None

            latest_run_dir, metadata = result
            # Should return the run with later timestamp
            assert metadata['run_id'] == run_id2


class TestTrainEvalSetup:
    """Test convenience functions for train/eval setup."""

    def test_setup_train_run(self):
        """Test training run setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'paths': {'work_dir': tmpdir},
                'model': {'type': 'AquaSOT3DTracker'}
            }

            run_dir, checkpoint_dir, log_dir = setup_train_run(config)

            # Check directories exist
            assert Path(run_dir).exists()
            assert Path(checkpoint_dir).exists()
            assert Path(log_dir).exists()

            # Check metadata was saved
            metadata = load_run_metadata(run_dir)
            assert metadata is not None
            assert 'config' in metadata

    def test_setup_eval_run(self):
        """Test evaluation run setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'evaluation': {'output_dir': tmpdir},
                'model': {'type': 'AquaSOT3DTracker'}
            }
            checkpoint_path = 'path/to/model.pth'

            run_dir, visualization_dir = setup_eval_run(
                config,
                checkpoint_path
            )

            # Check directories exist
            assert Path(run_dir).exists()
            assert Path(visualization_dir).exists()

            # Check metadata was saved with checkpoint path
            metadata = load_run_metadata(run_dir)
            assert metadata is not None
            assert 'additional' in metadata
            assert metadata['additional']['checkpoint_path'] == checkpoint_path


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_train_workflow(self):
        """Test complete training workflow with run management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup config
            config = {
                'paths': {'work_dir': tmpdir},
                'model': {'type': 'AquaSOT3DTracker'},
                'training': {'lr': 0.0001}
            }

            # Setup training run
            run_dir, checkpoint_dir, log_dir = setup_train_run(config)

            # Simulate training artifacts
            checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
            checkpoint_path.touch()

            log_path = Path(log_dir) / 'train.log'
            log_path.write_text('Training log content')

            # Verify structure
            assert Path(run_dir).exists()
            assert checkpoint_path.exists()
            assert log_path.exists()

            # Verify metadata
            metadata = load_run_metadata(run_dir)
            assert metadata is not None
            assert metadata['config']['training']['lr'] == 0.0001

            # Find latest run
            result = find_latest_run(tmpdir)
            assert result is not None
            latest_dir, latest_metadata = result
            assert latest_dir == run_dir

    def test_complete_eval_workflow(self):
        """Test complete evaluation workflow with run management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup config
            config = {
                'evaluation': {'output_dir': tmpdir},
                'model': {'type': 'AquaSOT3DTracker'}
            }
            checkpoint_path = 'path/to/best_model.pth'

            # Setup evaluation run
            run_dir, vis_dir = setup_eval_run(config, checkpoint_path)

            # Simulate evaluation artifacts
            results_path = Path(run_dir) / 'results.json'
            results_path.write_text('{"success_auc": 0.75}')

            vis_path = Path(vis_dir) / 'trajectory.png'
            vis_path.touch()

            # Verify structure
            assert Path(run_dir).exists()
            assert results_path.exists()
            assert vis_path.exists()

            # Verify metadata
            metadata = load_run_metadata(run_dir)
            assert metadata is not None
            assert metadata['additional']['checkpoint_path'] == checkpoint_path

    def test_legacy_compatibility(self):
        """Test handling of legacy outputs alongside new versioned outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy outputs
            (Path(tmpdir) / 'old_model.pth').touch()
            legacy_checkpoints = Path(tmpdir) / 'checkpoints'
            legacy_checkpoints.mkdir()
            (legacy_checkpoints / 'model_epoch_10.pth').touch()

            # Detect legacy outputs
            has_legacy, legacy_items = detect_legacy_outputs(tmpdir)
            assert has_legacy is True
            assert len(legacy_items) >= 2

            # Create new versioned run
            config = {'paths': {'work_dir': tmpdir}}
            run_dir, _, _ = setup_train_run(config, force=True)

            # Verify new run coexists with legacy outputs
            assert Path(run_dir).exists()
            assert (Path(tmpdir) / 'old_model.pth').exists()

            # Migrate legacy outputs
            report = migrate_legacy_outputs(tmpdir, dry_run=False)
            assert report['success'] is True

            # Verify legacy outputs are archived
            assert not (Path(tmpdir) / 'old_model.pth').exists()
            archive_dir = Path(tmpdir) / 'legacy_archive'
            assert (archive_dir / 'old_model.pth').exists()

            # Verify versioned run is unaffected
            assert Path(run_dir).exists()
            metadata = load_run_metadata(run_dir)
            assert metadata is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
