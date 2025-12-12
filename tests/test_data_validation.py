#!/usr/bin/env python3
"""
Test Data Validation Utilities

This test file demonstrates and validates the data validation utilities
used across train.py, eval.py, and analyze_dataset.py.

Author: AquaSOT-3D Team
Date: 2025-12-13
"""

import sys
import pytest
from pathlib import Path
import tempfile
import shutil
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from aquasot3d.utils.data_validation import (
    rect_data_check,
    validate_data_root,
    check_rect_readiness,
    block_polar_data,
    generate_metadata_dict,
    validate_sequences_or_raise,
    get_validation_summary,
    EXPECTED_SONAR_RECT_RESOLUTION,
)


class TestRectDataCheck:
    """Test rect_data_check function"""

    @pytest.fixture
    def temp_sequence_rect(self):
        """Create temporary sequence with rect data"""
        tmpdir = Path(tempfile.mkdtemp())

        # Create directory structure
        sonar_dir = tmpdir / "test_seq" / "sonar"
        img_rect_dir = sonar_dir / "img_rect"
        img_rect_dir.mkdir(parents=True)

        # Create sample images with correct resolution
        for i in range(1, 6):
            img = np.zeros((376, 682), dtype=np.uint8)
            cv2.imwrite(str(img_rect_dir / f"{i}.jpg"), img)

        # Create groundtruth file
        gt_file = sonar_dir / "groundtruth_rect.txt"
        with open(gt_file, 'w') as f:
            for i in range(5):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        yield tmpdir / "test_seq"

        # Cleanup
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def temp_sequence_no_rect(self):
        """Create temporary sequence without rect data"""
        tmpdir = Path(tempfile.mkdtemp())

        # Create directory with only polar data
        sonar_dir = tmpdir / "test_seq" / "sonar"
        img_dir = sonar_dir / "img"
        img_dir.mkdir(parents=True)

        # Create sample polar images
        for i in range(1, 6):
            img = np.zeros((512, 512), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{i}.jpg"), img)

        # Create groundtruth file
        gt_file = sonar_dir / "groundtruth.txt"
        with open(gt_file, 'w') as f:
            for i in range(5):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        yield tmpdir / "test_seq"

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_rect_data_check_valid(self, temp_sequence_rect):
        """Test rect_data_check with valid rect data"""
        result = rect_data_check(temp_sequence_rect, modality="sonar")

        assert result["has_rect_data"] is True
        assert result["img_dir_exists"] is True
        assert result["gt_file_exists"] is True
        assert result["resolution_ok"] is True
        assert result["actual_resolution"] == EXPECTED_SONAR_RECT_RESOLUTION
        assert result["num_images"] == 5
        assert result["num_annotations"] == 5
        assert len(result["errors"]) == 0

    def test_rect_data_check_missing_rect(self, temp_sequence_no_rect):
        """Test rect_data_check with missing rect data"""
        result = rect_data_check(temp_sequence_no_rect, modality="sonar")

        assert result["has_rect_data"] is False
        assert result["img_dir_exists"] is False
        assert len(result["errors"]) > 0

    def test_rect_data_check_wrong_resolution(self, temp_sequence_rect):
        """Test rect_data_check with wrong resolution"""
        # Replace one image with wrong resolution
        img_rect_dir = temp_sequence_rect / "sonar" / "img_rect"
        wrong_img = np.zeros((400, 600), dtype=np.uint8)
        cv2.imwrite(str(img_rect_dir / "1.jpg"), wrong_img)

        result = rect_data_check(temp_sequence_rect, modality="sonar", check_resolution=True)

        assert result["resolution_ok"] is False
        assert "Resolution mismatch" in str(result["errors"])

    def test_rect_data_check_rgb_modality(self, temp_sequence_rect):
        """Test rect_data_check for RGB modality"""
        # Create RGB directory structure
        rgb_dir = temp_sequence_rect / "light"
        img_dir = rgb_dir / "img"
        img_dir.mkdir(parents=True)

        # Create sample RGB images
        for i in range(1, 6):
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{i}.jpg"), img)

        # Create groundtruth file
        gt_file = rgb_dir / "groundtruth.txt"
        with open(gt_file, 'w') as f:
            for i in range(5):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        result = rect_data_check(temp_sequence_rect, modality="rgb")

        assert result["has_rect_data"] is True
        assert result["img_dir_exists"] is True
        assert result["gt_file_exists"] is True
        assert result["num_images"] == 5


class TestValidateDataRoot:
    """Test validate_data_root function"""

    @pytest.fixture
    def temp_data_root(self):
        """Create temporary data root with multiple sequences"""
        tmpdir = Path(tempfile.mkdtemp())

        # Create two sequences: one with rect, one without
        for seq_name in ["seq1", "seq2"]:
            sonar_dir = tmpdir / seq_name / "sonar"
            img_rect_dir = sonar_dir / "img_rect"
            img_rect_dir.mkdir(parents=True)

            # Create images
            for i in range(1, 4):
                img = np.zeros((376, 682), dtype=np.uint8)
                cv2.imwrite(str(img_rect_dir / f"{i}.jpg"), img)

            # Create groundtruth
            gt_file = sonar_dir / "groundtruth_rect.txt"
            with open(gt_file, 'w') as f:
                for i in range(3):
                    f.write(f"{i*10},{i*10},{50},{50}\n")

        yield tmpdir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_validate_data_root_with_sequence_names(self, temp_data_root):
        """Test validate_data_root with explicit sequence names"""
        results = validate_data_root(temp_data_root, sequence_names=["seq1", "seq2"])

        assert len(results) == 2
        assert "seq1" in results
        assert "seq2" in results
        assert results["seq1"]["has_rect_data"] is True
        assert results["seq2"]["has_rect_data"] is True

    def test_validate_data_root_auto_detect(self, temp_data_root):
        """Test validate_data_root with auto-detection"""
        results = validate_data_root(temp_data_root, sequence_names=None)

        assert len(results) == 2
        assert all(result["has_rect_data"] for result in results.values())


class TestCheckRectReadiness:
    """Test check_rect_readiness function"""

    @pytest.fixture
    def mixed_data_root(self):
        """Create data root with mixed sequence states"""
        tmpdir = Path(tempfile.mkdtemp())

        # Sequence 1: Properly preprocessed
        sonar_dir = tmpdir / "seq1" / "sonar"
        img_rect_dir = sonar_dir / "img_rect"
        img_rect_dir.mkdir(parents=True)

        for i in range(1, 4):
            img = np.zeros((376, 682), dtype=np.uint8)
            cv2.imwrite(str(img_rect_dir / f"{i}.jpg"), img)

        gt_file = sonar_dir / "groundtruth_rect.txt"
        with open(gt_file, 'w') as f:
            for i in range(3):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        # Sequence 2: Missing rect data
        sonar_dir2 = tmpdir / "seq2" / "sonar"
        img_dir2 = sonar_dir2 / "img"
        img_dir2.mkdir(parents=True)

        for i in range(1, 4):
            img = np.zeros((512, 512), dtype=np.uint8)
            cv2.imwrite(str(img_dir2 / f"{i}.jpg"), img)

        gt_file2 = sonar_dir2 / "groundtruth.txt"
        with open(gt_file2, 'w') as f:
            for i in range(3):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        yield tmpdir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_check_rect_readiness_mixed(self, mixed_data_root):
        """Test check_rect_readiness with mixed sequences"""
        ready, not_ready, issues = check_rect_readiness(mixed_data_root)

        assert "seq1" in ready
        assert "seq2" in not_ready
        assert "seq2" in issues
        assert len(issues["seq2"]) > 0


class TestBlockPolarData:
    """Test block_polar_data function"""

    @pytest.fixture
    def polar_only_sequence(self):
        """Create sequence with only polar data"""
        tmpdir = Path(tempfile.mkdtemp())

        sonar_dir = tmpdir / "test_seq" / "sonar"
        img_dir = sonar_dir / "img"
        img_dir.mkdir(parents=True)

        # Create polar images
        for i in range(1, 4):
            img = np.zeros((512, 512), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{i}.jpg"), img)

        gt_file = sonar_dir / "groundtruth.txt"
        with open(gt_file, 'w') as f:
            for i in range(3):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        yield tmpdir / "test_seq"

        # Cleanup
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def rect_sequence(self):
        """Create sequence with rect data"""
        tmpdir = Path(tempfile.mkdtemp())

        sonar_dir = tmpdir / "test_seq" / "sonar"
        img_rect_dir = sonar_dir / "img_rect"
        img_rect_dir.mkdir(parents=True)

        # Create rect images
        for i in range(1, 4):
            img = np.zeros((376, 682), dtype=np.uint8)
            cv2.imwrite(str(img_rect_dir / f"{i}.jpg"), img)

        gt_file = sonar_dir / "groundtruth_rect.txt"
        with open(gt_file, 'w') as f:
            for i in range(3):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        yield tmpdir / "test_seq"

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_block_polar_data_raises(self, polar_only_sequence):
        """Test block_polar_data raises error for polar-only data"""
        with pytest.raises(RuntimeError, match="POLAR DATA DETECTED"):
            block_polar_data(polar_only_sequence)

    def test_block_polar_data_passes(self, rect_sequence):
        """Test block_polar_data passes for rect data"""
        # Should not raise
        block_polar_data(rect_sequence)


class TestGenerateMetadataDict:
    """Test generate_metadata_dict function"""

    def test_generate_metadata_basic(self):
        """Test basic metadata generation"""
        metadata = generate_metadata_dict()

        assert "run_id" in metadata
        assert "timestamp" in metadata
        assert "git_hash" in metadata
        assert "git_branch" in metadata
        assert "git_dirty" in metadata

    def test_generate_metadata_with_config(self):
        """Test metadata generation with config file"""
        # Create temporary config file
        tmpdir = Path(tempfile.mkdtemp())
        config_file = tmpdir / "test_config.yaml"
        config_file.write_text("model:\n  type: test\n")

        metadata = generate_metadata_dict(config_path=config_file)

        assert "config_md5" in metadata
        assert metadata["config_md5"] is not None
        assert "config_path" in metadata

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_generate_metadata_with_rect_check(self):
        """Test metadata generation with rect check"""
        tmpdir = Path(tempfile.mkdtemp())

        # Create simple sequence
        sonar_dir = tmpdir / "seq1" / "sonar"
        img_rect_dir = sonar_dir / "img_rect"
        img_rect_dir.mkdir(parents=True)

        for i in range(1, 4):
            img = np.zeros((376, 682), dtype=np.uint8)
            cv2.imwrite(str(img_rect_dir / f"{i}.jpg"), img)

        gt_file = sonar_dir / "groundtruth_rect.txt"
        with open(gt_file, 'w') as f:
            for i in range(3):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        metadata = generate_metadata_dict(extra_info={"data_root": str(tmpdir)})

        assert "rect_check_passed" in metadata
        assert "rect_ready_sequences" in metadata
        assert "seq1" in metadata["rect_ready_sequences"]

        # Cleanup
        shutil.rmtree(tmpdir)


class TestValidateSequencesOrRaise:
    """Test validate_sequences_or_raise function"""

    @pytest.fixture
    def data_root_with_issues(self):
        """Create data root with some sequences missing rect data"""
        tmpdir = Path(tempfile.mkdtemp())

        # Good sequence
        sonar_dir = tmpdir / "seq1" / "sonar"
        img_rect_dir = sonar_dir / "img_rect"
        img_rect_dir.mkdir(parents=True)

        for i in range(1, 4):
            img = np.zeros((376, 682), dtype=np.uint8)
            cv2.imwrite(str(img_rect_dir / f"{i}.jpg"), img)

        gt_file = sonar_dir / "groundtruth_rect.txt"
        with open(gt_file, 'w') as f:
            for i in range(3):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        # Bad sequence (polar only)
        sonar_dir2 = tmpdir / "seq2" / "sonar"
        img_dir2 = sonar_dir2 / "img"
        img_dir2.mkdir(parents=True)

        yield tmpdir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_validate_sequences_or_raise_strict(self, data_root_with_issues):
        """Test validate_sequences_or_raise in strict mode"""
        with pytest.raises(RuntimeError, match="DATA VALIDATION FAILED"):
            validate_sequences_or_raise(
                data_root_with_issues,
                sequence_names=["seq1", "seq2"],
                strict=True
            )

    def test_validate_sequences_or_raise_non_strict(self, data_root_with_issues):
        """Test validate_sequences_or_raise in non-strict mode"""
        # Should not raise, just warn
        validate_sequences_or_raise(
            data_root_with_issues,
            sequence_names=["seq1", "seq2"],
            strict=False
        )


class TestGetValidationSummary:
    """Test get_validation_summary function"""

    @pytest.fixture
    def data_root(self):
        """Create data root for summary testing"""
        tmpdir = Path(tempfile.mkdtemp())

        # Create one good sequence
        sonar_dir = tmpdir / "seq1" / "sonar"
        img_rect_dir = sonar_dir / "img_rect"
        img_rect_dir.mkdir(parents=True)

        for i in range(1, 4):
            img = np.zeros((376, 682), dtype=np.uint8)
            cv2.imwrite(str(img_rect_dir / f"{i}.jpg"), img)

        gt_file = sonar_dir / "groundtruth_rect.txt"
        with open(gt_file, 'w') as f:
            for i in range(3):
                f.write(f"{i*10},{i*10},{50},{50}\n")

        yield tmpdir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_get_validation_summary(self, data_root):
        """Test get_validation_summary output"""
        summary = get_validation_summary(data_root)

        assert "DATA VALIDATION SUMMARY" in summary
        assert "Ready sequences:" in summary
        assert "seq1" in summary


def test_expected_sonar_resolution():
    """Test that expected resolution constant is correct"""
    assert EXPECTED_SONAR_RECT_RESOLUTION == (682, 376)
    assert EXPECTED_SONAR_RECT_RESOLUTION[0] == 682  # width
    assert EXPECTED_SONAR_RECT_RESOLUTION[1] == 376  # height


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
