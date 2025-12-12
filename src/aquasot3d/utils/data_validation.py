"""
Data Validation Utilities for AquaSOT-3D

This module provides comprehensive data validation functions to ensure dataset
integrity and prevent training/evaluation with improperly preprocessed data.

Key Features:
- Validate rect data preprocessing status
- Verify sonar resolution consistency (682x376)
- Block usage of polar data (geometrically incompatible)
- Generate metadata for reproducibility tracking
- Support for train.py, eval.py, and analyze_dataset.py

Audit References:
- Finding 4: Validate data root and rect preprocessing
- Finding 16: Verify sonar resolution consistency
- Finding 24: Block polar data usage
- Finding 26: Generate metadata with git hash and config md5
- Finding 27: Check rect readiness across sequences

Author: AquaSOT-3D Team
Date: 2025-12-13
"""

import cv2
import hashlib
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import uuid

logger = logging.getLogger(__name__)

# Expected rectified sonar resolution (width, height)
# Must match preprocessor output from scripts/preprocess_sonar.py
EXPECTED_SONAR_RECT_RESOLUTION = (682, 376)


def rect_data_check(
    sequence_path: Union[str, Path],
    modality: str = "sonar",
    check_resolution: bool = True,
) -> Dict[str, any]:
    """
    Validate that a sequence has rectified data with correct format.

    For sonar modality:
    - Checks for sonar_rect/img_rect directory
    - Checks for groundtruth_rect.txt
    - Validates resolution is 682x376 (if check_resolution=True)

    For RGB modality:
    - Checks for light/img directory
    - Checks for groundtruth.txt
    - Optionally validates resolution

    Args:
        sequence_path: Path to sequence directory (e.g., data/uuv1)
        modality: Modality to check ('sonar' or 'rgb')
        check_resolution: Whether to verify image resolution

    Returns:
        Dictionary with validation results:
            - has_rect_data: bool, whether rectified data exists
            - img_dir_exists: bool, whether image directory exists
            - gt_file_exists: bool, whether groundtruth file exists
            - resolution_ok: bool or None, resolution check result
            - actual_resolution: Tuple[int, int] or None, (width, height)
            - num_images: int, number of images found
            - num_annotations: int, number of annotation lines
            - errors: List[str], list of error messages

    Raises:
        ValueError: If invalid modality specified
        FileNotFoundError: If sequence directory does not exist
    """
    sequence_path = Path(sequence_path)

    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")

    if modality not in ["sonar", "rgb"]:
        raise ValueError(f"Invalid modality: {modality}. Must be 'sonar' or 'rgb'.")

    errors = []
    result = {
        "has_rect_data": False,
        "img_dir_exists": False,
        "gt_file_exists": False,
        "resolution_ok": None,
        "actual_resolution": None,
        "num_images": 0,
        "num_annotations": 0,
        "errors": errors,
    }

    # Define paths based on modality
    if modality == "sonar":
        modality_dir = sequence_path / "sonar"
        img_dir = modality_dir / "img_rect"
        gt_file = modality_dir / "groundtruth_rect.txt"
        expected_resolution = EXPECTED_SONAR_RECT_RESOLUTION
    else:  # rgb
        modality_dir = sequence_path / "light"
        img_dir = modality_dir / "img"
        gt_file = modality_dir / "groundtruth.txt"
        expected_resolution = None  # RGB can have varying resolutions

    # Check image directory
    if not img_dir.exists():
        errors.append(f"Image directory not found: {img_dir}")
        result["img_dir_exists"] = False
    else:
        result["img_dir_exists"] = True

        # Count images
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        result["num_images"] = len(img_files)

        if result["num_images"] == 0:
            errors.append(f"No images found in {img_dir}")

    # Check groundtruth file
    if not gt_file.exists():
        errors.append(f"Groundtruth file not found: {gt_file}")
        result["gt_file_exists"] = False
    else:
        result["gt_file_exists"] = True

        # Count annotations
        try:
            with open(gt_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                result["num_annotations"] = len(lines)
        except Exception as e:
            errors.append(f"Failed to read groundtruth file: {e}")

    # Check if both directory and groundtruth exist
    result["has_rect_data"] = result["img_dir_exists"] and result["gt_file_exists"]

    # Validate resolution if requested and data exists
    if check_resolution and result["img_dir_exists"] and result["num_images"] > 0:
        try:
            # Find first available image
            img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            first_img_path = img_files[0]

            # Load and check resolution
            img = cv2.imread(str(first_img_path), cv2.IMREAD_GRAYSCALE if modality == "sonar" else cv2.IMREAD_COLOR)

            if img is None:
                errors.append(f"Failed to load image: {first_img_path}")
                result["resolution_ok"] = False
            else:
                actual_resolution = (img.shape[1], img.shape[0])  # (width, height)
                result["actual_resolution"] = actual_resolution

                if expected_resolution is not None:
                    if actual_resolution == expected_resolution:
                        result["resolution_ok"] = True
                    else:
                        result["resolution_ok"] = False
                        errors.append(
                            f"Resolution mismatch: expected {expected_resolution[0]}x{expected_resolution[1]}, "
                            f"got {actual_resolution[0]}x{actual_resolution[1]}"
                        )
                else:
                    # RGB modality - just record resolution, don't validate
                    result["resolution_ok"] = True

        except Exception as e:
            errors.append(f"Resolution check failed: {e}")
            result["resolution_ok"] = False

    return result


def validate_data_root(
    data_root: Union[str, Path],
    sequence_names: Optional[List[str]] = None,
    check_resolution: bool = True,
) -> Dict[str, Dict]:
    """
    Check data root path and verify sequences have rect data.

    Args:
        data_root: Root directory containing sequence folders
        sequence_names: List of sequence names to check (if None, auto-detect)
        check_resolution: Whether to verify image resolution

    Returns:
        Dictionary mapping sequence_name to validation result dict

    Raises:
        FileNotFoundError: If data root does not exist
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    # Auto-detect sequences if not provided
    if sequence_names is None:
        sequence_names = []
        for item in data_root.iterdir():
            if item.is_dir() and (item / "sonar").exists():
                sequence_names.append(item.name)
        sequence_names.sort()
        logger.info(f"Auto-detected {len(sequence_names)} sequences: {sequence_names}")

    results = {}
    for seq_name in sequence_names:
        seq_path = data_root / seq_name

        if not seq_path.exists():
            logger.warning(f"Sequence directory not found: {seq_path}")
            results[seq_name] = {
                "has_rect_data": False,
                "errors": [f"Sequence directory not found: {seq_path}"],
            }
            continue

        # Check sonar rect data
        try:
            result = rect_data_check(seq_path, modality="sonar", check_resolution=check_resolution)
            results[seq_name] = result
        except Exception as e:
            logger.error(f"Error checking sequence {seq_name}: {e}")
            results[seq_name] = {
                "has_rect_data": False,
                "errors": [f"Validation error: {e}"],
            }

    return results


def check_rect_readiness(
    data_root: Union[str, Path],
    sequence_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Scan sequences and categorize by rect preprocessing status.

    Args:
        data_root: Root directory containing sequence folders
        sequence_names: List of sequence names to check (if None, auto-detect)

    Returns:
        Tuple of (ready_sequences, not_ready_sequences, issues_dict)
        - ready_sequences: List of sequences with proper rect data
        - not_ready_sequences: List of sequences missing rect data
        - issues_dict: Dictionary mapping sequence_name to list of issues

    Example:
        >>> ready, not_ready, issues = check_rect_readiness('data/')
        >>> if not_ready:
        ...     print(f"Need to preprocess: {not_ready}")
        ...     for seq in not_ready:
        ...         print(f"  {seq}: {issues[seq]}")
    """
    results = validate_data_root(data_root, sequence_names, check_resolution=True)

    ready_sequences = []
    not_ready_sequences = []
    issues_dict = {}

    for seq_name, result in results.items():
        if result.get("has_rect_data") and result.get("resolution_ok", False):
            ready_sequences.append(seq_name)
        else:
            not_ready_sequences.append(seq_name)
            issues_dict[seq_name] = result.get("errors", ["Unknown error"])

    return ready_sequences, not_ready_sequences, issues_dict


def block_polar_data(sequence_path: Union[str, Path]) -> None:
    """
    Raise error if sequence is using polar data instead of rectified data.

    This enforces the requirement that all training/evaluation must use
    geometrically correct rectified coordinates, not polar sonar data.

    Args:
        sequence_path: Path to sequence directory

    Raises:
        RuntimeError: If polar data is detected without proper rectification
    """
    sequence_path = Path(sequence_path)

    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")

    sonar_dir = sequence_path / "sonar"
    if not sonar_dir.exists():
        return  # No sonar directory, nothing to check

    # Check for polar data (img/ without img_rect/)
    polar_img_dir = sonar_dir / "img"
    rect_img_dir = sonar_dir / "img_rect"
    rect_gt_file = sonar_dir / "groundtruth_rect.txt"

    has_polar = polar_img_dir.exists()
    has_rect = rect_img_dir.exists() and rect_gt_file.exists()

    if has_polar and not has_rect:
        error_msg = (
            f"POLAR DATA DETECTED WITHOUT RECTIFICATION: {sequence_path.name}\n"
            f"\n"
            f"AquaSOT-3D requires geometrically correct rectified sonar data.\n"
            f"Using polar coordinates directly will produce incorrect training.\n"
            f"\n"
            f"Found:\n"
            f"  - Polar images: {polar_img_dir} (INCOMPATIBLE)\n"
            f"  - Rect images: {rect_img_dir} (MISSING)\n"
            f"  - Rect GT: {rect_gt_file} (MISSING)\n"
            f"\n"
            f"Required action:\n"
            f"  Run preprocessing to generate rectified data:\n"
            f"    python scripts/preprocess_sonar.py --sequence {sequence_path.name}\n"
            f"  or:\n"
            f"    python scripts/preprocess_sonar.py --all\n"
        )
        raise RuntimeError(error_msg)


def generate_metadata_dict(
    config_path: Optional[Union[str, Path]] = None,
    extra_info: Optional[Dict] = None,
) -> Dict[str, any]:
    """
    Generate metadata dictionary for reproducibility tracking.

    Creates a comprehensive metadata record including:
    - Git commit hash (if repository available)
    - Config file MD5 hash (if config_path provided)
    - Rect data validation status (if extra_info contains rect_check)
    - Unique run ID
    - Timestamp

    Args:
        config_path: Optional path to configuration file for MD5 hashing
        extra_info: Optional dictionary with additional metadata fields
            - rect_check: Result from check_rect_readiness()
            - data_root: Path to data root for validation
            - Any other custom fields

    Returns:
        Dictionary with metadata fields:
            - run_id: Unique identifier for this run
            - timestamp: ISO format timestamp
            - git_hash: Git commit hash (or 'unknown')
            - git_branch: Git branch name (or 'unknown')
            - git_dirty: Whether repository has uncommitted changes
            - config_md5: MD5 hash of config file (or None)
            - config_path: Path to config file (or None)
            - rect_check_passed: Whether rect validation passed (or None)
            - rect_ready_sequences: List of ready sequences (or None)
            - rect_issues: Dict of sequences with issues (or None)
            - Custom fields from extra_info

    Example:
        >>> metadata = generate_metadata_dict(
        ...     config_path='configs/train.yaml',
        ...     extra_info={'data_root': 'data/', 'rect_check': True}
        ... )
        >>> print(metadata['git_hash'])
        'a1b2c3d4'
    """
    metadata = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
    }

    # Get git information
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        metadata["git_hash"] = result.stdout.strip()[:8]  # Short hash

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        metadata["git_branch"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        metadata["git_dirty"] = bool(result.stdout.strip())

        if metadata["git_dirty"]:
            logger.warning("Git repository has uncommitted changes")

    except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError):
        metadata["git_hash"] = "unknown"
        metadata["git_branch"] = "unknown"
        metadata["git_dirty"] = False
        logger.debug("Could not retrieve git information")

    # Get config MD5
    if config_path is not None:
        config_path = Path(config_path)
        try:
            with open(config_path, 'rb') as f:
                config_content = f.read()
                config_md5 = hashlib.md5(config_content).hexdigest()
                metadata["config_md5"] = config_md5
                metadata["config_path"] = str(config_path)
        except Exception as e:
            logger.warning(f"Could not compute config MD5: {e}")
            metadata["config_md5"] = None
            metadata["config_path"] = str(config_path) if config_path else None
    else:
        metadata["config_md5"] = None
        metadata["config_path"] = None

    # Add rect validation info if requested
    if extra_info:
        # Check if rect validation was requested
        if "rect_check" in extra_info or "data_root" in extra_info:
            data_root = extra_info.get("data_root")

            if data_root:
                try:
                    ready, not_ready, issues = check_rect_readiness(data_root)
                    metadata["rect_check_passed"] = len(not_ready) == 0
                    metadata["rect_ready_sequences"] = ready
                    metadata["rect_not_ready_sequences"] = not_ready
                    metadata["rect_issues"] = issues
                except Exception as e:
                    logger.error(f"Rect validation failed: {e}")
                    metadata["rect_check_passed"] = None
                    metadata["rect_error"] = str(e)
            else:
                metadata["rect_check_passed"] = None

        # Add all other extra_info fields
        for key, value in extra_info.items():
            if key not in metadata:  # Don't override existing fields
                metadata[key] = value

    return metadata


def validate_sequences_or_raise(
    data_root: Union[str, Path],
    sequence_names: List[str],
    strict: bool = True,
) -> None:
    """
    Validate sequences have proper rect data or raise informative error.

    This is a convenience function for train.py/eval.py to fail fast with
    clear instructions if data is not properly preprocessed.

    Args:
        data_root: Root directory containing sequence folders
        sequence_names: List of sequence names to validate
        strict: If True, raise error on any validation failure.
                If False, only warn about issues.

    Raises:
        RuntimeError: If strict=True and any sequence fails validation

    Example:
        >>> # In train.py
        >>> validate_sequences_or_raise('data/', ['uuv1', 'uuv2'])
        >>> # Raises error if preprocessing needed
    """
    data_root = Path(data_root)

    ready, not_ready, issues = check_rect_readiness(data_root, sequence_names)

    if not_ready:
        error_lines = [
            f"DATA VALIDATION FAILED: {len(not_ready)}/{len(sequence_names)} sequences not ready",
            "",
            "Sequences missing proper rect preprocessing:",
        ]

        for seq_name in not_ready:
            seq_issues = issues.get(seq_name, ["Unknown error"])
            error_lines.append(f"  - {seq_name}:")
            for issue in seq_issues:
                error_lines.append(f"      {issue}")

        error_lines.extend([
            "",
            "Required action:",
            "  Run preprocessing to generate rectified data:",
            f"    python scripts/preprocess_sonar.py --all",
            "  or for specific sequences:",
        ])

        for seq_name in not_ready:
            error_lines.append(f"    python scripts/preprocess_sonar.py --sequence {seq_name}")

        error_msg = "\n".join(error_lines)

        if strict:
            raise RuntimeError(error_msg)
        else:
            logger.warning(error_msg)
    else:
        logger.info(
            f"Data validation passed: {len(ready)}/{len(sequence_names)} sequences ready"
        )


def get_validation_summary(data_root: Union[str, Path]) -> str:
    """
    Generate human-readable validation summary for logging.

    Args:
        data_root: Root directory containing sequence folders

    Returns:
        Multi-line string with validation summary

    Example:
        >>> summary = get_validation_summary('data/')
        >>> logger.info(summary)
    """
    ready, not_ready, issues = check_rect_readiness(data_root)

    lines = [
        "=" * 60,
        "DATA VALIDATION SUMMARY",
        "=" * 60,
        f"Data root: {data_root}",
        f"Total sequences: {len(ready) + len(not_ready)}",
        f"Ready sequences: {len(ready)}",
        f"Not ready sequences: {len(not_ready)}",
    ]

    if ready:
        lines.append("")
        lines.append("Ready sequences:")
        for seq_name in ready:
            lines.append(f"  ✓ {seq_name}")

    if not_ready:
        lines.append("")
        lines.append("Sequences needing preprocessing:")
        for seq_name in not_ready:
            seq_issues = issues.get(seq_name, ["Unknown error"])
            lines.append(f"  ✗ {seq_name}:")
            for issue in seq_issues:
                lines.append(f"      - {issue}")

    lines.append("=" * 60)

    return "\n".join(lines)
