"""
Run Manager for AquaSOT-3D

This module provides utilities for output directory versioning with run_id,
including unique run ID generation, versioned output directory creation,
metadata tracking, and legacy output detection.

Addresses audit findings:
- 13: Output directory versioning with run_id
- 21: Metadata tracking (git hash, config, rect_check result)
- 24: Legacy output detection and incompatibility warnings

Author: AquaSOT-3D Team
Date: 2025-12-13
"""

import os
import json
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """
    Generate unique run ID with format '{timestamp}_{short_hash}'.

    The run ID combines:
    - Timestamp: YYYYMMDD_HHMMSS for human readability
    - Short hash: 6-character random hash for uniqueness

    Returns:
        str: Unique run ID (e.g., '20251213_143022_a3f2b1')

    Example:
        >>> run_id = generate_run_id()
        >>> print(run_id)
        '20251213_143022_a3f2b1'
    """
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Generate short hash from timestamp + random seed
    hash_input = f"{timestamp}_{os.urandom(16).hex()}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:6]

    run_id = f"{timestamp}_{short_hash}"

    logger.info(f"Generated run ID: {run_id}")
    return run_id


def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        str: Short git commit hash (7 characters), or None if not in git repo

    Example:
        >>> git_hash = get_git_hash()
        >>> print(git_hash)
        'b18ea45'
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )
        git_hash = result.stdout.strip()
        logger.debug(f"Git hash: {git_hash}")
        return git_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to get git hash (not in git repo or git not available)")
        return None


def get_git_status() -> Dict[str, Any]:
    """
    Get git repository status information.

    Returns:
        dict: Git status information including:
            - hash: Current commit hash
            - branch: Current branch name
            - is_dirty: Whether working directory has uncommitted changes
            - uncommitted_files: List of uncommitted file paths

    Example:
        >>> status = get_git_status()
        >>> print(status['is_dirty'])
        False
    """
    status = {
        'hash': None,
        'branch': None,
        'is_dirty': False,
        'uncommitted_files': []
    }

    try:
        # Get commit hash
        status['hash'] = get_git_hash()

        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )
        status['branch'] = result.stdout.strip()

        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )

        if result.stdout.strip():
            status['is_dirty'] = True
            status['uncommitted_files'] = [
                line.strip() for line in result.stdout.strip().split('\n')
            ]
            logger.warning(
                f"Working directory has {len(status['uncommitted_files'])} uncommitted changes"
            )

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Failed to get git status: {e}")

    return status


def create_run_directory(
    base_output_dir: str,
    run_id: Optional[str] = None,
    create_subdirs: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Create versioned output directory with run_id subdirectory.

    Directory structure:
        base_output_dir/
            run_id/
                metadata.json
                checkpoints/
                logs/
                visualizations/
                ... (other subdirs)

    Args:
        base_output_dir: Base output directory path (e.g., 'outputs/train')
        run_id: Optional run ID (generated if not provided)
        create_subdirs: Optional list of subdirectory names to create
                       (e.g., ['checkpoints', 'logs', 'visualizations'])

    Returns:
        tuple: (run_directory_path, run_id)
            - run_directory_path: Full path to run-specific directory
            - run_id: The run ID used

    Example:
        >>> run_dir, run_id = create_run_directory(
        ...     'outputs/train',
        ...     create_subdirs=['checkpoints', 'logs']
        ... )
        >>> print(run_dir)
        'outputs/train/20251213_143022_a3f2b1'
    """
    # Generate run_id if not provided
    if run_id is None:
        run_id = generate_run_id()

    # Create run directory
    run_dir = Path(base_output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")

    # Create subdirectories if specified
    if create_subdirs:
        for subdir in create_subdirs:
            subdir_path = run_dir / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created subdirectory: {subdir_path}")

    return str(run_dir), run_id


def save_run_metadata(
    run_dir: str,
    config: Optional[Dict[str, Any]] = None,
    rect_check_result: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Write metadata.json with git hash, config, rect_check result, run_id, and timestamp.

    Metadata structure:
        {
            "run_id": "20251213_143022_a3f2b1",
            "timestamp": "2025-12-13T14:30:22",
            "git": {
                "hash": "b18ea45",
                "branch": "master",
                "is_dirty": false,
                "uncommitted_files": []
            },
            "config": {...},
            "rect_check": {...},
            "additional": {...}
        }

    Args:
        run_dir: Path to run directory
        config: Configuration dictionary
        rect_check_result: Rectangle check validation result
        run_id: Run ID (extracted from run_dir if not provided)
        additional_metadata: Additional metadata to include

    Returns:
        str: Path to saved metadata.json file

    Example:
        >>> metadata_path = save_run_metadata(
        ...     'outputs/train/20251213_143022_a3f2b1',
        ...     config={'model': {'type': 'AquaSOT3DTracker'}},
        ...     rect_check_result={'valid': True, 'errors': []}
        ... )
    """
    # Extract run_id from directory name if not provided
    if run_id is None:
        run_id = Path(run_dir).name

    # Build metadata dictionary
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'git': get_git_status(),
    }

    # Add config if provided
    if config is not None:
        # Convert ConfigDict to plain dict if needed
        from .config import ConfigDict
        if isinstance(config, ConfigDict):
            from .config import _config_to_dict
            metadata['config'] = _config_to_dict(config)
        else:
            metadata['config'] = config

    # Add rect_check result if provided
    if rect_check_result is not None:
        metadata['rect_check'] = rect_check_result

    # Add additional metadata if provided
    if additional_metadata is not None:
        metadata['additional'] = additional_metadata

    # Save metadata to JSON file
    metadata_path = Path(run_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=False)

    logger.info(f"Saved run metadata to: {metadata_path}")

    return str(metadata_path)


def load_run_metadata(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata.json from run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        dict: Metadata dictionary, or None if metadata.json doesn't exist

    Example:
        >>> metadata = load_run_metadata('outputs/train/20251213_143022_a3f2b1')
        >>> print(metadata['run_id'])
        '20251213_143022_a3f2b1'
    """
    metadata_path = Path(run_dir) / 'metadata.json'

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata from: {metadata_path}")
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None


def check_output_directory(
    output_dir: str,
    force: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check if output directory has existing results and warn/require --force.

    Checks for:
    - Versioned outputs: subdirectories matching run_id pattern
    - Legacy outputs: checkpoint files, log files, result files at top level

    Args:
        output_dir: Output directory path to check
        force: Whether to allow overwriting existing results

    Returns:
        tuple: (has_existing_results, existing_items)
            - has_existing_results: Whether directory has existing outputs
            - existing_items: List of existing item paths found

    Raises:
        FileExistsError: If existing results found and force=False

    Example:
        >>> has_existing, items = check_output_directory('outputs/train', force=False)
        >>> if has_existing:
        ...     print(f"Found {len(items)} existing results")
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.debug(f"Output directory does not exist: {output_dir}")
        return False, []

    existing_items = []

    # Check for versioned outputs (run_id subdirectories)
    # Pattern: YYYYMMDD_HHMMSS_xxxxxx
    for item in output_path.iterdir():
        if item.is_dir():
            # Check if directory name matches run_id pattern
            name = item.name
            if len(name) >= 22 and name[8] == '_' and name[15] == '_':
                # Looks like a run_id directory
                existing_items.append(str(item))

    # Check for legacy outputs (files at top level)
    legacy_patterns = [
        '*.pth',           # Checkpoint files
        '*.ckpt',          # Checkpoint files
        '*.pt',            # PyTorch files
        '*.log',           # Log files
        '*.json',          # Result files
        'checkpoints',     # Legacy checkpoint directory
        'logs',            # Legacy log directory
        'results',         # Legacy results directory
    ]

    for pattern in legacy_patterns:
        if '*' in pattern:
            # Glob pattern
            for item in output_path.glob(pattern):
                if item.is_file():
                    existing_items.append(str(item))
        else:
            # Directory name
            legacy_dir = output_path / pattern
            if legacy_dir.exists() and legacy_dir.is_dir():
                existing_items.append(str(legacy_dir))

    has_existing = len(existing_items) > 0

    if has_existing:
        logger.warning(
            f"Found {len(existing_items)} existing output(s) in {output_dir}"
        )

        if not force:
            error_msg = (
                f"Output directory contains existing results: {output_dir}\n"
                f"Found {len(existing_items)} item(s):\n"
            )
            for item in existing_items[:5]:  # Show first 5 items
                error_msg += f"  - {item}\n"
            if len(existing_items) > 5:
                error_msg += f"  ... and {len(existing_items) - 5} more\n"
            error_msg += "\nUse --force to overwrite existing results"

            logger.error(error_msg)
            raise FileExistsError(error_msg)
        else:
            logger.warning("--force enabled, will overwrite existing results")

    return has_existing, existing_items


def detect_legacy_outputs(
    output_dir: str
) -> Tuple[bool, List[str]]:
    """
    Check for old outputs without run_id and warn about incompatibility.

    Legacy outputs are files/directories at the top level of output_dir,
    which do not follow the new run_id versioning scheme.

    Args:
        output_dir: Output directory path to check

    Returns:
        tuple: (has_legacy_outputs, legacy_items)
            - has_legacy_outputs: Whether legacy outputs were found
            - legacy_items: List of legacy item paths found

    Example:
        >>> has_legacy, items = detect_legacy_outputs('outputs/train')
        >>> if has_legacy:
        ...     print("WARNING: Found legacy outputs without run_id")
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.debug(f"Output directory does not exist: {output_dir}")
        return False, []

    legacy_items = []

    # Check for files at top level
    legacy_file_patterns = [
        '*.pth',           # Checkpoint files
        '*.ckpt',          # Checkpoint files
        '*.pt',            # PyTorch files
        '*.log',           # Log files
        '*.json',          # Result files (except metadata.json in versioned runs)
    ]

    for pattern in legacy_file_patterns:
        for item in output_path.glob(pattern):
            if item.is_file():
                legacy_items.append(str(item))

    # Check for legacy directories (not matching run_id pattern)
    legacy_dir_names = ['checkpoints', 'logs', 'results', 'visualizations']

    for dir_name in legacy_dir_names:
        legacy_dir = output_path / dir_name
        if legacy_dir.exists() and legacy_dir.is_dir():
            # Check if it's actually a legacy directory (has contents)
            if any(legacy_dir.iterdir()):
                legacy_items.append(str(legacy_dir))

    has_legacy = len(legacy_items) > 0

    if has_legacy:
        logger.warning(
            f"INCOMPATIBILITY WARNING: Found {len(legacy_items)} legacy output(s) "
            f"without run_id in {output_dir}"
        )
        logger.warning(
            "Legacy outputs are not compatible with the new run_id versioning scheme."
        )
        logger.warning(
            "Consider migrating legacy outputs to versioned format or archiving them."
        )

        # Log first few legacy items
        for item in legacy_items[:5]:
            logger.warning(f"  - Legacy: {item}")
        if len(legacy_items) > 5:
            logger.warning(f"  ... and {len(legacy_items) - 5} more")

    return has_legacy, legacy_items


def migrate_legacy_outputs(
    output_dir: str,
    archive_dir: Optional[str] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Migrate legacy outputs to run_id versioned format.

    Creates a new run_id directory and moves legacy outputs into it,
    or archives them to a separate location.

    Args:
        output_dir: Output directory containing legacy outputs
        archive_dir: Optional archive directory (default: {output_dir}/legacy_archive)
        dry_run: If True, only report what would be done without actually moving files

    Returns:
        dict: Migration report with:
            - success: Whether migration completed successfully
            - migrated_items: List of migrated item paths
            - errors: List of error messages

    Example:
        >>> report = migrate_legacy_outputs('outputs/train', dry_run=True)
        >>> print(f"Would migrate {len(report['migrated_items'])} items")
    """
    import shutil

    output_path = Path(output_dir)
    has_legacy, legacy_items = detect_legacy_outputs(output_dir)

    if not has_legacy:
        logger.info("No legacy outputs found to migrate")
        return {'success': True, 'migrated_items': [], 'errors': []}

    # Determine archive directory
    if archive_dir is None:
        archive_path = output_path / 'legacy_archive'
    else:
        archive_path = Path(archive_dir)

    migrated_items = []
    errors = []

    if dry_run:
        logger.info(f"DRY RUN: Would migrate {len(legacy_items)} legacy items to {archive_path}")
        for item in legacy_items:
            logger.info(f"  Would move: {item}")
        return {
            'success': True,
            'migrated_items': legacy_items,
            'errors': [],
            'dry_run': True
        }

    # Create archive directory
    try:
        archive_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created archive directory: {archive_path}")
    except OSError as e:
        error_msg = f"Failed to create archive directory: {e}"
        logger.error(error_msg)
        return {'success': False, 'migrated_items': [], 'errors': [error_msg]}

    # Move legacy items to archive
    for item in legacy_items:
        item_path = Path(item)
        dest_path = archive_path / item_path.name

        try:
            if item_path.is_file():
                shutil.move(str(item_path), str(dest_path))
            else:
                shutil.move(str(item_path), str(dest_path))

            migrated_items.append(str(item_path))
            logger.info(f"Migrated: {item_path} -> {dest_path}")
        except (OSError, shutil.Error) as e:
            error_msg = f"Failed to migrate {item_path}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    success = len(errors) == 0

    if success:
        logger.info(f"Successfully migrated {len(migrated_items)} legacy items")
    else:
        logger.warning(
            f"Migration completed with {len(errors)} error(s), "
            f"{len(migrated_items)} items migrated"
        )

    return {
        'success': success,
        'migrated_items': migrated_items,
        'errors': errors
    }


def find_latest_run(
    base_output_dir: str,
    metric: Optional[str] = None
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Find the latest run directory in base output directory.

    Args:
        base_output_dir: Base output directory containing run subdirectories
        metric: Optional metric to use for ranking (from metadata.json)
                If None, uses most recent timestamp

    Returns:
        tuple: (run_directory_path, metadata) or None if no runs found

    Example:
        >>> run_dir, metadata = find_latest_run('outputs/train')
        >>> print(f"Latest run: {metadata['run_id']}")
    """
    output_path = Path(base_output_dir)

    if not output_path.exists():
        logger.debug(f"Output directory does not exist: {base_output_dir}")
        return None

    # Find all run directories
    run_dirs = []
    for item in output_path.iterdir():
        if item.is_dir():
            # Check if directory name matches run_id pattern
            name = item.name
            if len(name) >= 22 and name[8] == '_' and name[15] == '_':
                metadata = load_run_metadata(str(item))
                if metadata:
                    run_dirs.append((str(item), metadata))

    if not run_dirs:
        logger.debug(f"No run directories found in {base_output_dir}")
        return None

    # Sort by timestamp (most recent first)
    run_dirs.sort(
        key=lambda x: x[1].get('timestamp', ''),
        reverse=True
    )

    # If metric specified, rank by metric value
    if metric is not None:
        # Filter runs that have the metric
        runs_with_metric = [
            (run_dir, metadata)
            for run_dir, metadata in run_dirs
            if metric in metadata.get('additional', {})
        ]

        if runs_with_metric:
            # Sort by metric value (assumes lower is better)
            runs_with_metric.sort(
                key=lambda x: x[1]['additional'][metric]
            )
            latest_run = runs_with_metric[0]
            logger.info(
                f"Found best run by metric '{metric}': {latest_run[1]['run_id']}"
            )
            return latest_run

    # Return most recent run
    latest_run = run_dirs[0]
    logger.info(f"Found latest run: {latest_run[1]['run_id']}")
    return latest_run


# Convenience functions for train/eval workflows

def setup_train_run(
    config: Dict[str, Any],
    force: bool = False,
    run_id: Optional[str] = None,
    rect_check_result: Optional[Dict[str, Any]] = None,
    data_paths: Optional[Dict[str, str]] = None
) -> Tuple[str, str, str]:
    """
    Setup training run with versioned output directory.

    Creates run directory structure:
        {work_dir}/{run_id}/
            checkpoints/
            logs/
            metadata.json

    Args:
        config: Configuration dictionary (must have paths.work_dir)
        force: Whether to allow overwriting existing results
        run_id: Optional run ID (generated if not provided)
        rect_check_result: Optional rect validation result dict (Audit 2025121202 F21/24)
        data_paths: Optional dict of data paths {data_root, train_root, val_root}

    Returns:
        tuple: (run_dir, checkpoint_dir, log_dir)

    Example:
        >>> run_dir, ckpt_dir, log_dir = setup_train_run(config, rect_check_result={'passed': True})
        >>> print(f"Training outputs: {run_dir}")
    """
    # Get base output directory from config
    work_dir = config.get('paths', {}).get('work_dir', 'outputs/train')

    # Check for existing outputs
    check_output_directory(work_dir, force=force)

    # Detect legacy outputs
    detect_legacy_outputs(work_dir)

    # Create run directory with subdirectories
    run_dir, run_id = create_run_directory(
        work_dir,
        run_id=run_id,
        create_subdirs=['checkpoints', 'logs']
    )

    # CRITICAL FIX (Audit 2025121202 Finding 21/24): Include rect_check and data_paths in metadata
    additional_metadata = {}
    if data_paths:
        additional_metadata['data_paths'] = data_paths
    if rect_check_result:
        additional_metadata['rect_check'] = rect_check_result

    # Save metadata with rect_check and data_paths
    save_run_metadata(
        run_dir,
        config=config,
        rect_check_result=rect_check_result,
        run_id=run_id,
        additional_metadata=additional_metadata if additional_metadata else None
    )

    # Return paths
    checkpoint_dir = str(Path(run_dir) / 'checkpoints')
    log_dir = str(Path(run_dir) / 'logs')

    logger.info(f"Training run setup complete: {run_dir}")

    return run_dir, checkpoint_dir, log_dir


def setup_eval_run(
    config: Dict[str, Any],
    checkpoint_path: str,
    force: bool = False,
    run_id: Optional[str] = None,
    rect_check_result: Optional[Dict[str, Any]] = None,
    data_paths: Optional[Dict[str, str]] = None
) -> Tuple[str, str]:
    """
    Setup evaluation run with versioned output directory.

    Creates run directory structure:
        {output_dir}/{run_id}/
            visualization/
            metadata.json

    Args:
        config: Configuration dictionary (must have evaluation.output_dir)
        checkpoint_path: Path to checkpoint being evaluated
        force: Whether to allow overwriting existing results
        run_id: Optional run ID (generated if not provided)
        rect_check_result: Optional rect validation result dict (Audit 2025121202 F21/24)
        data_paths: Optional dict of data paths {data_root, test_root}

    Returns:
        tuple: (run_dir, visualization_dir)

    Example:
        >>> run_dir, vis_dir = setup_eval_run(config, 'model.pth', rect_check_result={'passed': True})
        >>> print(f"Evaluation outputs: {run_dir}")
    """
    # Get base output directory from config
    output_dir = config.get('evaluation', {}).get('output_dir', 'outputs/eval')

    # Check for existing outputs
    check_output_directory(output_dir, force=force)

    # Detect legacy outputs
    detect_legacy_outputs(output_dir)

    # Create run directory with subdirectories
    run_dir, run_id = create_run_directory(
        output_dir,
        run_id=run_id,
        create_subdirs=['visualization']
    )

    # CRITICAL FIX (Audit 2025121202 Finding 21/24): Include rect_check and data_paths in metadata
    additional_metadata = {'checkpoint_path': checkpoint_path}
    if data_paths:
        additional_metadata['data_paths'] = data_paths
    if rect_check_result:
        additional_metadata['rect_check'] = rect_check_result
    save_run_metadata(
        run_dir,
        config=config,
        rect_check_result=rect_check_result,
        run_id=run_id,
        additional_metadata=additional_metadata
    )

    # Return paths
    visualization_dir = str(Path(run_dir) / 'visualization')

    logger.info(f"Evaluation run setup complete: {run_dir}")

    return run_dir, visualization_dir
