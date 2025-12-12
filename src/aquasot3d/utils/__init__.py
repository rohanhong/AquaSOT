"""
Utility Modules

Bounding box utilities and configuration management.

Author: AquaSOT-3D Team
Date: 2025-11-20
"""

from .config import (
    ConfigDict,
    load_config,
    save_config,
    validate_config,
    print_config,
    setup_config_logging,
)

from .bbox_utils import (
    convert_bbox_format,
    normalize_bbox,
    denormalize_bbox,
    scale_bbox,
    convert_boxes,
)

from .bbox import (
    BBoxXYWH,
    BBoxCXCYWH,
    BBoxXYXY,
    bbox_from_xywh,
    bbox_from_cxcywh,
    bbox_from_xyxy,
    convert_bbox_format as convert_bbox_format_typed,
)

from .preprocessing import (
    letterbox_with_metadata,
    preprocess_for_inference,
    crop_metadata_to_letterbox,
)

from .run_manager import (
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

from .data_validation import (
    rect_data_check,
    validate_data_root,
    check_rect_readiness,
    block_polar_data,
    generate_metadata_dict,
    validate_sequences_or_raise,
    get_validation_summary,
    EXPECTED_SONAR_RECT_RESOLUTION,
)

__all__ = [
    # Bbox Format Conversion (Audit 2025111405 contract helpers)
    'convert_bbox_format',
    'normalize_bbox',
    'denormalize_bbox',
    'scale_bbox',
    'convert_boxes',

    # Type-safe Bbox Wrappers (Audit 2025111411 Finding 3)
    'BBoxXYWH',
    'BBoxCXCYWH',
    'BBoxXYXY',
    'bbox_from_xywh',
    'bbox_from_cxcywh',
    'bbox_from_xyxy',
    'convert_bbox_format_typed',

    # BEST PRACTICE: Use typed wrappers at module boundaries to prevent format confusion
    # Example:
    #   # In model forward (outputs normalized cxcywh):
    #   pred_bbox_cxcywh = BBoxCXCYWH(model_output['boxes'])
    #
    #   # In evaluation (expects xywh pixels):
    #   pred_bbox_xywh = pred_bbox_cxcywh.to_xywh()
    #   pred_pixels = denormalize_bbox(pred_bbox_xywh.tensor(), image_size)
    #   pred_bbox = BBoxXYWH(pred_pixels)  # Type-safe pixel coordinates

    # Configuration
    'ConfigDict',
    'load_config',
    'save_config',
    'validate_config',
    'print_config',
    'setup_config_logging',

    # Preprocessing (Audit 2025120401 unified metadata conversion)
    'letterbox_with_metadata',
    'preprocess_for_inference',
    'crop_metadata_to_letterbox',

    # Run Management (Audit 2025121301 output versioning and metadata tracking)
    'generate_run_id',
    'get_git_hash',
    'get_git_status',
    'create_run_directory',
    'save_run_metadata',
    'load_run_metadata',
    'check_output_directory',
    'detect_legacy_outputs',
    'migrate_legacy_outputs',
    'find_latest_run',
    'setup_train_run',
    'setup_eval_run',

    # Data Validation (Audit 2025121301 - Findings 4, 16, 24, 26, 27)
    'rect_data_check',
    'validate_data_root',
    'check_rect_readiness',
    'block_polar_data',
    'generate_metadata_dict',
    'validate_sequences_or_raise',
    'get_validation_summary',
    'EXPECTED_SONAR_RECT_RESOLUTION',
]
