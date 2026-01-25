"""
Default configuration for AquaSOT.

This module defines all default parameters following OSTrack pattern.
YAML config files only override these defaults - they cannot introduce new parameters.

Usage:
    from lib.config.aquasot.config import cfg, update_config_from_file
    update_config_from_file('experiments/aquasot/train.yaml')
"""

from easydict import EasyDict as edict
import yaml

cfg = edict()

# ==============================================================================
# MODEL
# ==============================================================================
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.PRETRAIN_FILE_SONAR = ""  # Separate pretrain for sonar (optional)
cfg.MODEL.RETURN_INTER = False  # Return intermediate features
cfg.MODEL.RETURN_STAGES = [2, 5, 8, 11]  # Stages to return features from

# MODEL.BACKBONE.RGB
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.RGB = edict()
cfg.MODEL.BACKBONE.RGB.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.RGB.STRIDE = 16
cfg.MODEL.BACKBONE.RGB.IN_CHANNELS = 3
cfg.MODEL.BACKBONE.RGB.EMBED_DIM = 768
cfg.MODEL.BACKBONE.RGB.DEPTH = 12
cfg.MODEL.BACKBONE.RGB.NUM_HEADS = 12
cfg.MODEL.BACKBONE.RGB.MLP_RATIO = 4.0
cfg.MODEL.BACKBONE.RGB.DROP_PATH_RATE = 0.1
cfg.MODEL.BACKBONE.RGB.FREEZE = False
cfg.MODEL.BACKBONE.RGB.CAT_MODE = 'direct'  # template-search concatenation mode
cfg.MODEL.BACKBONE.RGB.CE_LOC = []  # Candidate Elimination layer locations
cfg.MODEL.BACKBONE.RGB.CE_KEEP_RATIO = []  # CE keep ratios
cfg.MODEL.BACKBONE.RGB.CE_TEMPLATE_RANGE = 'ALL'
cfg.MODEL.BACKBONE.RGB.SEP_SEG = False  # Separate segment embedding

# MODEL.BACKBONE.SONAR
cfg.MODEL.BACKBONE.SONAR = edict()
cfg.MODEL.BACKBONE.SONAR.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.SONAR.STRIDE = 16
cfg.MODEL.BACKBONE.SONAR.IN_CHANNELS = 1  # Grayscale sonar
cfg.MODEL.BACKBONE.SONAR.EMBED_DIM = 768
cfg.MODEL.BACKBONE.SONAR.DEPTH = 12
cfg.MODEL.BACKBONE.SONAR.NUM_HEADS = 12
cfg.MODEL.BACKBONE.SONAR.MLP_RATIO = 4.0
cfg.MODEL.BACKBONE.SONAR.DROP_PATH_RATE = 0.1
cfg.MODEL.BACKBONE.SONAR.FREEZE = False
cfg.MODEL.BACKBONE.SONAR.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.SONAR.CE_LOC = []
cfg.MODEL.BACKBONE.SONAR.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.SONAR.CE_TEMPLATE_RANGE = 'ALL'
cfg.MODEL.BACKBONE.SONAR.SEP_SEG = False  # Separate segment embedding

# MODEL.FUSION
cfg.MODEL.FUSION = edict()
cfg.MODEL.FUSION.TYPE = "dcma"  # Cross-Modal Attention (standard attention, not deformable)
cfg.MODEL.FUSION.EMBED_DIM = 768
cfg.MODEL.FUSION.NUM_HEADS = 8
cfg.MODEL.FUSION.NUM_LAYERS = 2
cfg.MODEL.FUSION.DROPOUT = 0.1

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
cfg.MODEL.HEAD.FREEZE_BN = False

# ==============================================================================
# TRAIN
# ==============================================================================
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 100  # Reduced for small dataset (9,588 training frames)
cfg.TRAIN.LR_DROP_EPOCH = 80
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_WORKER = 12
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GRAD_CLIP_NORM = 0.1  # OSTrack standard, DO NOT change
cfg.TRAIN.AMP = False
cfg.TRAIN.GRADIENT_ACCUMULATION = 4
cfg.TRAIN.PRINT_INTERVAL = 10
cfg.TRAIN.VAL_EPOCH_INTERVAL = 1

# Loss weights (OSTrack standard)
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FOCAL_WEIGHT = 1.0

# Multi-modal specific loss weights
cfg.TRAIN.RGB_BOX_WEIGHT = 1.0
cfg.TRAIN.SONAR_BOX_WEIGHT = 1.0
cfg.TRAIN.CONSISTENCY_WEIGHT = 0.1

# Candidate Elimination schedule
cfg.TRAIN.CE_START_EPOCH = 20
cfg.TRAIN.CE_WARM_EPOCH = 80

# Drop path
cfg.TRAIN.DROP_PATH_RATE = 0.1

# Layer-wise LR decay
cfg.TRAIN.USE_LAYER_DECAY = True
cfg.TRAIN.LAYER_DECAY_RATE = 0.75

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"  # step/multistep/cosine
cfg.TRAIN.SCHEDULER.GAMMA = 0.1
cfg.TRAIN.SCHEDULER.MILESTONES = [80]  # For multistep
cfg.TRAIN.SCHEDULER.WARMUP_EPOCHS = 5
cfg.TRAIN.SCHEDULER.WARMUP_LR = 0.00001
cfg.TRAIN.SCHEDULER.MIN_LR = 0.000001
cfg.TRAIN.CHECKPOINT_INTERVAL = 5

# TRAIN.CHECKPOINT
cfg.TRAIN.CHECKPOINT = edict()
cfg.TRAIN.CHECKPOINT.SAVE_INTERVAL = 5
cfg.TRAIN.CHECKPOINT.SAVE_BEST = True
cfg.TRAIN.CHECKPOINT.SAVE_LAST = True
cfg.TRAIN.CHECKPOINT.MAX_KEEP = 5
cfg.TRAIN.CHECKPOINT.METRIC = "val_total"
cfg.TRAIN.CHECKPOINT.MODE = "min"

# ==============================================================================
# DATA
# ==============================================================================
cfg.DATA = edict()
cfg.DATA.ROOT = "data"
cfg.DATA.SAMPLER_MODE = "causal"
cfg.DATA.MAX_SAMPLE_INTERVAL = 20  # Max gap between template and search

# Normalization (ImageNet for RGB)
cfg.DATA.RGB_MEAN = [0.485, 0.456, 0.406]
cfg.DATA.RGB_STD = [0.229, 0.224, 0.225]
cfg.DATA.SONAR_MEAN = [0.5]
cfg.DATA.SONAR_STD = [0.5]

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.SEQUENCES = ["uuv1", "uuv2", "uuv4", "uuv5", "uuv7", "uuv8", "uuv9", "uuv11", "uuv12"]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 8000  # ~1x training frames to avoid over-sampling
cfg.DATA.TRAIN.VISIBILITY_POLICY = "all"  # all/any/both

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.SEQUENCES = ["uuv3", "uuv6", "uuv10"]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 2000  # ~1x validation frames

# DATA.SEARCH (shared defaults)
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 1.0  # Moderate jitter for training
cfg.DATA.SEARCH.SCALE_JITTER = 0.25
cfg.DATA.SEARCH.NUMBER = 1

# DATA.TEMPLATE (shared defaults)
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0.0  # No jitter for template
cfg.DATA.TEMPLATE.SCALE_JITTER = 0.0
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.UPDATE_POLICY = "mixformer"  # first_frame/recent/mixformer
cfg.DATA.TEMPLATE.ONLINE_NUMBER = 1  # Number of online templates during training

# DATA.RGB (RGB-specific settings)
cfg.DATA.RGB = edict()
cfg.DATA.RGB.TEMPLATE = edict()
cfg.DATA.RGB.TEMPLATE.SIZE = 128
cfg.DATA.RGB.TEMPLATE.FACTOR = 2.0
cfg.DATA.RGB.TEMPLATE.CENTER_JITTER = 0.0
cfg.DATA.RGB.TEMPLATE.SCALE_JITTER = 0.0
cfg.DATA.RGB.SEARCH = edict()
cfg.DATA.RGB.SEARCH.SIZE = 256
cfg.DATA.RGB.SEARCH.FACTOR = 4.0
cfg.DATA.RGB.SEARCH.CENTER_JITTER = 1.0
cfg.DATA.RGB.SEARCH.SCALE_JITTER = 0.25

# DATA.SONAR (Sonar-specific settings)
cfg.DATA.SONAR = edict()
cfg.DATA.SONAR.TEMPLATE = edict()
cfg.DATA.SONAR.TEMPLATE.SIZE = 128
cfg.DATA.SONAR.TEMPLATE.FACTOR = 2.0
cfg.DATA.SONAR.TEMPLATE.CENTER_JITTER = 0.0
cfg.DATA.SONAR.TEMPLATE.SCALE_JITTER = 0.0
cfg.DATA.SONAR.SEARCH = edict()
cfg.DATA.SONAR.SEARCH.SIZE = 256
cfg.DATA.SONAR.SEARCH.FACTOR = 4.0
cfg.DATA.SONAR.SEARCH.CENTER_JITTER = 1.0
cfg.DATA.SONAR.SEARCH.SCALE_JITTER = 0.25

# DATA.AUGMENTATION
cfg.DATA.AUGMENTATION = edict()
cfg.DATA.AUGMENTATION.ENABLED = True

# Joint augmentation (synchronized across modalities)
cfg.DATA.AUGMENTATION.HORIZONTAL_FLIP = True
cfg.DATA.AUGMENTATION.HORIZONTAL_FLIP_PROB = 0.5

# RGB augmentation
cfg.DATA.AUGMENTATION.RGB = edict()
cfg.DATA.AUGMENTATION.RGB.BRIGHTNESS = 0.2
cfg.DATA.AUGMENTATION.RGB.COLOR_JITTER = True
cfg.DATA.AUGMENTATION.RGB.CONTRAST = 0.3
cfg.DATA.AUGMENTATION.RGB.SATURATION = 0.3
cfg.DATA.AUGMENTATION.RGB.HUE = 0.2
cfg.DATA.AUGMENTATION.RGB.GRAYSCALE = True
cfg.DATA.AUGMENTATION.RGB.GRAYSCALE_PROB = 0.05
cfg.DATA.AUGMENTATION.RGB.BLUR = True
cfg.DATA.AUGMENTATION.RGB.BLUR_PROB = 0.3
cfg.DATA.AUGMENTATION.RGB.OCCLUSION = True
cfg.DATA.AUGMENTATION.RGB.OCCLUSION_PROB = 0.2

# Sonar augmentation
cfg.DATA.AUGMENTATION.SONAR = edict()
cfg.DATA.AUGMENTATION.SONAR.SPECKLE = True
cfg.DATA.AUGMENTATION.SONAR.SPECKLE_LEVEL = 0.1
cfg.DATA.AUGMENTATION.SONAR.SPECKLE_PROB = 0.5
cfg.DATA.AUGMENTATION.SONAR.OCCLUSION = True
cfg.DATA.AUGMENTATION.SONAR.OCCLUSION_PROB = 0.2

# Modality dropout (for robustness training)
cfg.DATA.MODALITY_DROPOUT = edict()
cfg.DATA.MODALITY_DROPOUT.ENABLED = False
cfg.DATA.MODALITY_DROPOUT.P_DROP_RGB = 0.1
cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR = 0.1
cfg.DATA.MODALITY_DROPOUT.P_DROP_BOTH = 0.02

# ==============================================================================
# TEST
# ==============================================================================
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 100

# Template update during tracking (MixFormer-style three-tier strategy)
cfg.TEST.TEMPLATE_UPDATE = edict()
cfg.TEST.TEMPLATE_UPDATE.ENABLED = True  # Enable by default
cfg.TEST.TEMPLATE_UPDATE.STRATEGY = "mixformer"  # mixformer/simple
cfg.TEST.TEMPLATE_UPDATE.ONLINE_SIZE = 2  # Number of online templates
cfg.TEST.TEMPLATE_UPDATE.UPDATE_INTERVAL = 200  # Frames between updates
cfg.TEST.TEMPLATE_UPDATE.SCORE_THRESHOLD = 0.5  # Min confidence to update
cfg.TEST.TEMPLATE_UPDATE.MAX_SCORE_DECAY = 1.0  # Decay rate per frame (1.0=no decay)

# Validity gating
cfg.TEST.VALIDITY_GATING = edict()
cfg.TEST.VALIDITY_GATING.ENABLED = False
cfg.TEST.VALIDITY_GATING.RGB_THRESHOLD = 0.35
cfg.TEST.VALIDITY_GATING.SONAR_THRESHOLD = 0.40

# Metrics
cfg.TEST.METRICS = ["success_auc", "precision_20", "avg_center_error"]

# ==============================================================================
# SYSTEM
# ==============================================================================
cfg.SYSTEM = edict()
cfg.SYSTEM.SEED = 42
cfg.SYSTEM.DEVICE = "cuda"
cfg.SYSTEM.NUM_GPUS = 1
cfg.SYSTEM.DISTRIBUTED = False
cfg.SYSTEM.DIST_BACKEND = "nccl"
cfg.SYSTEM.DETERMINISTIC = False
cfg.SYSTEM.BENCHMARK = True

# ==============================================================================
# PATHS
# ==============================================================================
cfg.PATHS = edict()
cfg.PATHS.WORK_DIR = "outputs"
cfg.PATHS.CHECKPOINT_DIR = "outputs/checkpoints"
cfg.PATHS.LOG_DIR = "outputs/logs"
cfg.PATHS.PRETRAINED_DIR = "pretrained"

# ==============================================================================
# EXPERIMENT
# ==============================================================================
cfg.EXPERIMENT = edict()
cfg.EXPERIMENT.NAME = "aquasot_default"
cfg.EXPERIMENT.GROUP = "default"
cfg.EXPERIMENT.TAGS = []

# ==============================================================================
# WANDB
# ==============================================================================
cfg.WANDB = edict()
cfg.WANDB.ENABLED = False
cfg.WANDB.PROJECT = "aquasot"
cfg.WANDB.ENTITY = ""  # Your wandb username or team name
cfg.WANDB.NAME = ""  # Run name (defaults to EXPERIMENT.NAME if empty)
cfg.WANDB.TAGS = []  # Additional tags
cfg.WANDB.NOTES = ""  # Run notes
cfg.WANDB.MODE = "online"  # online/offline/disabled


# ==============================================================================
# Helper Functions
# ==============================================================================

def _edict2dict(dest_dict, src_edict):
    """Convert easydict to standard dict recursively."""
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    """Generate YAML config file from current cfg."""
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    """Update base config with experiment config (only existing keys)."""
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, (dict, edict)):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError(
                    f"Key '{k}' does not exist in config.py. "
                    f"All parameters must be defined in config.py first."
                )
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    """
    Update config from YAML file.

    Args:
        filename: Path to YAML config file
        base_cfg: Base config to update (defaults to global cfg)
    """
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)


def get_config():
    """Get a copy of the current config."""
    return cfg.copy()


def print_config():
    """Print current configuration."""
    def _print_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, (dict, edict)):
                print(f"{prefix}{k}:")
                _print_dict(v, prefix + '  ')
            else:
                print(f"{prefix}{k}: {v}")

    print("=" * 60)
    print("AquaSOT Configuration")
    print("=" * 60)
    _print_dict(cfg)
    print("=" * 60)
