"""
AquaSOT Configuration Module

Usage:
    from lib.config.aquasot import cfg, update_config_from_file

    # Load config overrides from YAML
    update_config_from_file('experiments/aquasot/train.yaml')

    # Access config values
    print(cfg.TRAIN.LR)
    print(cfg.MODEL.BACKBONE.RGB.TYPE)
"""

from .config import cfg, update_config_from_file, get_config, print_config, gen_config

__all__ = ['cfg', 'update_config_from_file', 'get_config', 'print_config', 'gen_config']
