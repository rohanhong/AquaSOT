#!/usr/bin/env python3
"""
Model Export Script for AquaSOT-3D

Export trained model to ONNX or TorchScript format for deployment.

Usage:
    python scripts/export.py --checkpoint model.pth --format onnx --output model.onnx
    python scripts/export.py --checkpoint model.pth --format torchscript --output model.pt

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import argparse
import torch
import logging

from aquasot3d.utils.config import setup_config_logging
from aquasot3d.models.tracker import AquaSOT3DTracker

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='AquaSOT-3D Model Export')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--format', type=str, required=True,
                        choices=['onnx', 'torchscript'],
                        help='Export format: onnx or torchscript')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model (requires onnx-simplifier)')
    
    return parser.parse_args()


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        model = AquaSOT3DTracker(
            rgb_backbone_type=cfg.model.backbone.rgb.type,
            sonar_encoder_type=cfg.model.backbone.sonar.type,
            fusion_config=cfg.model.fusion,
            temporal_config=cfg.model.temporal,
            head_configs=cfg.model.heads
        )
    else:
        logger.warning("Config not in checkpoint, using default")
        model = AquaSOT3DTracker()
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def export_onnx(model, output_path, opset_version=11, simplify=False):
    """Export model to ONNX format"""
    logger.info(f"Exporting to ONNX (opset {opset_version})")
    
    # Dummy inputs
    batch_size = 1
    rgb_template = torch.randn(batch_size, 3, 127, 127)
    rgb_search = torch.randn(batch_size, 3, 255, 255)
    sonar_template = torch.randn(batch_size, 1, 127, 127)
    sonar_search = torch.randn(batch_size, 1, 255, 255)
    
    # Export
    torch.onnx.export(
        model,
        (rgb_template, rgb_search, sonar_template, sonar_search),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['rgb_template', 'rgb_search', 'sonar_template', 'sonar_search'],
        output_names=['rgb_box', 'sonar_box', 'fusion_box', 'weak3d'],
        dynamic_axes={
            'rgb_template': {0: 'batch_size'},
            'rgb_search': {0: 'batch_size'},
            'sonar_template': {0: 'batch_size'},
            'sonar_search': {0: 'batch_size'}
        }
    )
    
    logger.info(f"ONNX model saved to: {output_path}")
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            logger.info("Simplifying ONNX model...")
            model_onnx = onnx.load(output_path)
            model_simplified, check = onnx_simplify(model_onnx)
            
            if check:
                onnx.save(model_simplified, output_path)
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification check failed")
        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping simplification")


def export_torchscript(model, output_path):
    """Export model to TorchScript format"""
    logger.info("Exporting to TorchScript")
    
    # Dummy inputs
    rgb_template = torch.randn(1, 3, 127, 127)
    rgb_search = torch.randn(1, 3, 255, 255)
    sonar_template = torch.randn(1, 1, 127, 127)
    sonar_search = torch.randn(1, 1, 255, 255)
    
    # Trace model
    traced_model = torch.jit.trace(
        model,
        (rgb_template, rgb_search, sonar_template, sonar_search)
    )
    
    # Save
    traced_model.save(output_path)
    logger.info(f"TorchScript model saved to: {output_path}")


def main():
    """Main export function"""
    args = parse_args()
    
    # Setup logging
    setup_config_logging('INFO')
    
    logger.info("="*70)
    logger.info("AquaSOT-3D Model Export")
    logger.info("="*70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Output: {args.output}")
    
    # Load model (use CPU for export)
    model = load_model(args.checkpoint, device='cpu')
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    if args.format == 'onnx':
        export_onnx(model, args.output, args.opset, args.simplify)
    elif args.format == 'torchscript':
        export_torchscript(model, args.output)
    
    logger.info("="*70)
    logger.info("Export completed successfully")
    logger.info("="*70)


if __name__ == '__main__':
    main()
