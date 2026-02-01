"""
Evaluation CLI for AquaSOT.

Usage:
    python tracking/test.py --tracker aquasot --checkpoint path/to/checkpoint.pth
    python tracking/test.py --tracker aquasot --config experiments/aquasot/baseline.yaml
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AquaSOT Evaluation')

    # Tracker settings
    parser.add_argument('--tracker', type=str, default='aquasot',
                        help='Tracker name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='aquasot',
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to dataset')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequences to evaluate')

    # Output settings
    parser.add_argument('--result_dir', type=str, default='outputs/test/baseline/results',
                        help='Directory to save results')
    parser.add_argument('--report_dir', type=str, default='outputs/test/baseline/reports',
                        help='Directory to save reports')

    # Evaluation settings
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize tracking results (default: True)')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--report_only', action='store_true',
                        help='Only generate report from existing results')
    parser.add_argument('--plot_curves', action='store_true', default=True,
                        help='Plot success/precision curves (default: True)')
    parser.add_argument('--no_plot_curves', action='store_true',
                        help='Disable curve plotting')

    # Evaluation mode and visibility filtering
    parser.add_argument('--eval_mode', type=str, default='dual',
                        choices=['dual', 'rgb_only', 'sonar_only', 'no_fusion'],
                        help='Evaluation mode: dual (both), rgb_only, sonar_only, no_fusion')
    parser.add_argument('--no_filter_visibility', action='store_true',
                        help='Include all frames (legacy behavior, ignores GT absence)')
    parser.add_argument('--no_skip_init_frame', action='store_true',
                        help='Include frame 0 in evaluation (non-standard)')

    # Model settings
    parser.add_argument('--search_size', type=int, default=256,
                        help='Search region size')
    parser.add_argument('--template_size', type=int, default=128,
                        help='Template size')
    parser.add_argument('--search_factor', type=float, default=4.0,
                        help='Search area factor')
    parser.add_argument('--template_factor', type=float, default=2.0,
                        help='Template area factor')

    # BatchNorm adaptation for ablation experiments
    parser.add_argument('--bn_adapt', action='store_true',
                        help='Enable BatchNorm adaptation. Required for all ablation experiments '
                             '(--eval_mode rgb_only/sonar_only/no_fusion).')

    # Fusion weight override for ablation experiments
    parser.add_argument('--fusion_weight', type=float, default=None,
                        help='Override inference fusion weight for ablation experiments. '
                             'Default from config is 0.3. Use 0.0 for no fusion, 1.0 for full fusion.')

    return parser.parse_args()


def get_dataset_path(dataset_name, dataset_path=None):
    """Get dataset path from name or explicit path."""
    if dataset_path is not None:
        return dataset_path

    # Default paths
    default_paths = {
        'aquasot': os.path.join(project_root, 'data'),
        'uot100': os.path.join(project_root, 'data', 'UOT100'),
    }

    if dataset_name in default_paths:
        return default_paths[dataset_name]

    raise ValueError(f'Unknown dataset: {dataset_name}. '
                     f'Please provide --dataset_path')


def create_tracker(args):
    """Create tracker from arguments."""
    from lib.test.tracker import AquaSOTTracker

    params = {
        'checkpoint': args.checkpoint,
        'cfg_file': args.config,
        'search_size': args.search_size,
        'template_size': args.template_size,
        'search_area_factor': args.search_factor,
        'template_area_factor': args.template_factor,
    }

    return AquaSOTTracker(params, dataset_name=args.dataset)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Get dataset path
    dataset_path = get_dataset_path(args.dataset, args.dataset_path)
    print(f'[Evaluation] Dataset: {args.dataset}')
    print(f'[Evaluation] Dataset path: {dataset_path}')

    # Use default test sequences from config if not specified
    from lib.config.aquasot.config import cfg
    if args.sequences is None:
        args.sequences = cfg.DATA.TEST.SEQUENCES
        print(f'[Evaluation] Using default test sequences: {args.sequences}')

    # Configure modality dropout / fusion based on eval_mode
    if args.eval_mode == 'rgb_only':
        cfg.DATA.MODALITY_DROPOUT.ENABLED = True
        cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR = 1.0
        cfg.DATA.MODALITY_DROPOUT.P_DROP_RGB = 0.0
        print('[Evaluation] RGB-only mode: Sonar input will be zeroed')
    elif args.eval_mode == 'sonar_only':
        cfg.DATA.MODALITY_DROPOUT.ENABLED = True
        cfg.DATA.MODALITY_DROPOUT.P_DROP_RGB = 1.0
        cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR = 0.0
        print('[Evaluation] Sonar-only mode: RGB input will be zeroed')
    elif args.eval_mode == 'no_fusion':
        cfg.MODEL.FUSION.RGB_FUSION_WEIGHT = 0.0
        cfg.MODEL.FUSION.SONAR_FUSION_WEIGHT = 0.0
        print('[Evaluation] No-fusion mode: Fusion weights set to 0')

    # Apply fusion weight override if specified
    if args.fusion_weight is not None:
        cfg.MODEL.FUSION.RGB_FUSION_WEIGHT = args.fusion_weight
        cfg.MODEL.FUSION.SONAR_FUSION_WEIGHT = args.fusion_weight
        print(f'[Evaluation] Fusion weight override: {args.fusion_weight}')

    # Configure BN adaptation
    if args.bn_adapt:
        cfg.TEST.BN_ADAPTATION.ENABLED = True
        print('[Evaluation] BatchNorm adaptation enabled')

    # Handle --no_visualize and --no_plot_curves flags
    if args.no_visualize:
        args.visualize = False
    if args.no_plot_curves:
        args.plot_curves = False

    # Create experiment
    from lib.test.evaluation import ExperimentAquaSOT

    experiment = ExperimentAquaSOT(
        root_dir=dataset_path,
        result_dir=args.result_dir,
        report_dir=args.report_dir,
        sequence_names=args.sequences,
        filter_visibility=not args.no_filter_visibility,
        skip_init_frame=not args.no_skip_init_frame,
        eval_mode=args.eval_mode,
    )

    print(f'[Evaluation] Found {len(experiment.sequence_names)} sequences')

    if not args.report_only:
        # Create tracker
        tracker = create_tracker(args)
        print(f'[Evaluation] Tracker: {tracker.name}')

        # Run evaluation
        experiment.run(
            tracker,
            visualize=args.visualize,
            overwrite=args.overwrite,
        )

    # Generate report
    tracker_name = args.tracker if args.report_only else tracker.name
    performance = experiment.report(
        [tracker_name],
        plot_curves=args.plot_curves,
    )

    # Print summary
    if tracker_name in performance and 'overall' in performance[tracker_name]:
        overall = performance[tracker_name]['overall']
        print(f'\n[Results] {tracker_name}:')
        print(f'  Success AUC:  {overall["success_auc"]:.3f}')
        print(f'  Precision@20: {overall["precision_score"]:.3f}')
        print(f'  Average IoU:  {overall["avg_iou"]:.3f}')
        print(f'  Center Error: {overall["avg_center_error"]:.1f} px')


if __name__ == '__main__':
    main()
