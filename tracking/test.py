"""
Evaluation CLI for AquaSOT-3D.

Usage:
    python tracking/test.py --tracker aquasot3d --checkpoint path/to/checkpoint.pth
    python tracking/test.py --tracker aquasot3d --config experiments/aquasot3d/baseline.yaml
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AquaSOT-3D Evaluation')

    # Tracker settings
    parser.add_argument('--tracker', type=str, default='aquasot3d',
                        help='Tracker name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='aquasot3d',
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to dataset')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequences to evaluate')

    # Output settings
    parser.add_argument('--result_dir', type=str, default='outputs/test/results',
                        help='Directory to save results')
    parser.add_argument('--report_dir', type=str, default='outputs/test/reports',
                        help='Directory to save reports')

    # Evaluation settings
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize tracking results')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--report_only', action='store_true',
                        help='Only generate report from existing results')
    parser.add_argument('--plot_curves', action='store_true',
                        help='Plot success/precision curves')

    # Model settings
    parser.add_argument('--search_size', type=int, default=256,
                        help='Search region size')
    parser.add_argument('--template_size', type=int, default=128,
                        help='Template size')
    parser.add_argument('--search_factor', type=float, default=4.0,
                        help='Search area factor')
    parser.add_argument('--template_factor', type=float, default=2.0,
                        help='Template area factor')

    return parser.parse_args()


def get_dataset_path(dataset_name, dataset_path=None):
    """Get dataset path from name or explicit path."""
    if dataset_path is not None:
        return dataset_path

    # Default paths
    default_paths = {
        'aquasot3d': os.path.join(project_root, 'data'),
        'uot100': os.path.join(project_root, 'data', 'UOT100'),
    }

    if dataset_name in default_paths:
        return default_paths[dataset_name]

    raise ValueError(f'Unknown dataset: {dataset_name}. '
                     f'Please provide --dataset_path')


def create_tracker(args):
    """Create tracker from arguments."""
    from lib.test.tracker import AquaSOT3DTracker

    params = {
        'checkpoint': args.checkpoint,
        'cfg_file': args.config,
        'search_size': args.search_size,
        'template_size': args.template_size,
        'search_area_factor': args.search_factor,
        'template_area_factor': args.template_factor,
    }

    return AquaSOT3DTracker(params, dataset_name=args.dataset)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Get dataset path
    dataset_path = get_dataset_path(args.dataset, args.dataset_path)
    print(f'[Evaluation] Dataset: {args.dataset}')
    print(f'[Evaluation] Dataset path: {dataset_path}')

    # Create experiment
    from lib.test.evaluation import ExperimentAquaSOT3D

    experiment = ExperimentAquaSOT3D(
        root_dir=dataset_path,
        result_dir=args.result_dir,
        report_dir=args.report_dir,
        sequence_names=args.sequences,
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
