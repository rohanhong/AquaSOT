"""
Experiment class for AquaSOT evaluation.

Based on got10k-toolkit ExperimentOTB.
"""

import os
import json
import numpy as np
from collections import OrderedDict

from .metrics import rect_iou, center_error, success_curve, precision_curve, calculate_auc


class ExperimentAquaSOT:
    """Experiment for evaluating trackers on AquaSOT dataset.

    Supports:
        - Running trackers on sequences
        - Computing Success/Precision metrics for RGB and Sonar modalities
        - Generating evaluation reports

    Args:
        root_dir: Root directory of AquaSOT dataset
        result_dir: Directory to save tracking results
        report_dir: Directory to save reports
        sequence_names: List of sequence names to evaluate (None = all)
    """

    def __init__(
        self,
        root_dir,
        result_dir='results',
        report_dir='reports',
        sequence_names=None,
    ):
        self.root_dir = root_dir
        self.result_dir = result_dir
        self.report_dir = report_dir

        # Discover sequences
        self.sequence_names = sequence_names
        if self.sequence_names is None:
            self.sequence_names = self._discover_sequences()

        # Create output directories
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

    def _discover_sequences(self):
        """Discover available sequences in dataset."""
        sequences = []
        if os.path.exists(self.root_dir):
            for name in sorted(os.listdir(self.root_dir)):
                seq_path = os.path.join(self.root_dir, name)
                if os.path.isdir(seq_path):
                    # Check for required files (actual structure)
                    rgb_gt = os.path.join(seq_path, 'light', 'groundtruth.txt')
                    sonar_gt = os.path.join(seq_path, 'sonar', 'groundtruth_rect.txt')
                    if os.path.exists(rgb_gt) or os.path.exists(sonar_gt):
                        sequences.append(name)
        return sequences

    def run(self, tracker, visualize=False, overwrite=False):
        """Run tracker on all sequences.

        Args:
            tracker: Tracker instance with init() and update() methods
            visualize: Whether to visualize results
            overwrite: Whether to overwrite existing results
        """
        print(f'[ExperimentAquaSOT] Running {tracker.name} on {len(self.sequence_names)} sequences')

        for seq_idx, seq_name in enumerate(self.sequence_names):
            print(f'  [{seq_idx + 1}/{len(self.sequence_names)}] {seq_name}')

            # Check if result exists
            result_file = os.path.join(
                self.result_dir, tracker.name, f'{seq_name}.txt'
            )
            if os.path.exists(result_file) and not overwrite:
                print(f'    Result exists, skipping')
                continue

            # Load sequence data
            seq_data = self._load_sequence(seq_name)
            if seq_data is None:
                print(f'    Failed to load sequence, skipping')
                continue

            # Get image files and ground truth
            rgb_files = seq_data['rgb_files']
            sonar_files = seq_data['sonar_files']
            rgb_gt = seq_data['rgb_gt']
            sonar_gt = seq_data['sonar_gt']

            # Create multi-modal image list
            img_files = []
            for i in range(len(rgb_files)):
                img_files.append({
                    'rgb': rgb_files[i] if i < len(rgb_files) else None,
                    'sonar': sonar_files[i] if i < len(sonar_files) else None,
                })

            # Initial box (use RGB ground truth if available, else sonar)
            if rgb_gt is not None and len(rgb_gt) > 0:
                init_box = rgb_gt[0]
            elif sonar_gt is not None and len(sonar_gt) > 0:
                init_box = sonar_gt[0]
            else:
                print(f'    No ground truth available, skipping')
                continue

            # Run tracker
            try:
                boxes, times = tracker.track(img_files, init_box, visualize=visualize)
            except Exception as e:
                print(f'    Tracker failed: {e}')
                continue

            # Save results
            self._save_results(tracker.name, seq_name, boxes, times)

    def _load_sequence(self, seq_name):
        """Load sequence data.

        Args:
            seq_name: Sequence name

        Returns:
            dict with rgb_files, sonar_files, rgb_gt, sonar_gt
        """
        seq_path = os.path.join(self.root_dir, seq_name)

        def extract_frame_number(fname):
            """Extract numeric frame number from filename for sorting."""
            try:
                return int(os.path.splitext(fname)[0])
            except ValueError:
                return float('inf')

        # Load RGB images (actual structure: light/img/)
        rgb_dir = os.path.join(seq_path, 'light', 'img')
        rgb_files = []
        if os.path.exists(rgb_dir):
            files = [f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            files.sort(key=extract_frame_number)  # Numerical sort
            rgb_files = [os.path.join(rgb_dir, f) for f in files]

        # Load Sonar images (actual structure: sonar/img_rect/)
        sonar_dir = os.path.join(seq_path, 'sonar', 'img_rect')
        sonar_files = []
        if os.path.exists(sonar_dir):
            files = [f for f in os.listdir(sonar_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            files.sort(key=extract_frame_number)  # Numerical sort
            sonar_files = [os.path.join(sonar_dir, f) for f in files]

        # Load RGB ground truth (actual structure: light/groundtruth.txt)
        rgb_gt_file = os.path.join(seq_path, 'light', 'groundtruth.txt')
        rgb_gt = self._load_groundtruth(rgb_gt_file)

        # Load Sonar ground truth (actual structure: sonar/groundtruth_rect.txt)
        sonar_gt_file = os.path.join(seq_path, 'sonar', 'groundtruth_rect.txt')
        sonar_gt = self._load_groundtruth(sonar_gt_file)

        if len(rgb_files) == 0 and len(sonar_files) == 0:
            return None

        return {
            'rgb_files': rgb_files,
            'sonar_files': sonar_files,
            'rgb_gt': rgb_gt,
            'sonar_gt': sonar_gt,
        }

    def _load_groundtruth(self, gt_file):
        """Load ground truth from file.

        Args:
            gt_file: Path to ground truth file

        Returns:
            Array of boxes (N, 4) in [x, y, w, h] format
        """
        if not os.path.exists(gt_file):
            return None

        gt = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support comma or space separated
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()
                if len(parts) >= 4:
                    box = [float(p) for p in parts[:4]]
                    gt.append(box)

        return np.array(gt) if gt else None

    def _save_results(self, tracker_name, seq_name, boxes, times):
        """Save tracking results.

        Args:
            tracker_name: Tracker name
            seq_name: Sequence name
            boxes: Predicted boxes (N, 4)
            times: Frame times (N,)
        """
        # Create tracker directory
        tracker_dir = os.path.join(self.result_dir, tracker_name)
        os.makedirs(tracker_dir, exist_ok=True)

        # Save boxes
        result_file = os.path.join(tracker_dir, f'{seq_name}.txt')
        np.savetxt(result_file, boxes, delimiter=',', fmt='%.4f')

        # Save times
        time_file = os.path.join(tracker_dir, f'{seq_name}_time.txt')
        np.savetxt(time_file, times, fmt='%.6f')

    def report(self, tracker_names, plot_curves=False):
        """Generate evaluation report.

        Args:
            tracker_names: List of tracker names to report
            plot_curves: Whether to plot curves

        Returns:
            dict with metrics for each tracker
        """
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]

        # Compute metrics for each tracker
        performance = OrderedDict()
        for name in tracker_names:
            performance[name] = self._compute_metrics(name)

        # Save report
        report_file = os.path.join(self.report_dir, 'performance.json')
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=2)

        # Print summary
        self._print_summary(performance)

        # Plot curves if requested
        if plot_curves:
            self._plot_curves(performance)

        return performance

    def _compute_metrics(self, tracker_name):
        """Compute metrics for a tracker.

        Args:
            tracker_name: Tracker name

        Returns:
            dict with metrics
        """
        tracker_dir = os.path.join(self.result_dir, tracker_name)
        if not os.path.exists(tracker_dir):
            print(f'[Warning] Results not found for {tracker_name}')
            return {}

        all_ious = []
        all_errors = []
        seq_metrics = {}

        for seq_name in self.sequence_names:
            # Load predicted boxes
            result_file = os.path.join(tracker_dir, f'{seq_name}.txt')
            if not os.path.exists(result_file):
                continue

            pred_boxes = np.loadtxt(result_file, delimiter=',')
            if pred_boxes.ndim == 1:
                pred_boxes = pred_boxes[np.newaxis, :]

            # Load ground truth
            seq_data = self._load_sequence(seq_name)
            if seq_data is None:
                continue

            gt_boxes = seq_data['rgb_gt']
            if gt_boxes is None:
                gt_boxes = seq_data['sonar_gt']
            if gt_boxes is None:
                continue

            # Ensure same length
            n_frames = min(len(pred_boxes), len(gt_boxes))
            pred_boxes = pred_boxes[:n_frames]
            gt_boxes = gt_boxes[:n_frames]

            # Compute IoU and center error
            ious = rect_iou(pred_boxes, gt_boxes)
            errors = center_error(pred_boxes, gt_boxes)

            all_ious.append(ious)
            all_errors.append(errors)

            # Per-sequence metrics
            seq_metrics[seq_name] = {
                'success_score': np.mean(ious),
                'precision_score': np.mean(errors <= 20),  # 20 pixel threshold
                'avg_iou': float(np.mean(ious)),
                'avg_center_error': float(np.mean(errors)),
            }

        if not all_ious:
            return {}

        # Compute overall curves
        all_ious = np.concatenate(all_ious)
        all_errors = np.concatenate(all_errors)

        success = success_curve(all_ious)
        precision = precision_curve(all_errors)

        # Compute AUC
        success_auc = calculate_auc(success)
        precision_auc = calculate_auc(precision, start=0.0, end=50.0)

        return {
            'overall': {
                'success_score': float(success_auc),
                'precision_score': float(precision[20]),  # 20 pixel threshold
                'success_auc': float(success_auc),
                'avg_iou': float(np.mean(all_ious)),
                'avg_center_error': float(np.mean(all_errors)),
            },
            'sequences': seq_metrics,
            'success_curve': success.tolist(),
            'precision_curve': precision.tolist(),
        }

    def _print_summary(self, performance):
        """Print evaluation summary.

        Args:
            performance: dict with metrics for each tracker
        """
        print('\n' + '=' * 70)
        print('Evaluation Summary')
        print('=' * 70)
        print(f'{"Tracker":<20} {"Success AUC":>15} {"Precision@20":>15} {"Avg IoU":>12}')
        print('-' * 70)

        for name, metrics in performance.items():
            if 'overall' not in metrics:
                print(f'{name:<20} {"N/A":>15} {"N/A":>15} {"N/A":>12}')
                continue

            overall = metrics['overall']
            print(f'{name:<20} {overall["success_auc"]:>15.3f} '
                  f'{overall["precision_score"]:>15.3f} {overall["avg_iou"]:>12.3f}')

        print('=' * 70)

    def _plot_curves(self, performance):
        """Plot success and precision curves.

        Args:
            performance: dict with metrics for each tracker
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('[Warning] matplotlib not available, skipping curve plots')
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(performance)))

        for (name, metrics), color in zip(performance.items(), colors):
            if 'success_curve' not in metrics:
                continue

            success = np.array(metrics['success_curve'])
            precision = np.array(metrics['precision_curve'])
            auc = metrics['overall']['success_auc']

            # Success plot
            ax1.plot(
                np.linspace(0, 1, len(success)),
                success,
                color=color,
                label=f'{name} [{auc:.3f}]'
            )

            # Precision plot
            ax2.plot(
                np.linspace(0, 50, len(precision)),
                precision,
                color=color,
                label=f'{name}'
            )

        ax1.set_xlabel('Overlap Threshold')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Plot')
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel('Center Error Threshold (pixels)')
        ax2.set_ylabel('Precision Rate')
        ax2.set_title('Precision Plot')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save figure
        fig_file = os.path.join(self.report_dir, 'curves.png')
        plt.savefig(fig_file, dpi=150)
        print(f'[ExperimentAquaSOT] Curves saved to {fig_file}')

        plt.close()
