"""
Experiment class for AquaSOT evaluation.

Based on got10k-toolkit ExperimentOTB with visibility filtering support.

Key features:
- Visibility filtering: excludes frames where GT is (0,0,0,0) or invalid
- Multi-modal evaluation: separate RGB and Sonar metrics
- Skip frame 0: initialization frame not evaluated (GOT-10k standard)

Reference: GOT-10k (got10k/experiments/got10k.py:181)
    seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
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
        - Visibility filtering (excludes absent frames)
        - Generating evaluation reports

    Args:
        root_dir: Root directory of AquaSOT dataset
        result_dir: Directory to save tracking results
        report_dir: Directory to save reports
        sequence_names: List of sequence names to evaluate (None = all)
        filter_visibility: Whether to filter out absent frames (default: True)
        skip_init_frame: Whether to skip frame 0 in evaluation (default: True)
        eval_mode: Evaluation mode - 'dual', 'rgb_only', or 'sonar_only' (default: 'dual')
    """

    def __init__(
        self,
        root_dir,
        result_dir='results',
        report_dir='reports',
        sequence_names=None,
        filter_visibility=True,
        skip_init_frame=True,
        eval_mode='dual',
    ):
        self.root_dir = root_dir
        self.result_dir = result_dir
        self.report_dir = report_dir
        self.filter_visibility = filter_visibility
        self.skip_init_frame = skip_init_frame
        self.eval_mode = eval_mode  # 'dual', 'rgb_only', 'sonar_only'

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
                self.result_dir, tracker.name, f'{seq_name}_rgb.txt'
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

            # Initial box (use dict format for dual-modal tracking)
            # Each modality uses its own ground truth for initialization
            if rgb_gt is not None and len(rgb_gt) > 0:
                rgb_init = rgb_gt[0]
            else:
                rgb_init = None

            if sonar_gt is not None and len(sonar_gt) > 0:
                sonar_init = sonar_gt[0]
            else:
                sonar_init = None

            if rgb_init is None and sonar_init is None:
                print(f'    No ground truth available, skipping')
                continue

            # Pass as dict for dual-modal initialization (always dict for consistency)
            # Tracker will use each modality's GT to initialize its template
            if rgb_init is not None and sonar_init is not None:
                init_box = {'rgb': rgb_init, 'sonar': sonar_init}
            elif rgb_init is not None:
                init_box = {'rgb': rgb_init}
            else:
                init_box = {'sonar': sonar_init}

            # Run tracker
            try:
                results = tracker.track(img_files, init_box, visualize=visualize)
            except Exception as e:
                print(f'    Tracker failed: {e}')
                continue

            # Save results (handle both new dict format and legacy tuple format)
            if isinstance(results, dict):
                self._save_results(tracker.name, seq_name, results)
            else:
                # Legacy format: (boxes, times)
                boxes, times = results
                self._save_results(tracker.name, seq_name, {
                    'boxes': boxes,
                    'rgb_boxes': boxes,
                    'sonar_boxes': boxes,
                    'times': times,
                })

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

    def _compute_visibility_mask(self, gt_boxes):
        """Compute visibility mask from ground truth boxes.

        AquaSOT encodes target absence as (0,0,0,0) boxes.
        Following GOT-10k pattern: filter frames where targets are invisible.

        Args:
            gt_boxes: Array of boxes (N, 4) in [x, y, w, h] format

        Returns:
            visibility: Boolean array (N,), True = visible, False = absent
        """
        if gt_boxes is None or len(gt_boxes) == 0:
            return np.array([], dtype=bool)

        # Method 1: Exact (0,0,0,0) match
        is_zero = np.all(gt_boxes == 0, axis=1)

        # Method 2: Invalid box detection (w <= 0 or h <= 0)
        has_invalid_size = (gt_boxes[:, 2] <= 0) | (gt_boxes[:, 3] <= 0)

        # Combine: visible = not zero AND has valid size
        visibility = ~is_zero & ~has_invalid_size

        return visibility

    def _save_results(self, tracker_name, seq_name, results):
        """Save tracking results.

        Args:
            tracker_name: Tracker name
            seq_name: Sequence name
            results: dict with keys:
                - boxes/rgb_boxes: RGB predicted boxes (N, 4)
                - sonar_boxes: Sonar predicted boxes (N, 4)
                - rgb_scores: RGB confidence scores (N,) (optional)
                - sonar_scores: Sonar confidence scores (N,) (optional)
                - times: Frame times (N,)
        """
        # Create tracker directory
        tracker_dir = os.path.join(self.result_dir, tracker_name)
        os.makedirs(tracker_dir, exist_ok=True)

        # Get boxes (support both 'boxes' and 'rgb_boxes' keys)
        rgb_boxes = results.get('rgb_boxes', results.get('boxes'))
        sonar_boxes = results.get('sonar_boxes', rgb_boxes)
        times = results.get('times', np.zeros(len(rgb_boxes)))

        # Save RGB boxes
        rgb_file = os.path.join(tracker_dir, f'{seq_name}_rgb.txt')
        np.savetxt(rgb_file, rgb_boxes, delimiter=',', fmt='%.4f')

        # Save Sonar boxes
        sonar_file = os.path.join(tracker_dir, f'{seq_name}_sonar.txt')
        np.savetxt(sonar_file, sonar_boxes, delimiter=',', fmt='%.4f')

        # Save times
        time_file = os.path.join(tracker_dir, f'{seq_name}_time.txt')
        np.savetxt(time_file, times, fmt='%.6f')

        # Save scores if available
        if 'rgb_scores' in results:
            rgb_scores_file = os.path.join(tracker_dir, f'{seq_name}_rgb_scores.txt')
            np.savetxt(rgb_scores_file, results['rgb_scores'], fmt='%.6f')

        if 'sonar_scores' in results:
            sonar_scores_file = os.path.join(tracker_dir, f'{seq_name}_sonar_scores.txt')
            np.savetxt(sonar_scores_file, results['sonar_scores'], fmt='%.6f')

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
        """Compute metrics for a tracker with visibility filtering and multi-modal support.

        Following GOT-10k pattern (line 181):
            seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]

        Args:
            tracker_name: Tracker name

        Returns:
            dict with keys:
                - rgb: RGB-specific metrics
                - sonar: Sonar-specific metrics
                - overall: Combined metrics (weighted by visible frames)
                - sequences: Per-sequence breakdown
        """
        tracker_dir = os.path.join(self.result_dir, tracker_name)
        if not os.path.exists(tracker_dir):
            print(f'[Warning] Results not found for {tracker_name}')
            return {}

        # Accumulators for each modality
        rgb_ious_all = []
        rgb_errors_all = []
        sonar_ious_all = []
        sonar_errors_all = []

        # Frame counters for visibility stats
        rgb_visible_frames = 0
        rgb_total_frames = 0
        sonar_visible_frames = 0
        sonar_total_frames = 0

        seq_metrics = {}

        for seq_name in self.sequence_names:
            # Load predictions for both modalities
            rgb_result_file = os.path.join(tracker_dir, f'{seq_name}_rgb.txt')
            sonar_result_file = os.path.join(tracker_dir, f'{seq_name}_sonar.txt')

            if not os.path.exists(rgb_result_file):
                continue

            rgb_pred = np.loadtxt(rgb_result_file, delimiter=',')
            if rgb_pred.ndim == 1:
                rgb_pred = rgb_pred[np.newaxis, :]

            sonar_pred = None
            if os.path.exists(sonar_result_file):
                sonar_pred = np.loadtxt(sonar_result_file, delimiter=',')
                if sonar_pred.ndim == 1:
                    sonar_pred = sonar_pred[np.newaxis, :]

            # Load ground truth
            seq_data = self._load_sequence(seq_name)
            if seq_data is None:
                continue

            rgb_gt = seq_data['rgb_gt']
            sonar_gt = seq_data['sonar_gt']

            # Compute visibility masks
            rgb_vis = self._compute_visibility_mask(rgb_gt) if rgb_gt is not None else None
            sonar_vis = self._compute_visibility_mask(sonar_gt) if sonar_gt is not None else None

            # Initialize per-sequence metrics
            seq_metrics[seq_name] = {}

            # === RGB Evaluation ===
            if rgb_gt is not None and len(rgb_gt) > 0 and self.eval_mode != 'sonar_only':
                n_frames = min(len(rgb_pred), len(rgb_gt))

                # Determine start index (skip frame 0 if configured)
                start_idx = 1 if self.skip_init_frame and n_frames > 1 else 0

                if n_frames > start_idx:
                    pred_eval = rgb_pred[start_idx:n_frames]
                    gt_eval = rgb_gt[start_idx:n_frames]
                    vis_eval = rgb_vis[start_idx:n_frames] if rgb_vis is not None else np.ones(n_frames - start_idx, dtype=bool)

                    # Apply visibility filter if enabled
                    if not self.filter_visibility:
                        vis_eval = np.ones(len(vis_eval), dtype=bool)

                    # Compute raw metrics
                    ious = rect_iou(pred_eval, gt_eval)
                    errors = center_error(pred_eval, gt_eval)

                    # Filter by visibility (GOT-10k pattern)
                    n_visible = vis_eval.sum()
                    if n_visible > 0:
                        ious_filtered = ious[vis_eval]
                        errors_filtered = errors[vis_eval]
                        rgb_ious_all.append(ious_filtered)
                        rgb_errors_all.append(errors_filtered)

                    # Update counters
                    rgb_total_frames += len(vis_eval)
                    rgb_visible_frames += n_visible

                    # Per-sequence RGB metrics
                    if n_visible > 0:
                        seq_metrics[seq_name]['rgb'] = {
                            'avg_iou': float(np.mean(ious_filtered)),
                            'precision_20': float(np.mean(errors_filtered <= 20)),
                            'avg_center_error': float(np.mean(errors_filtered)),
                            'visible_frames': int(n_visible),
                            'total_frames': int(len(vis_eval)),
                        }

            # === Sonar Evaluation ===
            if sonar_gt is not None and sonar_pred is not None and len(sonar_gt) > 0 and self.eval_mode != 'rgb_only':
                n_frames = min(len(sonar_pred), len(sonar_gt))

                # Determine start index
                start_idx = 1 if self.skip_init_frame and n_frames > 1 else 0

                if n_frames > start_idx:
                    pred_eval = sonar_pred[start_idx:n_frames]
                    gt_eval = sonar_gt[start_idx:n_frames]
                    vis_eval = sonar_vis[start_idx:n_frames] if sonar_vis is not None else np.ones(n_frames - start_idx, dtype=bool)

                    # Apply visibility filter if enabled
                    if not self.filter_visibility:
                        vis_eval = np.ones(len(vis_eval), dtype=bool)

                    # Compute raw metrics
                    ious = rect_iou(pred_eval, gt_eval)
                    errors = center_error(pred_eval, gt_eval)

                    # Filter by visibility
                    n_visible = vis_eval.sum()
                    if n_visible > 0:
                        ious_filtered = ious[vis_eval]
                        errors_filtered = errors[vis_eval]
                        sonar_ious_all.append(ious_filtered)
                        sonar_errors_all.append(errors_filtered)

                    # Update counters
                    sonar_total_frames += len(vis_eval)
                    sonar_visible_frames += n_visible

                    # Per-sequence Sonar metrics
                    if n_visible > 0:
                        seq_metrics[seq_name]['sonar'] = {
                            'avg_iou': float(np.mean(ious_filtered)),
                            'precision_20': float(np.mean(errors_filtered <= 20)),
                            'avg_center_error': float(np.mean(errors_filtered)),
                            'visible_frames': int(n_visible),
                            'total_frames': int(len(vis_eval)),
                        }

        # Aggregate results
        result = {
            'rgb': self._aggregate_modality_metrics(
                rgb_ious_all, rgb_errors_all, rgb_visible_frames, rgb_total_frames
            ),
            'sonar': self._aggregate_modality_metrics(
                sonar_ious_all, sonar_errors_all, sonar_visible_frames, sonar_total_frames
            ),
            'sequences': seq_metrics,
        }

        # Compute overall metrics (weighted by visible frames)
        result['overall'] = self._compute_overall_metrics(result['rgb'], result['sonar'])

        # Backward compatibility: add legacy 'overall' fields
        # Use RGB curves by default, fall back to Sonar in sonar_only mode
        result['overall']['success_score'] = result['overall']['success_auc']
        if result['rgb'].get('success_auc') is not None:
            result['success_curve'] = result['rgb'].get('success_curve', [])
            result['precision_curve'] = result['rgb'].get('precision_curve', [])
        elif result['sonar'].get('success_auc') is not None:
            result['success_curve'] = result['sonar'].get('success_curve', [])
            result['precision_curve'] = result['sonar'].get('precision_curve', [])
        else:
            result['success_curve'] = []
            result['precision_curve'] = []

        return result

    def _aggregate_modality_metrics(self, ious_list, errors_list, visible_frames, total_frames):
        """Aggregate metrics for a single modality.

        Args:
            ious_list: List of IoU arrays (one per sequence)
            errors_list: List of center error arrays (one per sequence)
            visible_frames: Total visible frames across all sequences
            total_frames: Total frames across all sequences

        Returns:
            dict with aggregated metrics
        """
        if not ious_list:
            return {
                'success_auc': None,
                'precision_score': None,
                'avg_iou': None,
                'avg_center_error': None,
                'visible_frames': 0,
                'total_frames': total_frames,
                'visibility_ratio': 0.0,
            }

        all_ious = np.concatenate(ious_list)
        all_errors = np.concatenate(errors_list)

        success = success_curve(all_ious)
        precision = precision_curve(all_errors)
        success_auc = calculate_auc(success)

        return {
            'success_auc': float(success_auc),
            'precision_score': float(precision[20]) if len(precision) > 20 else 0.0,
            'avg_iou': float(np.mean(all_ious)),
            'avg_center_error': float(np.mean(all_errors)),
            'visible_frames': int(visible_frames),
            'total_frames': int(total_frames),
            'visibility_ratio': float(visible_frames / total_frames) if total_frames > 0 else 0.0,
            'success_curve': success.tolist(),
            'precision_curve': precision.tolist(),
        }

    def _compute_overall_metrics(self, rgb_metrics, sonar_metrics):
        """Compute overall metrics from RGB and Sonar metrics.

        Uses weighted average by number of visible frames (GOT-10k style).

        Args:
            rgb_metrics: dict with RGB metrics
            sonar_metrics: dict with Sonar metrics

        Returns:
            dict with overall metrics
        """
        rgb_vis = rgb_metrics.get('visible_frames', 0) or 0
        sonar_vis = sonar_metrics.get('visible_frames', 0) or 0
        total_vis = rgb_vis + sonar_vis

        if total_vis == 0:
            return {
                'success_auc': 0.0,
                'precision_score': 0.0,
                'avg_iou': 0.0,
                'avg_center_error': float('inf'),
                'visible_frames': 0,
                'rgb_weight': 0.0,
                'sonar_weight': 0.0,
            }

        # Weighted average by visible frames
        w_rgb = rgb_vis / total_vis
        w_sonar = sonar_vis / total_vis

        def weighted_avg(rgb_val, sonar_val):
            rgb_v = rgb_val if rgb_val is not None else 0.0
            sonar_v = sonar_val if sonar_val is not None else 0.0
            return w_rgb * rgb_v + w_sonar * sonar_v

        return {
            'success_auc': weighted_avg(
                rgb_metrics.get('success_auc'),
                sonar_metrics.get('success_auc')
            ),
            'precision_score': weighted_avg(
                rgb_metrics.get('precision_score'),
                sonar_metrics.get('precision_score')
            ),
            'avg_iou': weighted_avg(
                rgb_metrics.get('avg_iou'),
                sonar_metrics.get('avg_iou')
            ),
            'avg_center_error': weighted_avg(
                rgb_metrics.get('avg_center_error', float('inf')),
                sonar_metrics.get('avg_center_error', float('inf'))
            ),
            'visible_frames': total_vis,
            'rgb_weight': w_rgb,
            'sonar_weight': w_sonar,
        }

    def _print_summary(self, performance):
        """Print evaluation summary with multi-modal breakdown.

        Args:
            performance: dict with metrics for each tracker
        """
        print('\n' + '=' * 90)
        print('AquaSOT Evaluation Summary (Visibility-Filtered)')
        print('=' * 90)
        print(f'{"Tracker":<20} {"Modality":<10} {"Success AUC":>12} {"Precision@20":>13} '
              f'{"Avg IoU":>10} {"Vis Frames":>12}')
        print('-' * 90)

        for name, metrics in performance.items():
            if 'rgb' not in metrics:
                print(f'{name:<20} {"N/A":<10}')
                continue

            # RGB metrics
            rgb = metrics.get('rgb', {})
            rgb_auc = rgb.get('success_auc')
            if rgb_auc is not None:
                print(f'{name:<20} {"RGB":<10} {rgb_auc:>12.3f} '
                      f'{rgb.get("precision_score", 0):>13.3f} {rgb.get("avg_iou", 0):>10.3f} '
                      f'{rgb.get("visible_frames", 0):>12d}')
            else:
                print(f'{name:<20} {"RGB":<10} {"N/A":>12} {"N/A":>13} {"N/A":>10} '
                      f'{rgb.get("total_frames", 0):>12d}')

            # Sonar metrics
            sonar = metrics.get('sonar', {})
            sonar_auc = sonar.get('success_auc')
            if sonar_auc is not None:
                print(f'{"":<20} {"Sonar":<10} {sonar_auc:>12.3f} '
                      f'{sonar.get("precision_score", 0):>13.3f} {sonar.get("avg_iou", 0):>10.3f} '
                      f'{sonar.get("visible_frames", 0):>12d}')
            else:
                print(f'{"":<20} {"Sonar":<10} {"N/A":>12} {"N/A":>13} {"N/A":>10} '
                      f'{sonar.get("total_frames", 0):>12d}')

            # Overall metrics
            overall = metrics.get('overall', {})
            print(f'{"":<20} {"Overall":<10} {overall.get("success_auc", 0):>12.3f} '
                  f'{overall.get("precision_score", 0):>13.3f} {overall.get("avg_iou", 0):>10.3f} '
                  f'{overall.get("visible_frames", 0):>12d}')
            print('-' * 90)

        print('=' * 90)
        vis_note = "Metrics computed only on visible frames (GT != [0,0,0,0])"
        if not self.filter_visibility:
            vis_note = "Visibility filtering DISABLED (all frames included)"
        init_note = ", frame 0 skipped" if self.skip_init_frame else ""
        mode_note = f", eval_mode={self.eval_mode}" if self.eval_mode != 'dual' else ""
        print(f'Note: {vis_note}{init_note}{mode_note}')
        print('=' * 90)

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

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(performance)))

        for (name, metrics), color in zip(performance.items(), colors):
            # RGB curves
            rgb = metrics.get('rgb', {})
            if rgb.get('success_curve'):
                success = np.array(rgb['success_curve'])
                precision = np.array(rgb['precision_curve'])
                auc = rgb.get('success_auc', 0)

                axes[0, 0].plot(
                    np.linspace(0, 1, len(success)),
                    success,
                    color=color,
                    label=f'{name} RGB [{auc:.3f}]'
                )
                axes[0, 1].plot(
                    np.linspace(0, 50, len(precision)),
                    precision,
                    color=color,
                    label=f'{name} RGB'
                )

            # Sonar curves
            sonar = metrics.get('sonar', {})
            if sonar.get('success_curve'):
                success = np.array(sonar['success_curve'])
                precision = np.array(sonar['precision_curve'])
                auc = sonar.get('success_auc', 0)

                axes[1, 0].plot(
                    np.linspace(0, 1, len(success)),
                    success,
                    color=color,
                    linestyle='--',
                    label=f'{name} Sonar [{auc:.3f}]'
                )
                axes[1, 1].plot(
                    np.linspace(0, 50, len(precision)),
                    precision,
                    color=color,
                    linestyle='--',
                    label=f'{name} Sonar'
                )

        # Configure axes
        for i, title in enumerate(['RGB Success', 'RGB Precision', 'Sonar Success', 'Sonar Precision']):
            ax = axes[i // 2, i % 2]
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            if 'Success' in title:
                ax.set_xlabel('Overlap Threshold')
                ax.set_ylabel('Success Rate')
            else:
                ax.set_xlabel('Center Error Threshold (pixels)')
                ax.set_ylabel('Precision Rate')

        plt.tight_layout()

        # Save figure
        fig_file = os.path.join(self.report_dir, 'curves.png')
        plt.savefig(fig_file, dpi=150)
        print(f'[ExperimentAquaSOT] Curves saved to {fig_file}')

        plt.close()
