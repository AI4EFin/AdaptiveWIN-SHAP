"""
LPA Hyperparameter Sensitivity Analysis

Tests sensitivity of Adaptive WIN-SHAP to LPA parameters:
- N0 (initial window size)
- Jump (detection stride)
- Alpha (significance level)
- Num_bootstrap (bootstrap iterations)

Runs full pipeline: LPA window detection → SHAP computation → Metric evaluation
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from itertools import product
from tqdm import tqdm
from pathlib import Path

# Add parent directory and examples directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptivewinshap import AdaptiveModel, ChangeDetector, store_init_kwargs
from benchmark import run_benchmark


class AdaptiveLSTM(AdaptiveModel):
    """LSTM model for AdaptiveWinShap (copied from lstm_simulation.py)."""

    @store_init_kwargs
    def __init__(self, device, seq_length=3, input_size=1, hidden=16, layers=1,
                 dropout=0.2, batch_size=512, lr=1e-12, epochs=50, type_precision=np.float32):
        super().__init__(device=device, batch_size=batch_size, lr=lr, epochs=epochs,
                         type_precision=type_precision)
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout

    def forward(self, x):
        out, _ = self.lstm(x)
        yhat = self.fc(out[:, -1, :])
        return yhat.squeeze(-1)

    def prepare_data(self, window, start_abs_idx):
        """Prepare sequences from window."""
        L = self.seq_length
        F = window.shape[1] if window.ndim == 2 else 1
        n = len(window)

        if n <= L:
            return None, None, None

        if window.ndim == 1:
            window = window[:, None]

        X_list = []
        y_list = []
        for i in range(L, n):
            X_list.append(window[i-L:i])
            y_list.append(window[i, 0])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        return X_tensor, y_tensor, t_abs

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        preds = self(xt)
        return preds.detach().cpu().numpy().reshape(-1, 1)


class LPASensitivityExperiment:
    """Run LPA sensitivity analysis."""

    DATASET_CONFIGS = {
        "piecewise_ar3": {"seq_length": 3, "n_covariates": 0},
        "arx_rotating": {"seq_length": 3, "n_covariates": 3},
        "trend_season": {"seq_length": 3, "n_covariates": 0},
        "spike_process": {"seq_length": 3, "n_covariates": 2},
        "garch_regime": {"seq_length": 1, "n_covariates": 2},
    }

    def __init__(self, output_dir='examples/results/robustness/lpa_sensitivity'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.device = self._get_device()

    def _get_device(self):
        """Get best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        return "cpu"

    def run_lpa_detection(self, dataset_name, N0, alpha, num_bootstrap,
                          temp_dir, n_runs=1, growth="geometric", growth_base=2.0):
        """
        Run LPA window detection with specified parameters.

        Note: jump is fixed at 1 (no skipping) as it's only a computational
        parameter for development, not a methodological parameter.

        Parameters:
        -----------
        growth : str
            Window growth strategy: "arithmetic" or "geometric"
        growth_base : float
            Base for geometric growth (only used if growth="geometric")

        Returns:
        --------
        dict with keys:
            - window_csv: path to windows CSV
            - window_mean: mean window size
            - window_std: std of window size
            - detection_time: time taken for detection
        """
        # Load dataset
        dataset_path = f"examples/datasets/simulated/{dataset_name}/data.csv"
        df = pd.read_csv(dataset_path)

        # Get configuration
        config = self.DATASET_CONFIGS[dataset_name]
        seq_length = config["seq_length"]
        n_covariates = config["n_covariates"]
        input_size = 1 + n_covariates

        # Extract target and covariates
        target = df["N"].to_numpy(dtype=np.float64)

        if n_covariates > 0:
            cov_cols = [f"Z_{i}" for i in range(n_covariates)]
            covariates = df[cov_cols].to_numpy(dtype=np.float64)
            data = np.column_stack([target, covariates])
        else:
            data = target

        # Initialize model
        model = AdaptiveLSTM(
            self.device,
            seq_length=seq_length,
            input_size=input_size,
            hidden=16,
            layers=1,
            dropout=0.0,
            batch_size=64,
            lr=1e-2,
            epochs=15,
            type_precision=np.float64
        )

        cd = ChangeDetector(model, data, debug=False, force_cpu=True)

        # Run detection (jump fixed at 1 for full detection)
        start_time = time.time()

        run_csvs = []
        for run in range(n_runs):
            # Save run results
            run_csv = temp_dir / f"run_{run}.csv"

            if Path(run_csv).exists():
                print(f"Skipping run {run} as results already exist")
                run_csvs.append(run_csv)
                continue

            results = cd.detect(
                min_window=4,
                n_0=N0,
                jump=1,  # Fixed at 1 - not a methodological parameter
                search_step=2,
                alpha=alpha,
                num_bootstrap=num_bootstrap,
                t_workers=10,
                b_workers=10,
                one_b_threads=1,
                debug_anim=False,
                save_path=None,
                growth=growth,
                growth_base=growth_base
            )


            pd.DataFrame(results).to_csv(run_csv, index=False)
            run_csvs.append(run_csv)

        detection_time = time.time() - start_time

        # Aggregate windows from runs
        dfs = []
        for i, csv_path in enumerate(run_csvs):
            win_df = pd.read_csv(csv_path, usecols=["windows"])
            win_df = win_df.rename(columns={"windows": f"windows_run_{i}"})
            dfs.append(win_df)

        windows_df = pd.concat(dfs, axis=1)
        windows_df["window_mean"] = windows_df.mean(axis=1)

        # PADDING FIX: Ensure windows align with original data length
        # Windows start at seq_length due to LSTM sequence preparation
        config = self.DATASET_CONFIGS[dataset_name]
        seq_length = config["seq_length"]

        # Expected data length (from original dataset)
        expected_length = len(target)

        # If windows are shorter than expected, pad at the beginning
        if len(windows_df) < expected_length:
            n_pad = expected_length - len(windows_df)

            # Create padding rows with first window value
            first_window = windows_df.iloc[0]
            pad_df = pd.DataFrame([first_window] * n_pad, columns=windows_df.columns)

            # Prepend padding
            windows_df = pd.concat([pad_df, windows_df], ignore_index=True)

        # Save aggregated windows
        windows_csv = temp_dir / "windows.csv"
        windows_df.to_csv(windows_csv, index=False)

        return {
            'window_csv': str(windows_csv),
            'window_mean': float(windows_df['window_mean'].mean()),
            'window_std': float(windows_df['window_mean'].std()),
            'window_min': float(windows_df['window_mean'].min()),
            'window_max': float(windows_df['window_mean'].max()),
            'detection_time': detection_time,
            'n_timepoints': len(windows_df)
        }

    def ensure_window_padding(self, windows_csv, dataset_name):
        """
        Ensure windows CSV has correct length with padding if needed.

        This fixes the alignment issue where LSTM seq_length causes
        windows to be shorter than the original data.

        Returns the path to the (potentially updated) windows CSV.
        """
        # Load windows
        windows_df = pd.read_csv(windows_csv)

        # Load original dataset to get expected length
        dataset_path = f"examples/datasets/simulated/{dataset_name}/data.csv"
        df = pd.read_csv(dataset_path)
        expected_length = len(df)

        # Check if padding is needed
        if len(windows_df) < expected_length:
            n_pad = expected_length - len(windows_df)
            print(f"  Padding windows from {len(windows_df)} to {expected_length} rows (adding {n_pad} rows)")

            # Create padding rows with first window value
            first_window = windows_df.iloc[0]
            pad_df = pd.DataFrame([first_window] * n_pad, columns=windows_df.columns)

            # Prepend padding
            windows_df = pd.concat([pad_df, windows_df], ignore_index=True)

            # Save updated windows
            windows_df.to_csv(windows_csv, index=False)

        return windows_csv

    def compute_breakpoint_accuracy(self, windows_csv, dataset_name):
        """
        Compute how accurately detected windows align with true breakpoints.

        For datasets with known regime changes, calculate the lag/lead
        from true breakpoints.
        """
        # Load detected windows
        windows_df = pd.read_csv(windows_csv)
        window_sizes = windows_df['window_mean'].values

        # Define true breakpoints for each dataset
        true_breakpoints = {
            'piecewise_ar3': [500, 1000],  # regime changes at t=500, 1000
            'arx_rotating': [500, 1000],
            'trend_season': [500, 1000],
            'spike_process': [750],  # single breakpoint at t=750
            'garch_regime': [750]
        }

        if dataset_name not in true_breakpoints:
            return {
                'breakpoint_detection_lag_mean': np.nan,
                'breakpoint_detection_lag_std': np.nan
            }

        breakpoints = true_breakpoints[dataset_name]

        # Detect changes in window size (indicator of detected breakpoint)
        # A detected breakpoint is where window size drops significantly
        window_changes = np.abs(np.diff(window_sizes))
        detected_breakpoints = np.where(window_changes > window_sizes[:-1].mean() * 0.3)[0]

        if len(detected_breakpoints) == 0:
            return {
                'breakpoint_detection_lag_mean': 999.0,  # Large penalty for no detection
                'breakpoint_detection_lag_std': 0.0
            }

        # For each true breakpoint, find closest detected breakpoint
        lags = []
        for true_bp in breakpoints:
            if len(detected_breakpoints) > 0:
                distances = np.abs(detected_breakpoints - true_bp)
                min_distance = np.min(distances)
                lags.append(min_distance)
            else:
                lags.append(999.0)

        return {
            'breakpoint_detection_lag_mean': float(np.mean(lags)),
            'breakpoint_detection_lag_std': float(np.std(lags)) if len(lags) > 1 else 0.0
        }

    def run_benchmark_with_windows(self, dataset_name, windows_csv, temp_dir):
        """
        Run full benchmark pipeline with detected windows.

        Returns dict of metrics from benchmark.
        """
        dataset_path = f"examples/datasets/simulated/{dataset_name}/data.csv"
        benchmark_output = temp_dir / "benchmark"

        # Run benchmark (this computes SHAP and all metrics)
        try:
            summary_df = run_benchmark(
                dataset_path=dataset_path,
                output_dir=str(benchmark_output),
                device=self.device,
                verbose=False,
                dataset_type='simulated',
                column_name='N',
                seq_length=self.DATASET_CONFIGS[dataset_name]['seq_length'],
                precomputed_windows_path=windows_csv,
                rolling_mean_window=75
            )

            # Extract metrics for adaptive_shap method
            adaptive_metrics = summary_df[summary_df['method'] == 'adaptive_shap']

            if len(adaptive_metrics) == 0:
                print(f"Warning: No adaptive_shap results found for {dataset_name}")
                return {}

            # Convert to dictionary
            metrics = {}
            for _, row in adaptive_metrics.iterrows():
                metric_key = f"{row['metric_type']}_{row['evaluation']}"
                metrics[metric_key] = float(row['score'])

            # Also get correlation with true importance if available
            adaptive_results_path = benchmark_output / "adaptive_shap_results.csv"
            if adaptive_results_path.exists():
                adaptive_df = pd.read_csv(adaptive_results_path)

                # Load true importances
                true_imp_path = f"examples/datasets/simulated/{dataset_name}/true_importances.csv"
                if os.path.exists(true_imp_path):
                    true_imp_df = pd.read_csv(true_imp_path)

                    # Compute correlation for each feature
                    shap_cols = [c for c in adaptive_df.columns if c.startswith('shap_lag_') or c.startswith('shap_cov_')]
                    true_cols = [c for c in true_imp_df.columns if c.startswith('true_imp_')]

                    if len(shap_cols) == len(true_cols):
                        correlations = []
                        for shap_col, true_col in zip(shap_cols, true_cols):
                            # Get SHAP values
                            shap_vals = adaptive_df[shap_col].values

                            # Align true importances with SHAP results using end_index
                            if 'end_index' in adaptive_df.columns:
                                end_indices = adaptive_df['end_index'].values.astype(int)
                                # Clip to valid range
                                end_indices = np.clip(end_indices, 0, len(true_imp_df) - 1)
                                true_vals = true_imp_df.iloc[end_indices][true_col].values
                            else:
                                # Fallback: use first len(shap_vals) values
                                true_vals = true_imp_df[true_col].iloc[:len(shap_vals)].values

                            # Handle NaN values
                            mask = ~(np.isnan(shap_vals) | np.isnan(true_vals))
                            if mask.sum() > 0:
                                corr = np.corrcoef(
                                    np.abs(shap_vals[mask]),
                                    true_vals[mask]
                                )[0, 1]
                                correlations.append(corr)

                        if correlations:
                            metrics['correlation_true_imp_mean'] = float(np.mean(correlations))
                            metrics['correlation_true_imp_std'] = float(np.std(correlations))

            return metrics

        except Exception as e:
            print(f"Error running benchmark for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def run_grid_search(self, dataset_name, param_grid, n_runs=1, benchmark_only=False):
        """
        Run grid search over LPA parameters.

        Parameters:
        -----------
        dataset_name : str
            Dataset to test on
        param_grid : dict
            Dictionary with keys: N0, alpha, num_bootstrap
            Each value is a list of parameters to test
            Note: jump is fixed at 1 (not tested)
        n_runs : int
            Number of LPA detection runs per parameter set (for stability)
        benchmark_only : bool
            If True, skip LPA detection and only run benchmark on existing windows
        """
        print(f"\n{'='*60}")
        print(f"LPA Sensitivity Analysis: {dataset_name}")
        if benchmark_only:
            print("Mode: BENCHMARK ONLY (using existing windows)")
        print(f"{'='*60}")

        # Generate all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        print(f"Total combinations: {len(combinations)}")
        print(f"Parameters: {keys}")
        if not benchmark_only:
            print(f"Detection runs per combination: {n_runs}")
        print(f"{'='*60}\n")

        # Create dataset-specific output directory with growth strategy
        dataset_output = self.output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)

        # Run grid search
        for combo in tqdm(combinations, desc=f"Grid search on {dataset_name}"):
            params = dict(zip(keys, combo))

            # Get growth strategy and create subdirectory
            growth_strategy = params.get('growth', 'arithmetic')
            growth_output = dataset_output / growth_strategy
            growth_output.mkdir(parents=True, exist_ok=True)

            # Create temp directory for this run
            param_str = "_".join([f"{k}{v}" for k, v in params.items()])
            temp_dir = growth_output / f"temp_{param_str}"

            if benchmark_only:
                # Check if temp directory and windows.csv exist
                windows_csv = temp_dir / "windows.csv"
                if not windows_csv.exists():
                    print(f"\nWarning: Windows file not found: {windows_csv}")
                    print(f"Skipping {params}")
                    continue
            else:
                temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                if not benchmark_only:
                    # Step 1: Run LPA detection (jump=1 fixed)
                    detection_results = self.run_lpa_detection(
                        dataset_name=dataset_name,
                        N0=params['N0'],
                        alpha=params['alpha'],
                        num_bootstrap=params['num_bootstrap'],
                        temp_dir=temp_dir,
                        n_runs=n_runs,
                        growth=params.get('growth', 'geometric'),
                        growth_base=params.get('growth_base', 2.0)
                    )
                else:
                    # Load existing detection results from windows.csv
                    windows_csv = temp_dir / "windows.csv"

                    # Ensure windows have correct padding (fixes alignment issue)
                    windows_csv = self.ensure_window_padding(windows_csv, dataset_name)

                    windows_df = pd.read_csv(windows_csv)
                    detection_results = {
                        'window_csv': str(windows_csv),
                        'window_mean': float(windows_df['window_mean'].mean()),
                        'window_std': float(windows_df['window_mean'].std()),
                        'window_min': float(windows_df['window_mean'].min()),
                        'window_max': float(windows_df['window_mean'].max()),
                        'detection_time': 0.0,  # Not re-running detection
                        'n_timepoints': len(windows_df)
                    }

                # Step 2: Compute breakpoint accuracy
                breakpoint_metrics = self.compute_breakpoint_accuracy(
                    windows_csv=detection_results['window_csv'],
                    dataset_name=dataset_name
                )

                # Step 3: Run full benchmark with these windows
                benchmark_metrics = self.run_benchmark_with_windows(
                    dataset_name=dataset_name,
                    windows_csv=detection_results['window_csv'],
                    temp_dir=temp_dir
                )

                # Combine all results
                result = {
                    'dataset': dataset_name,
                    **params,
                    **detection_results,
                    **breakpoint_metrics,
                    **benchmark_metrics,
                    'success': True
                }

            except Exception as e:
                print(f"\nError with params {params}: {e}")
                result = {
                    'dataset': dataset_name,
                    **params,
                    'success': False,
                    'error': str(e)
                }

            self.results.append(result)

            # Save intermediate results
            self.save_results()

    def save_results(self):
        """Save results to CSV."""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / 'results.csv', index=False)

    def save_config(self, config):
        """Save experiment configuration."""
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='LPA Sensitivity Analysis')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['piecewise_ar3', 'arx_rotating'],
                       help='Datasets to test')
    parser.add_argument('--n-runs', type=int, default=1,
                       help='Number of LPA detection runs per parameter set')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameter grid')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Skip LPA detection and only run benchmark on existing windows')
    parser.add_argument('--output-dir', type=str,
                       default='examples/results/robustness/lpa_sensitivity',
                       help='Output directory')

    args = parser.parse_args()

    # Define parameter grid
    if args.quick_test:
        # Quick test: fewer parameters
        param_grid = {
            'N0': [50, 100],
            'alpha': [0.95],
            'num_bootstrap': [10, 50],
            'growth': ['geometric'],
            'growth_base': [1.41421356237]  # sqrt(2)
        }
    else:
        # Full grid
        param_grid = {
            'N0': [25, 50, 75, 100],
            'alpha': [0.90, 0.95, 0.99],
            'num_bootstrap': [10, 30, 50],
            'growth': ['geometric'],
            'growth_base': [1.41421356237]  # 2.0 and sqrt(2)
        }

    print("="*60)
    print("LPA Hyperparameter Sensitivity Analysis")
    print("="*60)
    print(f"Datasets: {args.datasets}")
    print(f"Parameter grid: {param_grid}")
    print(f"Runs per combination: {args.n_runs}")
    print(f"Quick test mode: {args.quick_test}")
    print(f"Benchmark only: {args.benchmark_only}")
    print("="*60)

    # Initialize experiment
    experiment = LPASensitivityExperiment(output_dir=args.output_dir)

    # Save configuration
    config = {
        'datasets': args.datasets,
        'param_grid': param_grid,
        'n_runs': args.n_runs,
        'quick_test': args.quick_test,
        'benchmark_only': args.benchmark_only,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    experiment.save_config(config)

    # Run grid search for each dataset
    for dataset in args.datasets:
        experiment.run_grid_search(
            dataset_name=dataset,
            param_grid=param_grid,
            n_runs=args.n_runs,
            benchmark_only=args.benchmark_only
        )

    # Final save
    experiment.save_results()

    print("\n" + "="*60)
    print("Experiment Complete!")
    print(f"Results saved to: {experiment.output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Run visualization script:")
    print(f"   python examples/robustness_viz.py --results-dir {args.output_dir}")
    print("\n2. Analyze sensitivity indices:")
    print(f"   python examples/robustness/analyze_sensitivity.py --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()