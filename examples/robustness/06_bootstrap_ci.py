"""
Bootstrap Confidence Intervals for Adaptive WIN-SHAP

Tests statistical robustness by running the full pipeline on multiple
realizations of each dataset with different random seeds.

Computes 95% confidence intervals for:
- Faithfulness scores
- MIF/LIF ratios
- Correlation with true importance
- Window size statistics
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
from tqdm import tqdm

# Add parent directory and examples directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptivewinshap import AdaptiveModel, ChangeDetector, store_init_kwargs
from benchmark import run_benchmark
from generate_simulated_datasets import save_dataset
from generate_simulated_datasets_random import (
    sim_piecewise_ar3_rotating,
    sim_piecewise_arx_rotating_drivers,
    sim_trend_season_ar_break,
    sim_spike_process,
    sim_regime_garch_factors
)


class AdaptiveLSTM(AdaptiveModel):
    """LSTM model for AdaptiveWinShap."""

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


class BootstrapExperiment:
    """Run bootstrap confidence interval experiments."""

    # Dataset generation functions
    DATASET_GENERATORS = {
        "piecewise_ar3": sim_piecewise_ar3_rotating,
        "arx_rotating": sim_piecewise_arx_rotating_drivers,
        "trend_season": sim_trend_season_ar_break,
        "spike_process": sim_spike_process,
        "garch_regime": sim_regime_garch_factors,
    }

    # Dataset configurations (must match generate_simulated_datasets.py)
    DATASET_CONFIGS = {
        "piecewise_ar3": {"seq_length": 3, "n_covariates": 0},
        "arx_rotating": {"seq_length": 3, "n_covariates": 3},
        "trend_season": {"seq_length": 3, "n_covariates": 0},
        "spike_process": {"seq_length": 3, "n_covariates": 2},
        "garch_regime": {"seq_length": 1, "n_covariates": 2},
    }

    def __init__(self, output_dir='examples/results/robustness/bootstrap_ci',
                 N0=75, alpha=0.95, num_bootstrap=100, randomize_params=True):
        """
        Initialize bootstrap experiment.

        Parameters
        ----------
        output_dir : str
            Output directory for results
        N0 : int
            Initial window size for LPA
        alpha : float
            Significance level for LPA
        num_bootstrap : int
            Number of bootstrap iterations for LPA
        randomize_params : bool
            If True, randomize DGP parameters across seeds while maintaining structure
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.device = self._get_device()

        # Fixed LPA hyperparameters
        self.N0 = N0
        self.alpha = alpha
        self.num_bootstrap = num_bootstrap
        self.randomize_params = randomize_params

    def _get_device(self):
        """Get best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        return "cpu"

    def generate_dataset_realization(self, dataset_name, seed, temp_dir, T=1500):
        """
        Generate a new realization of a dataset with a different seed.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        seed : int
            Random seed for generation
        temp_dir : Path
            Temporary directory to save dataset
        T : int
            Number of time points

        Returns
        -------
        dataset_path : str
            Path to the generated dataset CSV
        """
        if dataset_name not in self.DATASET_GENERATORS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        generator = self.DATASET_GENERATORS[dataset_name]

        # Generate dataset with this seed (with parameter randomization)
        if dataset_name == "piecewise_ar3":
            Y, Z, true_imp, params = generator(T=T, seed=seed, randomize_params=self.randomize_params)
        elif dataset_name == "arx_rotating":
            Y, Z, true_imp, params = generator(T=T, p=3, seed=seed, randomize_params=self.randomize_params)
        elif dataset_name == "trend_season":
            Y, Z, true_imp, params = generator(T=T, p=3, seed=seed, randomize_params=self.randomize_params)
        elif dataset_name == "spike_process":
            Y, Z, true_imp, params = generator(T=T, p=3, seed=seed, randomize_params=self.randomize_params)
        elif dataset_name == "garch_regime":
            Y, Z, true_imp, params = generator(T=T, seed=seed, randomize_params=self.randomize_params)

        # Save to temporary directory
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Main data file
        data_df = pd.DataFrame({'N': Y})
        if Z is not None:
            for i in range(Z.shape[1]):
                data_df[f'Z_{i}'] = Z[:, i]
        data_csv = dataset_dir / 'data.csv'
        data_df.to_csv(data_csv, index=False)

        # True importances
        imp_cols = {f'true_imp_{i}': true_imp[:, i] for i in range(true_imp.shape[1])}
        imp_df = pd.DataFrame(imp_cols)
        true_imp_csv = dataset_dir / 'true_importances.csv'
        imp_df.to_csv(true_imp_csv, index=False)

        # Save generation parameters for documentation
        params_json = dataset_dir / 'generation_params.json'
        with open(params_json, 'w') as f:
            json.dump(params, f, indent=2)

        return str(data_csv), str(true_imp_csv), params

    def run_single_realization(self, dataset_name, seed, temp_dir):
        """
        Run full pipeline (LPA detection + SHAP + metrics) on one realization.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        seed : int
            Random seed for this realization
        temp_dir : Path
            Temporary directory for this run

        Returns
        -------
        dict
            Results containing all metrics
        """
        # Step 1: Generate dataset realization
        data_csv, true_imp_csv, params = self.generate_dataset_realization(
            dataset_name, seed, temp_dir
        )

        # Step 2: Load data
        df = pd.read_csv(data_csv)
        target = df["N"].to_numpy(dtype=np.float64)

        config = self.DATASET_CONFIGS[dataset_name]
        seq_length = config["seq_length"]
        n_covariates = config["n_covariates"]
        input_size = 1 + n_covariates

        if n_covariates > 0:
            cov_cols = [f"Z_{i}" for i in range(n_covariates)]
            covariates = df[cov_cols].to_numpy(dtype=np.float64)
            data = np.column_stack([target, covariates])
        else:
            data = target

        # Step 3: Run LPA detection
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

        start_time = time.time()
        detection_results = cd.detect(
            min_window=4,
            n_0=self.N0,
            jump=1,  # Fixed at 1 for full detection
            search_step=5,
            alpha=self.alpha,
            num_bootstrap=self.num_bootstrap,
            t_workers=10,
            b_workers=10,
            one_b_threads=1,
            debug_anim=False,
            save_path=None
        )
        detection_time = time.time() - start_time

        # Save windows
        windows_csv = temp_dir / "windows.csv"
        pd.DataFrame(detection_results).to_csv(windows_csv, index=False)

        # Step 4: Run benchmark (SHAP + metrics)
        benchmark_output = temp_dir / "benchmark"
        try:
            summary_df = run_benchmark(
                dataset_path=data_csv,
                output_dir=str(benchmark_output),
                device=self.device,
                verbose=False,
                dataset_type='simulated',
                column_name='N',
                seq_length=seq_length,
                precomputed_windows_path=str(windows_csv),
                rolling_mean_window=75
            )

            # Extract metrics for adaptive_shap
            adaptive_metrics = summary_df[summary_df['method'] == 'adaptive_shap']

            metrics = {}
            if len(adaptive_metrics) > 0:
                for _, row in adaptive_metrics.iterrows():
                    metric_key = f"{row['metric_type']}_{row['evaluation']}"
                    metrics[metric_key] = float(row['score'])

            # Get correlation with true importance
            adaptive_results_path = benchmark_output / "adaptive_shap_results.csv"
            if adaptive_results_path.exists():
                adaptive_df = pd.read_csv(adaptive_results_path)
                true_imp_df = pd.read_csv(true_imp_csv)

                shap_cols = [c for c in adaptive_df.columns if c.startswith('shap_lag_') or c.startswith('shap_cov_')]
                true_cols = [c for c in true_imp_df.columns if c.startswith('true_imp_')]

                if len(shap_cols) == len(true_cols):
                    correlations = []
                    for shap_col, true_col in zip(shap_cols, true_cols):
                        shap_vals = adaptive_df[shap_col].values
                        true_vals = true_imp_df[true_col].values

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

            # Window size statistics
            windows_df = pd.read_csv(windows_csv)
            window_sizes = windows_df['windows'].values

            result = {
                'dataset': dataset_name,
                'seed': seed,
                'N0': self.N0,
                'alpha': self.alpha,
                'num_bootstrap': self.num_bootstrap,
                'window_mean': float(window_sizes.mean()),
                'window_std': float(window_sizes.std()),
                'window_min': float(window_sizes.min()),
                'window_max': float(window_sizes.max()),
                'detection_time': detection_time,
                **metrics,
                'success': True
            }

            return result

        except Exception as e:
            print(f"\nError in realization {seed}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'dataset': dataset_name,
                'seed': seed,
                'N0': self.N0,
                'alpha': self.alpha,
                'num_bootstrap': self.num_bootstrap,
                'success': False,
                'error': str(e)
            }

    def run_bootstrap(self, dataset_name, n_realizations=10, start_seed=1000):
        """
        Run bootstrap experiment on a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of dataset to test
        n_realizations : int
            Number of bootstrap realizations
        start_seed : int
            Starting seed value (will use start_seed, start_seed+1, ...)
        """
        print(f"\n{'='*60}")
        print(f"Bootstrap CI Experiment: {dataset_name}")
        print(f"{'='*60}")
        print(f"Number of realizations: {n_realizations}")
        print(f"LPA parameters: N0={self.N0}, alpha={self.alpha}, num_bootstrap={self.num_bootstrap}")
        print(f"Seeds: {start_seed} to {start_seed + n_realizations - 1}")
        print(f"{'='*60}\n")

        # Create dataset-specific output directory
        dataset_output = self.output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)

        # Run realizations
        for i in tqdm(range(n_realizations), desc=f"Bootstrap on {dataset_name}"):
            seed = start_seed + i

            # Create temp directory for this realization
            temp_dir = dataset_output / f"temp_seed_{seed}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = self.run_single_realization(
                    dataset_name=dataset_name,
                    seed=seed,
                    temp_dir=temp_dir
                )
                self.results.append(result)

            except Exception as e:
                print(f"\nFailed on seed {seed}: {e}")
                self.results.append({
                    'dataset': dataset_name,
                    'seed': seed,
                    'success': False,
                    'error': str(e)
                })

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
    parser = argparse.ArgumentParser(description='Bootstrap Confidence Intervals for Adaptive WIN-SHAP')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['piecewise_ar3', 'arx_rotating'],
                       help='Datasets to test')
    parser.add_argument('--n-realizations', type=int, default=2,
                       help='Number of bootstrap realizations per dataset')
    parser.add_argument('--start-seed', type=int, default=1000,
                       help='Starting seed value')
    parser.add_argument('--N0', type=int, default=75,
                       help='Initial window size for LPA')
    parser.add_argument('--alpha', type=float, default=0.95,
                       help='Significance level for LPA')
    parser.add_argument('--num-bootstrap', type=int, default=10,
                       help='Bootstrap iterations for LPA')
    parser.add_argument('--output-dir', type=str,
                       default='examples/results/robustness/bootstrap_ci',
                       help='Output directory')

    args = parser.parse_args()
    n_realizations = args.n_realizations

    print("="*60)
    print("Bootstrap Confidence Intervals - Adaptive WIN-SHAP")
    print("="*60)
    print(f"Datasets: {args.datasets}")
    print(f"Realizations per dataset: {n_realizations}")
    print(f"LPA parameters: N0={args.N0}, alpha={args.alpha}, num_bootstrap={args.num_bootstrap}")
    print(f"Starting seed: {args.start_seed}")
    print("="*60)

    # Initialize experiment
    experiment = BootstrapExperiment(
        output_dir=args.output_dir,
        N0=args.N0,
        alpha=args.alpha,
        num_bootstrap=args.num_bootstrap
    )

    # Save configuration
    config = {
        'datasets': args.datasets,
        'n_realizations': n_realizations,
        'start_seed': args.start_seed,
        'N0': args.N0,
        'alpha': args.alpha,
        'num_bootstrap': args.num_bootstrap,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    experiment.save_config(config)

    # Run bootstrap for each dataset
    for dataset in args.datasets:
        experiment.run_bootstrap(
            dataset_name=dataset,
            n_realizations=n_realizations,
            start_seed=args.start_seed
        )

    # Final save
    experiment.save_results()

    print("\n" + "="*60)
    print("Bootstrap Experiment Complete!")
    print(f"Results saved to: {experiment.output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Run analysis script:")
    print(f"   python examples/robustness/analyze_bootstrap_ci.py --results-dir {args.output_dir}")

if __name__ == "__main__":
    main()