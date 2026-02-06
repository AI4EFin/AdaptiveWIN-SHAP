"""
Comprehensive benchmarking of SHAP explanation methods on time series data.

Compares:
1. Vanilla SHAP (GlobalSHAP) - Vanilla kernel SHAP on a single global model
2. RollingWindowSHAP - Fixed rolling window approach
3. AdaptiveWinShap - Adaptive window detection and SHAP

Evaluates using perturbation and sequence analysis metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarking import (
    GlobalSHAP,
    RollingWindowSHAP,
    TimeShapWrapper
)
from adaptivewinshap import AdaptiveWinShap, AdaptiveLSTM


def run_benchmark(dataset_path, output_dir, device='cpu', verbose=True,
                  dataset_type='simulated', column_name='N', date_start=None, date_end=None,
                  seq_length=3, hidden_size=16, num_layers=1, dropout=0.0, epochs=15,
                  batch_size=64, lr=1e-2, max_background=100, shap_nsamples=500,
                  rolling_window_size=100, rolling_stride=1, precomputed_windows_path=None,
                  rolling_mean_window=75):
    """
    Run comprehensive benchmark comparing different SHAP methods.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset CSV file
    output_dir : str
        Directory to save results
    device : str
        Device for PyTorch models
    verbose : bool
        Print progress information
    dataset_type : str
        Type of dataset ('simulated' or 'empirical')
    column_name : str
        Name of the data column in the CSV
    date_start : str or pd.Timestamp, optional
        Start date for empirical datasets
    date_end : str or pd.Timestamp, optional
        End date for empirical datasets
    seq_length : int
        Sequence length for LSTM
    hidden_size : int
        Hidden size for LSTM
    num_layers : int
        Number of LSTM layers
    dropout : float
        Dropout rate
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    max_background : int
        Maximum background samples for SHAP
    shap_nsamples : int
        Number of SHAP samples
    rolling_window_size : int
        Window size for rolling SHAP
    rolling_stride : int
        Stride for rolling windows
    precomputed_windows_path : str, optional
        Path to pre-computed adaptive windows
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if verbose:
        print(f"Loading dataset from {dataset_path}...")

    if dataset_type == 'empirical':
        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

        # Apply date filtering if specified
        if date_start is not None and date_end is not None:
            if verbose:
                print(f"Filtering data from {date_start} to {date_end}...")
            df = df.loc[date_start:date_end]

        data = df[column_name].to_numpy(dtype=np.float32)
    else:
        df = pd.read_csv(dataset_path)
        data = df[column_name].to_numpy(dtype=np.float64)

    # Load covariates if present (columns starting with Z_)
    covariates = None
    covariate_cols = [c for c in df.columns if c.startswith('Z_')]
    if covariate_cols:
        covariates = df[covariate_cols].to_numpy(dtype=np.float32)
        if verbose:
            print(f"Found {len(covariate_cols)} covariate(s): {covariate_cols}")

    if verbose:
        if covariates is not None:
            print(f"Dataset loaded: {len(data)} time points with {covariates.shape[1]} covariates")
        else:
            print(f"Dataset loaded: {len(data)} time points (no covariates)")

    # Hyperparameters (now passed as arguments)
    SEQ_LENGTH = seq_length
    HIDDEN_SIZE = hidden_size
    NUM_LAYERS = num_layers
    DROPOUT = dropout
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LR = lr
    MAX_BACKGROUND = max_background
    SHAP_NSAMPLES = shap_nsamples
    ROLLING_WINDOW_SIZE = rolling_window_size
    ROLLING_STRIDE = rolling_stride

    results = {}

    # ========== Method 1: Vanilla SHAP ==========
    if verbose:
        print("\n" + "="*60)
        print("Method 1: Vanilla SHAP (Vanilla Kernel SHAP on Global Model)")
        print("="*60)

    global_shap = GlobalSHAP(
        seq_length=SEQ_LENGTH,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=device,
        max_background=MAX_BACKGROUND,
        shap_nsamples=SHAP_NSAMPLES
    )

    if verbose:
        print("Training global model...")
    global_shap.fit(data, covariates=covariates, verbose=verbose)

    if verbose:
        print("Computing SHAP values...")
    global_results = global_shap.explain(data, covariates=covariates)
    global_results.to_csv(os.path.join(output_dir, 'global_shap_results.csv'), index=False)

    if verbose:
        print(f"Vanilla SHAP computed for {len(global_results)} time points")
        print("Faithfulness and ablation scores computed inline during SHAP computation")

    # ========== Method 1b: TimeShap ==========
    if verbose:
        print("\n" + "="*60)
        print("Method 1b: TimeShap (Optional - if library available)")
        print("="*60)

    try:
        timeshap_wrapper = TimeShapWrapper(
            seq_length=SEQ_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            device=device
        )

        if verbose:
            print("Training TimeShap model...")
        timeshap_wrapper.fit(data, covariates=covariates, verbose=verbose)

        if verbose:
            print("Computing TimeShap explanations...")
        timeshap_results = timeshap_wrapper.explain(data, covariates=covariates)
        timeshap_results.to_csv(os.path.join(output_dir, 'timeshap_results.csv'), index=False)

        if verbose:
            print(f"TimeShap computed for {len(timeshap_results)} time points")
            print("Faithfulness and ablation scores computed inline during SHAP computation")
    except ImportError as e:
        if verbose:
            print(f"TimeShap library not available: {e}")
            print("Skipping TimeShap method.")
    except NotImplementedError as e:
        if verbose:
            print(f"TimeShap not yet fully implemented: {e}")
            print("Skipping TimeShap method.")
    except Exception as e:
        if verbose:
            print(f"Error running TimeShap: {e}")
            print("Skipping TimeShap method.")

    # ========== Method 2: Rolling Window SHAP (Multiple Window Sizes) ==========
    if verbose:
        print("\n" + "="*60)
        print("Method 2: Rolling Window SHAP (Multiple Window Sizes)")
        print("="*60)

    # Determine window sizes to use
    rolling_window_configs = [
        (ROLLING_WINDOW_SIZE, 'rolling', 'Fixed Window (100)')
    ]

    # Add adaptive-based window sizes if available
    if precomputed_windows_path and os.path.exists(precomputed_windows_path):
        windows_df = pd.read_csv(precomputed_windows_path)
        if 'window_mean' in windows_df.columns:
            window_sizes = windows_df['window_mean'].dropna()
            if len(window_sizes) > 0:
                max_window = int(window_sizes.max())
                mean_window = int(window_sizes.mean())

                rolling_window_configs.extend([
                    (max_window, 'max', f'Max Adaptive Window ({max_window})'),
                    (mean_window, 'mean', f'Mean Adaptive Window ({mean_window})')
                ])

                if verbose:
                    print(f"Adaptive window statistics:")
                    print(f"  Max: {max_window}")
                    print(f"  Mean: {mean_window}")

    # Run rolling SHAP for each window size configuration
    for window_size, suffix, description in rolling_window_configs:
        if verbose:
            print(f"\nComputing rolling SHAP with {description}...")

        rolling_shap = RollingWindowSHAP(
            seq_length=SEQ_LENGTH,
            window_size=window_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            device=device,
            max_background=MAX_BACKGROUND,
            shap_nsamples=SHAP_NSAMPLES
        )

        rolling_results = rolling_shap.rolling_explain(data, covariates=covariates, stride=ROLLING_STRIDE, verbose=verbose)

        # Save with appropriate suffix - use adaptive_shap for max/mean, rolling_shap for fixed
        if suffix == 'rolling':
            output_filename = 'rolling_shap_results.csv'
        else:
            # max and mean use adaptive window sizes, so name them as adaptive_shap
            output_filename = f'adaptive_shap_{suffix}_results.csv'
        rolling_results.to_csv(os.path.join(output_dir, output_filename), index=False)

        if verbose:
            print(f"Rolling SHAP ({description}) computed for {len(rolling_results)} windows")
            print("Faithfulness and ablation scores computed inline during SHAP computation")

    # ========== Method 3: Adaptive SHAP ==========
    if verbose:
        print("\n" + "="*60)
        print("Method 3: Adaptive SHAP (Pre-computed Adaptive Windows)")
        print("="*60)

    if verbose and precomputed_windows_path:
        print(f"Loading pre-computed windows from {precomputed_windows_path}...")

    if precomputed_windows_path is None or not os.path.exists(precomputed_windows_path):
        if precomputed_windows_path:
            print(f"ERROR: Pre-computed windows file not found: {precomputed_windows_path}")
        else:
            print("WARNING: No pre-computed windows path provided.")
        print("Skipping Adaptive SHAP method.")
    else:
        detection_df = pd.read_csv(precomputed_windows_path)

        if verbose:
            print(f"Loaded {len(detection_df)} window sizes")
            if 'window_mean' in detection_df.columns:
                print(f"Window sizes: min={detection_df['window_mean'].min():.0f}, "
                      f"max={detection_df['window_mean'].max():.0f}, "
                      f"mean={detection_df['window_mean'].mean():.0f}")
            else:
                print(f"Available columns: {list(detection_df.columns)}")

        # Now compute SHAP with adaptive windows
        if verbose:
            print("Computing SHAP with adaptive windows...")

        # Save windows to temporary CSV for AdaptiveWinShap
        temp_windows_csv = os.path.join(output_dir, 'temp_windows.csv')
        if 'window_mean' in detection_df.columns:
            detection_df[['window_mean']].rename(columns={'window_mean': 'windows'}).to_csv(temp_windows_csv, index=False)
        else:
            # If window_mean doesn't exist, try to find windows column
            windows_col = [c for c in detection_df.columns if 'window' in c.lower()][0]
            detection_df[[windows_col]].rename(columns={windows_col: 'windows'}).to_csv(temp_windows_csv, index=False)

        adaptive_runner = AdaptiveWinShap(
            seq_length=SEQ_LENGTH,
            lstm_hidden=HIDDEN_SIZE,
            lstm_layers=NUM_LAYERS,
            lstm_dropout=DROPOUT,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LR,
            max_background=MAX_BACKGROUND,
            shap_nsamples=SHAP_NSAMPLES,
            aggregate_lag=True
        )

        # Prepare data with covariates if available
        if covariates is not None:
            # Concatenate target + covariates: shape [T, 1+n_cov]
            adaptive_data = np.column_stack([data.astype(np.float32)[:, None], covariates])
        else:
            adaptive_data = data.astype(np.float32)

        adaptive_results = adaptive_runner.rolling_explain(
            data=adaptive_data,
            window_sizes_csv=temp_windows_csv,
            csv_column='windows'
        )
        adaptive_results.to_csv(os.path.join(output_dir, 'adaptive_shap_results.csv'), index=False)

        if verbose:
            print(f"Adaptive SHAP computed for {len(adaptive_results)} windows")
            print("Faithfulness and ablation scores computed inline during SHAP computation")

        # ========== Method 3b: Adaptive SHAP with Rolling Mean Windows ==========
        if verbose:
            print("\n" + "="*60)
            print("Method 3b: Adaptive SHAP (Rolling Mean of Adaptive Windows)")
            print("="*60)

        # Use the rolling_mean_window parameter passed to the function

        if 'window_mean' in detection_df.columns:
            window_series = detection_df['window_mean']
        else:
            windows_col = [c for c in detection_df.columns if 'window' in c.lower()][0]
            window_series = detection_df[windows_col]

        # Compute rolling mean with center=True for smoother transitions
        smoothed_windows = window_series.rolling(
            window=rolling_mean_window,
            center=True,
            min_periods=1
        ).mean()

        if verbose:
            print(f"Smoothing window sizes with rolling mean (window={rolling_mean_window})...")
            print(f"Original windows: min={window_series.min():.0f}, max={window_series.max():.0f}, mean={window_series.mean():.0f}")
            print(f"Smoothed windows: min={smoothed_windows.min():.0f}, max={smoothed_windows.max():.0f}, mean={smoothed_windows.mean():.0f}")

        # Save smoothed windows to temporary CSV
        temp_smoothed_csv = os.path.join(output_dir, 'temp_windows_smoothed.csv')
        pd.DataFrame({'windows': smoothed_windows}).to_csv(temp_smoothed_csv, index=False)

        adaptive_smoothed_runner = AdaptiveWinShap(
            seq_length=SEQ_LENGTH,
            lstm_hidden=HIDDEN_SIZE,
            lstm_layers=NUM_LAYERS,
            lstm_dropout=DROPOUT,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LR,
            max_background=MAX_BACKGROUND,
            shap_nsamples=SHAP_NSAMPLES,
            aggregate_lag=True
        )

        adaptive_smoothed_results = adaptive_smoothed_runner.rolling_explain(
            data=adaptive_data,
            window_sizes_csv=temp_smoothed_csv,
            csv_column='windows'
        )
        adaptive_smoothed_results.to_csv(os.path.join(output_dir, 'adaptive_shap_rolling_mean_results.csv'), index=False)

        if verbose:
            print(f"Adaptive SHAP (rolling mean) computed for {len(adaptive_smoothed_results)} windows")
            print("Faithfulness and ablation scores computed inline during SHAP computation")

    # ========== Aggregate Pre-computed Faithfulness and Ablation Scores ==========
    if verbose:
        print("\n" + "="*60)
        print("Summary of Results")
        print("="*60)
        print("Aggregating pre-computed faithfulness and ablation scores from CSV files...")

    # Read pre-computed faithfulness and ablation scores from CSV files
    summary_data = []

    # Helper function to extract metrics columns
    def extract_metrics(df, method_name):
        """Extract and aggregate faithfulness and ablation metrics from a dataframe."""
        # Find all faithfulness and ablation columns
        faith_cols = [c for c in df.columns if c.startswith('faithfulness_')]
        ablation_cols = [c for c in df.columns if c.startswith('ablation_')]

        if not faith_cols and not ablation_cols:
            if verbose:
                print(f"  Warning: No metrics columns found in {method_name}")
            return []

        rows = []

        # Process faithfulness columns
        faith_eval_types = set()
        for col in faith_cols:
            # Extract evaluation type: faithfulness_prtb_p90 -> prtb_p90
            eval_type = col.replace('faithfulness_', '')
            faith_eval_types.add(eval_type)

        for eval_type in sorted(faith_eval_types):
            col_name = f'faithfulness_{eval_type}'
            if col_name in df.columns:
                # Compute mean of this metric across all time points
                mean_value = df[col_name].mean()
                rows.append({
                    'method': method_name,
                    'metric_type': 'faithfulness',
                    'evaluation': eval_type,
                    'score': mean_value
                })

        # Process ablation columns
        ablation_eval_types = set()
        for col in ablation_cols:
            # Extract evaluation type: ablation_mif_p90 -> mif_p90
            eval_type = col.replace('ablation_', '')
            ablation_eval_types.add(eval_type)

        for eval_type in sorted(ablation_eval_types):
            col_name = f'ablation_{eval_type}'
            if col_name in df.columns:
                # Compute mean of this metric across all time points
                mean_value = df[col_name].mean()
                rows.append({
                    'method': method_name,
                    'metric_type': 'ablation',
                    'evaluation': eval_type,
                    'score': mean_value
                })

        return rows

    # Process Vanilla SHAP
    global_path = os.path.join(output_dir, 'global_shap_results.csv')
    if os.path.exists(global_path):
        global_df = pd.read_csv(global_path)
        summary_data.extend(extract_metrics(global_df, 'global_shap'))
        if verbose:
            print(f"  Vanilla SHAP: {len(global_df)} time points")

    # Process TimeShap
    timeshap_path = os.path.join(output_dir, 'timeshap_results.csv')
    if os.path.exists(timeshap_path):
        timeshap_df = pd.read_csv(timeshap_path)
        summary_data.extend(extract_metrics(timeshap_df, 'timeshap'))
        if verbose:
            print(f"  TimeShap: {len(timeshap_df)} time points")

    # Process Rolling SHAP (Fixed window)
    rolling_path = os.path.join(output_dir, 'rolling_shap_results.csv')
    if os.path.exists(rolling_path):
        rolling_df = pd.read_csv(rolling_path)
        summary_data.extend(extract_metrics(rolling_df, 'rolling_shap'))
        if verbose:
            print(f"  Rolling SHAP (Fixed): {len(rolling_df)} windows")

    # Process Adaptive SHAP variants (Max and Mean window sizes)
    for suffix, name in [('max', 'Max'), ('mean', 'Mean')]:
        adaptive_path = os.path.join(output_dir, f'adaptive_shap_{suffix}_results.csv')
        if os.path.exists(adaptive_path):
            adaptive_df = pd.read_csv(adaptive_path)
            method_name = f'adaptive_shap_{suffix}'
            summary_data.extend(extract_metrics(adaptive_df, method_name))
            if verbose:
                print(f"  Adaptive SHAP ({name}): {len(adaptive_df)} windows")

    # Process Adaptive SHAP
    adaptive_path = os.path.join(output_dir, 'adaptive_shap_results.csv')
    if os.path.exists(adaptive_path):
        adaptive_df = pd.read_csv(adaptive_path)
        summary_data.extend(extract_metrics(adaptive_df, 'adaptive_shap'))
        if verbose:
            print(f"  Adaptive SHAP: {len(adaptive_df)} windows")

    # Process Adaptive SHAP (Rolling Mean)
    adaptive_rm_path = os.path.join(output_dir, 'adaptive_shap_rolling_mean_results.csv')
    if os.path.exists(adaptive_rm_path):
        adaptive_rm_df = pd.read_csv(adaptive_rm_path)
        summary_data.extend(extract_metrics(adaptive_rm_df, 'adaptive_shap_rolling_mean'))
        if verbose:
            print(f"  Adaptive SHAP (Rolling Mean): {len(adaptive_rm_df)} windows")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'benchmark_summary.csv'), index=False)

    if verbose:
        print("\nBenchmark Summary:")
        print(summary_df.to_string(index=False))
        print(f"\nAll results saved to: {output_dir}")

    # Save configuration
    config = {
        'dataset': dataset_path,
        'seq_length': SEQ_LENGTH,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'max_background': MAX_BACKGROUND,
        'shap_nsamples': SHAP_NSAMPLES,
        'rolling_window_size': ROLLING_WINDOW_SIZE,
        'rolling_stride': ROLLING_STRIDE,
        'rolling_mean_window': rolling_mean_window,
        'device': device
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return summary_df


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run SHAP methods benchmark')
    parser.add_argument('--dataset', type=str, default='piecewise_ar3',
                        help='Dataset name (default: piecewise_ar3)')
    parser.add_argument('--data-type', type=str, default='simulated',
                        choices=['simulated', 'empirical'],
                        help='Dataset type (default: simulated)')
    parser.add_argument('--n0', type=int, default=75,
                        help='N0 value used in window detection (default: 75)')
    parser.add_argument('--jump', type=int, default=1,
                        help='Jump value used in window detection (default: 1)')
    parser.add_argument('--rolling-mean-window', type=int, default=75,
                        help='Window size for rolling mean smoothing of adaptive windows (default: 75)')
    parser.add_argument('--growth', type=str, default='geometric',
                        help='Window growth strategy (default: geometric)')
    parser.add_argument('--penalty-factor', type=float, default=0.05,
                        help='Spokoiny penalty factor lambda used in window detection (default: 0.05)')
    args = parser.parse_args()

    # Configuration
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
        DEVICE = "mps"

    if args.data_type == "simulated":
        # ========== Simulated Dataset Configuration ==========
        dataset_name = args.dataset

        # Dataset-specific configurations (must match lstm_simulation.py)
        DATASET_CONFIGS = {
            "piecewise_ar3": {"seq_length": 3, "n_covariates": 0},  # AR(3)
            "arx_rotating": {"seq_length": 3, "n_covariates": 3},   # AR(3) + 3 covariates (D,F,R)
            "trend_season": {"seq_length": 3, "n_covariates": 0},   # AR(3) with structural break
            "spike_process": {"seq_length": 3, "n_covariates": 2},  # AR(3) + 2 covariates (D,R) + spikes
            "tvp_arx": {"seq_length": 3, "n_covariates": 2},        # AR(3) + 2 covariates (Z1,Z2), time-varying
            "garch_regime": {"seq_length": 1, "n_covariates": 2},   # GARCH(1,1) + 2 factors (M,V)
            "cointegration": {"seq_length": 1, "n_covariates": 4},  # Cointegration: 2 important (X1,X2) + 2 noise (X3,X4)
        }

        # Get dataset-specific configuration
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

        config = DATASET_CONFIGS[dataset_name]
        SEQ_LENGTH = config["seq_length"]

        # Paths
        dataset_path = f"examples/datasets/simulated/{dataset_name}/data.csv"
        output_dir = f"examples/results/benchmark_{dataset_name}/N0_{args.n0}_lambda_{args.penalty_factor}"

        # Hyperparameters for simulated data
        HIDDEN_SIZE = 16
        NUM_LAYERS = 1
        DROPOUT = 0.0
        EPOCHS = 15
        BATCH_SIZE = 64
        LR = 1e-2
        MAX_BACKGROUND = 100
        SHAP_NSAMPLES = 500
        ROLLING_WINDOW_SIZE = 100
        ROLLING_STRIDE = 1

        # Pre-computed windows path (includes lambda parameter)
        PRECOMPUTED_WINDOWS = f"examples/results/LSTM/{dataset_name}/Jump_{args.jump}_N0_{args.n0}_lambda_{args.penalty_factor}/windows.csv"

        print("="*60)
        print("SHAP Methods Benchmarking - Simulated Data")
        print("="*60)
        print(f"Device: {DEVICE}")
        print(f"Dataset: {dataset_name}")
        print(f"Dataset config: seq_length={SEQ_LENGTH}, n_covariates={config['n_covariates']}")
        print(f"LPA params: N0={args.n0}, Jump={args.jump}, Lambda={args.penalty_factor}")
        print(f"Dataset path: {dataset_path}")
        print(f"Windows path: {PRECOMPUTED_WINDOWS}")
        print(f"Output: {output_dir}")
        print("="*60)

        # Run benchmark
        summary = run_benchmark(
            dataset_path=dataset_path,
            output_dir=output_dir,
            device=DEVICE,
            verbose=True,
            dataset_type='simulated',
            column_name='N',
            seq_length=SEQ_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_background=MAX_BACKGROUND,
            shap_nsamples=SHAP_NSAMPLES,
            rolling_window_size=ROLLING_WINDOW_SIZE,
            rolling_stride=ROLLING_STRIDE,
            precomputed_windows_path=PRECOMPUTED_WINDOWS,
            rolling_mean_window=args.rolling_mean_window
        )

    elif args.data_type == "empirical":
        # ========== Empirical Dataset Configuration ==========
        # Paths
        dataset_path = "examples/datasets/empirical/dataset.csv"
        output_dir = "examples/results/benchmark_empirical"

        # Date range for filtering (matching lstm_empirical.py)
        DATE_START = pd.Timestamp("2021-05-01 00:00:00", tz="Europe/Bucharest")
        DATE_END = pd.Timestamp("2021-08-01 00:00:00", tz="Europe/Bucharest")

        # Hyperparameters for empirical data (matching lstm_empirical.py)
        SEQ_LENGTH = 24  # 1 day (hourly data)
        HIDDEN_SIZE = 16
        NUM_LAYERS = 1
        DROPOUT = 0.0
        EPOCHS = 15
        BATCH_SIZE = 64
        LR = 1e-2
        MAX_BACKGROUND = 500
        SHAP_NSAMPLES = 500
        ROLLING_WINDOW_SIZE = 168  # 1 week
        ROLLING_STRIDE = 1

        # Pre-computed windows path (matching lstm_empirical.py output)
        JUMP = 1
        N_0 = 72
        PRECOMPUTED_WINDOWS = f"examples/results/LSTM/empirical/Jump_{JUMP}_N0_{N_0}_lambda_{args.penalty_factor}/windows.csv"

        print("="*60)
        print("SHAP Methods Benchmarking - Empirical Data")
        print("="*60)
        print(f"Device: {DEVICE}")
        print(f"Dataset: {dataset_path}")
        print(f"Date Range: {DATE_START} to {DATE_END}")
        print(f"LPA params: N0={N_0}, Jump={JUMP}, Lambda={args.penalty_factor}")
        print(f"Windows path: {PRECOMPUTED_WINDOWS}")
        print(f"Output: {output_dir}")
        print("="*60)

        # Run benchmark
        summary = run_benchmark(
            dataset_path=dataset_path,
            output_dir=output_dir,
            device=DEVICE,
            verbose=True,
            dataset_type='empirical',
            column_name='Price Day Ahead',
            date_start=DATE_START,
            date_end=DATE_END,
            seq_length=SEQ_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_background=MAX_BACKGROUND,
            shap_nsamples=SHAP_NSAMPLES,
            rolling_window_size=ROLLING_WINDOW_SIZE,
            rolling_stride=ROLLING_STRIDE,
            precomputed_windows_path=PRECOMPUTED_WINDOWS,
            rolling_mean_window=args.rolling_mean_window
        )

    print("\n" + "="*60)
    print("Benchmarking Complete!")
    print("="*60)
