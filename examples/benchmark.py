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
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarking import (
    GlobalSHAP,
    RollingWindowSHAP,
    create_sequences
)
from adaptivewinshap import AdaptiveWinShap, ChangeDetector, AdaptiveModel, store_init_kwargs


class AdaptiveLSTM(AdaptiveModel):
    """LSTM model for AdaptiveWinShap."""

    @store_init_kwargs
    def __init__(self, device, seq_length=3, input_size=1, hidden=16, layers=1,
                 dropout=0.2, batch_size=512, lr=1e-12, epochs=50, type_precision=np.float32):
        import torch.nn as nn
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
        L = self.seq_length
        n = len(window)
        if n <= L:
            return None, None
        X = np.lib.stride_tricks.sliding_window_view(window, L + 1)
        X, y = X[:, :-1], X[:, -1]
        X = X[..., None].astype(np.float32)
        y = y.astype(np.float32)
        t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        return X_tensor, y_tensor, t_abs

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        preds = self(xt)
        return preds.detach().cpu().numpy().reshape(-1, 1)


def run_benchmark(dataset_path, output_dir, device='cpu', verbose=True):
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
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if verbose:
        print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    data = df['N'].to_numpy(dtype=np.float64)

    if verbose:
        print(f"Dataset loaded: {len(data)} time points")

    # Hyperparameters
    SEQ_LENGTH = 3
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
    global_shap.fit(data, verbose=verbose)

    if verbose:
        print("Computing SHAP values...")
    global_results = global_shap.explain(data)
    global_results.to_csv(os.path.join(output_dir, 'global_shap_results.csv'), index=False)

    if verbose:
        print(f"Vanilla SHAP computed for {len(global_results)} time points")
        print("Faithfulness scores computed inline during SHAP computation")

    # ========== Method 2: Rolling Window SHAP ==========
    if verbose:
        print("\n" + "="*60)
        print("Method 2: Rolling Window SHAP (Fixed Window Size)")
        print("="*60)

    rolling_shap = RollingWindowSHAP(
        seq_length=SEQ_LENGTH,
        window_size=ROLLING_WINDOW_SIZE,
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
        print(f"Computing rolling SHAP with window size {ROLLING_WINDOW_SIZE}, stride {ROLLING_STRIDE}...")
    rolling_results = rolling_shap.rolling_explain(data, stride=ROLLING_STRIDE, verbose=verbose)
    rolling_results.to_csv(os.path.join(output_dir, 'rolling_shap_results.csv'), index=False)

    if verbose:
        print(f"Rolling SHAP computed for {len(rolling_results)} windows")
        print("Faithfulness scores computed inline during SHAP computation")

    # ========== Method 3: Adaptive SHAP ==========
    if verbose:
        print("\n" + "="*60)
        print("Method 3: Adaptive SHAP (Pre-computed Adaptive Windows)")
        print("="*60)

    # Use pre-computed windows
    precomputed_windows_path = "examples/results/LSTM/ar_3/Jump_1_N0_100/windows.csv"

    if verbose:
        print(f"Loading pre-computed windows from {precomputed_windows_path}...")

    if not os.path.exists(precomputed_windows_path):
        print(f"ERROR: Pre-computed windows file not found: {precomputed_windows_path}")
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

        adaptive_results = adaptive_runner.rolling_explain(
            data=data.astype(np.float32),
            window_sizes_csv=temp_windows_csv,
            csv_column='windows'
        )
        adaptive_results.to_csv(os.path.join(output_dir, 'adaptive_shap_results.csv'), index=False)

        if verbose:
            print(f"Adaptive SHAP computed for {len(adaptive_results)} windows")
            print("Faithfulness scores computed inline during SHAP computation")

    # ========== Aggregate Pre-computed Faithfulness Scores ==========
    if verbose:
        print("\n" + "="*60)
        print("Summary of Results")
        print("="*60)
        print("Aggregating pre-computed faithfulness scores from CSV files...")

    # Read pre-computed faithfulness scores from CSV files
    summary_data = []

    # Helper function to extract faithfulness columns
    def extract_faithfulness_metrics(df, method_name):
        """Extract and aggregate faithfulness metrics from a dataframe."""
        # Find all faithfulness columns
        faith_cols = [c for c in df.columns if c.startswith('faithfulness_')]

        if not faith_cols:
            if verbose:
                print(f"  Warning: No faithfulness columns found in {method_name}")
            return []

        # Group by evaluation type (e.g., 'prtb_p90', 'sqnc_p50')
        eval_types = set()
        for col in faith_cols:
            # Extract evaluation type: faithfulness_prtb_p90 -> prtb_p90
            eval_type = col.replace('faithfulness_', '')
            eval_types.add(eval_type)

        rows = []
        for eval_type in sorted(eval_types):
            col_name = f'faithfulness_{eval_type}'
            if col_name in df.columns:
                # Compute mean of this metric across all time points
                mean_value = df[col_name].mean()
                rows.append({
                    'method': method_name,
                    'evaluation': eval_type,
                    'faithfulness_score': mean_value
                })

        return rows

    # Process Vanilla SHAP
    global_path = os.path.join(output_dir, 'global_shap_results.csv')
    if os.path.exists(global_path):
        global_df = pd.read_csv(global_path)
        summary_data.extend(extract_faithfulness_metrics(global_df, 'global_shap'))
        if verbose:
            print(f"  Vanilla SHAP: {len(global_df)} time points")

    # Process Rolling SHAP
    rolling_path = os.path.join(output_dir, 'rolling_shap_results.csv')
    if os.path.exists(rolling_path):
        rolling_df = pd.read_csv(rolling_path)
        summary_data.extend(extract_faithfulness_metrics(rolling_df, 'rolling_shap'))
        if verbose:
            print(f"  Rolling SHAP: {len(rolling_df)} windows")

    # Process Adaptive SHAP
    adaptive_path = os.path.join(output_dir, 'adaptive_shap_results.csv')
    if os.path.exists(adaptive_path):
        adaptive_df = pd.read_csv(adaptive_path)
        summary_data.extend(extract_faithfulness_metrics(adaptive_df, 'adaptive_shap'))
        if verbose:
            print(f"  Adaptive SHAP: {len(adaptive_df)} windows")

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
        'device': device
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return summary_df


if __name__ == "__main__":
    # Configuration
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
        DEVICE = "mps"

    dataset_type = "ar"
    order = "3"

    # Paths
    dataset_path = f"examples/datasets/simulated/{dataset_type}/{order}.csv"
    output_dir = f"examples/results/benchmark_{dataset_type}_{order}"

    print("="*60)
    print("SHAP Methods Benchmarking")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Run benchmark
    summary = run_benchmark(
        dataset_path=dataset_path,
        output_dir=output_dir,
        device=DEVICE,
        verbose=True
    )

    print("\n" + "="*60)
    print("Benchmarking Complete!")
    print("="*60)
