"""
AdaptiveWIN-SHAP Window Detection Example

This script demonstrates how to use the AdaptiveWIN-SHAP library with
Monte Carlo pre-computed critical values for change point detection.
"""
import glob
import os
import timeit

import numpy as np
import torch
import pandas as pd

from adaptivewinshap import AdaptiveLSTM, ChangeDetector, AdaptiveWinShap


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run AdaptiveWIN-SHAP window detection')
    parser.add_argument('--dataset', type=str, default='piecewise_ar3',
                        help='Dataset name (default: piecewise_ar3)')
    parser.add_argument('--n0', type=int, default=100,
                        help='Initial window size (default: 100)')
    parser.add_argument('--jump', type=int, default=1,
                        help='Jump size for detection (default: 1)')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size for detection. Represents how many points we should skip in J_k for faster computation (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.95,
                        help='The quantile of the critical value distribution (default: 0.95)')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of detection runs (default: 1)')
    parser.add_argument('--growth', type=str, default='geometric',
                        help='Window growth strategy (default: geometric)')
    parser.add_argument('--growth-base', type=float, default=1.41421356237,
                        help='Base for geometric growth (default: ~np.sqrt(2)')
    # Monte Carlo critical value arguments
    parser.add_argument('--mc-reps', type=int, default=300,
                        help='Number of Monte Carlo replications for CV computation (default: 100)')
    parser.add_argument('--penalty-factor', type=float, default=0.05,
                        help='Spokoiny penalty factor lambda for CV adjustment (default: 0.05)')
    parser.add_argument('--cv-file', type=str, default=None,
                        help='Path to pre-computed critical values CSV (skip computation if provided)')
    parser.add_argument('--save-cv', action='store_true',
                        help='Save computed critical values to file')
    parser.add_argument('--video-format', type=str, default='mp4', choices=['mp4', 'webm', 'gif', 'png'],
                        help='Video format for animation (default: mp4). Use webm for transparent background.')
    args = parser.parse_args()

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"

    # ============================================================
    # DATASET-SPECIFIC CONFIGURATIONS
    # ============================================================
    # Each dataset has specific structure: (seq_length, n_covariates)
    # seq_length: number of lags in the AR process
    # n_covariates: number of exogenous covariates
    # input_size = 1 (target) + n_covariates
    DATASET_CONFIGS = {
        "piecewise_ar3": {"seq_length": 3, "n_covariates": 0},  # AR(3)
        "arx_rotating": {"seq_length": 3, "n_covariates": 3},   # AR(3) + 3 covariates (D,F,R)
        "trend_season": {"seq_length": 3, "n_covariates": 0},   # AR(3) with structural break
        "spike_process": {"seq_length": 3, "n_covariates": 2},  # AR(3) + 2 covariates (D,R) + spikes
        "tvp_arx": {"seq_length": 3, "n_covariates": 2},        # AR(3) + 2 covariates (Z1,Z2), time-varying
        "switching_factor": {"seq_length": 1, "n_covariates": 3},  # Factor model + 3 factors (Market, Supply, Credit)
        "cointegration": {"seq_length": 1, "n_covariates": 4},  # Cointegration: 2 important (X1,X2) + 2 noise (X3,X4)
    }

    # LSTM hyperparameters (same for all datasets)
    LSTM_HIDDEN = 16
    LSTM_LAYERS = 1
    LSTM_EPOCHS = 15
    LSTM_BATCH = 64
    LSTM_LR = 1e-2
    LSTM_DROPOUT = 0.0

    DATASET = args.dataset

    # Load dataset
    if "/" in DATASET:
        # Legacy format: "ar/3"
        dataset_type, order = DATASET.split("/")
        dataset_path = f"examples/datasets/simulated/{dataset_type}/{order}.csv"
        dataset_name = f"{dataset_type}_{order}"
    else:
        # New format: "piecewise_ar3", "arx_rotating", etc.
        dataset_path = f"examples/datasets/simulated/{DATASET}/data.csv"
        dataset_name = DATASET

    print("="*60)
    print(f"AdaptiveWIN-SHAP Window Detection - {dataset_name}")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_path}")
    print("="*60)

    # Get dataset-specific configuration
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    LSTM_SEQ_LEN = config["seq_length"]
    n_covariates = config["n_covariates"]
    LSTM_INPUT_SIZE = 1 + n_covariates  # target + covariates

    print(f"Dataset config: seq_length={LSTM_SEQ_LEN}, n_covariates={n_covariates}, input_size={LSTM_INPUT_SIZE}")
    print("="*60)

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Extract target and covariates
    target = df["N"].to_numpy(dtype=np.float64)

    if n_covariates > 0:
        # Get covariate columns in order: Z_0, Z_1, Z_2, ...
        cov_cols = [f"Z_{i}" for i in range(n_covariates)]
        covariates = df[cov_cols].to_numpy(dtype=np.float64)

        # Combine target and covariates: [T, 1+n_covariates]
        data = np.column_stack([target, covariates])
        print(f"Data shape: {data.shape} (target + {n_covariates} covariates)")
    else:
        # Univariate case
        data = target
        print(f"Data shape: {data.shape} (univariate)")

    # Initialize model with dataset-specific parameters
    model = AdaptiveLSTM(
        DEVICE,
        seq_length=LSTM_SEQ_LEN,
        input_size=LSTM_INPUT_SIZE,
        hidden=LSTM_HIDDEN,
        layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
        batch_size=LSTM_BATCH,
        lr=LSTM_LR,
        epochs=LSTM_EPOCHS,
        type_precision=np.float64
    )

    cd = ChangeDetector(model, data, debug=False, force_cpu=True)

    # Use command-line arguments or defaults
    MIN_SEG = 4
    N_0 = args.n0
    JUMP = args.jump
    STEP = args.step
    ALPHA = args.alpha

    out_dir = os.path.join("examples", f"results/LSTM/{dataset_name}/Jump_{JUMP}_N0_{N_0}_lambda_{args.penalty_factor}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Parameters: N_0={N_0}, JUMP={JUMP}, NUM_RUNS={args.num_runs}, GROWTH_BASE={args.growth_base}")
    print(f"MC parameters: mc_reps={args.mc_reps}, penalty_factor={args.penalty_factor}")
    print(f"Video format: {args.video_format}" + (" (transparent)" if args.video_format == 'webm' else ""))
    print()

    # ============================================================
    # Critical Value Computation or Loading
    # ============================================================
    cv_path = os.path.join(out_dir, "critical_values.csv")

    if args.cv_file:
        # Load pre-computed critical values
        print(f"Loading critical values from: {args.cv_file}")
        cd.load_critical_values(args.cv_file)
    elif os.path.exists(cv_path) and not args.save_cv:
        # Load existing CV file
        print(f"Loading existing critical values from: {cv_path}")
        cd.load_critical_values(cv_path)
    else:
        # Compute critical values
        print("Computing Monte Carlo critical values...")
        cd.precompute_critical_values(
            data=data,
            n_0=N_0,
            mc_reps=args.mc_reps,
            alpha=ALPHA,
            search_step=STEP,
            min_seg=MIN_SEG,
            penalty_factor=args.penalty_factor,
            growth_base=args.growth_base,
            verbose=True
        )

        if args.save_cv or True:  # Always save for reproducibility
            cd.save_critical_values(cv_path)

    # ============================================================
    # Window Size Detection
    # ============================================================
    print("\nStarting window size detection...")
    num_runs = args.num_runs
    for run in range(num_runs):
        print(f"\nRun {run}")
        out_csv = os.path.join(out_dir, f"run_{run}.csv")

        # Determine video save path based on format
        if args.video_format == 'png':
            video_path = f"{out_dir}/frames/run_{run}.png"  # Creates run_0_0001.png, etc.
        else:
            video_path = f"{out_dir}/run_{run}.{args.video_format}"

        results = cd.detect(
            min_window=MIN_SEG,
            n_0=N_0,
            jump=JUMP,
            search_step=STEP,
            alpha=ALPHA,
            t_workers=10,
            save_path=video_path,
            growth=args.growth,
            growth_base=args.growth_base
        )

        pd.DataFrame(results).to_csv(out_csv)
        print(f"Saved results to: {out_csv}")

    # Aggregate windows from multiple runs
    print("\nAggregating window sizes...")
    all_files = glob.glob(os.path.join(out_dir, "run*.csv"))

    dfs = []
    for file in all_files:
        # Read only the column you care about
        win_df = pd.read_csv(file, usecols=["windows"])

        # Rename the column to something unique (e.g., filename without extension)
        name = os.path.splitext(os.path.basename(file))[0]
        win_df = win_df.rename(columns={"windows": f"windows_{name}"})

        dfs.append(win_df)

    windows_df = pd.concat(dfs, axis=1)
    windows_df["window_mean"] = windows_df.mean(axis=1)
    windows_df.to_csv(f"{out_dir}/windows.csv")
    print(f"Saved aggregated windows to: {out_dir}/windows.csv")

    print("\n" + "="*60)
    print("Window detection complete!")
    print("="*60)
    print(f"Results saved to: {out_dir}")
    print(f"Critical values saved to: {cv_path}")
    print(f"Use these windows for benchmarking with benchmark.py")
    print("="*60)
