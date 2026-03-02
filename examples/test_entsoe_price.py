"""
Quick test of the ENTSO-E Romania Day-Ahead Price pipeline.

Runs the full pipeline (download -> preprocess -> detect -> benchmark)
on only 6 months of data (~4,380 points) to verify everything works
before committing to the multi-day full run.

All outputs go to a separate _test directory so they don't collide
with the real run.

Usage:
    python examples/test_entsoe_price.py --api-key YOUR_KEY
    python examples/test_entsoe_price.py --api-key YOUR_KEY --skip-download
"""

import glob
import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptivewinshap import AdaptiveLSTM, ChangeDetector
from examples.benchmark import run_benchmark


# ============================================================
# TEST configuration -- 6 months, lower MC reps, same structure
# ============================================================
DATA_DIR = "examples/datasets/empirical/entsoe_ro_price_test"
RAW_PRICE_PATH = os.path.join(DATA_DIR, "raw_price.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "data.csv")
NORM_PARAMS_PATH = os.path.join(DATA_DIR, "normalization_params.json")

# 6-month window: 2025-07-01 to 2026-01-01
START = pd.Timestamp('20250701', tz='Europe/Bucharest')
END = pd.Timestamp('20260101', tz='Europe/Bucharest')

# LSTM -- same as production
SEQ_LENGTH = 24
INPUT_SIZE = 1
HIDDEN_SIZE = 16
NUM_LAYERS = 1
DROPOUT = 0.0
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-2

# LPA -- lower MC reps for speed, rest identical
N0 = 168
JUMP = 1
STEP = 5
ALPHA = 0.95
MC_REPS = 100        # reduced from 100 for faster test
PENALTY_FACTOR = 0.2
MIN_SEG = 4


def download_entsoe_data(api_key):
    from entsoe import EntsoePandasClient

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_PRICE_PATH):
        print(f"Cached test data found at {DATA_DIR}, skipping download.")
        return

    client = EntsoePandasClient(api_key=api_key)
    country_code = 'RO'

    print(f"[TEST] Downloading day-ahead prices for {country_code} from {START} to {END}...")
    prices = client.query_day_ahead_prices(country_code, start=START, end=END)
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices.to_csv(RAW_PRICE_PATH, header=True)
    print(f"  Saved raw prices to {RAW_PRICE_PATH} ({len(prices)} rows)")


def preprocess():
    print("\n" + "="*60)
    print("[TEST] Preprocessing")
    print("="*60)

    prices = pd.read_csv(RAW_PRICE_PATH, index_col=0, parse_dates=True).squeeze()
    prices.name = 'price'

    df = prices.to_frame()
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    n_missing_before = df.isna().sum().sum()
    df = df.ffill(limit=3)
    n_missing_after = df.isna().sum().sum()
    df = df.dropna()

    print(f"Missing values: {n_missing_before} before ffill, {n_missing_after} after ffill, "
          f"{len(df)} rows after dropna")

    price_mean = df['price'].mean()
    price_std = df['price'].std()

    df_norm = pd.DataFrame(index=df.index)
    df_norm['N'] = (df['price'] - price_mean) / price_std

    df_norm.to_csv(PROCESSED_PATH)
    print(f"Saved: {PROCESSED_PATH}  ({df_norm.shape})")

    norm_params = {
        'price_mean': float(price_mean),
        'price_std': float(price_std),
    }
    with open(NORM_PARAMS_PATH, 'w') as f:
        json.dump(norm_params, f, indent=2)

    return df_norm


def run_detection(args):
    print("\n" + "="*60)
    print("[TEST] LPA Window Detection")
    print("="*60)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    df = pd.read_csv(PROCESSED_PATH, index_col=0, parse_dates=True)
    data = df['N'].to_numpy(dtype=np.float64)

    print(f"Data shape: {data.shape} (univariate)")
    print(f"Device: {device}")

    model = AdaptiveLSTM(
        device,
        seq_length=SEQ_LENGTH,
        input_size=INPUT_SIZE,
        hidden=HIDDEN_SIZE,
        layers=NUM_LAYERS,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        type_precision=np.float64
    )

    cd = ChangeDetector(model, data, debug=False, force_cpu=True)

    out_dir = os.path.join(
        "examples",
        f"results/LSTM/entsoe_ro_price_test/Jump_{JUMP}_N0_{N0}_lambda_{PENALTY_FACTOR}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Parameters: N0={N0}, JUMP={JUMP}, STEP={STEP}, MC_REPS={MC_REPS}")

    cv_path = os.path.join(out_dir, "critical_values.csv")
    if os.path.exists(cv_path):
        print(f"Loading existing critical values from: {cv_path}")
        cd.load_critical_values(cv_path)
    else:
        print("Computing Monte Carlo critical values...")
        cd.precompute_critical_values(
            data=data,
            n_0=N0,
            mc_reps=MC_REPS,
            alpha=ALPHA,
            search_step=STEP,
            min_seg=MIN_SEG,
            penalty_factor=PENALTY_FACTOR,
            verbose=True
        )
        cd.save_critical_values(cv_path)

    print("\nStarting window size detection...")
    num_runs = args.num_runs
    for run in range(num_runs):
        print(f"\nRun {run}")
        out_csv = os.path.join(out_dir, f"run_{run}.csv")

        results = cd.detect(
            min_window=MIN_SEG,
            n_0=N0,
            jump=JUMP,
            search_step=STEP,
            alpha=ALPHA,
            t_workers=10,
            debug_anim=False
        )

        pd.DataFrame(results).to_csv(out_csv)
        print(f"Saved results to: {out_csv}")

    print("\nAggregating window sizes...")
    all_files = glob.glob(os.path.join(out_dir, "run*.csv"))

    dfs = []
    for file in all_files:
        win_df = pd.read_csv(file, usecols=["windows"])
        name = os.path.splitext(os.path.basename(file))[0]
        win_df = win_df.rename(columns={"windows": f"windows_{name}"})
        dfs.append(win_df)

    windows_df = pd.concat(dfs, axis=1)
    windows_df["window_mean"] = windows_df.mean(axis=1)
    windows_path = os.path.join(out_dir, "windows.csv")
    windows_df.to_csv(windows_path)
    print(f"Saved aggregated windows to: {windows_path}")

    return out_dir


def run_full_benchmark():
    print("\n" + "="*60)
    print("[TEST] Running SHAP Benchmark")
    print("="*60)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    windows_path = os.path.join(
        "examples",
        f"results/LSTM/entsoe_ro_price_test/Jump_{JUMP}_N0_{N0}_lambda_{PENALTY_FACTOR}",
        "windows.csv"
    )
    output_dir = os.path.join(
        "examples",
        f"results/benchmark_entsoe_ro_price_test/N0_{N0}_lambda_{PENALTY_FACTOR}"
    )

    if not os.path.exists(windows_path):
        print(f"ERROR: Windows file not found: {windows_path}")
        print("Run detection first (without --skip-detection).")
        sys.exit(1)

    print(f"Dataset: {PROCESSED_PATH}")
    print(f"Windows: {windows_path}")
    print(f"Output: {output_dir}")

    summary = run_benchmark(
        dataset_path=PROCESSED_PATH,
        output_dir=output_dir,
        device=device,
        verbose=True,
        dataset_type='empirical',
        column_name='N',
        seq_length=SEQ_LENGTH,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        max_background=100,
        shap_nsamples=500,
        rolling_window_size=N0,
        rolling_stride=1,
        precomputed_windows_path=windows_path,
        rolling_mean_window=75
    )

    print("\n" + "="*60)
    print("[TEST] Benchmark Complete!")
    print("="*60)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='QUICK TEST: ENTSO-E Romania Price (6 months)'
    )
    parser.add_argument('--api-key', type=str, required=True,
                        help='ENTSO-E Transparency Platform API key')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download (use cached files)')
    parser.add_argument('--skip-detection', action='store_true',
                        help='Skip LPA detection (use existing windows.csv)')
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='Skip the full benchmark')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of LPA detection runs (default: 1)')
    args = parser.parse_args()

    print("="*60)
    print("[TEST] ENTSO-E Romania Price -- 6 month quick test")
    print("="*60)
    print(f"Date range: {START} to {END}")
    print(f"MC reps: {MC_REPS} (reduced for speed)")
    print(f"LPA: N0={N0}, Step={STEP}, Lambda={PENALTY_FACTOR}")
    print("="*60)

    if not args.skip_download:
        download_entsoe_data(args.api_key)

    if not os.path.exists(PROCESSED_PATH):
        preprocess()
    else:
        print(f"\nProcessed test data already exists: {PROCESSED_PATH}")

    if not args.skip_detection:
        run_detection(args)

    if not args.skip_benchmark:
        run_full_benchmark()

    print("\n[TEST] All steps complete!")
