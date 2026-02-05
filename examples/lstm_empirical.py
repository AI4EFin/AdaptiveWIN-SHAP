"""
AdaptiveWIN-SHAP Empirical Data Example

This script demonstrates AdaptiveWIN-SHAP on real-world energy price data.
"""
import glob
import os

import numpy as np
import torch
import pandas as pd

from adaptivewinshap import AdaptiveLSTM, ChangeDetector, AdaptiveWinShap


if __name__ == "__main__":
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"

    LSTM_SEQ_LEN = 24  # 1 day
    LSTM_HIDDEN = 16
    LSTM_LAYERS = 1
    LSTM_EPOCHS = 15
    LSTM_BATCH = 64
    LSTM_LR = 1e-2
    LSTM_DROPOUT = 0.0
    start = pd.Timestamp("2021-05-01 00:00:00", tz="Europe/Bucharest")
    end = pd.Timestamp("2021-08-01 00:00:00", tz="Europe/Bucharest")

    df = pd.read_csv("examples/datasets/empirical/dataset.csv", index_col=0, parse_dates=True)
    df = df.loc[start:end]

    data = df["Price Day Ahead"].to_numpy(dtype=np.float32)

    model = AdaptiveLSTM(
        DEVICE,
        seq_length=LSTM_SEQ_LEN,
        input_size=1,
        hidden=LSTM_HIDDEN,
        layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
        batch_size=LSTM_BATCH,
        lr=LSTM_LR,
        epochs=LSTM_EPOCHS,
        type_precision=np.float32
    )

    cd = ChangeDetector(model, data, debug=False, force_cpu=True)

    MIN_SEG = 30
    N_0 = 72  # 3 days
    JUMP = 1
    STEP = 5
    ALPHA = 0.95
    MC_REPS = 100

    out_dir = os.path.join("examples", f"results/LSTM/empirical/Jump_{JUMP}_N0_{N_0}/")
    os.makedirs(out_dir, exist_ok=True)

    # Compute or load critical values
    cv_path = os.path.join(out_dir, "critical_values.csv")
    if os.path.exists(cv_path):
        print(f"Loading critical values from: {cv_path}")
        cd.load_critical_values(cv_path)
    else:
        print("Computing Monte Carlo critical values...")
        cd.precompute_critical_values(
            data=data,
            n_0=N_0,
            mc_reps=MC_REPS,
            alpha=ALPHA,
            search_step=STEP,
            min_seg=MIN_SEG,
            penalty_factor=0.25,
            verbose=True
        )
        cd.save_critical_values(cv_path)

    # Run detection
    num_runs = 1
    for run in range(num_runs):
        print(f"Run {run}")
        out_csv = os.path.join(out_dir, f"run_{run}.csv")
        results = cd.detect(
            min_window=MIN_SEG,
            n_0=N_0,
            jump=JUMP,
            search_step=STEP,
            alpha=ALPHA,
            t_workers=10,
            debug_anim=True,
            save_path=f"{out_dir}/run_{run}.mp4"
        )

        pd.DataFrame(results).to_csv(out_csv)
        print(f"Saved results to: {out_csv}")

    # Aggregate windows
    all_files = glob.glob(os.path.join(out_dir, "run*.csv"))

    dfs = []
    for file in all_files:
        win_df = pd.read_csv(file, usecols=["windows"])
        name = os.path.splitext(os.path.basename(file))[0]
        win_df = win_df.rename(columns={"windows": f"windows_{name}"})
        dfs.append(win_df)

    windows_df = pd.concat(dfs, axis=1)
    windows_df["window_mean"] = windows_df.mean(axis=1)
    windows_df.to_csv(f"{out_dir}/windows.csv")

    # SHAP computation
    runner = AdaptiveWinShap(
        seq_length=LSTM_SEQ_LEN,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        lstm_dropout=LSTM_DROPOUT,
        batch_size=LSTM_BATCH,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        max_background=150,
        shap_nsamples=500,
        aggregate_lag=True
    )

    print(df.head())
    df_results_csv = runner.rolling_explain(
        data=data,
        window_sizes_csv=f"{out_dir}/windows.csv",
        csv_column="window_mean"
    )

    df_results_csv.to_csv(f"{out_dir}/results.csv")
    print(f"SHAP results saved to: {out_dir}/results.csv")
