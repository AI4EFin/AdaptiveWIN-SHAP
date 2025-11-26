import glob
import os
import timeit

import numpy as np

import torch
import torch.nn as nn
import pandas as pd

from adaptivewinshap import AdaptiveModel, ChangeDetector, store_init_kwargs, AdaptiveWinShap

class AdaptiveLSTM(AdaptiveModel):
    @store_init_kwargs
    def __init__(self, device, seq_length=3, input_size=1, hidden=16, layers=1, dropout=0.2, batch_size=512, lr=1e-12, epochs=50, type_precision=np.float32):
        super().__init__(device=device, batch_size=batch_size, lr=lr, epochs=epochs, type_precision=type_precision)
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout


    def forward(self, x):
        out, _ = self.lstm(x)           # [B,L,H]
        yhat = self.fc(out[:, -1, :])   # [B,1]
        return yhat.squeeze(-1)

    def prepare_data(self, window, start_abs_idx):
        """
        window: [n, F] array where F = 1 (target only) or F > 1 (target + covariates)
        """
        L = self.seq_length
        F = window.shape[1] if window.ndim == 2 else 1
        n = len(window)

        if n <= L:
            return None, None, None

        # Ensure window is 2D
        if window.ndim == 1:
            window = window[:, None]  # [n, 1]

        # Create sequences: for each time t, use [t-L:t] to predict t
        X_list = []
        y_list = []
        for i in range(L, n):
            X_list.append(window[i-L:i])  # [L, F]
            y_list.append(window[i, 0])   # target is always first column

        X = np.array(X_list, dtype=np.float32)  # [N, L, F]
        y = np.array(y_list, dtype=np.float32)  # [N]

        t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        return X_tensor, y_tensor, t_abs

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        # x_flat: [N, L*F] -> [N, L, F]
        # N = x_flat.shape[0]
        # F = input_size
        # x = x_flat.reshape(N, L, F)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        preds = model(xt)  # [N]
        return preds.detach().cpu().numpy().reshape(-1, 1)  # SHAP expects 2D

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
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of detection runs (default: 1)')
    args = parser.parse_args()

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
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
        "garch_regime": {"seq_length": 1, "n_covariates": 2},   # GARCH(1,1) + 2 factors (M,V)
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
    model = AdaptiveLSTM(DEVICE, seq_length=LSTM_SEQ_LEN, input_size=LSTM_INPUT_SIZE,
                         hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, dropout=LSTM_DROPOUT,
                         batch_size=LSTM_BATCH, lr=LSTM_LR, epochs=LSTM_EPOCHS,
                         type_precision=np.float64)

    cd = ChangeDetector(model, data, debug=False, force_cpu=True)

    # Use command-line arguments or defaults
    MIN_SEG = 4
    N_0 = args.n0
    JUMP = args.jump
    STEP = 5
    ALPHA = 0.95
    NUM_BOOTSTRAP = 1

    out_dir = os.path.join("examples", f"results/LSTM/{dataset_name}/Jump_{JUMP}_N0_{N_0}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Parameters: N_0={N_0}, JUMP={JUMP}, NUM_RUNS={args.num_runs}")
    print()

    # ============================================================
    # Window Size Detection
    # ============================================================
    print("Starting window size detection...")
    num_runs = args.num_runs
    for run in range(num_runs):
        print(f"Run {run}")
        out_csv = os.path.join(out_dir, f"run_{run}.csv")
        results = cd.detect(min_window=MIN_SEG, n_0=N_0, jump=JUMP, search_step=STEP, alpha=ALPHA, num_bootstrap=NUM_BOOTSTRAP,
                        t_workers=10, b_workers=10, one_b_threads=1, save_path=f"{out_dir}/run_{run}.mp4")

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
    print(f"Use these windows for benchmarking with benchmark.py")
    print("="*60)

