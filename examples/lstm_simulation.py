import os

import numpy as np

import torch
import torch.nn as nn
import pandas as pd

from adaptivewinshap import AdaptiveModel, ChangeDetector, store_init_kwargs


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
        L = self.seq_length
        n = len(window)
        if n <= L: return None, None
        X = np.lib.stride_tricks.sliding_window_view(window, L + 1)  # [N, L+1]
        X, y = X[:, :-1], X[:, -1]  # [N,L], [N]

        X = X[..., None].astype(np.float32)  # [N,L,1]
        y = y.astype(np.float32)

        t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        return X_tensor, y_tensor, t_abs

if __name__ == "__main__":
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
        DEVICE = "mps"
    LSTM_SEQ_LEN = 3
    LSTM_HIDDEN = 16
    LSTM_LAYERS = 1
    LSTM_EPOCHS = 15
    LSTM_BATCH = 64
    LSTM_LR = 1e-2
    LSTM_DROPOUT = 0.0

    dataset_type = "line_std"
    order = "25"

    df = pd.read_csv(f"examples/datasets/simulated/{dataset_type}/{order}.csv")

    model = AdaptiveLSTM(DEVICE, seq_length=LSTM_SEQ_LEN, input_size=1, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, dropout=LSTM_DROPOUT,
                         batch_size=LSTM_BATCH, lr=LSTM_LR, epochs=LSTM_EPOCHS, type_precision=np.float64)

    cd = ChangeDetector(model, df["N"].to_numpy(dtype=np.float64), debug=False, force_cpu=True)

    MIN_SEG = 4
    N_0=20
    JUMP=1
    STEP=2
    ALPHA=0.95
    NUM_BOOTSTRAP = 10

    out_dir = os.path.join("examples", f"results/LSTM/{dataset_type}_{order}/Jump_{JUMP}_N0_{N_0}")
    os.makedirs(out_dir, exist_ok=True)

    num_runs = 10
    for run in range(num_runs):
        print(f"Run {run}")
        out_csv = os.path.join(out_dir, f"run_{run}.csv")
        results = cd.detect(min_window=MIN_SEG, n_0=N_0, jump=JUMP, search_step=STEP, alpha=ALPHA, num_bootstrap=NUM_BOOTSTRAP,
                        t_workers=10, b_workers=10, one_b_threads=1)

        pd.DataFrame(results).to_csv(out_csv)
        print(f"Saved results to: {out_csv}")
