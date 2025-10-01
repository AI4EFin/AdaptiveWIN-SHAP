import os

import numpy as np

import torch
import torch.nn as nn
import pandas as pd

from adaptivewinshap import AdaptiveModel, ChangeDetector, store_init_kwargs


class AdaptiveLSTM(AdaptiveModel):
    @store_init_kwargs
    def __init__(self, seq_length=3, input_size=1, hidden=16, layers=1, dropout=0.2, device="cpu", batch_size=512, lr=1e-12, epochs=50, type_precision=np.float32):
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
    LSTM_EPOCHS = 50
    LSTM_BATCH = 64
    LSTM_LR = 1e-2
    LSTM_DROPOUT = 0.0

    df = pd.read_csv("examples/datasets/data.csv")
    print(df.head())

    model = AdaptiveLSTM(seq_length=LSTM_SEQ_LEN, input_size=1, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, dropout=LSTM_DROPOUT,
                         device=DEVICE, batch_size=LSTM_BATCH, lr=LSTM_LR, epochs=LSTM_EPOCHS, type_precision=np.float32)

    cd = ChangeDetector(model, df["N"].to_numpy(dtype=np.float32), debug=True)

    out_dir = os.path.join("examples", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "lstm_simulation.csv")

    MIN_SEG = 20
    N_0=100
    JUMP=1
    STEP=5
    ALPHA=0.95
    NUM_BOOTSTRAP = 1

    results = cd.detect(min_window=MIN_SEG, n_0=N_0, jump=JUMP, search_step=STEP, alpha=ALPHA, num_bootstrap=NUM_BOOTSTRAP)
    print(f"Saved results to: {out_csv}")
