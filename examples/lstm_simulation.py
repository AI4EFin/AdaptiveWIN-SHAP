import os

import numpy as np

import torch
import torch.nn as nn
import pandas as pd

from adaptivewinshap import AdaptiveModel, ChangeDetector

class AdaptiveLSTM(AdaptiveModel):
    def __init__(self, seq_length=3, input_size=1, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)
        self.seq_len = seq_length


    def forward(self, x):
        out, _ = self.lstm(x)           # [B,L,H]
        yhat = self.fc(out[:, -1, :])   # [B,1]
        return yhat.squeeze(-1)

    def prepare_data(self, window, start_abs_idx):
        L = self.seq_len
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
    if torch.mps.is_available():
        DEVICE = "mps"

    LSTM_SEQ_LEN = 3
    LSTM_HIDDEN = 16
    LSTM_LAYERS = 1
    LSTM_EPOCHS = 15
    LSTM_BATCH = 64
    LSTM_LR = 1e-2
    LSTM_DROPOUT = 0.0
    MIN_SEG = 20

    df = pd.read_csv("examples/data.csv")
    print(df.head())

    model = AdaptiveLSTM()

    cd = ChangeDetector(model, df["N"].to_numpy(dtype=np.float32), debug=True)

    out_dir = os.path.join("examples", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "lstm_simulation.csv")

    results = cd.detect()
    print(f"Saved results to: {out_csv}")
