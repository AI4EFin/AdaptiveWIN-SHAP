import os

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from adaptivewinshap import AdaptiveModel, ChangeDetector


class AdaptiveRNN(AdaptiveModel):
    def __init__(self, seq_length, input_size, hidden_size, num_layers=1, dropout=0.0,
                 device="cpu", batch_size=512, lr=1e-3, epochs=50, type_precision=np.float32):
        super().__init__(device=device, batch_size=batch_size, lr=lr, epochs=epochs,
                         type_precision=type_precision)

        self.seq_len = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Simple RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Forward through RNN
        rnn_out, _ = self.rnn(x, h0)  # [batch_size, seq_len, hidden_size]

        # Apply dropout
        rnn_out = self.dropout(rnn_out)

        # Use the output from the last time step
        last_output = rnn_out[:, -1, :]  # [batch_size, hidden_size]

        # Generate prediction
        output = self.fc(last_output)  # [batch_size, 1]

        return output.squeeze(-1)  # [batch_size]

    def prepare_data(self, window, start_abs_idx):
        L = self.seq_len
        n = len(window)
        if n <= L:
            return None, None, None

        # Create sliding windows
        X = np.lib.stride_tricks.sliding_window_view(window, L + 1)  # [N, L+1]
        X, y = X[:, :-1], X[:, -1]  # [N,L], [N]

        # Add feature dimension for RNN input
        X = X[..., None].astype(self.type_precision)  # [N, L, 1]
        y = y.astype(self.type_precision)

        # Absolute time indices
        t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        return X_tensor, y_tensor, t_abs


if __name__ == "__main__":
    # Device selection
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
        DEVICE = "mps"

    # RNN hyperparameters
    RNN_SEQ_LEN = 3
    RNN_HIDDEN_SIZE = 32
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.2
    RNN_EPOCHS = 50
    RNN_BATCH = 64
    RNN_LR = 1e-3

    # Load data
    df = pd.read_csv("examples/datasets/data.csv")
    print("Data shape:", df.shape)
    print(df.head())

    # Create RNN model
    model = AdaptiveRNN(
        seq_length=RNN_SEQ_LEN,
        input_size=1,  # Single feature (N column)
        hidden_size=RNN_HIDDEN_SIZE,
        num_layers=RNN_NUM_LAYERS,
        dropout=RNN_DROPOUT,
        device=DEVICE,
        batch_size=RNN_BATCH,
        lr=RNN_LR,
        epochs=RNN_EPOCHS,
        type_precision=np.float32
    )

    print(f"Using device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create change detector
    cd = ChangeDetector(model, df["N"].to_numpy(dtype=np.float32), debug=False)

    # Output directory
    out_dir = os.path.join("examples", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "rnn_simulation.csv")

    # Detection parameters
    MIN_SEG = 25
    N_0 = 100
    JUMP = 1
    STEP = 5
    ALPHA = 0.95
    NUM_BOOTSTRAP = 1

    print(f"Running change detection with RNN...")
    print(f"Sequence length: {RNN_SEQ_LEN}")
    print(f"Hidden size: {RNN_HIDDEN_SIZE}")
    print(f"Number of layers: {RNN_NUM_LAYERS}")
    print(f"Dropout: {RNN_DROPOUT}")

    # Run detection
    results = cd.detect(
        min_window=MIN_SEG,
        n_0=N_0,
        jump=JUMP,
        search_step=STEP,
        alpha=ALPHA,
        num_bootstrap=NUM_BOOTSTRAP
    )

    # Save results
    results[0].to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")

    # Print some statistics
    print(f"\nResults summary:")
    print(f"Total data points: {len(df)}")
    print(f"Results shape: {results[0].shape}")
    print(results[0].head())