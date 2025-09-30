import os
import math

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from adaptivewinshap import AdaptiveModel, ChangeDetector


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class AdaptiveTransformer(AdaptiveModel):
    def __init__(self, seq_length, input_size, d_model=64, nhead=4, num_layers=2,
                 dropout=0.1, device="cpu", batch_size=512, lr=1e-3, epochs=50,
                 type_precision=np.float32):
        super().__init__(device=device, batch_size=batch_size, lr=lr, epochs=epochs,
                         type_precision=type_precision)

        self.seq_len = seq_length
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len = x.size(0), x.size(1)

        # Project input to d_model dimensions
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model] for pos encoding
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]

        # Use the last time step for prediction
        last_output = transformer_output[:, -1, :]  # [batch_size, d_model]

        # Project to output
        output = self.output_projection(last_output)  # [batch_size, 1]

        return output.squeeze(-1)  # [batch_size]

    def prepare_data(self, window, start_abs_idx):
        L = self.seq_len
        n = len(window)
        if n <= L:
            return None, None, None

        # Create sliding windows
        X = np.lib.stride_tricks.sliding_window_view(window, L + 1)  # [N, L+1]
        X, y = X[:, :-1], X[:, -1]  # [N,L], [N]

        # Add feature dimension for transformer input
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

    # Transformer hyperparameters
    TRANSFORMER_SEQ_LEN = 3  # Longer sequences often work better with transformers
    TRANSFORMER_D_MODEL = 64
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_EPOCHS = 50
    TRANSFORMER_BATCH = 64
    TRANSFORMER_LR = 1e-3

    # Load data
    df = pd.read_csv("examples/datasets/data.csv")
    print("Data shape:", df.shape)
    print(df.head())

    # Create transformer model
    model = AdaptiveTransformer(
        seq_length=TRANSFORMER_SEQ_LEN,
        input_size=1,  # Single feature (N column)
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dropout=TRANSFORMER_DROPOUT,
        device=DEVICE,
        batch_size=TRANSFORMER_BATCH,
        lr=TRANSFORMER_LR,
        epochs=TRANSFORMER_EPOCHS,
        type_precision=np.float32
    )

    print(f"Using device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create change detector
    cd = ChangeDetector(model, df["N"].to_numpy(dtype=np.float32), debug=False)

    # Output directory
    out_dir = os.path.join("examples", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "transformer_simulation.csv")

    # Detection parameters
    MIN_SEG = 30  # Increased minimum segment size for transformers
    N_0 = 100
    JUMP = 1
    STEP = 5
    ALPHA = 0.95
    NUM_BOOTSTRAP = 1

    print(f"Running change detection with transformer...")
    print(f"Sequence length: {TRANSFORMER_SEQ_LEN}")
    print(f"Model dimension: {TRANSFORMER_D_MODEL}")
    print(f"Number of heads: {TRANSFORMER_NHEAD}")
    print(f"Number of layers: {TRANSFORMER_NUM_LAYERS}")

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