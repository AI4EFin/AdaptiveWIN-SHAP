import os

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from adaptivewinshap import AdaptiveModel, ChangeDetector


class AdaptiveMLP(AdaptiveModel):
    def __init__(self, seq_length, input_size, hidden_layers=[64, 32], dropout=0.2,
                 activation='relu', device="cpu", batch_size=512, lr=1e-3, epochs=50,
                 type_precision=np.float32):
        super().__init__(device=device, batch_size=batch_size, lr=lr, epochs=epochs,
                         type_precision=type_precision)

        self.seq_len = seq_length
        self.input_size = input_size

        # Flatten the sequence for MLP input
        flattened_input_size = seq_length * input_size

        # Build the MLP layers
        layers = []
        prev_size = flattened_input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization

            # Activation function
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())  # Default to ReLU

            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        # Create the sequential model
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.size(0)

        # Flatten the sequence for MLP input
        x_flat = x.view(batch_size, -1)  # [batch_size, seq_len * input_size]

        # Forward through MLP
        output = self.mlp(x_flat)  # [batch_size, 1]

        return output.squeeze(-1)  # [batch_size]

    def prepare_data(self, window, start_abs_idx):
        L = self.seq_len
        n = len(window)
        if n <= L:
            return None, None, None

        # Create sliding windows
        X = np.lib.stride_tricks.sliding_window_view(window, L + 1)  # [N, L+1]
        X, y = X[:, :-1], X[:, -1]  # [N,L], [N]

        # Add feature dimension for consistency with other models
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

    # MLP hyperparameters
    MLP_SEQ_LEN = 7  # Window size for MLP
    MLP_HIDDEN_LAYERS = [128, 64, 32]  # Multiple hidden layers
    MLP_DROPOUT = 0.3
    MLP_ACTIVATION = 'relu'  # Can be 'relu', 'tanh', 'leaky_relu'
    MLP_EPOCHS = 100
    MLP_BATCH = 128
    MLP_LR = 1e-3

    # Load data
    df = pd.read_csv("examples/datasets/data.csv")
    print("Data shape:", df.shape)
    print(df.head())

    # Create MLP model
    model = AdaptiveMLP(
        seq_length=MLP_SEQ_LEN,
        input_size=1,  # Single feature (N column)
        hidden_layers=MLP_HIDDEN_LAYERS,
        dropout=MLP_DROPOUT,
        activation=MLP_ACTIVATION,
        device=DEVICE,
        batch_size=MLP_BATCH,
        lr=MLP_LR,
        epochs=MLP_EPOCHS,
        type_precision=np.float32
    )

    print(f"Using device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create change detector
    cd = ChangeDetector(model, df["N"].to_numpy(dtype=np.float32), debug=False)

    # Output directory
    out_dir = os.path.join("examples", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "mlp_simulation.csv")

    # Detection parameters
    MIN_SEG = 25
    N_0 = 100
    JUMP = 1
    STEP = 5
    ALPHA = 0.95
    NUM_BOOTSTRAP = 1

    print(f"Running change detection with MLP...")
    print(f"Sequence length: {MLP_SEQ_LEN}")
    print(f"Hidden layers: {MLP_HIDDEN_LAYERS}")
    print(f"Activation: {MLP_ACTIVATION}")
    print(f"Dropout: {MLP_DROPOUT}")

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