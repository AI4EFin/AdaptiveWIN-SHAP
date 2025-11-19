"""
Baseline SHAP methods for benchmarking against AdaptiveWinShap.

Includes:
- GlobalSHAP: Vanilla kernel SHAP on a single global model
- RollingWindowSHAP: Simple fixed rolling window approach
- TimeShapWrapper: Integration with TimeShap library (if available)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Union
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available")

try:
    import timeshap
    TIMESHAP_AVAILABLE = True
except ImportError:
    TIMESHAP_AVAILABLE = False
    warnings.warn("TimeShap library not available - TimeShap baseline will be skipped")


class LSTMModel(nn.Module):
    """Simple LSTM model for baseline comparisons."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        yhat = self.fc(out[:, -1, :])  # Use last time step
        return yhat.squeeze(-1)

    def predict(self, x):
        """Prediction method compatible with SHAP."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            # Ensure input is on the same device as model
            x = x.to(self.device)

            # Handle both 2D and 3D inputs
            # If 2D [N, seq_len], expand to 3D [N, seq_len, 1]
            if x.dim() == 2:
                x = x.unsqueeze(-1)

            preds = self.forward(x)
            return preds.cpu().numpy().reshape(-1, 1)


def train_lstm_model(X, y, model, device='cpu', epochs=50, batch_size=64, lr=1e-2, verbose=False):
    """
    Train an LSTM model on the given data.

    Parameters
    ----------
    X : array of shape [N, seq_len, features]
    y : array of shape [N]
    model : LSTMModel instance
    device : str
    epochs : int
    batch_size : int
    lr : float
    verbose : bool

    Returns
    -------
    model : trained model
    train_loss : final training loss
    """
    model = model.to(device)
    model.train()

    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

    model.eval()
    return model, epoch_loss / len(loader)


def create_sequences(data, seq_length):
    """
    Create sequences for LSTM training/prediction.

    Parameters
    ----------
    data : 1D array
    seq_length : int

    Returns
    -------
    X : array of shape [N, seq_length, 1]
    y : array of shape [N]
    """
    n = len(data)
    if n <= seq_length:
        raise ValueError(f"Data length {n} must be > seq_length {seq_length}")

    X = np.lib.stride_tricks.sliding_window_view(data, seq_length + 1)
    X, y = X[:, :-1], X[:, -1]
    X = X[..., None].astype(np.float32)  # Add feature dimension
    y = y.astype(np.float32)

    return X, y


class GlobalSHAP:
    """
    Vanilla kernel SHAP on a single global model trained on all data.
    """

    def __init__(self, seq_length=3, hidden_size=64, num_layers=1, dropout=0.0,
                 epochs=50, batch_size=64, lr=1e-2, device='cpu',
                 max_background=100, shap_nsamples=500):
        """
        Parameters
        ----------
        seq_length : int
        hidden_size : int
        num_layers : int
        dropout : float
        epochs : int
        batch_size : int
        lr : float
        device : str
        max_background : int - maximum background samples for SHAP
        shap_nsamples : int - number of samples for SHAP kernel
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.max_background = max_background
        self.shap_nsamples = shap_nsamples
        self.model = None
        self.explainer = None

    def fit(self, data, verbose=False):
        """
        Train a global model on all data.

        Parameters
        ----------
        data : 1D array
        verbose : bool
        """
        X, y = create_sequences(data, self.seq_length)

        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device
        )

        self.model, train_loss = train_lstm_model(
            X, y, self.model,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=verbose
        )

        # Create SHAP explainer with background data
        n_bg = min(len(X), self.max_background)
        bg_indices = np.random.choice(len(X), size=n_bg, replace=False)
        background = X[bg_indices]

        # Flatten background for SHAP (2D input expected)
        background_2d = background.squeeze(-1)  # [N, seq_length, 1] -> [N, seq_length]

        self.explainer = shap.KernelExplainer(self.model.predict, background_2d)

        if verbose:
            print(f"Global model trained. Final loss: {train_loss:.6f}")

        return self

    def explain(self, data, start_idx=None, end_idx=None):
        """
        Compute SHAP values for a portion of the data.

        Parameters
        ----------
        data : 1D array - full data series
        start_idx : int - start index for explanation (default: seq_length)
        end_idx : int - end index for explanation (default: len(data))

        Returns
        -------
        DataFrame with columns: end_index, shap_lag_0, shap_lag_1, ..., y_hat
        """
        if self.model is None or self.explainer is None:
            raise ValueError("Model not trained. Call fit() first.")

        X, y = create_sequences(data, self.seq_length)

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(X)

        X_explain = X[start_idx:end_idx]

        # Flatten for SHAP (SHAP expects 2D input)
        # X_explain shape: [N, seq_length, 1] -> [N, seq_length]
        X_explain_2d = X_explain.squeeze(-1)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_explain_2d, nsamples=self.shap_nsamples)

        # Get predictions
        y_hat = self.model.predict(X_explain).flatten()

        # shap_values shape: [N, seq_length] or could be list
        # Convert to numpy array if needed
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        # Ensure it's 2D
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, self.seq_length)

        shap_per_lag = np.abs(shap_values)  # [N, seq_length]

        # Compute faithfulness for each point
        from .metrics import compute_point_faithfulness

        faithfulness_scores = []
        print(f"Computing faithfulness for {len(X_explain)} points...")
        for i in range(len(X_explain)):
            faith = compute_point_faithfulness(
                model=self.model,
                shap_values=shap_per_lag[i],  # [seq_length]
                input_sequence=X_explain[i:i+1],  # [1, seq_length, 1]
                percentiles=[90, 70, 50],
                eval_types=['prtb', 'sqnc'],
                seq_len=self.seq_length
            )
            faithfulness_scores.append(faith)

        # Create results DataFrame
        results = {
            'end_index': np.arange(start_idx + self.seq_length, start_idx + self.seq_length + len(X_explain)).tolist(),
            'y_hat': y_hat.tolist() if hasattr(y_hat, 'tolist') else list(y_hat),
        }

        # Add per-lag SHAP values with correct semantic names
        # Position 0 in sequence = t-seq_length (oldest), Position seq_length-1 = t-1 (newest)
        # Store with names matching adaptive: shap_lag_t-1, shap_lag_t-2, shap_lag_t-3
        # Use same indexing as adaptive: t-1 gets last position (seq_length-1), t-3 gets first position (0)
        for lag in range(1, self.seq_length + 1):
            results[f'shap_lag_t-{lag}'] = shap_per_lag[:, self.seq_length - lag].flatten().tolist()

        # Add faithfulness scores - ensure they are scalar values
        for key in faithfulness_scores[0].keys():
            results[key] = [float(f[key]) for f in faithfulness_scores]

        return pd.DataFrame(results)


class RollingWindowSHAP:
    """
    Simple rolling window SHAP with fixed window size.
    Trains a new model on each window and computes SHAP.
    """

    def __init__(self, seq_length=3, window_size=100, hidden_size=64, num_layers=1,
                 dropout=0.0, epochs=50, batch_size=64, lr=1e-2, device='cpu',
                 max_background=100, shap_nsamples=500):
        """
        Parameters
        ----------
        seq_length : int
        window_size : int - fixed window size for training
        hidden_size : int
        num_layers : int
        dropout : float
        epochs : int
        batch_size : int
        lr : float
        device : str
        max_background : int
        shap_nsamples : int
        """
        self.seq_length = seq_length
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.max_background = max_background
        self.shap_nsamples = shap_nsamples

    def rolling_explain(self, data, stride=1, verbose=False):
        """
        Compute SHAP values using rolling windows.

        Parameters
        ----------
        data : 1D array
        stride : int - step size between windows
        verbose : bool

        Returns
        -------
        DataFrame with columns: end_index, window_len, shap_lag_0, ..., y_hat
        """
        results = []
        n = len(data)

        # Start from window_size and slide forward
        for end_idx in range(self.window_size, n, stride):
            start_idx = end_idx - self.window_size
            window_data = data[start_idx:end_idx]

            # Train model on this window
            X_win, y_win = create_sequences(window_data, self.seq_length)

            if len(X_win) < self.seq_length + 1:
                continue

            model = LSTMModel(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                device=self.device
            )

            model, _ = train_lstm_model(
                X_win, y_win, model,
                device=self.device,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                verbose=False
            )

            # Create explainer for this window
            n_bg = min(len(X_win), self.max_background)
            bg_indices = np.random.choice(len(X_win), size=n_bg, replace=False)
            background = X_win[bg_indices]

            # Flatten background for SHAP (2D input expected)
            background_2d = background.squeeze(-1)

            explainer = shap.KernelExplainer(model.predict, background_2d)

            # Explain the last point in the window
            X_last = X_win[-1:, :, :]
            X_last_2d = X_last.squeeze(-1)  # Flatten for SHAP
            shap_values = explainer.shap_values(X_last_2d, nsamples=self.shap_nsamples)
            y_hat = model.predict(X_last)[0, 0]

            # Compute faithfulness for this point (using the rolling window model)
            from .metrics import compute_point_faithfulness
            shap_per_lag = np.abs(shap_values).squeeze()  # [seq_length]

            faithfulness = compute_point_faithfulness(
                model=model,  # Use the rolling window model
                shap_values=shap_per_lag,
                input_sequence=X_last,  # [1, seq_length, 1]
                percentiles=[90, 70, 50],
                eval_types=['prtb', 'sqnc'],
                seq_len=self.seq_length
            )

            # Store results
            result_row = {
                'end_index': end_idx,
                'window_len': self.window_size,
                'y_hat': y_hat
            }

            # Add per-lag SHAP values with correct semantic names
            # Position 0 in sequence = t-seq_length (oldest), Position seq_length-1 = t-1 (newest)
            # Use same indexing as adaptive: t-1 gets last position (seq_length-1), t-3 gets first position (0)
            for lag in range(1, self.seq_length + 1):
                result_row[f'shap_lag_t-{lag}'] = shap_per_lag[self.seq_length - lag]

            # Add faithfulness scores
            result_row.update(faithfulness)

            results.append(result_row)

            if verbose and len(results) % 10 == 0:
                print(f"Processed {len(results)} windows...")

        return pd.DataFrame(results)


class TimeShapWrapper:
    """
    Wrapper for TimeShap library (if available).
    """

    def __init__(self, seq_length=3, hidden_size=64, num_layers=1, dropout=0.0,
                 epochs=50, batch_size=64, lr=1e-2, device='cpu'):
        if not TIMESHAP_AVAILABLE:
            raise ImportError("TimeShap library not available. Install with: pip install timeshap")

        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None

    def fit(self, data, verbose=False):
        """Train a global model."""
        X, y = create_sequences(data, self.seq_length)

        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device
        )

        self.model, train_loss = train_lstm_model(
            X, y, self.model,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=verbose
        )

        if verbose:
            print(f"TimeShap model trained. Final loss: {train_loss:.6f}")

        return self

    def explain(self, data, start_idx=None, end_idx=None):
        """
        Compute TimeShap explanations.

        Note: TimeShap has a different API and this is a placeholder
        for actual implementation.
        """
        raise NotImplementedError("TimeShap integration not yet implemented. "
                                  "Requires specific TimeShap API adaptation.")
