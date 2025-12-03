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
            # If 2D, it could be:
            # - [N, seq_len] for univariate (input_size=1)
            # - [N, seq_len * n_features] from SHAP (needs reshaping)
            if x.dim() == 2:
                n_samples = x.shape[0]
                total_features = x.shape[1]

                # If total features matches input_size, it's already a flat sequence (univariate case)
                if total_features == self.input_size and self.input_size == 1:
                    x = x.unsqueeze(-1)  # [N, seq_len] -> [N, seq_len, 1]
                else:
                    # Otherwise, reshape from SHAP's flattened format
                    # [N, seq_len * n_features] -> [N, seq_len, n_features]
                    seq_len = total_features // self.input_size
                    x = x.reshape(n_samples, seq_len, self.input_size)

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


def create_sequences(data, seq_length, covariates=None):
    """
    Create sequences for LSTM training/prediction with optional covariates.

    Parameters
    ----------
    data : 1D array - target time series
    seq_length : int
    covariates : 2D array of shape [T, n_covariates], optional
        Additional features (e.g., exogenous variables) aligned with data.
        If provided, each sequence will include both lags AND current covariates.

    Returns
    -------
    X : array of shape [N, seq_length, features]
        where features = 1 (lag only) or 1 + n_covariates (lags + covariates)
    y : array of shape [N]
    feature_names : list of str
        Names of features in order: ['lag_t-1', 'lag_t-2', ..., 'Z_0', 'Z_1', ...]
    """
    n = len(data)
    if n <= seq_length:
        raise ValueError(f"Data length {n} must be > seq_length {seq_length}")

    # Create lagged sequences from target
    X_lags = np.lib.stride_tricks.sliding_window_view(data, seq_length + 1)
    X_lags, y = X_lags[:, :-1], X_lags[:, -1]
    X_lags = X_lags[..., None].astype(np.float32)  # [N, seq_length, 1]

    feature_names = [f'lag_t-{i+1}' for i in range(seq_length)]

    # Add covariates if provided
    if covariates is not None:
        # covariates shape: [T, n_cov]
        # We want to include current timestep covariates for each sequence
        # Sequence at position i predicts y[i+seq_length], uses covariates[i+seq_length]
        n_cov = covariates.shape[1]
        n_sequences = len(X_lags)

        # Get covariates at prediction timepoints
        cov_at_pred = covariates[seq_length:seq_length+n_sequences]  # [N, n_cov]

        # Expand covariates to match sequence format: [N, seq_length, n_cov]
        # Repeat the same covariate values across all timesteps in the sequence
        # (This assumes covariates are contemporaneous, not lagged)
        cov_expanded = np.repeat(cov_at_pred[:, np.newaxis, :], seq_length, axis=1)

        # Concatenate lags and covariates along feature dimension
        X = np.concatenate([X_lags, cov_expanded], axis=2)  # [N, seq_length, 1+n_cov]

        feature_names += [f'Z_{i}' for i in range(n_cov)]
    else:
        X = X_lags

    y = y.astype(np.float32)

    return X, y, feature_names


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
        self.feature_names = None
        self.n_features = None

    def fit(self, data, covariates=None, verbose=False):
        """
        Train a global model on all data.

        Parameters
        ----------
        data : 1D array - target time series
        covariates : 2D array of shape [T, n_cov], optional
        verbose : bool
        """
        X, y, feature_names = create_sequences(data, self.seq_length, covariates=covariates)
        self.feature_names = feature_names
        self.n_features = X.shape[2]

        self.model = LSTMModel(
            input_size=self.n_features,
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

        # Flatten background for SHAP: [N, seq_length, n_features] -> [N, seq_length * n_features]
        background_2d = background.reshape(len(background), -1)

        self.explainer = shap.KernelExplainer(self.model.predict, background_2d)

        if verbose:
            print(f"Global model trained. Final loss: {train_loss:.6f}")

        return self

    def explain(self, data, covariates=None, start_idx=None, end_idx=None):
        """
        Compute SHAP values for a portion of the data.

        Parameters
        ----------
        data : 1D array - full data series
        covariates : 2D array of shape [T, n_cov], optional
        start_idx : int - start index for explanation (default: 0)
        end_idx : int - end index for explanation (default: len(data))

        Returns
        -------
        DataFrame with columns: end_index, y_hat, shap_lag_t-*, shap_Z_*, faithfulness_*, ablation_*
        """
        if self.model is None or self.explainer is None:
            raise ValueError("Model not trained. Call fit() first.")

        X, y, feature_names = create_sequences(data, self.seq_length, covariates=covariates)

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(X)

        X_explain = X[start_idx:end_idx]

        # Reshape for SHAP (SHAP expects 2D input: [N, seq_length * n_features])
        # X_explain shape: [N, seq_length, n_features] -> [N, seq_length * n_features]
        n_samples = len(X_explain)
        X_explain_2d = X_explain.reshape(n_samples, -1)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_explain_2d, nsamples=self.shap_nsamples)

        # Get predictions
        y_hat = self.model.predict(X_explain).flatten()

        # shap_values shape: [N, seq_length * n_features]
        # Convert to numpy array if needed
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        # Reshape to [N, seq_length, n_features]
        shap_values_3d = shap_values.reshape(n_samples, self.seq_length, self.n_features)

        # Take absolute values
        shap_per_feature = np.abs(shap_values_3d)  # [N, seq_length, n_features]

        # Sum across sequence length to get per-feature importance: [N, n_features]
        shap_per_feature_sum = shap_per_feature.sum(axis=1)  # [N, n_features]

        # For faithfulness/ablation metrics, we need per-timestep importance
        # Sum across features to get [N, seq_length] importance per timestep
        shap_per_timestep = shap_per_feature.sum(axis=2)  # [N, seq_length]

        # Compute faithfulness and ablation for each point
        from .metrics import compute_point_faithfulness, compute_point_ablation

        faithfulness_scores = []
        ablation_scores = []
        print(f"Computing faithfulness and ablation for {len(X_explain)} points...")
        for i in range(len(X_explain)):
            faith = compute_point_faithfulness(
                model=self.model,
                shap_values=shap_per_timestep[i],  # [seq_length]
                input_sequence=X_explain[i:i+1],  # [1, seq_length, n_features]
                percentiles=[90, 70, 50],
                eval_types=['prtb', 'sqnc'],
                seq_len=self.seq_length
            )
            faithfulness_scores.append(faith)

            ablation = compute_point_ablation(
                model=self.model,
                shap_values=shap_per_timestep[i],  # [seq_length]
                input_sequence=X_explain[i:i+1],  # [1, seq_length, n_features]
                percentiles=[90, 70, 50],
                ablation_types=['mif', 'lif']
            )
            ablation_scores.append(ablation)

        # Create results DataFrame
        results = {
            'end_index': np.arange(start_idx + self.seq_length, start_idx + self.seq_length + len(X_explain)).tolist(),
            'y_true': y[start_idx:end_idx].tolist(),
            'y_hat': y_hat.tolist() if hasattr(y_hat, 'tolist') else list(y_hat),
        }

        # Add SHAP values
        # shap_per_feature has shape [N, seq_length, n_features]
        # For lags (feature 0): report per-timestep SHAP values
        # For covariates (features 1+): sum across timesteps

        # Lag SHAP values per timestep
        lag_shap = shap_per_feature[:, :, 0]  # [N, seq_length]
        for t in range(self.seq_length):
            # Store with semantic naming: t-1 is most recent (last position)
            lag_idx = self.seq_length - 1 - t  # Map t=0 to most recent
            results[f'shap_lag_t-{t+1}'] = lag_shap[:, lag_idx].tolist()

        # Covariate SHAP values (sum across sequence)
        if self.n_features > 1:
            # Start from feature index 1 (after lags)
            cov_shap = shap_per_feature[:, :, 1:]  # [N, seq_length, n_cov]
            cov_shap_sum = cov_shap.sum(axis=1)  # [N, n_cov]

            n_cov = self.n_features - 1
            for cov_idx in range(n_cov):
                results[f'shap_Z_{cov_idx}'] = cov_shap_sum[:, cov_idx].tolist()

        # Add faithfulness scores - ensure they are scalar values
        for key in faithfulness_scores[0].keys():
            results[key] = [float(f[key]) for f in faithfulness_scores]

        # Add ablation scores - ensure they are scalar values
        for key in ablation_scores[0].keys():
            results[key] = [float(a[key]) for a in ablation_scores]

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

    def rolling_explain(self, data, covariates=None, stride=1, verbose=False):
        """
        Compute SHAP values using rolling windows.

        Parameters
        ----------
        data : 1D array
        covariates : 2D array of shape [T, n_cov], optional
        stride : int - step size between windows
        verbose : bool

        Returns
        -------
        DataFrame with columns: end_index, window_len, shap_lag_*, shap_Z_*, y_hat, faithfulness_*, ablation_*
        """
        results = []
        n = len(data)

        # Start from window_size and slide forward
        for end_idx in range(self.window_size, n, stride):
            start_idx = end_idx - self.window_size
            window_data = data[start_idx:end_idx]
            window_cov = covariates[start_idx:end_idx] if covariates is not None else None

            # Train model on this window
            X_win, y_win, feature_names = create_sequences(window_data, self.seq_length, covariates=window_cov)

            if len(X_win) < self.seq_length + 1:
                continue

            n_features = X_win.shape[2]

            model = LSTMModel(
                input_size=n_features,
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

            # Flatten background for SHAP: [N, seq_length, n_features] -> [N, seq_length * n_features]
            background_2d = background.reshape(len(background), -1)

            explainer = shap.KernelExplainer(model.predict, background_2d)

            # Explain the last point in the window
            X_last = X_win[-1:, :, :]
            X_last_2d = X_last.reshape(1, -1)  # Flatten for SHAP: [1, seq_length * n_features]
            shap_values = explainer.shap_values(X_last_2d, nsamples=self.shap_nsamples)
            y_hat = model.predict(X_last)[0, 0]

            # Process SHAP values
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)

            # Reshape: [1, seq_length * n_features] -> [1, seq_length, n_features]
            shap_values_3d = shap_values.reshape(1, self.seq_length, n_features)
            shap_per_feature = np.abs(shap_values_3d)  # [1, seq_length, n_features]

            # Sum across sequence length: [1, n_features]
            shap_per_feature_sum = shap_per_feature.sum(axis=1).squeeze()  # [n_features]

            # For faithfulness/ablation, sum across features: [1, seq_length]
            shap_per_timestep = shap_per_feature.sum(axis=2).squeeze()  # [seq_length]

            # Compute faithfulness and ablation for this point (using the rolling window model)
            from .metrics import compute_point_faithfulness, compute_point_ablation

            faithfulness = compute_point_faithfulness(
                model=model,  # Use the rolling window model
                shap_values=shap_per_timestep,  # [seq_length]
                input_sequence=X_last,  # [1, seq_length, n_features]
                percentiles=[90, 70, 50],
                eval_types=['prtb', 'sqnc'],
                seq_len=self.seq_length
            )

            ablation = compute_point_ablation(
                model=model,  # Use the rolling window model
                shap_values=shap_per_timestep,  # [seq_length]
                input_sequence=X_last,  # [1, seq_length, n_features]
                percentiles=[90, 70, 50],
                ablation_types=['mif', 'lif']
            )

            # Store results
            result_row = {
                'end_index': end_idx,
                'window_len': self.window_size,
                'y_true': float(data[end_idx]),
                'y_hat': y_hat
            }

            # Add SHAP values
            # shap_per_feature has shape [1, seq_length, n_features]
            # For lags (feature 0): report per-timestep SHAP values
            # For covariates (features 1+): sum across timesteps

            # Lag SHAP values per timestep
            lag_shap = shap_per_feature[0, :, 0]  # [seq_length]
            for t in range(self.seq_length):
                # Store with semantic naming: t-1 is most recent (last position)
                lag_idx = self.seq_length - 1 - t  # Map t=0 to most recent
                result_row[f'shap_lag_t-{t+1}'] = float(lag_shap[lag_idx])

            # Covariate SHAP values (sum across sequence)
            if n_features > 1:
                # Start from feature index 1 (after lags)
                cov_shap = shap_per_feature[0, :, 1:]  # [seq_length, n_cov]
                cov_shap_sum = cov_shap.sum(axis=0)  # [n_cov]

                n_cov = n_features - 1
                for cov_idx in range(n_cov):
                    result_row[f'shap_Z_{cov_idx}'] = float(cov_shap_sum[cov_idx])

            # Add faithfulness and ablation scores
            result_row.update(faithfulness)
            result_row.update(ablation)

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
        self.feature_names = None

    def fit(self, data, verbose=False):
        """Train a global model."""
        X, y, feature_names = create_sequences(data, self.seq_length)
        self.feature_names = feature_names

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
        Compute TimeShap explanations using local explanations.

        Parameters
        ----------
        data : 1D array - full data series
        start_idx : int - start index for explanation (default: 0)
        end_idx : int - end index for explanation (default: len(data))

        Returns
        -------
        DataFrame with columns: end_index, y_true, y_hat, shap_lag_t-*, faithfulness_*, ablation_*
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        from timeshap.explainer import local_pruning, local_event, local_feat
        from timeshap.wrappers import TorchModelWrapper
        from timeshap.utils import calc_avg_event

        # Create sequences
        X, y, feature_names = create_sequences(data, self.seq_length)

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(X)

        # Wrap model for TimeShap
        model_wrapped = TorchModelWrapper(self.model)
        f_hs = lambda x, y=None: model_wrapped.predict_last_hs(x, y)

        # Calculate average event (baseline) using all training data
        # For simplicity, use mean of all features
        average_event = np.mean(X, axis=(0, 1)).reshape(1, -1)

        # Process each sequence
        results = []
        for i in range(start_idx, end_idx):
            # Get single sequence as [1, seq_length, features]
            x_instance = X[i:i+1]

            # Get prediction
            y_hat = self.model.predict(x_instance)[0, 0]

            # Compute pruning and explanations
            try:
                pruning_dict = {'tol': 0.025}
                coal_plot_data, coal_prun_idx = local_pruning(
                    f_hs, x_instance, pruning_dict, average_event,
                    f'instance_{i}', 'id', False
                )
                pruning_idx = x_instance.shape[1] + coal_prun_idx

                # Event-level explanations
                event_dict = {'rs': 42, 'nsamples': 1000}
                event_data = local_event(
                    f_hs, x_instance, event_dict, f'instance_{i}', 'id',
                    average_event, pruning_idx
                )

                # Extract SHAP values from event data
                # event_data is a DataFrame with columns for each event
                result_row = {
                    'end_index': start_idx + self.seq_length + i,
                    'y_true': float(y[i]),
                    'y_hat': float(y_hat)
                }

                # Add SHAP values per timestep (lag features)
                # Assuming event_data has 'Shapley Value' column per event
                for t in range(min(self.seq_length, len(event_data))):
                    event_shap = event_data.iloc[t]['Shapley Value'] if t < len(event_data) else 0.0
                    result_row[f'shap_lag_t-{t+1}'] = float(event_shap)

                # Pad with zeros if we have fewer events than seq_length
                for t in range(len(event_data), self.seq_length):
                    result_row[f'shap_lag_t-{t+1}'] = 0.0

                # Compute proper faithfulness and ablation metrics
                # Extract SHAP values for this point
                shap_vals = np.array([result_row[f'shap_lag_t-{t+1}'] for t in range(self.seq_length)])

                # Get input sequence for this point: X[i] has shape [seq_length, features]
                input_seq = x_instance  # Already [1, seq_length, features]

                # Compute faithfulness metrics
                from .metrics import compute_point_faithfulness, compute_point_ablation
                faith_metrics = compute_point_faithfulness(
                    self.model, shap_vals, input_seq,
                    percentiles=[90, 70, 50], eval_types=['prtb', 'sqnc'],
                    seq_len=self.seq_length
                )

                # Compute ablation metrics
                ablation_metrics = compute_point_ablation(
                    self.model, shap_vals, input_seq,
                    percentiles=[90, 70, 50], ablation_types=['mif', 'lif']
                )

                # Add metrics to result row
                result_row.update(faith_metrics)
                result_row.update(ablation_metrics)

                results.append(result_row)

            except Exception as e:
                # If TimeShap fails, use simple prediction
                print(f"Warning: TimeShap explanation failed for instance {i}: {e}")
                result_row = {
                    'end_index': start_idx + self.seq_length + i,
                    'y_true': float(y[i]),
                    'y_hat': float(y_hat)
                }
                # Fill with zeros
                for t in range(self.seq_length):
                    result_row[f'shap_lag_t-{t+1}'] = 0.0
                # Fill faithfulness metrics with zeros for all percentiles
                for p in [90, 70, 50]:
                    result_row[f'faithfulness_prtb_p{p}'] = 0.0
                    result_row[f'faithfulness_sqnc_p{p}'] = 0.0
                    result_row[f'ablation_mif_p{p}'] = 0.0
                    result_row[f'ablation_lif_p{p}'] = 0.0
                results.append(result_row)

        return pd.DataFrame(results)
