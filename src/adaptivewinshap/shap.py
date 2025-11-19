import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import shap
from typing import Callable, Optional, Tuple, Iterable, Dict, Any


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Small LSTM forecaster
# -------------------------
class LSTMRegressor(nn.Module):
    """
    Minimal LSTM regressor: input [B, L, F] -> output [B] via last hidden step.
    """
    def __init__(self, input_size: int = 1, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)             # [B, L, H]
        h_last = out[:, -1, :]            # [B, H]
        yhat   = self.fc(h_last)          # [B, 1]
        return yhat.squeeze(-1)           # [B]

    def predict(self, x):
        """Prediction method compatible with faithfulness computation."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

            # Get device from model parameters
            device = next(self.parameters()).device
            x = x.to(device)

            # Handle both 2D and 3D inputs
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # [N, seq_len] -> [N, seq_len, 1]

            preds = self.forward(x)
            return preds.cpu().numpy().reshape(-1, 1)


# -------------------------
# Utilities
# -------------------------
def create_sequences(arr_2d: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build next-step forecast sequences from (T, F):
      X: (T-L, L, F), y: (T-L, 1) predicting the first feature at t.
    """
    T, F = arr_2d.shape
    if T <= seq_length:
        return np.empty((0, seq_length, F), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    X, y = [], []
    for t in range(seq_length, T):
        X.append(arr_2d[t - seq_length:t, :])
        y.append(arr_2d[t, 0])  # predict first feature at time t
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def fixed_rolling_windows(arr: np.ndarray, window_size: int, step: int = 1) -> Iterable[Tuple[int, np.ndarray]]:
    """
    Yield (end_index_in_original_series, window_array) for fixed rolling windows.
    end_index corresponds to the raw series index of the last item in the window.
    """
    n = len(arr)
    for start in range(0, n - window_size + 1, step):
        end = start + window_size - 1
        yield end, arr[start:end + 1]


def windows_from_csv_sizes(data: np.ndarray, window_sizes: np.ndarray, min_len: int) -> Iterable[Tuple[int, np.ndarray]]:
    """
    Yield windows according to per-index sizes: at index i, use window_sizes[i].
    """
    n = len(data)
    for i in range(n):
        w = window_sizes[i]
        if pd.isna(w):
            continue
        w = int(w)
        if w < min_len:
            continue
        start = i - w + 1
        if start < 0:
            continue
        yield i, data[start:i + 1]


# -------------------------
# Reusable SHAP Runner
# -------------------------
class AdaptiveWinShap:
    """
    Train a small LSTM per window (in-sample) and compute SHAP contributions
    for the **last prediction** in that window.

    - Background data for Kernel SHAP = all sequences in the window (optionally subsampled).
    - Supports univariate or multivariate inputs.
    - Returns either per-lag aggregated SHAP or full (lag x feature) matrices.

    Parameters
    ----------
    seq_length : int
        Number of lags (sequence length).
    lstm_hidden : int
        LSTM hidden units.
    lstm_layers : int
        Number of LSTM layers.
    lstm_dropout : float
        Dropout for LSTM (only active if layers > 1).
    batch_size : int
        Training batch size.
    epochs : int
        Training epochs per window.
    lr : float
        Learning rate for AdamW.
    max_background : Optional[int]
        If set, randomly subsample background sequences to this many rows per window.
    shap_nsamples : int
        Kernel SHAP nsamples (trade speed/variance).
    aggregate_lag : bool
        If True, return per-lag SHAP by summing |contrib| across features for each lag.
    device : Optional[torch.device]
        Torch device; autodetected if None.
    """

    def __init__(
        self,
        seq_length: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        batch_size: int = 128,
        epochs: int = 50,
        lr: float = 1e-3,
        max_background: Optional[int] = None,
        shap_nsamples: int = 1000,
        aggregate_lag: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.seq_length = seq_length
        self.hidden = lstm_hidden
        self.layers = lstm_layers
        self.dropout = lstm_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_background = max_background
        self.shap_nsamples = shap_nsamples
        self.aggregate_lag = aggregate_lag
        self.device = device or get_device()

    # ---------- training ----------
    def _fit_model(self, X: np.ndarray, y: np.ndarray, input_size: int,
                   warm_state: Optional[Dict[str, Any]] = None) -> LSTMRegressor:
        model = LSTMRegressor(
            input_size=input_size,
            hidden=self.hidden,
            layers=self.layers,
            dropout=self.dropout
        ).to(self.device)

        if warm_state is not None:
            model.load_state_dict(warm_state)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.squeeze(-1)))
        dl = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            pin_memory=(self.device.type == "cuda"), num_workers=0
        )
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return model

    # ---------- SHAP helpers ----------
    def _predict_fn(self, model: nn.Module, input_size: int) -> Callable[[np.ndarray], np.ndarray]:
        L = self.seq_length
        dev = self.device

        @torch.no_grad()
        def predict(x_flat: np.ndarray) -> np.ndarray:
            # x_flat: [N, L*F] -> [N, L, F]
            N = x_flat.shape[0]
            F = input_size
            x = x_flat.reshape(N, L, F)
            xt = torch.tensor(x, dtype=torch.float32, device=dev)
            preds = model(xt)                     # [N]
            return preds.detach().cpu().numpy().reshape(-1, 1)  # SHAP expects 2D
        return predict

    def _build_background(self, X_flat: np.ndarray) -> np.ndarray:
        if self.max_background is None or X_flat.shape[0] <= self.max_background:
            return X_flat
        idx = np.random.default_rng(0).choice(X_flat.shape[0], self.max_background, replace=False)
        return X_flat[idx]

    def _explain_last(self, model: nn.Module, X_seq: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP for the last sequence in X_seq.
        Returns (shap_matrix[L, F], y_hat_last).
        """
        assert X_seq.ndim == 3  # [N, L, F]
        N, L, F = X_seq.shape
        assert L == self.seq_length

        X_flat = X_seq.reshape(N, L * F)
        bg = self._build_background(X_flat)

        predict = self._predict_fn(model, input_size=F)
        y_hat_last = predict(X_flat[-1:].copy())[0, 0]

        explainer = shap.KernelExplainer(predict, bg)
        shap_vals = explainer.shap_values(X_flat[-1:].copy(), nsamples=self.shap_nsamples, silent=True)
        # shap_vals shape: [1, L*F]
        shap_vec = np.asarray(shap_vals).reshape(L * F)
        shap_mat = shap_vec.reshape(L, F)
        return shap_mat, float(y_hat_last)

    # ---------- public API ----------
    def explain_window(
        self,
        window: np.ndarray,
        warm_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train on 'window' in-sample and compute SHAP for the last prediction.

        Parameters
        ----------
        window : np.ndarray
            Raw series window, shape (T,) for univariate or (T, F) for multivariate.
            If univariate, it will be expanded to (T, 1).
        warm_state : dict, optional
            Optional state_dict to warm-start the per-window model.

        Returns
        -------
        dict with keys:
          - 'y_hat': float
          - 'shap_lag': np.ndarray [L] (if aggregate_lag=True)
          - 'shap_full': np.ndarray [L, F] (always returned)
        """
        w = np.asarray(window, dtype=np.float32)
        if w.ndim == 1:
            w = w[:, None]  # (T, 1)

        X_seq, y_seq = create_sequences(w, self.seq_length)
        if X_seq.shape[0] == 0:
            return {"y_hat": np.nan, "shap_lag": np.array([]), "shap_full": np.zeros((self.seq_length, w.shape[1]))}

        model = self._fit_model(X_seq, y_seq, input_size=w.shape[1], warm_state=warm_state)
        shap_full, y_hat = self._explain_last(model, X_seq)

        if self.aggregate_lag:
            # L1 aggregate across features per lag (common for attribution per lag)
            shap_lag = np.sum(np.abs(shap_full), axis=1)
        else:
            shap_lag = np.zeros((0,))  # not used

        return {
            "y_hat": y_hat,
            "shap_lag": shap_lag,
            "shap_full": shap_full,
            "_model": model,  # Internal: model for faithfulness computation
            "_X_last": X_seq[-1:]  # Internal: last sequence used for prediction
        }

    def rolling_explain(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
        step: int = 1,
        window_sizes_csv: Optional[str] = None,
        csv_column: str = "window_size",
        min_window_len: Optional[int] = None,
        return_full: bool = True
    ) -> pd.DataFrame:
        """
        Iterate windows over 'data' and compute SHAP per window for the last prediction.

        Choose exactly one of (window_size) or (window_sizes_csv).

        Parameters
        ----------
        data : np.ndarray
            Raw series (T,) or (T, F).
        window_size : int, optional
            Fixed rolling window size.
        step : int
            Step for fixed rolling windows.
        window_sizes_csv : str, optional
            Path to CSV with a per-index 'window_size' column.
        csv_column : str
            Column name in CSV that holds sizes (default 'window_size').
        min_window_len : int, optional
            Minimum window length; default is seq_length + 1.
        return_full : bool
            If True, includes per-lag columns; if False, returns just summary cols.

        Returns
        -------
        pd.DataFrame
            Columns: end_index, window_len, y_hat, and either:
              - shap_lstm_t-1, shap_lstm_t-2, ... (when aggregate_lag=True)
              - or flattened shap_full_l{lag}_f{feat} columns when aggregate_lag=False
        """
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]

        if min_window_len is None:
            min_window_len = self.seq_length + 1  # need at least one (X,y)

        # Build iterator
        if window_sizes_csv is not None:
            ws = pd.read_csv(window_sizes_csv)
            if csv_column not in ws.columns:
                # try to discover a plausible column
                candidates = [c for c in ws.columns if "window" in c.lower()]
                if not candidates:
                    raise ValueError(f"CSV must contain '{csv_column}' column.")
                csv_column = candidates[0]
            sizes = ws[csv_column].to_numpy()
            iterator = windows_from_csv_sizes(arr, sizes, min_len=min_window_len)
        else:
            if window_size is None:
                raise ValueError("Provide either window_size or window_sizes_csv.")
            iterator = fixed_rolling_windows(arr, window_size=window_size, step=step)

        records = []
        for end_idx, window in iterator:
            # train + shap for this window
            out = self.explain_window(window)
            row = {
                "end_index": int(end_idx),
                "window_len": int(window.shape[0]),
                "y_hat": float(out["y_hat"]),
            }

            L, F = out["shap_full"].shape

            if self.aggregate_lag:
                # name: shap_lstm_t-1, shap_lstm_t-2, ...
                for lag in range(1, L + 1):
                    row[f"shap_lstm_t-{lag}"] = float(out["shap_lag"][L - lag])  # t-1 is last row
            else:
                # export full matrix
                for l in range(L):
                    for f in range(F):
                        row[f"shap_full_l{l}_f{f}"] = float(out["shap_full"][l, f])

            # Compute faithfulness and ablation if model and sequence are available
            if "_model" in out and "_X_last" in out:
                try:
                    # Import here to avoid circular dependencies
                    import sys
                    import os
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples'))
                    from benchmarking.metrics import compute_point_faithfulness, compute_point_ablation

                    # Use the aggregated SHAP values (per lag)
                    shap_vals = out["shap_lag"] if self.aggregate_lag else np.sum(np.abs(out["shap_full"]), axis=1)

                    faithfulness = compute_point_faithfulness(
                        model=out["_model"],
                        shap_values=shap_vals,
                        input_sequence=out["_X_last"],
                        percentiles=[90, 70, 50],
                        eval_types=['prtb', 'sqnc'],
                        seq_len=self.seq_length
                    )
                    row.update(faithfulness)

                    ablation = compute_point_ablation(
                        model=out["_model"],
                        shap_values=shap_vals,
                        input_sequence=out["_X_last"],
                        percentiles=[90, 70, 50],
                        ablation_types=['mif', 'lif']
                    )
                    row.update(ablation)
                except Exception as e:
                    # If metric computation fails, continue without it
                    import warnings
                    warnings.warn(f"Metric computation failed: {e}")

            records.append(row)

        df = pd.DataFrame(records).sort_values("end_index").reset_index(drop=True)
        if not return_full and self.aggregate_lag:
            # keep summary columns only
            keep = ["end_index", "window_len", "y_hat"] + [c for c in df.columns if c.startswith("shap_lstm_t-")]
            return df[keep]
        return df
