import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import store_init_kwargs


class AdaptiveModel(nn.Module):
    """
    Base class for adaptive models used in LPA-based change detection.

    Subclasses must implement:
    - forward(x): PyTorch forward pass
    - prepare_data(window, start_abs_idx): Convert window to (X, y, t_abs) tensors
    - simulate_series(...): Generate synthetic series for Monte Carlo CV computation
    """

    def __init__(self, device, batch_size=512, lr=1e-2, epochs=50, type_precision=np.float32):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.type_precision = type_precision

    def fit(self, X, y):
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        model = self.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            model.train()
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return self

    @staticmethod
    def residuals(y, yhat):
        return y - yhat

    @staticmethod
    def sse(residuals):
        return np.sum(residuals**2)

    @staticmethod
    def mse(sse, n):
        return sse / max(n, 1)

    @staticmethod
    def likelihood(sse, n):
        return -(n / 2) * np.log(sse)

    def prepare_data(self, window, start_abs_idx):
        """
        :param window: The input data.
        :param start_abs_idx: The id from which the target data starts in the original time series. This function takes the window data.
        :return: X_tensor, y_tensor, t_abs
        """
        X, y = window[:, :-1], window[:, -1]
        X_tensor = X if isinstance(X, torch.Tensor) else torch.from_numpy(X)
        y_tensor = y if isinstance(y, torch.Tensor) else torch.from_numpy(y)
        t_abs = np.arange(start_abs_idx, start_abs_idx + len(y), dtype=np.int64)
        return X_tensor, y_tensor, t_abs

    def diagnostics(self, X, y, batch_size=512):
        ds = TensorDataset(X, y)

        self.eval()
        with torch.no_grad():
            all_pred, all_y = [], []
            evl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            for xb, yb in evl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self(xb)
                all_pred.append(pred.cpu().numpy())
                all_y.append(yb.cpu().numpy())
            yhat = np.concatenate(all_pred)
            y_true = np.concatenate(all_y)
        resid = AdaptiveModel.residuals(y_true, yhat)
        sse = AdaptiveModel.sse(resid)
        m = int(y_true.shape[0])
        mse = AdaptiveModel.mse(sse, m)
        likelihood = AdaptiveModel.likelihood(sse, m)
        return likelihood, yhat, resid, sse, mse, m

    @property
    def seq_length(self) -> int:
        """
        Return the sequence length (number of lags) used by the model.
        Must be implemented by subclasses that support simulation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement 'seq_length' property "
            "to support Monte Carlo critical value computation."
        )

    @staticmethod
    def draw_mammen_weights(m: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw weights from Mammen's two-point distribution for wild bootstrap.

        The distribution is:
            w_t = -1/φ  with probability φ/√5
            w_t = φ     with probability 1 - φ/√5

        where φ = (1 + √5) / 2 (golden ratio).

        This ensures E[w_t] = 0 and E[w_t²] = 1, which is required for the
        wild bootstrap to correctly mimic the first two moments of the error process.

        Parameters
        ----------
        m : int
            Number of weights to draw.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Array of m weights (dtype=float32).

        References
        ----------
        Härdle & Mammen (1991). Comparing nonparametric versus parametric regression fits.
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
        p = phi / np.sqrt(5)        # Probability of -1/φ ≈ 0.7236
        a = -1 / phi                # ≈ -0.618
        b = phi                     # ≈ 1.618
        u = rng.random(m)
        return np.where(u < p, a, b).astype(np.float32)

    def simulate_series(
        self,
        seed_values: np.ndarray,
        sigma: float,
        n_total: int,
        rng: np.random.Generator,
        method: str = "wild_bootstrap",
        residuals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate a homogeneous time series from the fitted model.

        This method is used for Monte Carlo computation of critical values
        under the null hypothesis (no change points).

        Parameters
        ----------
        seed_values : np.ndarray
            Initial values to seed the simulation (at least seq_length points).
        sigma : float
            Standard deviation of innovations (from fitted model residuals).
        n_total : int
            Total length of series to generate.
        rng : np.random.Generator
            Random number generator for reproducibility.
        method : str, default 'wild_bootstrap'
            Simulation method:
            - 'wild_bootstrap': y_t = predict(history) + w_t * resampled_residual
              where w_t ~ Mammen distribution (recommended for heteroskedasticity)
            - 'gaussian': y_t = predict(history) + N(0, sigma)
            - 'residual_bootstrap': y_t = predict(history) + resampled_residual
        residuals : np.ndarray, optional
            Array of residuals to resample from (required for 'wild_bootstrap'
            and 'residual_bootstrap' methods).

        Returns
        -------
        np.ndarray
            Simulated series of length n_total (dtype=float32).

        Raises
        ------
        NotImplementedError
            If the model subclass doesn't implement this method.
        ValueError
            If method requires residuals but residuals is None.

        References
        ----------
        Härdle & Mammen (1991). Comparing nonparametric versus parametric regression fits.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement 'simulate_series()' method "
            "to support Monte Carlo critical value computation."
        )


class AdaptiveLSTM(AdaptiveModel):
    """
    LSTM-based adaptive model for time series forecasting and change detection.

    This model uses an LSTM to predict the next value in a time series based on
    the previous `seq_length` values. It supports both univariate and multivariate
    time series (with exogenous covariates).

    Parameters
    ----------
    device : torch.device or str
        Device to run the model on ('cpu', 'cuda', 'mps').
    seq_length : int, default 3
        Number of lags (sequence length) for LSTM input.
    input_size : int, default 1
        Number of features per timestep (1 for univariate, >1 for multivariate).
    hidden : int, default 16
        LSTM hidden state size.
    layers : int, default 1
        Number of LSTM layers.
    dropout : float, default 0.0
        Dropout rate (only active if layers > 1).
    batch_size : int, default 64
        Training batch size.
    lr : float, default 1e-2
        Learning rate for AdamW optimizer.
    epochs : int, default 15
        Number of training epochs.
    type_precision : np.dtype, default np.float32
        Data type precision.
    """

    @store_init_kwargs
    def __init__(
        self,
        device,
        seq_length: int = 3,
        input_size: int = 1,
        hidden: int = 16,
        layers: int = 1,
        dropout: float = 0.0,
        batch_size: int = 64,
        lr: float = 1e-2,
        epochs: int = 15,
        type_precision=np.float32
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            type_precision=type_precision
        )
        self._seq_length = seq_length
        self.input_size = input_size
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, 1)

    @property
    def seq_length(self) -> int:
        return self._seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, L, F] -> [B]"""
        out, _ = self.lstm(x)           # [B, L, H]
        yhat = self.fc(out[:, -1, :])   # [B, 1]
        return yhat.squeeze(-1)         # [B]

    def prepare_data(self, window: np.ndarray, start_abs_idx: int):
        """
        Convert a time series window to LSTM input format.

        Parameters
        ----------
        window : np.ndarray
            Time series window, shape [n] for univariate or [n, F] for multivariate.
        start_abs_idx : int
            Absolute index of the first element in the original time series.

        Returns
        -------
        X_tensor : torch.Tensor or None
            Input sequences [N, L, F], or None if window too short.
        y_tensor : torch.Tensor or None
            Target values [N], or None if window too short.
        t_abs : np.ndarray or None
            Absolute target indices [N], or None if window too short.
        """
        L = self._seq_length
        n = len(window)

        if n <= L:
            return None, None, None

        # Ensure window is 2D: [n, F]
        if window.ndim == 1:
            window = window[:, None]

        F = window.shape[1]

        # Create sequences: for each time t, use [t-L:t] to predict t
        X_list = []
        y_list = []
        for i in range(L, n):
            X_list.append(window[i - L:i])  # [L, F]
            y_list.append(window[i, 0])     # target is always first column

        X = np.array(X_list, dtype=np.float32)  # [N, L, F]
        y = np.array(y_list, dtype=np.float32)  # [N]

        t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        return X_tensor, y_tensor, t_abs

    def simulate_series(
        self,
        seed_values: np.ndarray,
        sigma: float,
        n_total: int,
        rng: np.random.Generator,
        method: str = "wild_bootstrap",
        residuals: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate a homogeneous time series via recursive 1-step LSTM forecasts.

        Uses the fitted LSTM model for predictions and adds innovations:
        - Wild bootstrap: y_t = LSTM(y_{t-L:t}) + w_t * resampled_residual
          where w_t ~ Mammen distribution (recommended for heteroskedasticity)
        - Gaussian: y_t = LSTM(y_{t-L:t}) + N(0, sigma)
        - Residual bootstrap: y_t = LSTM(y_{t-L:t}) + resampled_residual

        Parameters
        ----------
        seed_values : np.ndarray
            Initial values to seed the simulation (at least seq_length points).
            Should be from the I_0 segment (e.g., last seq_length points).
        sigma : float
            Standard deviation of innovations (estimated from model residuals).
        n_total : int
            Total length of series to generate.
        rng : np.random.Generator
            Random number generator for reproducibility.
        method : str, default 'wild_bootstrap'
            'wild_bootstrap': innovations = w_t * resampled_residual (Mammen weights)
            'gaussian': innovations ~ N(0, sigma)
            'residual_bootstrap': innovations resampled IID from residuals array
        residuals : np.ndarray, optional
            Array of residuals to resample from (required for 'wild_bootstrap'
            and 'residual_bootstrap' methods).
        covariates : np.ndarray, optional
            Exogenous covariate values for the simulation period,
            shape [n_total, n_cov]. Required when model has input_size > 1.
            These are treated as fixed/observed (not simulated).

        Returns
        -------
        np.ndarray
            Simulated target series of length n_total (dtype=float32).

        References
        ----------
        Härdle & Mammen (1991). Comparing nonparametric versus parametric regression fits.
        """
        if n_total <= 0:
            return np.array([], dtype=np.float32)

        L = self._seq_length
        if len(seed_values) < L:
            raise ValueError(f"Need at least {L} seed values, got {len(seed_values)}")

        if method in ("residual_bootstrap", "wild_bootstrap") and residuals is None:
            raise ValueError(f"residuals array required for method='{method}'")

        y_sim = np.empty(n_total, dtype=np.float32)

        # Initialize with seed values
        init = np.asarray(seed_values[-L:], dtype=np.float32)
        n_init = min(L, n_total)
        y_sim[:n_init] = init[:n_init]

        if n_total <= L:
            return y_sim[:n_total]

        # Pre-generate all random values for efficiency
        n_sim = n_total - L
        if method == "gaussian":
            innovations = rng.normal(0, sigma, size=n_sim).astype(np.float32)
        elif method == "residual_bootstrap":
            innovations = rng.choice(residuals, size=n_sim).astype(np.float32)
        elif method == "wild_bootstrap":
            # Wild bootstrap: w_t * e_π(t)
            # where w_t ~ Mammen distribution and e_π(t) is resampled residual
            weights = AdaptiveModel.draw_mammen_weights(n_sim, rng)
            resampled = rng.choice(residuals, size=n_sim).astype(np.float32)
            innovations = weights * resampled
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Use 'wild_bootstrap', 'gaussian', or 'residual_bootstrap'."
            )

        # Recursive simulation
        self.eval()
        with torch.no_grad():
            for i, t in enumerate(range(L, n_total)):
                if covariates is not None:
                    # Multivariate: combine target history with covariate history
                    target_hist = y_sim[t - L:t].reshape(L, 1)
                    cov_hist = covariates[t - L:t]  # [L, n_cov]
                    x = np.concatenate([target_hist, cov_hist], axis=1)
                    x = x.reshape(1, L, -1)
                else:
                    x = y_sim[t - L:t].reshape(1, L, 1)
                xb = torch.from_numpy(x).to(self.device)
                pred = float(self(xb).item())
                y_sim[t] = np.float32(pred + innovations[i])

        return y_sim

    @torch.no_grad()
    def predict_single(self, x: np.ndarray) -> float:
        """
        Predict a single value from input sequence.

        Parameters
        ----------
        x : np.ndarray
            Input sequence, shape [L] or [L, F].

        Returns
        -------
        float
            Predicted value.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)  # [1, L, 1]
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])  # [1, L, F]

        xb = torch.from_numpy(x.astype(np.float32)).to(self.device)
        return float(self(xb).item())
