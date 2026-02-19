import time
import math
import os
from typing import Optional, Dict, Tuple, Callable

import numpy as np
import pandas as pd
import torch

from joblib import Parallel, delayed

# Try to import tqdm for progress bars
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

from .model import AdaptiveModel


class ChangeDetector:
    """
    LPA-based change point detector with Monte Carlo pre-computed critical values.

    This detector implements the Local Parametric Approach (Spokoiny 1998, Cizek 2009)
    for adaptive window selection in time series analysis. Critical values are computed
    via Monte Carlo simulation under the null hypothesis of homogeneity.

    Parameters
    ----------
    model : AdaptiveModel
        The model to use for fitting (must support simulate_series for MC).
    data : np.ndarray
        The time series data (1D or 2D with target in first column).
    debug : bool, default False
        Enable debug output.
    force_cpu : bool, default False
        Force CPU computation even if GPU is available.

    Attributes
    ----------
    _critical_values : pd.DataFrame or None
        Pre-computed critical values table (after precompute or load).
    _cv_params : dict or None
        Parameters used for critical value computation.

    Example
    -------
    >>> model = AdaptiveLSTM(device='cpu', seq_length=3)
    >>> detector = ChangeDetector(model, data)
    >>> detector.precompute_critical_values(data, mc_reps=300, alpha=0.95)
    >>> detector.save_critical_values("cv_table.csv")
    >>> results = detector.detect(n_0=100, jump=10, alpha=0.95)
    """

    def __init__(self, model: AdaptiveModel, data, debug=False, force_cpu=False):
        self.model = model
        self.data = data
        self.debug = debug
        self.previous_device = 0
        self.force_cpu = force_cpu

        # Critical value storage
        self._critical_values: Optional[pd.DataFrame] = None
        self._cv_params: Optional[Dict] = None

    # =========================================================================
    # GPU Memory Management
    # =========================================================================
    @staticmethod
    def _estimate_gpu_memory_per_scale(seq_length: int, hidden: int, batch_size: int) -> int:
        """Estimate GPU memory usage per scale in bytes (rough estimate)."""
        # Model params: ~4 * hidden * (input_size + hidden + 1) for LSTM + FC
        model_params = 4 * hidden * (1 + hidden + 1) * 4  # float32
        # Activations: batch_size * seq_length * hidden * 2 (forward + backward)
        activations = batch_size * seq_length * hidden * 2 * 4
        # Buffer for gradients, optimizer states (~2x model params)
        overhead = model_params * 3
        return model_params + activations + overhead

    @staticmethod
    def _get_max_concurrent_scales() -> int:
        """Auto-detect maximum concurrent scales based on GPU memory."""
        if not torch.cuda.is_available():
            # CPU: use number of cores
            import multiprocessing
            return max(1, multiprocessing.cpu_count() // 2)

        try:
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory
            # Reserve 20% for system overhead
            available = int(total_memory * 0.8)
            # Estimate per-scale usage (conservative: ~200MB per scale)
            per_scale = 200 * 1024 * 1024
            max_scales = max(1, available // per_scale)
            return min(max_scales, 8)  # Cap at 8 to avoid oversubscription
        except Exception:
            return 4  # Safe default

    # =========================================================================
    # Critical Value Computation (Monte Carlo)
    # =========================================================================
    def _fit_i0_model(
        self,
        data: np.ndarray,
        n_0: int
    ) -> Tuple["AdaptiveModel", float, np.ndarray, Optional[np.ndarray]]:
        """
        Fit model on I_0 (last n_0 points) and compute residual statistics.

        Parameters
        ----------
        data : np.ndarray
            Full time series.
        n_0 : int
            Initial window size.

        Returns
        -------
        model : AdaptiveModel
            Fitted model on I_0.
        sigma : float
            Standard deviation of residuals.
        seed_values : np.ndarray
            Last seq_length points for simulation seeding.
        residuals : np.ndarray
            Residuals from I_0 fit (for residual bootstrap).
        """
        # Extract I_0 (last n_0 points)
        i0_data = data[-n_0:].astype(np.float32)
        if i0_data.ndim == 1:
            i0_data = i0_data[:, None]

        # Prepare sequences
        X, y, _ = self.model.prepare_data(i0_data, 0)
        if X is None:
            raise ValueError(f"I_0 too short: need > {self.model.seq_length} points, got {n_0}")

        # Fit model
        model = self.construct_new_model()
        model.fit(X, y)

        # Get residuals
        _, yhat, resid, _, _, _ = model.diagnostics(X, y)
        sigma = float(np.std(resid))

        # Seed values for simulation
        target = i0_data[:, 0] if i0_data.ndim == 2 else i0_data
        seed_values = target[-self.model.seq_length:].astype(np.float32)

        return model, sigma, seed_values, resid.astype(np.float32)

    def _compute_suplr_for_series(
        self,
        y: np.ndarray,
        j_start_pos: int,
        j_end_pos: int,
        search_step: int,
        min_seg: int
    ) -> float:
        """
        Compute SupLR for a single series over specified J_k range.

        Parameters
        ----------
        y : np.ndarray
            Time series array (1D).
        j_start_pos : int
            Start of J_k range (in series coordinates).
        j_end_pos : int
            End of J_k range (in series coordinates).
        search_step : int
            Split search granularity.
        min_seg : int
            Minimum segment size.

        Returns
        -------
        float
            Maximum LR statistic over splits in J_k.
        """
        # Ensure 2D
        if y.ndim == 1:
            y = y[:, None]

        # Prepare data
        X, y_target, t_idx = self.model.prepare_data(y, 0)
        if X is None:
            return 0.0

        # Fit on full window
        model_full = self.construct_new_model()
        model_full.fit(X, y_target)
        ll_full, _, _, _, _, _ = model_full.diagnostics(X, y_target)

        n_targets = len(y_target)
        seq_len = self.model.seq_length

        # Convert J_k range to target indices
        split_start = max(min_seg, j_start_pos - seq_len)
        split_end = min(n_targets - min_seg, j_end_pos - seq_len)

        if split_end <= split_start:
            return 0.0

        T_vals = []
        for split_idx in range(split_start, split_end, search_step):
            X_left, y_left = X[:split_idx], y_target[:split_idx]
            X_right, y_right = X[split_idx:], y_target[split_idx:]

            if len(y_left) < min_seg or len(y_right) < min_seg:
                continue

            model_left = self.construct_new_model()
            model_left.fit(X_left, y_left)
            ll_left, _, _, _, _, _ = model_left.diagnostics(X_left, y_left)

            model_right = self.construct_new_model()
            model_right.fit(X_right, y_right)
            ll_right, _, _, _, _, _ = model_right.diagnostics(X_right, y_right)

            T_i = ll_left + ll_right - ll_full
            T_vals.append(max(0.0, T_i))

        return max(T_vals) if T_vals else 0.0

    def _compute_single_mc_rep(
        self,
        i0_model: "AdaptiveModel",
        seed_values: np.ndarray,
        sigma: float,
        n_total: int,
        j_start: int,
        j_end: int,
        search_step: int,
        min_seg: int,
        simulation_method: str,
        residuals: Optional[np.ndarray],
        rep_seed: int,
        covariates: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute SupLR for a single MC replication.

        This is the unit of parallelization - each rep is independent.
        """
        rng = np.random.default_rng(rep_seed)

        # Simulate homogeneous target series
        y_sim = i0_model.simulate_series(
            seed_values=seed_values,
            sigma=sigma,
            n_total=n_total,
            rng=rng,
            method=simulation_method,
            residuals=residuals,
            covariates=covariates
        )

        # For multivariate models, combine simulated target with observed covariates
        if covariates is not None:
            sim_data = np.column_stack([y_sim[:, None], covariates[:len(y_sim)]])
        else:
            sim_data = y_sim

        # Compute SupLR over J_k
        suplr = self._compute_suplr_for_series(
            sim_data, j_start, j_end, search_step, min_seg
        )
        return max(0.0, suplr)

    def _compute_cv_for_scale(
        self,
        k: int,
        n_0: int,
        c: float,
        i0_model: "AdaptiveModel",
        sigma: float,
        seed_values: np.ndarray,
        residuals: Optional[np.ndarray],
        mc_reps: int,
        search_step: int,
        min_seg: int,
        simulation_method: str,
        seed: int,
        n_jobs: int = -1,
        show_progress: bool = False,
        covariates: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute critical value for a single scale k via Monte Carlo.

        Parameters
        ----------
        k : int
            Scale index.
        n_0 : int
            Initial window size.
        c : float
            Geometric ratio.
        i0_model : AdaptiveModel
            Fitted I_0 model for simulation.
        sigma : float
            Residual standard deviation.
        seed_values : np.ndarray
            Initial values for simulation.
        residuals : np.ndarray or None
            Residuals for bootstrap method.
        mc_reps : int
            Number of Monte Carlo replications.
        search_step : int
            Split search step.
        min_seg : int
            Minimum segment size.
        simulation_method : str
            'gaussian' or 'residual_bootstrap'.
        seed : int
            Random seed (offset by k for different scales).
        n_jobs : int
            Number of parallel jobs for MC reps (-1 = all cores).
        show_progress : bool
            Show progress bar for MC replications.

        Returns
        -------
        dict
            Dictionary with scale statistics.
        """
        n_k_minus1 = n_0 if k == 1 else int(n_0 * c ** (k - 1))
        n_k = int(n_0 * c ** k)
        n_k_plus1 = int(n_0 * c ** (k + 1))

        # J_k range (in series coordinates)
        j_start = n_k_plus1 - n_k
        j_end = n_k_plus1 - n_k_minus1

        # Generate unique seeds for each rep
        base_seed = seed + k * 10000
        rep_seeds = [base_seed + r for r in range(mc_reps)]

        # Prepare residuals for bootstrap methods (wild_bootstrap and residual_bootstrap)
        resid = residuals if simulation_method in ("wild_bootstrap", "residual_bootstrap") else None

        # Prepare covariate slice for this scale's series length
        # Use last n_k_plus1 observed covariate values (exogenous, fixed under null)
        scale_covariates = None
        if covariates is not None:
            if len(covariates) >= n_k_plus1:
                scale_covariates = covariates[-n_k_plus1:]
            else:
                # Data shorter than needed: tile covariates
                n_tiles = (n_k_plus1 // len(covariates)) + 1
                scale_covariates = np.tile(covariates, (n_tiles, 1))[:n_k_plus1]

        if show_progress and HAS_TQDM:
            # Parallel with progress bar - use tqdm to track completions
            pbar = tqdm(
                total=mc_reps,
                desc=f"    k={k} (n={n_k_plus1})",
                unit="rep",
                leave=False,
                dynamic_ncols=True
            )

            # Collect results and update progress
            suplr_values = []
            with Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator") as parallel:
                results_gen = parallel(
                    delayed(self._compute_single_mc_rep)(
                        i0_model, seed_values, sigma, n_k_plus1,
                        j_start, j_end, search_step, min_seg,
                        simulation_method, resid, rep_seed,
                        scale_covariates
                    )
                    for rep_seed in rep_seeds
                )
                for result in results_gen:
                    suplr_values.append(result)
                    pbar.update(1)

            pbar.close()
        else:
            # Parallel without progress
            suplr_values = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(self._compute_single_mc_rep)(
                    i0_model, seed_values, sigma, n_k_plus1,
                    j_start, j_end, search_step, min_seg,
                    simulation_method, resid, rep_seed,
                    scale_covariates
                )
                for rep_seed in rep_seeds
            )

        suplr_arr = np.array(suplr_values)

        return {
            'k': k,
            'n_k': n_k,
            'n_k_plus1': n_k_plus1,
            'j_start': j_start,
            'j_end': j_end,
            'critical_value_95': float(np.quantile(suplr_arr, 0.95)),
            'critical_value_99': float(np.quantile(suplr_arr, 0.99)),
            'mean': float(np.mean(suplr_arr)),
            'std': float(np.std(suplr_arr)),
        }

    def precompute_critical_values(
        self,
        data: np.ndarray,
        n_0: int = 100,
        mc_reps: int = 300,
        alpha: float = 0.95,
        search_step: int = 1,
        min_seg: int = 20,
        penalty_factor: float = 0.25,
        simulation_method: str = "wild_bootstrap",
        growth_base: float = 1.4142135623730951,  # sqrt(2)
        seed: int = 42,
        verbose: bool = True
    ) -> "ChangeDetector":
        """
        Pre-compute critical values via Monte Carlo simulation under the null hypothesis.

        This method:
        1. Fits an I_0 model on the last n_0 points of `data`
        2. For each scale k, simulates mc_reps homogeneous series of length n_{k+1}
        3. Computes SupLR distribution for each scale
        4. Extracts alpha-quantile as raw critical value
        5. Applies Spokoiny adjustment

        Parameters
        ----------
        data : np.ndarray
            The time series data. Used to determine K_max and fit I_0 model.
        n_0 : int, default 100
            Initial window size.
        mc_reps : int, default 300
            Number of Monte Carlo replications per scale.
        alpha : float, default 0.95
            Significance level (e.g., 0.95 for 95th percentile).
        search_step : int, default 1
            Step size for split search within J_k range.
        min_seg : int, default 20
            Minimum segment size for valid splits.
        penalty_factor : float, default 0.25
            Lambda for Spokoiny adjustment. Higher = more conservative for small windows.
            Set to 0 to disable adjustment.
        simulation_method : str, default 'wild_bootstrap'
            'wild_bootstrap': y_t = predict(history) + w_t * resampled_residual
                where w_t ~ Mammen distribution. Recommended for heteroskedasticity.
            'gaussian': Use Gaussian innovations N(0, sigma)
            'residual_bootstrap': Resample from I_0 residuals IID
        growth_base : float, default sqrt(2)
            Geometric ratio for window sizes.
        seed : int, default 42
            Random seed for reproducibility.
        verbose : bool, default True
            Print progress during computation.

        Returns
        -------
        self
            Returns self for method chaining.

        Notes
        -----
        - I_0 must be homogeneous (no change points). This is the user's responsibility.
        - Computation is parallelized across MC replications with auto-detected concurrency.
        - Call save_critical_values() to persist results.
        - The wild bootstrap uses Mammen's two-point distribution for weights, which
          ensures E[w_t]=0 and E[w_t²]=1 to correctly mimic error moments.

        References
        ----------
        Härdle & Mammen (1991). Comparing nonparametric versus parametric regression fits.
        Spokoiny (1998). Estimation of a function with discontinuities via local polynomial fit.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            target = data[:, 0]
        else:
            target = data

        T = len(target)
        c = growth_base

        if verbose:
            print("=" * 60)
            print("Monte Carlo Critical Value Computation")
            print("=" * 60)
            print(f"  Data length: {T}")
            print(f"  n_0: {n_0}")
            print(f"  MC replications: {mc_reps}")
            print(f"  Alpha: {alpha}")
            print(f"  Penalty factor (lambda): {penalty_factor}")
            print(f"  Simulation method: {simulation_method}")
            print(f"  Growth base: {c:.4f}")
            print(f"  Seed: {seed}")
            print()

        # Step 1: Fit I_0 model (pass full data for multivariate support)
        if verbose:
            print("Step 1: Fitting model on I_0...")
        i0_model, sigma, seed_values, residuals = self._fit_i0_model(data, n_0)

        # Extract covariates for MC simulation (exogenous, treated as fixed)
        if data.ndim == 2 and data.shape[1] > 1:
            covariates = data[:, 1:].astype(np.float32)
        else:
            covariates = None

        if verbose:
            print(f"  Residual sigma: {sigma:.4f}")
            if covariates is not None:
                print(f"  Covariates: {covariates.shape[1]} columns (used in MC simulation)")
            print()

        # Step 2: Determine scales
        K_max = 0
        for k in range(1, 100):
            n_k_plus1 = int(n_0 * c ** (k + 1))
            if n_k_plus1 > T:
                break
            K_max = k

        if K_max == 0:
            raise ValueError(f"Data too short for any scales. Need at least {int(n_0 * c ** 2)} points.")

        if verbose:
            print(f"Step 2: Computing critical values for k=1..{K_max}")

        # Step 3: MC simulation - parallelize across MC reps within each scale
        max_workers = self._get_max_concurrent_scales()

        if verbose:
            print(f"  Parallel workers: {max_workers}")
            print(f"  Total MC fits: {K_max} scales x {mc_reps} reps = {K_max * mc_reps}")
            print()

        t0 = time.time()

        # Run scales sequentially, parallelize MC reps within each scale
        results = []
        for k in range(1, K_max + 1):
            n_k = int(n_0 * c ** k)
            n_k_plus1 = int(n_0 * c ** (k + 1))

            if verbose:
                print(f"  Scale {k}/{K_max}: n_k={n_k}, simulating {mc_reps} series of length {n_k_plus1}")

            result = self._compute_cv_for_scale(
                k=k, n_0=n_0, c=c, i0_model=i0_model,
                sigma=sigma, seed_values=seed_values,
                residuals=residuals, mc_reps=mc_reps,
                search_step=search_step, min_seg=min_seg,
                simulation_method=simulation_method, seed=seed,
                n_jobs=max_workers,
                show_progress=verbose,
                covariates=covariates
            )
            results.append(result)

            if verbose:
                print(f"    -> CV(95%)={result['critical_value_95']:.2f}, "
                      f"CV(99%)={result['critical_value_99']:.2f}, "
                      f"mean={result['mean']:.2f}, std={result['std']:.2f}")

        if verbose:
            print()

        t1 = time.time()

        # Build DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('k').reset_index(drop=True)

        # Step 4: Apply Spokoiny adjustment
        if verbose:
            print(f"Step 3: Applying Spokoiny adjustment (lambda={penalty_factor})...")

        n_k_values = df['n_k'].values
        n_K_max = n_k_values.max()

        ratio = n_K_max / n_k_values  # >= 1, largest for k=1
        adjustment = 1 + penalty_factor * np.sqrt(np.log(ratio))

        df['adjustment_factor'] = adjustment
        df['critical_value_95'] = df['critical_value_95'] * adjustment
        df['critical_value_99'] = df['critical_value_99'] * adjustment
        df['penalty_factor'] = penalty_factor

        # Store results
        self._critical_values = df
        self._cv_params = {
            'n_0': n_0,
            'alpha': alpha,
            'mc_reps': mc_reps,
            'growth_base': c,
            'penalty_factor': penalty_factor,
            'simulation_method': simulation_method,
            'seed': seed,
            'data_length': T
        }

        if verbose:
            print()
            print(f"Critical value computation completed in {t1 - t0:.1f}s")
            print()
            print("Critical Values Summary:")
            print("-" * 50)
            for _, row in df.iterrows():
                print(f"  k={int(row['k'])}: n_k={int(row['n_k'])}, "
                      f"CV(95%)={row['critical_value_95']:.2f} "
                      f"(adj={row['adjustment_factor']:.3f})")
            print("=" * 60)

        return self

    def save_critical_values(self, path: str) -> None:
        """
        Save pre-computed critical values to CSV file.

        Parameters
        ----------
        path : str
            File path (should end in .csv).

        Raises
        ------
        ValueError
            If critical values haven't been computed yet.
        """
        if self._critical_values is None:
            raise ValueError(
                "Critical values not computed. Call precompute_critical_values() first."
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        self._critical_values.to_csv(path, index=False)
        print(f"Critical values saved to: {path}")

    def load_critical_values(self, path: str) -> "ChangeDetector":
        """
        Load pre-computed critical values from CSV file.

        Parameters
        ----------
        path : str
            Path to CSV file (from save_critical_values or notebook output).

        Returns
        -------
        self
            Returns self for method chaining.

        Notes
        -----
        Expected CSV columns: k, n_k, n_k_plus1, j_start, j_end,
                              critical_value_95, adjustment_factor, penalty_factor
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical values file not found: {path}")

        df = pd.read_csv(path)

        # Validate required columns
        required = ['k', 'n_k', 'critical_value_95']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        self._critical_values = df

        # Extract parameters if available
        if 'penalty_factor' in df.columns:
            penalty = df['penalty_factor'].iloc[0]
        else:
            penalty = 0.25  # default

        self._cv_params = {
            'loaded_from': path,
            'penalty_factor': penalty
        }

        print(f"Critical values loaded from: {path}")
        print(f"  Scales: k=1..{int(df['k'].max())}")
        return self

    def get_critical_value(self, k: int, alpha: float = 0.95) -> float:
        """
        Get critical value for scale k.

        Parameters
        ----------
        k : int
            Scale index.
        alpha : float, default 0.95
            Significance level (0.95 or 0.99).

        Returns
        -------
        float
            Critical value, or inf if scale not found.
        """
        if self._critical_values is None:
            raise ValueError("Critical values not computed/loaded.")

        col = f"critical_value_{int(alpha * 100)}"
        if col not in self._critical_values.columns:
            col = "critical_value_95"  # fallback

        row = self._critical_values[self._critical_values['k'] == k]
        if len(row) == 0:
            return math.inf
        return float(row[col].iloc[0])

    # =========================================================================
    # Legacy methods (kept for reference, will be removed)
    # =========================================================================
    @staticmethod
    def draw_rademacher(m, rng):
        return (rng.integers(0, 2, size=m) * 2 - 1).astype(np.float32)

    @staticmethod
    def draw_mammen(m, rng):
        p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        a = (1 - np.sqrt(5)) / 2  # ≈ -0.618
        b = (1 + np.sqrt(5)) / 2  # ≈  1.618
        u = rng.random(m)
        w = np.where(u < p, a, b).astype(np.float32)
        return w

    # A small factory so each worker uses a fresh model instance.
    # If your model can be reconstructed via something like `type(self.model)(**self.model.init_kwargs)`,
    # encode that here. Otherwise, if self.model is stateless & picklable, you can pass a lambda returning self.model.
    def construct_new_model(self):
        # Example: clone by re-calling the class with same config
        # Adjust to your actual model construction needs.
        # CUDA first (multi-GPU)
        kwargs = self.model.init_kwargs
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            kwargs["device"] = torch.device(f"cuda:{self.previous_device}")
            self.previous_device = (self.previous_device + 1) if self.previous_device < torch.cuda.device_count() - 1 else 0
        if self.force_cpu == True:
            kwargs["device"] = torch.device("cpu")
        return type(self.model)(**kwargs)

    # =========================================================================
    # Detection (using pre-computed critical values)
    # =========================================================================
    def detect(
        self,
        min_window: int = 3,
        n_0: int = 200,
        jump: int = 10,
        search_step: int = 5,
        alpha: float = 0.95,
        t_workers: int = -1,
        debug_anim: bool = True,
        pause: float = 0.05,
        save_path: Optional[str] = None,
        fps: int = 16,
        boomerang: bool = False,
        growth: str = "geometric",
        growth_base: float = 1.4142135623730951  # sqrt(2)
    ) -> pd.DataFrame:
        """
        Run LPA change detection using pre-computed critical values.

        This method detects change points by testing progressively larger windows
        against pre-computed Monte Carlo critical values.

        Parameters
        ----------
        min_window : int, default 3
            Minimum segment size for splits.
        n_0 : int, default 200
            Initial window size (I_0 size).
        jump : int, default 10
            Step size for moving through time series.
        search_step : int, default 5
            Step size for split search within J_k.
        alpha : float, default 0.95
            Significance level for critical value lookup.
        t_workers : int, default -1
            Workers for parallel T statistic computation (-1 = all cores).
        debug_anim : bool, default True
            Enable animation visualization.
        pause : float, default 0.05
            Pause between animation frames.
        save_path : str, optional
            Path to save animation video.
        fps : int, default 16
            Animation frames per second.
        boomerang : bool, default False
            Enable boomerang animation effect.
        growth : str, default 'geometric'
            Window growth strategy (geometric recommended).
        growth_base : float, default sqrt(2)
            Base for geometric growth.

        Returns
        -------
        pd.DataFrame
            Detection results with columns: Date, N, windows, scaled_windows, MSE, RMSE

        Raises
        ------
        ValueError
            If critical values haven't been computed/loaded.

        Notes
        -----
        Critical values must be pre-computed via precompute_critical_values()
        or loaded via load_critical_values() before calling detect().
        """
        # Validate that critical values are available
        if self._critical_values is None:
            raise ValueError(
                "Critical values not available. Call precompute_critical_values(data) "
                "or load_critical_values(path) before detect()."
            )
        # Extract target (first column if 2D, or the whole array if 1D)
        target_data = self.data[:, 0] if self.data.ndim == 2 else self.data
        DT_N = pd.DataFrame({"Date": np.arange(len(self.data)), "N": target_data})
        windows, mse_vals, rmse_vals, likelihoods, scaled_windows = [], [], [], [], []
        x_axis = np.arange(len(self.data), dtype=float)  # absolute index as x
        animator = None  # Initialize to None
        if debug_anim:
            from .animator import SlidingWindowAnimator
            animator = SlidingWindowAnimator(
                x_axis, np.asarray(target_data, dtype=float),
                title="Adaptive WIN-SHAP",
                pause=pause, record=True
            )

        io = self.data.shape[0]  # starting from final time
        I_0 = list(self.data[max(0, io - n_0):io])

        n_k_minus1 = n_0
        I_k_minus1 = I_0

        for l in range(0, self.data.shape[0], jump):
            t0 = time.time()
            io = self.data.shape[0] - l

            # Skip time points where io <= n_0 (not enough data for initial window).
            # Matches LPA_LSTM reference: these points get NaN windows.
            if io <= n_0:
                windows.append(np.nan)
                mse_vals.append(np.nan)
                rmse_vals.append(np.nan)
                likelihoods.append(np.nan)
                scaled_windows.append(np.nan)
                continue

            # Determine window sizes based on geometric growth strategy (Spokoiny 1998, Example 2.1)
            # K = log_a(io/n_0) where a = growth_base
            K = int(np.emath.logn(growth_base, io / n_0)) + 1

            I_0 = list(self.data[max(0, io - n_0):io])
            I_k = I_0
            I_k_minus1 = I_0
            n_k_minus1 = n_0
            any_scale_tested = False

            for k in range(1, K + 1):
                start_k_time = time.time()
                # Calculate window sizes using geometric growth
                n_k = int(np.power(growth_base, k) * n_0)
                n_k_plus1 = int(np.power(growth_base, k + 1) * n_0)

                # Stop if I_{k+1} would extend beyond available data
                if n_k_plus1 > io:
                    break

                start_kp1_abs = max(0, io - n_k_plus1)
                end_kp1_abs = io  # exclusive upper bound

                I_k = list(self.data[max(0, io - n_k):io])
                I_k_plus1 = list(self.data[max(0, io - n_k_plus1):io])

                # Pooled (observed) on I_k_plus1, out-of-sample
                # === Precompute all sequences for the CURRENT window I_k_plus1 once ===
                y_win = np.asarray(I_k_plus1, dtype=np.float32)
                start_abs = max(0, io - n_k_plus1)
                X_all, y_all, t_abs = self.model.prepare_data(y_win, start_abs)
                if X_all is None:
                    print("Window too small for sequences")
                    continue

                # --- Global null fit on I_{k+1} (observed) ---
                likelihood_i, yhat_i, resid_i, _, _, _ = self.construct_new_model().fit(X_all, y_all).diagnostics(X_all, y_all)

                # Candidate split range
                J_start = max(min_window, io - n_k)  # have enough left history
                J_end = io - n_k_minus1  # ensure right side at least n_0
                if J_end <= J_start:
                    print(f"J_end ({J_end}) < J_start ({J_start})")
                    continue

                # Candidate split range in ABSOLUTE target indices
                J_abs = np.arange(J_start, J_end, search_step, dtype=np.int64)

                # --- Observed T(i) across splits ---
                T_vals = self.compute_T_vals(X_all, y_all, likelihood_i, J_abs, t_abs, t_workers, min_seg=min_window)
                any_scale_tested = True

                # Best split τ
                best_tau = None
                if len(T_vals):
                    j_best_idx = int(np.argmax(T_vals))
                    best_tau = int(J_abs[j_best_idx])

                # --- Get pre-computed critical value for this scale ---
                critical_value = self.get_critical_value(k, alpha)

                end_k_time = time.time()
                # --- Decision for the current window ---
                if len(T_vals) > 0:
                    test_value = float(np.max(T_vals))
                    print(
                        f"[QLR] step={l} |I_k+1=[{max(0, io - n_k_plus1)}, {io}] | I_k=[{max(0, io - n_k)}, {io}]  |"
                        f"I_k-1=[{max(0, io - n_k_minus1)}, {io}] | J_k=[{J_start}, {J_end}] | k={k} | "
                        f"SupLR={test_value:.3f} | crit({alpha:.2f})={critical_value:.3f} | #splits={len(T_vals)} | time/k={end_k_time - start_k_time:.2f}s")
                else:
                    test_value, critical_value = 0.0, math.inf

                # --- FINAL (show crit) ---
                if animator:
                    animator.update(
                        io=io,
                        start_idx=start_kp1_abs,
                        end_idx=end_kp1_abs,
                        J_abs=J_abs,
                        sup_lr=test_value,
                        crit=critical_value,
                        k=k,
                        l=l,
                        tau=best_tau,
                        n_k_plus1=n_k_plus1,
                        n_k=n_k,
                        n_k_minus1=n_k_minus1,
                        J_start=J_start,
                        J_end=J_end,
                    )

                if test_value > critical_value:
                    print(f"Found break at step {l} (window size {len(I_k)}).")
                    break
                else:
                    I_k_minus1 = I_k
                    n_k_minus1 = n_k
                    continue

            # If no scale was actually tested, skip this point (NaN)
            if not any_scale_tested:
                windows.append(np.nan)
                mse_vals.append(np.nan)
                rmse_vals.append(np.nan)
                likelihoods.append(np.nan)
                scaled_windows.append(np.nan)
                continue

            # For the per-step diagnostic, compute OOS MSE on I_k
            MSE_I_k = 0  # segment_oos_mse(I_k, seq_len=seq_len, epochs=epochs, error_type=error_type)
            RMSE_I_k = 0  # math.sqrt(max(MSE_I_k, 0.0))
            Likelihood_I_k = 0

            windows.append(len(I_k))
            mse_vals.append(MSE_I_k)
            rmse_vals.append(RMSE_I_k)
            likelihoods.append(Likelihood_I_k)

            scaled_windows.append(len(I_k) / io)

            t1 = time.time()
            print(f"Step {l:4d} | time/step={t1 - t0:.2f}s | window={len(I_k)} | RMSE={RMSE_I_k:.4f}")

        # Reverse to align like your original flow
        windows.reverse()
        mse_vals.reverse()
        rmse_vals.reverse()
        scaled_windows.reverse()
        DT_N["windows"] = pd.Series(windows)
        DT_N["scaled_windows"] = pd.Series(scaled_windows)
        DT_N["MSE"] = pd.Series(mse_vals)
        DT_N["RMSE"] = pd.Series(rmse_vals)

        # Save (optional)
        if animator is not None:
            if save_path:
                animator.save(save_path, fps=fps, boomerang=boomerang)
            animator.close()

        return DT_N

    def compute_T_vals(self, X_all, y_all, likelihood_i, J_abs, t_abs, max_processes, min_seg=20):
        def safe_calc(i_abs):
            try:
                Ti = self.calculate_t(X_all, y_all, likelihood_i, i_abs, t_abs, min_seg=min_seg)
                return max(0.0, Ti)
            except ValueError:
                return 0.0

        # prefer="processes" → multi-process;
        T_vals = Parallel(n_jobs=max_processes, prefer="processes", batch_size='auto')(delayed(safe_calc)(i) for i in J_abs)
        return T_vals

    def calculate_t(self, X_all, y_all, likelihood_i, i_abs, t_abs, min_seg=20):
        if self.debug == True:
            print(f"tau={i_abs}")
        # Strict no-leak masks by target index
        Lmask = t_abs <= i_abs
        Rmask = t_abs > i_abs
        mA = int(np.sum(Lmask))
        mB = int(np.sum(Rmask))
        if mA < min_seg or mB < min_seg:  # min targets per side; tune as needed
            raise ValueError(f"Too few targets for split {i_abs}")
        likelihood_a, _, _, _, _, _ = self.construct_new_model().fit(X_all[Lmask], y_all[Lmask]).diagnostics(X_all[Lmask],
                                                                                             y_all[Lmask])
        likelihood_b, _, _, _, _, _ = self.construct_new_model().fit(X_all[Rmask], y_all[Rmask]).diagnostics(X_all[Rmask],
                                                                                             y_all[Rmask])

        Ti = likelihood_a + likelihood_b - likelihood_i
        return Ti

    def _Ti_for_mask(self, X_all, y_star_t, Lmask, Rmask, min_seg, likelihood_i_b):
        # returns 0.0 if segment too small
        mA = int(Lmask.sum())
        mB = int(Rmask.sum())
        if mA < min_seg or mB < min_seg:
            return 0.0
        likelihood_a_b, _, _, _, _, _ = self.construct_new_model().fit(X_all[Lmask], y_star_t[Lmask]).diagnostics(
            X_all[Lmask], y_star_t[Lmask]
        )
        likelihood_b_b, _, _, _, _, _ = self.construct_new_model().fit(X_all[Rmask], y_star_t[Rmask]).diagnostics(
            X_all[Rmask], y_star_t[Rmask]
        )
        return likelihood_a_b + likelihood_b_b - likelihood_i_b

    def _draw_weights(self, kind: str, n: int, rng: np.random.Generator):
        if kind == "mammen":
            # assuming these static methods exist on ChangeDetector
            return ChangeDetector.draw_mammen(n, rng)
        elif kind == "rademacher":
            return ChangeDetector.draw_rademacher(n, rng)
        else:
            raise ValueError(f"Weights {kind} not supported. Use 'mammen' or 'rademacher'.")

    def _one_bootstrap(
            self,
            X_all,
            yhat_i,
            resid_i,
            masks,  # list[(Lmask, Rmask)]
            weights_kind: str,
            min_seg: int = 20,
            inner_jobs: int = 1,  # NEW: threads per worker for masks
    ):
        rng = np.random.default_rng()

        # 1) multipliers & pseudo response
        w = self._draw_weights(weights_kind, len(resid_i), rng)
        y_star = (yhat_i + w * resid_i)
        y_star_t = torch.from_numpy(y_star)

        # 2) global null likelihood (shared within this bootstrap)
        model = self.construct_new_model()
        likelihood_i_b, *_ = model.fit(X_all, y_star_t).diagnostics(X_all, y_star_t)

        # 3) parallel sweep across masks using threads
        if inner_jobs == 1:
            # fast path: no threading, just a loop
            sup_b = 0.0
            for Lmask, Rmask in masks:
                Ti_b = self._Ti_for_mask(X_all, y_star_t, Lmask, Rmask, min_seg, likelihood_i_b)
                if Ti_b > sup_b:
                    sup_b = Ti_b
        else:
            # threaded inner parallelism (avoid processes-inside-processes)
            Ti_list = Parallel(n_jobs=inner_jobs, backend="threading", batch_size="auto")(
                delayed(self._Ti_for_mask)(
                    X_all, y_star_t, Lmask, Rmask, min_seg, likelihood_i_b
                )
                for (Lmask, Rmask) in masks
            )
            sup_b = max(Ti_list) if Ti_list else 0.0

        return max(0.0, sup_b)

    def calculate_t_bootstrap(self, X_all, yhat_i, resid_i, J_abs, t_abs, num_bootstrap, min_seg=20, n_jobs=-1, n_inner_threads=-1, batch_size=512):
        """
                Parallel bootstrap over `b` replicates using joblib.
                Returns: array shape (num_bootstrap,)
                """

        # Precompute masks for every split once (depends only on t_abs & J_abs)
        masks = []
        for i_abs in J_abs:
            Lmask = (t_abs <= i_abs)
            Rmask = ~Lmask  # faster than recomputing
            masks.append((Lmask, Rmask))

        # Parallel loop over bootstrap draws
        Sup_boot_list = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=batch_size)(
            delayed(self._one_bootstrap)(
                X_all=X_all,
                yhat_i=yhat_i,
                resid_i=resid_i,
                masks=masks,
                weights_kind=self.weights,
                min_seg=min_seg,
                inner_jobs=n_inner_threads,
            )
            for _ in range(num_bootstrap)
        )
        return np.asarray(Sup_boot_list, dtype=np.float64)
