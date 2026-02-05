# Specification: Monte Carlo Pre-computed Critical Values for LPA

**Status**: Implemented
**Author**: Interview-derived specification
**Date**: 2026-02-04

## Overview

This specification describes the replacement of the current dynamic wild bootstrap approach for computing critical values in the LPA-based change detection with a pre-computed Monte Carlo simulation approach, based on the implementation in `LPA_LSTM/LPA_LSTM.ipynb`.

### Motivation

The current implementation (`ChangeDetector.detect()`) computes critical values via wild bootstrap at **every step and scale**, which is computationally expensive. The new approach:

1. Pre-computes critical values once via Monte Carlo simulation under the null (homogeneity) hypothesis
2. Uses these fixed thresholds during detection
3. Eliminates the bootstrap loop entirely from `detect()`

## Design Decisions (from Interview)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| I_0 reference point | Single fit at series end | Fit I_0 model once on final n_0 points, reuse for all MC simulations |
| Simulation method | Configurable (Wild bootstrap default) | Wild: `y_t = LSTM(history) + w_t * resampled_residual` where w_t ~ Mammen. Also supports Gaussian and residual bootstrap |
| Calibration | Global critical values only | Accept trade-off: faster but no local variance adaptation |
| Parallelism | Parallel across scales | Compute all k=1..K scales simultaneously for maximum throughput |
| CV adjustment | Always apply Spokoiny, configurable lambda | `adjusted_cv = raw_cv * (1 + lambda * sqrt(log(n_K_max / n_k)))`, default lambda=0.25 |
| CV lifecycle | Explicit precompute + file save/load | `precompute_critical_values(data)`, `save_critical_values(path)`, `load_critical_values(path)` |
| Precompute input | Require data array | Extract length and I_0 from the same array passed to precompute |
| Simulation API | Method on AdaptiveModel | `simulate_series(seed_values, sigma, n_total, rng)` on base class |
| Randomness | Deterministic default | seed=42 by default, user can override |
| Split computation | Keep current parallel approach | Proven correct, already parallelized via joblib |
| CV storage format | CSV only | Notebook-compatible, human-readable, portable |
| Verbosity | Keep current verbose | Print step-by-step comparisons during detect() |
| Residual bootstrap init | Same seed + IID residuals | Initialize with I_0 tail, resample residuals with replacement |
| Backward compatibility | Replace bootstrap entirely | Breaking change, no backward compatibility mode |
| I_0 quality | Trust user (document requirement) | Document that I_0 must be homogeneous |
| Adjustment base | Pre-computed K_max | Fixed at CV computation time, baked into saved CVs |
| GPU memory | Auto-detect limits | Query GPU memory, estimate per-scale usage, auto-limit concurrent scales |
| CV object design | Internal state only | CVs managed by ChangeDetector, accessed via methods |
| SHAP integration | Decoupled | SHAP consumes window sizes only, no access to CV internals |

---

## Architecture

### File Structure Changes

```
src/adaptivewinshap/
├── detector.py          # Modified: add MC CV methods, remove bootstrap
├── model.py             # Modified: add simulate_series() abstract method
├── critical_values.py   # NEW: CV computation utilities (optional, or inline in detector)
├── shap.py              # Unchanged
├── utils.py             # Unchanged
└── animator.py          # Unchanged
```

### Class Diagram

```
AdaptiveModel (base class)
├── fit(X, y) -> self
├── diagnostics(X, y) -> tuple
├── prepare_data(window, start_idx) -> tuple
└── simulate_series(seed_values, sigma, n_total, rng) -> np.ndarray  # NEW: abstract

LSTMModel(AdaptiveModel)
└── simulate_series(...)  # Implements recursive 1-step forecasting

ChangeDetector
├── __init__(model, data, ...)
├── precompute_critical_values(data, mc_reps=300, alpha=0.95, ...) -> self  # NEW
├── save_critical_values(path: str) -> None  # NEW
├── load_critical_values(path: str) -> self  # NEW
├── detect(min_window, n_0, jump, ...) -> pd.DataFrame  # MODIFIED: uses pre-computed CVs
└── _critical_values: pd.DataFrame  # INTERNAL STATE
```

---

## API Specification

### AdaptiveModel.simulate_series()

```python
def simulate_series(
    self,
    seed_values: np.ndarray,
    sigma: float,
    n_total: int,
    rng: np.random.Generator,
    method: str = "gaussian"  # or "residual_bootstrap"
) -> np.ndarray:
    """
    Simulate a homogeneous time series from the fitted model.

    Parameters
    ----------
    seed_values : np.ndarray
        Initial values to seed the simulation (at least seq_len points).
    sigma : float
        Standard deviation of innovations (from fitted model residuals).
    n_total : int
        Total length of series to generate.
    rng : np.random.Generator
        Random number generator for reproducibility.
    method : str
        'gaussian': y_t = model_predict(history) + N(0, sigma)
        'residual_bootstrap': y_t = model_predict(history) + resampled_residual

    Returns
    -------
    np.ndarray
        Simulated series of length n_total.

    Raises
    ------
    NotImplementedError
        If the model subclass doesn't implement this method.
    """
```

### ChangeDetector.precompute_critical_values()

```python
def precompute_critical_values(
    self,
    data: np.ndarray,
    mc_reps: int = 300,
    alpha: float = 0.95,
    search_step: int = 1,
    penalty_factor: float = 0.25,
    simulation_method: str = "gaussian",  # or "residual_bootstrap"
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
        The time series data. Used to:
        - Determine K_max from len(data)
        - Fit I_0 model on last n_0 points
    mc_reps : int
        Number of Monte Carlo replications per scale.
    alpha : float
        Significance level (e.g., 0.95 for 95th percentile).
    search_step : int
        Step size for split search within J_k range.
    penalty_factor : float
        Lambda for Spokoiny adjustment. Higher = more conservative for small windows.
        Set to 0 to disable adjustment.
    simulation_method : str
        'gaussian' (default): Use Gaussian innovations N(0, sigma)
        'residual_bootstrap': Resample from I_0 residuals IID
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress during computation.

    Returns
    -------
    self
        Returns self for method chaining.

    Notes
    -----
    - I_0 must be homogeneous (no change points). This is the user's responsibility.
    - Computation is parallelized across scales with auto-detected GPU memory limits.
    - Call save_critical_values() to persist results.
    """
```

### ChangeDetector.save_critical_values() / load_critical_values()

```python
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
                          critical_value_95 (or critical_value_{alpha*100}),
                          adjustment_factor, penalty_factor
    """
```

### Modified ChangeDetector.detect()

```python
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
    growth_base: float = 2.0
) -> pd.DataFrame:
    """
    Run LPA change detection using pre-computed critical values.

    BREAKING CHANGE: This method no longer accepts bootstrap parameters
    (num_bootstrap, b_workers, one_b_threads). Critical values must be
    pre-computed via precompute_critical_values() or loaded via
    load_critical_values() before calling detect().

    Parameters
    ----------
    min_window : int
        Minimum segment size for splits.
    n_0 : int
        Initial window size (I_0 size). Must match the n_0 used during
        critical value pre-computation.
    jump : int
        Step size for moving through time series.
    search_step : int
        Step size for split search within J_k.
    alpha : float
        Significance level for critical value lookup. Must match the
        alpha used during pre-computation.
    t_workers : int
        Workers for parallel T statistic computation.
    debug_anim : bool
        Enable animation visualization.
    pause, save_path, fps, boomerang : animation parameters
    growth : str
        'geometric' or 'arithmetic' window growth.
    growth_base : float
        Base for geometric growth (only used if growth='geometric').

    Returns
    -------
    pd.DataFrame
        Detection results with columns: Date, N, windows, scaled_windows, MSE, RMSE

    Raises
    ------
    ValueError
        If critical values haven't been computed/loaded, or if n_0/alpha
        mismatch between detection and pre-computation.
    """
```

---

## Critical Value Table Schema

CSV columns:

| Column | Type | Description |
|--------|------|-------------|
| `k` | int | Scale index (1, 2, 3, ...) |
| `n_k` | int | Window size at scale k: `n_0 * c^k` |
| `n_k_plus1` | int | Window size at scale k+1: `n_0 * c^(k+1)` |
| `j_start` | int | Start of J_k split range |
| `j_end` | int | End of J_k split range |
| `critical_value_95` | float | Adjusted critical value at alpha=0.95 |
| `critical_value_99` | float | Adjusted critical value at alpha=0.99 (optional) |
| `mean` | float | Mean of SupLR distribution (diagnostic) |
| `std` | float | Std of SupLR distribution (diagnostic) |
| `adjustment_factor` | float | Spokoiny adjustment multiplier |
| `penalty_factor` | float | Lambda used for adjustment |

---

## Algorithm: Monte Carlo Critical Value Computation

```
Algorithm: compute_critical_values(data, n_0, c, mc_reps, alpha, lambda)

Input:
  - data: time series of length T
  - n_0: initial window size
  - c: geometric ratio (default sqrt(2))
  - mc_reps: number of MC replications (default 300)
  - alpha: significance level (default 0.95)
  - lambda: penalty factor for adjustment (default 0.25)

Output:
  - CV_table: DataFrame with critical values per scale

1. FIT I_0 MODEL:
   i0_data = data[-n_0:]
   model_i0 = fit_model(i0_data)
   residuals = model_i0.residuals(i0_data)
   sigma = std(residuals)
   seed_values = i0_data[-seq_len:]

2. DETERMINE SCALES:
   K_max = floor(log_c(T / n_0))

3. FOR k = 1 to K_max (in parallel):
   n_k = n_0 * c^k
   n_{k+1} = n_0 * c^{k+1}
   j_start = n_{k+1} - n_k
   j_end = n_{k+1} - n_{k-1}

   suplr_values = []
   FOR b = 1 to mc_reps:
     y_sim = model_i0.simulate_series(seed_values, sigma, n_{k+1})
     suplr = compute_suplr(y_sim, j_start, j_end)
     suplr_values.append(suplr)

   raw_cv = quantile(suplr_values, alpha)
   CV_table[k] = raw_cv

4. APPLY SPOKOINY ADJUSTMENT:
   n_K_max = max(CV_table.n_k)
   FOR each row in CV_table:
     ratio = n_K_max / row.n_k
     adjustment = 1 + lambda * sqrt(log(ratio))
     row.critical_value *= adjustment
     row.adjustment_factor = adjustment

5. RETURN CV_table
```

---

## GPU Memory Management

For parallel computation across scales, the implementation will:

1. Query available GPU memory via `torch.cuda.get_device_properties()`
2. Estimate per-scale memory usage based on:
   - Model size (LSTM parameters)
   - Sequence length
   - Batch size
   - MC replication count
3. Calculate `max_concurrent_scales = available_memory / per_scale_estimate`
4. Use joblib with `n_jobs=min(K_max, max_concurrent_scales)`

Fallback: If GPU memory cannot be estimated (CPU-only), use `n_jobs=-1` (all cores).

---

## Migration Guide

### Before (current API):

```python
detector = ChangeDetector(model, data)
results = detector.detect(
    n_0=200,
    jump=10,
    alpha=0.95,
    num_bootstrap=50,  # REMOVED
    b_workers=-1,       # REMOVED
    one_b_threads=-1    # REMOVED
)
```

### After (new API):

```python
detector = ChangeDetector(model, data)

# Option 1: Compute fresh
detector.precompute_critical_values(
    data,
    mc_reps=300,
    alpha=0.95,
    penalty_factor=0.25
)
detector.save_critical_values("critical_values.csv")

# Option 2: Load existing
detector.load_critical_values("critical_values.csv")

# Run detection (no bootstrap params)
results = detector.detect(n_0=200, jump=10, alpha=0.95)
```

---

## Implementation Checklist

### Phase 1: Model Layer
- [x] Add `simulate_series()` abstract method to `AdaptiveModel`
- [x] Implement `simulate_series()` for LSTM model (recursive forecasting + innovations)
- [x] Support both 'gaussian' and 'residual_bootstrap' methods
- [x] Unit tests for simulation correctness

### Phase 2: Critical Value Computation
- [x] Add `_fit_i0_model()` internal method
- [x] Add `_compute_suplr_for_scale()` with shared SupLR computation logic
- [x] Add `precompute_critical_values()` with parallel-across-scales
- [x] Implement GPU memory auto-detection and limiting
- [x] Add `save_critical_values()` / `load_critical_values()` (CSV format)
- [x] Unit tests for CV computation

### Phase 3: Detection Integration
- [x] Remove bootstrap-related code from `detect()`
- [x] Remove `num_bootstrap`, `b_workers`, `one_b_threads` parameters
- [x] Add CV lookup in `detect()` main loop
- [x] Add validation: CVs must be computed/loaded before detect()
- [x] Update verbose output to show CV source (pre-computed)
- [x] Integration tests

### Phase 4: Documentation & Cleanup
- [x] Update docstrings
- [x] Update examples/
- [ ] Update README (optional)
- [ ] Remove dead bootstrap code (kept as legacy for now)
- [ ] Performance benchmarks (before/after)

---

## Performance Expectations

| Operation | Current (bootstrap) | New (MC pre-computed) |
|-----------|--------------------|-----------------------|
| Pre-computation | N/A | O(K * mc_reps * LSTM_fit) - one-time |
| Per-step detection | O(K * num_bootstrap * LSTM_fit) | O(K * LSTM_fit) - CV lookup is O(1) |
| Total for T steps | O(T/jump * K * num_bootstrap) | O(mc_reps * K) + O(T/jump * K) |

With typical values (T=1000, K=5, num_bootstrap=50, mc_reps=300, jump=10):
- Current: ~100 steps * 5 scales * 50 bootstrap = 25,000 LSTM fits
- New: 300 MC * 5 scales = 1,500 fits (pre-compute) + 100 * 5 = 500 fits (detect) = 2,000 total

**Expected speedup: ~10-15x** (varies with parameters)

---

## Appendix: Spokoiny Adjustment Formula

From Spokoiny (1998), the adjustment controls false positive rates across scales:

```
adjusted_cv(k) = raw_cv(k) * (1 + λ * sqrt(log(n_K / n_k)))
```

Where:
- `n_K` = largest window size in the table
- `n_k` = window size at scale k
- `λ` = penalty factor (default 0.25)

This inflates critical values for smaller windows (larger ratio), making it harder to reject the null at small scales. The largest scale (k=K) has adjustment factor = 1 (unchanged).

---

## Appendix: Wild Bootstrap Methodology

The default simulation method uses a wild residual bootstrap that perturbs fitted values by resampled residual innovations, which is well-suited for heteroskedastic and non-Gaussian errors.

A bootstrap pseudo-series is generated as:

```
y_t^○ = f_θ(y°_{t-p:t-1}) + w_t * e_π(t)
```

where:
- `f_θ(...)` is the fitted model prediction
- `e_π(t)` is drawn with replacement from the I_0 residual pool
- `w_t` is drawn from Mammen's two-point distribution

### Mammen's Two-Point Distribution

```
w_t = -1/φ  with probability φ/√5
w_t = φ     with probability 1 - φ/√5

where φ = (1 + √5) / 2 ≈ 1.618 (golden ratio)
```

This distribution ensures E[w_t] = 0 and E[w_t²] = 1, which is required for the wild bootstrap to correctly mimic the first two moments of the underlying error process.

---

## References

1. Spokoiny, V. (1998). Estimation of a function with discontinuities via local polynomial fit with an adaptive window choice. *Annals of Statistics*.

2. Cizek, P., Härdle, W., & Spokoiny, V. (2009). Adaptive pointwise estimation in time-inhomogeneous conditional heteroscedasticity models. *Econometrics Journal*.

3. Härdle, W. & Mammen, E. (1991). Comparing nonparametric versus parametric regression fits. *Annals of Statistics*.

4. LPA_LSTM.ipynb - Reference implementation of MC critical value computation.
