"""
Generate all simulated datasets for benchmarking AdaptiveWIN-SHAP methods.

Creates 5 different datasets with various non-stationary characteristics:
1. piecewise_ar3 - Rotating dominant AR(3) lag (baseline, 3 equal regimes)
2. arx_rotating - ARX with rotating covariate drivers (3 equal regimes)
3. trend_season - Trend + seasonality + AR break (3 equal regimes)
4. piecewise_ar3_long - Like piecewise_ar3 but 7 unequal-length regimes
5. arx_rotating_long - Like arx_rotating but 7 unequal-length regimes
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _make_lag_matrix(y, p):
    """Return lagged design matrix: rows t, cols [y_{t-1}..y_{t-p}]."""
    T = len(y)
    X = np.zeros((T, p))
    for j in range(1, p+1):
        X[j:, j-1] = y[:-j]
    return X


def _std_importance(coefs, sigmas=None):
    """
    Standardized absolute importance for a coefficient vector.
    coefs: (d,)
    sigmas: (d,) or None -> assumes 1.
    """
    a = np.abs(coefs)
    if sigmas is not None:
        a = a * sigmas
    s = a.sum()
    return a / s if s > 0 else a


def sim_piecewise_ar3_rotating(T=1500, seed=123, noise_sigma=1.0,
                               regime_lengths=(500, 500, 500)):
    """
    Baseline univariate locally-stationary AR(3) with rotating dominant lag:

    Regime 1 (t=0..500):   X_t = 0.9 X_{t-1} + 0.01 X_{t-2} + 0.01 X_{t-3} + e_t
    Regime 2 (t=501..1000):X_t = 0.01 X_{t-1} + 0.9 X_{t-2} + 0.01 X_{t-3} + e_t
    Regime 3 (t=1001..1500):X_t = 0.01 X_{t-1} + 0.01 X_{t-2} + 0.9 X_{t-3} + e_t

    Returns:
      X: (T,) target series
      Z: None (no covariates)
      true_imp: (T,3) true lag importances over time
    """
    rng = np.random.default_rng(seed)

    # regime-specific AR(3) coefficients
    phis = [
        np.array([0.9, 0.01, 0.01]),
        np.array([0.01, 0.9, 0.01]),
        np.array([0.01, 0.01, 0.9]),
    ]

    # build regime index
    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    # simulate piecewise AR(3)
    for t in range(T):
        k = reg_idx[t]
        ar_part = 0.0
        for j in range(1, 4):
            if t - j >= 0:
                ar_part += phis[k][j-1] * X[t-j]
        X[t] = ar_part + eps[t]

    # true importance per time = normalized |phi|
    true_imp = np.zeros((T, 3))
    for t in range(T):
        k = reg_idx[t]
        a = np.abs(phis[k])
        true_imp[t] = a / a.sum()

    return X, None, true_imp


def sim_piecewise_arx_rotating_drivers(T=1500, p=3, seed=0,
                                       regime_lengths=(500, 500, 500),
                                       noise_sigma=0.5):
    """
    Y_t depends on p lags + 3 covariates (D, F, R),
    with regime-wise rotating dominance in covariate coefficients.
    """
    rng = np.random.default_rng(seed)

    # Covariates: iid standard normal (clean signal, regime change comes from betas)
    D = rng.normal(0, 1.0, size=T)
    F = rng.normal(0, 1.0, size=T)
    R = rng.normal(0, 1.0, size=T)

    Z = np.vstack([D, F, R]).T  # (T,3)

    # Regime-specific coefficients
    # lags roughly stable, covariates rotate
    phi = np.array([0.10, 0.05, 0.02])  # AR(3) base (weak, so covariates dominate)
    betas = [
        np.array([0.2, 1.2, -0.1]),  # Regime 1: fuel dominates
        np.array([1.2, 0.2, -0.1]),  # Regime 2: demand dominates
        np.array([0.2, -0.1, 1.2])   # Regime 3: RES dominates
    ]

    # Build regime index
    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    Y = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    for t in range(T):
        k = reg_idx[t]
        ar_part = 0.0
        for j in range(1, p+1):
            if t-j >= 0:
                ar_part += phi[j-1]*Y[t-j]
        Y[t] = ar_part + Z[t] @ betas[k] + eps[t]

    # True importance per t over features: [lags p] + [D,F,R]
    true_imp = np.zeros((T, p+3))
    sig_lags = np.ones(p)
    sig_Z = Z.std(axis=0)
    for t in range(T):
        k = reg_idx[t]
        coefs = np.concatenate([phi, betas[k]])
        sigmas = np.concatenate([sig_lags, sig_Z])
        true_imp[t] = _std_importance(coefs, sigmas)

    return Y, Z, true_imp


def sim_trend_season_ar_break(T=1500, p=3, seed=1,
                              regime_lengths=(500, 500, 500),
                              noise_sigma=0.4):
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # deterministic nonstationarity
    trend = 0.002 * t
    seas_day = 1.0*np.sin(2*np.pi*t/24) + 0.5*np.cos(2*np.pi*t/24)
    seas_week = 0.6*np.sin(2*np.pi*t/168)

    # AR(3) coefficients rotate like your univariate example
    phis = [
        np.array([0.9, 0.01, 0.01]),
        np.array([0.01, 0.9, 0.01]),
        np.array([0.01, 0.01, 0.9]),
    ]

    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    for tt in range(T):
        k = reg_idx[tt]
        ar_part = 0.0
        for j in range(1, p+1):
            if tt-j >= 0:
                ar_part += phis[k][j-1]*X[tt-j]
        X[tt] = ar_part + eps[tt]

    Y = trend + seas_day + seas_week + X

    true_imp = np.zeros((T, p))
    for tt in range(T):
        k = reg_idx[tt]
        true_imp[tt] = _std_importance(phis[k], np.ones(p))

    return Y, None, true_imp


def sim_piecewise_ar3_long(seed=10, noise_sigma=1.0,
                           regime_lengths=(500, 400, 300, 600, 800, 1000, 150)):
    """
    Like piecewise_ar3 but with 7 unequal-length regimes.
    AR(3) coefficients cycle through the 3 dominant-lag patterns.
    T = sum(regime_lengths) = 3750.
    """
    T = sum(regime_lengths)
    rng = np.random.default_rng(seed)

    # 3 coefficient patterns, cycled across 7 regimes
    phi_patterns = [
        np.array([0.9, 0.01, 0.01]),   # lag-1 dominates
        np.array([0.01, 0.9, 0.01]),   # lag-2 dominates
        np.array([0.01, 0.01, 0.9]),   # lag-3 dominates
    ]
    # regime k uses pattern k % 3
    phis = [phi_patterns[k % 3] for k in range(len(regime_lengths))]

    # build regime index
    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    for t in range(T):
        k = reg_idx[t]
        ar_part = 0.0
        for j in range(1, 4):
            if t - j >= 0:
                ar_part += phis[k][j-1] * X[t-j]
        X[t] = ar_part + eps[t]

    true_imp = np.zeros((T, 3))
    for t in range(T):
        k = reg_idx[t]
        a = np.abs(phis[k])
        true_imp[t] = a / a.sum()

    return X, None, true_imp


def sim_arx_rotating_long(p=3, seed=11, noise_sigma=0.5,
                           regime_lengths=(500, 400, 300, 600, 800, 1000, 150)):
    """
    Like arx_rotating but with 7 unequal-length regimes.
    Weak AR(3) + 3 iid covariates with rotating dominant beta.
    T = sum(regime_lengths) = 3750.
    """
    T = sum(regime_lengths)
    rng = np.random.default_rng(seed)

    # Covariates: iid standard normal
    D = rng.normal(0, 1.0, size=T)
    F = rng.normal(0, 1.0, size=T)
    R = rng.normal(0, 1.0, size=T)
    Z = np.vstack([D, F, R]).T  # (T,3)

    phi = np.array([0.10, 0.05, 0.02])  # weak AR

    # 3 beta patterns, cycled across 7 regimes
    beta_patterns = [
        np.array([0.2, 1.2, -0.1]),   # fuel dominates
        np.array([1.2, 0.2, -0.1]),   # demand dominates
        np.array([0.2, 0.1, 1.2]),   # RES dominates
    ]
    betas = [beta_patterns[k % 3] for k in range(len(regime_lengths))]

    # build regime index
    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    Y = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    for t in range(T):
        k = reg_idx[t]
        ar_part = 0.0
        for j in range(1, p+1):
            if t-j >= 0:
                ar_part += phi[j-1]*Y[t-j]
        Y[t] = ar_part + Z[t] @ betas[k] + eps[t]

    true_imp = np.zeros((T, p+3))
    sig_lags = np.ones(p)
    sig_Z = Z.std(axis=0)
    for t in range(T):
        k = reg_idx[t]
        coefs = np.concatenate([phi, betas[k]])
        sigmas = np.concatenate([sig_lags, sig_Z])
        true_imp[t] = _std_importance(coefs, sigmas)

    return Y, Z, true_imp

def save_dataset(name, Y, Z, true_imp, output_dir="examples/datasets/simulated"):
    """Save a dataset to CSV files."""
    dataset_dir = Path(output_dir) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Main data file - always save Y as 'N'
    data_df = pd.DataFrame({'N': Y})

    # Add covariates if they exist
    if Z is not None:
        for i in range(Z.shape[1]):
            data_df[f'Z_{i}'] = Z[:, i]

    data_df.to_csv(dataset_dir / 'data.csv', index=False)

    # Save true importances
    imp_cols = {f'true_imp_{i}': true_imp[:, i] for i in range(true_imp.shape[1])}
    imp_df = pd.DataFrame(imp_cols)
    imp_df.to_csv(dataset_dir / 'true_importances.csv', index=False)

    print(f"Saved {name}:")
    print(f"  - Data shape: {Y.shape}")
    if Z is not None:
        print(f"  - Covariates shape: {Z.shape}")
    print(f"  - True importances shape: {true_imp.shape}")
    print(f"  - Directory: {dataset_dir}")


if __name__ == "__main__":
    print("="*60)
    print("Generating Simulated Datasets for AdaptiveWIN-SHAP")
    print("="*60)

    T = 1500  # Number of time points

    # 1. Piecewise AR(3) with rotating dominant lag (baseline)
    print("\n1. Generating piecewise_ar3...")
    Y, Z, true_imp = sim_piecewise_ar3_rotating(T=T, seed=123)
    save_dataset("piecewise_ar3", Y, Z, true_imp)

    # 2. ARX with rotating covariate drivers
    print("\n2. Generating arx_rotating...")
    Y, Z, true_imp = sim_piecewise_arx_rotating_drivers(T=T, seed=0)
    save_dataset("arx_rotating", Y, Z, true_imp)

    # 3. Trend + seasonality + AR break
    print("\n3. Generating trend_season...")
    Y, Z, true_imp = sim_trend_season_ar_break(T=T, seed=1)
    save_dataset("trend_season", Y, Z, true_imp)

    # 4. Piecewise AR(3) with 7 unequal-length regimes
    print("\n4. Generating piecewise_ar3_long...")
    Y, Z, true_imp = sim_piecewise_ar3_long(seed=10)
    save_dataset("piecewise_ar3_long", Y, Z, true_imp)

    # 5. ARX rotating with 7 unequal-length regimes
    print("\n5. Generating arx_rotating_long...")
    Y, Z, true_imp = sim_arx_rotating_long(seed=11)
    save_dataset("arx_rotating_long", Y, Z, true_imp)

    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("="*60)