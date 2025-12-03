"""
Generate all simulated datasets for benchmarking AdaptiveWIN-SHAP methods.

Creates 6 different datasets with various non-stationary characteristics:
1. piecewise_ar3 - Rotating dominant AR(3) lag (baseline)
2. arx_rotating - ARX with rotating covariate drivers
3. trend_season - Trend + seasonality + AR break
4. spike_process - Jump/spike regime with changing covariate role
5. garch_regime - GARCH returns with regime-shifting factor loadings
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

    # Covariates: demand D, fuel F, renewables R (rough stylized)
    D = rng.normal(0, 1.0, size=T) + 0.3*np.sin(2*np.pi*np.arange(T)/24)
    F = rng.normal(0, 1.0, size=T) + 0.1*np.arange(T)/T  # slight drift
    R = rng.normal(0, 1.0, size=T) + 0.5*np.sin(2*np.pi*np.arange(T)/168)

    Z = np.vstack([D, F, R]).T  # (T,3)

    # Regime-specific coefficients
    # lags roughly stable, covariates rotate
    phi = np.array([0.6, 0.2, 0.1])  # AR(3) base
    betas = [
        np.array([0.2, 1.2, -0.1]),  # Regime 1: fuel dominates
        np.array([1.2, 0.2, -0.1]),  # Regime 2: demand dominates
        np.array([0.2, 0.1, -1.2])   # Regime 3: RES dominates
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


def sim_spike_process(T=1500, p=3, seed=2,
                      regime_lengths=(750, 750),
                      noise_sigma=0.3):
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # covariates
    D = rng.normal(0, 1.0, size=T) + 0.3*np.sin(2*np.pi*t/24)
    R = rng.normal(0, 1.0, size=T) + 0.5*np.sin(2*np.pi*t/168)
    Z = np.vstack([D, R]).T  # (T,2)

    phi = np.array([0.7, 0.2, 0.05])

    # spike probability depends on different covariate per regime
    gammas = [
        np.array([1.5, 0.0]),  # Regime 1: spikes driven by demand
        np.array([0.0, 1.5]),  # Regime 2: spikes driven by RES scarcity
    ]

    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    # baseline ARX without covariates for clarity
    for tt in range(T):
        ar_part = 0.0
        for j in range(1, p+1):
            if tt-j >= 0:
                ar_part += phi[j-1]*X[tt-j]
        X[tt] = ar_part + eps[tt]

    # spikes
    J = np.zeros(T)
    S = rng.lognormal(mean=1.0, sigma=0.6, size=T)  # spike size
    for tt in range(T):
        k = reg_idx[tt]
        logit_p = Z[tt] @ gammas[k] - 2.0  # baseline rarity
        p_t = 1/(1 + np.exp(-logit_p))
        J[tt] = rng.binomial(1, p_t)

    Y = X + J*S

    # True importance for features = [lags p] + [D, R] (through spike prob)
    true_imp = np.zeros((T, p+2))
    sig_Z = Z.std(axis=0)
    for tt in range(T):
        k = reg_idx[tt]
        coefs = np.concatenate([phi, gammas[k]])
        sigmas = np.concatenate([np.ones(p), sig_Z])
        true_imp[tt] = _std_importance(coefs, sigmas)

    return Y, Z, true_imp




def sim_regime_garch_factors(T=1500, seed=4,
                             regime_lengths=(750, 750)):
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # Factors: market M and vol-risk V
    M = rng.normal(0, 1.0, size=T)
    V = rng.normal(0, 1.0, size=T)
    Z = np.vstack([M, V]).T

    # regime-specific betas and GARCH params
    betas = [
        np.array([1.2, 0.2]),  # calm: market dominates
        np.array([0.3, 1.2])   # crisis: vol dominates
    ]
    garch = [
        (0.02, 0.05, 0.9),     # (omega, a, b)
        (0.05, 0.12, 0.85)
    ]

    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    h = np.zeros(T)
    eta = np.zeros(T)
    r = np.zeros(T)

    h[0] = 0.1
    for tt in range(T):
        k = reg_idx[tt]
        omega, a, b = garch[k]
        if tt > 0:
            h[tt] = omega + a*eta[tt-1]**2 + b*h[tt-1]
        eta[tt] = rng.normal(0, np.sqrt(h[tt]))
        r[tt] = Z[tt] @ betas[k] + eta[tt]

    true_imp = np.zeros((T, 2))
    sig_Z = Z.std(axis=0)
    for tt in range(T):
        k = reg_idx[tt]
        true_imp[tt] = _std_importance(betas[k], sig_Z)

    return r, Z, true_imp

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

    # 4. Spike/jump process
    print("\n4. Generating spike_process...")
    Y, Z, true_imp = sim_spike_process(T=T, seed=2)
    save_dataset("spike_process", Y, Z, true_imp)

    # 5. GARCH with regime-shifting factors
    print("\n6. Generating garch_regime...")
    Y, Z, true_imp = sim_regime_garch_factors(T=T, seed=4)
    save_dataset("garch_regime", Y, Z, true_imp)

    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("="*60)