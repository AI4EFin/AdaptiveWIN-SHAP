"""
Generate simulated datasets with RANDOMIZED PARAMETERS.

This module extends generate_simulated_datasets.py by randomizing the DGP parameters
while maintaining the structural properties (e.g., which feature dominates in which regime).

This is useful for bootstrap robustness testing to ensure methods work across
different parameter magnitudes, not just different noise realizations.
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


def sim_piecewise_ar3_rotating(T=1500, seed=123, noise_sigma=None,
                               regime_lengths=(500, 500, 500),
                               randomize_params=False):
    """
    Piecewise AR(3) with rotating dominant lag.

    Parameters
    ----------
    T : int
        Number of time points
    seed : int
        Random seed (used for both noise and parameter randomization)
    noise_sigma : float or None
        Noise standard deviation. If None and randomize_params=True, will be random
    regime_lengths : tuple
        Length of each regime
    randomize_params : bool
        If True, randomize AR coefficients while maintaining dominance structure

    Returns
    -------
    X : array (T,)
        Target series
    Z : None
        No covariates
    true_imp : array (T, 3)
        True lag importances
    params : dict
        Parameters used for generation (for documentation)
    """
    rng = np.random.default_rng(seed)

    if randomize_params:
        # Randomize parameters while maintaining structure
        # Dominant coefficient: uniform[0.7, 0.95]
        # Non-dominant coefficients: Dirichlet to ensure they sum to something reasonable

        dom_coef = rng.uniform(0.7, 0.95)  # Dominant lag coefficient

        # Non-dominant coefficients: split remaining "importance"
        # Use range [0.05, 0.25] total for non-dominant
        remaining = rng.uniform(0.05, 0.25)
        # Split between two non-dominant lags
        alpha = rng.dirichlet([1, 1])  # Equal prior
        non_dom = remaining * alpha

        # Construct regime-specific coefficients
        # Rotate which position gets the dominant coefficient
        phis = [
            np.array([dom_coef, non_dom[0], non_dom[1]]),  # Lag 1 dominant
            np.array([non_dom[0], dom_coef, non_dom[1]]),  # Lag 2 dominant
            np.array([non_dom[0], non_dom[1], dom_coef]),  # Lag 3 dominant
        ]

        # Randomize noise if requested
        if noise_sigma is None:
            noise_sigma = rng.uniform(0.5, 1.5)
    else:
        # Default parameters from original
        phis = [
            np.array([0.9, 0.01, 0.01]),
            np.array([0.01, 0.9, 0.01]),
            np.array([0.01, 0.01, 0.9]),
        ]
        if noise_sigma is None:
            noise_sigma = 1.0

    # Build regime index
    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    # Simulate piecewise AR(3)
    for t in range(T):
        k = reg_idx[t]
        ar_part = 0.0
        for j in range(1, 4):
            if t - j >= 0:
                ar_part += phis[k][j-1] * X[t-j]
        X[t] = ar_part + eps[t]

    # True importance per time = normalized |phi|
    true_imp = np.zeros((T, 3))
    for t in range(T):
        k = reg_idx[t]
        a = np.abs(phis[k])
        true_imp[t] = a / a.sum()

    # Document parameters
    params = {
        'regime_1_coefs': phis[0].tolist(),
        'regime_2_coefs': phis[1].tolist(),
        'regime_3_coefs': phis[2].tolist(),
        'noise_sigma': noise_sigma,
        'regime_lengths': regime_lengths,
        'randomized': randomize_params
    }

    return X, None, true_imp, params


def sim_piecewise_arx_rotating_drivers(T=1500, p=3, seed=0,
                                       regime_lengths=(500, 500, 500),
                                       noise_sigma=None,
                                       randomize_params=False):
    """
    ARX with rotating covariate drivers.

    Parameters
    ----------
    randomize_params : bool
        If True, randomize coefficients while maintaining dominance structure
    """
    rng = np.random.default_rng(seed)

    # Covariates: demand D, fuel F, renewables R
    D = rng.normal(0, 1.0, size=T) + 0.3*np.sin(2*np.pi*np.arange(T)/24)
    F = rng.normal(0, 1.0, size=T) + 0.1*np.arange(T)/T
    R = rng.normal(0, 1.0, size=T) + 0.5*np.sin(2*np.pi*np.arange(T)/168)
    Z = np.vstack([D, F, R]).T

    if randomize_params:
        # Randomize AR coefficients (keep roughly stationary)
        # phi[0] dominant, others smaller
        phi_1 = rng.uniform(0.5, 0.7)
        remaining = rng.uniform(0.1, 0.3)
        phi_rest = remaining * rng.dirichlet([1, 1])
        phi = np.array([phi_1, phi_rest[0], phi_rest[1]])

        # Randomize covariate coefficients
        # Dominant: [0.8, 1.5], non-dominant: [-0.2, 0.3]
        dom_coef = rng.uniform(0.8, 1.5)
        non_dom_1 = rng.uniform(-0.2, 0.3)
        non_dom_2 = rng.uniform(-0.2, 0.3)

        betas = [
            np.array([non_dom_1, dom_coef, non_dom_2]),     # Fuel dominates
            np.array([dom_coef, non_dom_1, non_dom_2]),     # Demand dominates
            np.array([non_dom_1, non_dom_2, -dom_coef])     # RES dominates (negative)
        ]

        if noise_sigma is None:
            noise_sigma = rng.uniform(0.3, 0.7)
    else:
        phi = np.array([0.6, 0.2, 0.1])
        betas = [
            np.array([0.2, 1.2, -0.1]),
            np.array([1.2, 0.2, -0.1]),
            np.array([0.2, 0.1, -1.2])
        ]
        if noise_sigma is None:
            noise_sigma = 0.5

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

    # True importance
    true_imp = np.zeros((T, p+3))
    sig_lags = np.ones(p)
    sig_Z = Z.std(axis=0)
    for t in range(T):
        k = reg_idx[t]
        coefs = np.concatenate([phi, betas[k]])
        sigmas = np.concatenate([sig_lags, sig_Z])
        true_imp[t] = _std_importance(coefs, sigmas)

    params = {
        'ar_coefs': phi.tolist(),
        'regime_1_betas': betas[0].tolist(),
        'regime_2_betas': betas[1].tolist(),
        'regime_3_betas': betas[2].tolist(),
        'noise_sigma': noise_sigma,
        'regime_lengths': regime_lengths,
        'randomized': randomize_params
    }

    return Y, Z, true_imp, params


def sim_trend_season_ar_break(T=1500, p=3, seed=1,
                              regime_lengths=(500, 500, 500),
                              noise_sigma=None,
                              randomize_params=False):
    """Trend + seasonality + AR break with optional parameter randomization."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # Deterministic components (keep fixed)
    trend = 0.002 * t
    seas_day = 1.0*np.sin(2*np.pi*t/24) + 0.5*np.cos(2*np.pi*t/24)
    seas_week = 0.6*np.sin(2*np.pi*t/168)

    # Randomize AR coefficients if requested
    if randomize_params:
        dom_coef = rng.uniform(0.7, 0.95)
        remaining = rng.uniform(0.05, 0.25)
        alpha = rng.dirichlet([1, 1])
        non_dom = remaining * alpha

        phis = [
            np.array([dom_coef, non_dom[0], non_dom[1]]),
            np.array([non_dom[0], dom_coef, non_dom[1]]),
            np.array([non_dom[0], non_dom[1], dom_coef]),
        ]

        if noise_sigma is None:
            noise_sigma = rng.uniform(0.2, 0.6)
    else:
        phis = [
            np.array([0.9, 0.01, 0.01]),
            np.array([0.01, 0.9, 0.01]),
            np.array([0.01, 0.01, 0.9]),
        ]
        if noise_sigma is None:
            noise_sigma = 0.4

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

    params = {
        'regime_1_coefs': phis[0].tolist(),
        'regime_2_coefs': phis[1].tolist(),
        'regime_3_coefs': phis[2].tolist(),
        'noise_sigma': noise_sigma,
        'regime_lengths': regime_lengths,
        'randomized': randomize_params
    }

    return Y, None, true_imp, params


def sim_spike_process(T=1500, p=3, seed=2,
                      regime_lengths=(750, 750),
                      noise_sigma=None,
                      randomize_params=False):
    """Spike process with optional parameter randomization."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # Covariates
    D = rng.normal(0, 1.0, size=T) + 0.3*np.sin(2*np.pi*t/24)
    R = rng.normal(0, 1.0, size=T) + 0.5*np.sin(2*np.pi*t/168)
    Z = np.vstack([D, R]).T

    if randomize_params:
        phi_1 = rng.uniform(0.6, 0.8)
        remaining = rng.uniform(0.1, 0.3)
        phi_rest = remaining * rng.dirichlet([2, 1])  # Bias toward lag 2
        phi = np.array([phi_1, phi_rest[0], phi_rest[1]])

        # Spike driver coefficients
        gamma_dom = rng.uniform(1.0, 2.0)
        gammas = [
            np.array([gamma_dom, 0.0]),
            np.array([0.0, gamma_dom]),
        ]

        if noise_sigma is None:
            noise_sigma = rng.uniform(0.2, 0.5)
    else:
        phi = np.array([0.7, 0.2, 0.05])
        gammas = [
            np.array([1.5, 0.0]),
            np.array([0.0, 1.5]),
        ]
        if noise_sigma is None:
            noise_sigma = 0.3

    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, L in enumerate(regime_lengths):
        reg_idx[start:start+L] = k
        start += L

    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    for tt in range(T):
        ar_part = 0.0
        for j in range(1, p+1):
            if tt-j >= 0:
                ar_part += phi[j-1]*X[tt-j]
        X[tt] = ar_part + eps[tt]

    # Spikes
    J = np.zeros(T)
    S = rng.lognormal(mean=1.0, sigma=0.6, size=T)
    for tt in range(T):
        k = reg_idx[tt]
        logit_p = Z[tt] @ gammas[k] - 2.0
        p_t = 1/(1 + np.exp(-logit_p))
        J[tt] = rng.binomial(1, p_t)

    Y = X + J*S

    true_imp = np.zeros((T, p+2))
    sig_Z = Z.std(axis=0)
    for tt in range(T):
        k = reg_idx[tt]
        coefs = np.concatenate([phi, gammas[k]])
        sigmas = np.concatenate([np.ones(p), sig_Z])
        true_imp[tt] = _std_importance(coefs, sigmas)

    params = {
        'ar_coefs': phi.tolist(),
        'regime_1_gammas': gammas[0].tolist(),
        'regime_2_gammas': gammas[1].tolist(),
        'noise_sigma': noise_sigma,
        'regime_lengths': regime_lengths,
        'randomized': randomize_params
    }

    return Y, Z, true_imp, params


def sim_regime_garch_factors(T=1500, seed=4,
                             regime_lengths=(750, 750),
                             randomize_params=False):
    """GARCH with optional parameter randomization."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # Factors
    M = rng.normal(0, 1.0, size=T)
    V = rng.normal(0, 1.0, size=T)
    Z = np.vstack([M, V]).T

    if randomize_params:
        # Randomize factor loadings
        beta_dom = rng.uniform(1.0, 1.5)
        beta_non = rng.uniform(0.1, 0.4)

        betas = [
            np.array([beta_dom, beta_non]),   # Market dominates
            np.array([beta_non, beta_dom])    # Vol dominates
        ]

        # Randomize GARCH parameters (keep stationary: α + β < 1)
        # Calm period: low α, high β (persistent, low volatility)
        omega_1 = rng.uniform(0.01, 0.03)
        alpha_1 = rng.uniform(0.03, 0.08)
        beta_1 = rng.uniform(0.85, 0.95)
        beta_1 = min(beta_1, 0.98 - alpha_1)  # Ensure stationarity

        # Crisis period: higher α, lower β (more reactive, higher volatility)
        omega_2 = rng.uniform(0.03, 0.07)
        alpha_2 = rng.uniform(0.10, 0.18)
        beta_2 = rng.uniform(0.75, 0.88)
        beta_2 = min(beta_2, 0.98 - alpha_2)

        garch = [
            (omega_1, alpha_1, beta_1),
            (omega_2, alpha_2, beta_2)
        ]
    else:
        betas = [
            np.array([1.2, 0.2]),
            np.array([0.3, 1.2])
        ]
        garch = [
            (0.02, 0.05, 0.9),
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

    params = {
        'regime_1_betas': betas[0].tolist(),
        'regime_2_betas': betas[1].tolist(),
        'regime_1_garch': garch[0],
        'regime_2_garch': garch[1],
        'regime_lengths': regime_lengths,
        'randomized': randomize_params
    }

    return r, Z, true_imp, params