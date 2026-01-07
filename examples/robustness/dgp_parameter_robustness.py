"""
DGP Parameter Robustness Test for Adaptive WIN-SHAP

Tests robustness of the method to variations in data generating process parameters.
Generates multiple scenarios with different AR(3) coefficients and evaluates
performance across changepoint detection, SHAP accuracy, and explanation quality.

Usage:
    python examples/robustness/dgp_parameter_robustness.py [options]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================================
# PARAMETER GENERATION FUNCTIONS
# ============================================================================

def is_stationary(phi: np.ndarray) -> bool:
    """
    Check if AR(3) parameters are stationary.

    Parameters
    ----------
    phi : np.ndarray
        AR(3) coefficients [phi_1, phi_2, phi_3]

    Returns
    -------
    bool
        True if stationary (all roots outside unit circle)
    """
    # Reject if all zeros (degenerate case)
    if np.allclose(phi, 0, atol=1e-8):
        return False

    try:
        # Characteristic polynomial: 1 - phi_1*z - phi_2*z^2 - phi_3*z^3 = 0
        # Rearranged: -phi_3*z^3 - phi_2*z^2 - phi_1*z + 1 = 0
        # For numpy.roots (descending powers): [-phi_3, -phi_2, -phi_1, 1]
        coefficients = [-phi[2], -phi[1], -phi[0], 1]
        roots = np.roots(coefficients)

        # All roots must be outside unit circle (magnitude > 1)
        # Use small tolerance to handle numerical precision
        return np.all(np.abs(roots) > 1.0)
    except:
        # If roots computation fails, assume non-stationary
        return False


def project_to_stationary(phi: np.ndarray, tolerance: float = 1e-4,
                         min_scale: float = 0.1) -> np.ndarray:
    """
    Project potentially non-stationary parameters onto stationary region.

    Uses binary search to find largest scaling factor that maintains stationarity.

    Parameters
    ----------
    phi : np.ndarray
        AR(3) coefficients (potentially non-stationary)
    tolerance : float
        Convergence tolerance for binary search
    min_scale : float
        Minimum scaling factor to avoid degenerate solutions

    Returns
    -------
    np.ndarray
        Projected stationary parameters
    """
    if is_stationary(phi):
        return phi

    # Try scaling down first
    scale_min = min_scale
    scale_max = 1.0

    # Check if even min_scale works
    if not is_stationary(phi * min_scale):
        # If even small scaling doesn't work, try a different approach
        # Use a simple heuristic: scale coefficients to sum to ~0.5
        phi_sum = np.sum(np.abs(phi))
        if phi_sum > 0:
            phi_scaled = phi * (0.5 / phi_sum)
            if is_stationary(phi_scaled):
                return phi_scaled

        # Last resort: return a known stationary configuration
        print(f"    Warning: Could not project {phi} to stationary region, using fallback")
        return np.array([0.6, 0.2, 0.1])

    # Binary search for largest valid scale
    best_phi = phi * min_scale
    while scale_max - scale_min > tolerance:
        scale = (scale_min + scale_max) / 2
        phi_scaled = phi * scale

        if is_stationary(phi_scaled):
            scale_min = scale
            best_phi = phi_scaled
        else:
            scale_max = scale

    return best_phi


def compute_l2_distance(phi1: np.ndarray, phi2: np.ndarray) -> float:
    """Compute L2 distance between two parameter vectors."""
    return np.linalg.norm(phi1 - phi2)


def generate_closer_scenario(baseline_regimes: List[np.ndarray],
                             target_reduction: float = 0.5) -> List[np.ndarray]:
    """
    Generate parameters closer together via linear interpolation toward centroid.

    Parameters
    ----------
    baseline_regimes : List[np.ndarray]
        Original regime parameters
    target_reduction : float
        Target reduction fraction (0.5 = reduce to 50%)

    Returns
    -------
    List[np.ndarray]
        New regime parameters with reduced L2 distances
    """
    # Compute centroid
    centroid = np.mean(baseline_regimes, axis=0)

    # Interpolate each regime toward centroid
    closer_regimes = []
    for phi in baseline_regimes:
        phi_closer = phi + target_reduction * (centroid - phi)

        # Ensure stationarity
        if not is_stationary(phi_closer):
            phi_closer = project_to_stationary(phi_closer)

        closer_regimes.append(phi_closer)

    return closer_regimes


def generate_further_scenario() -> List[np.ndarray]:
    """
    Generate parameters further apart with mixed dynamics.

    Manually specified regimes with extreme positive and negative coefficients.

    Returns
    -------
    List[np.ndarray]
        Three regime parameters with increased L2 distances
    """
    # These are pre-verified stationary parameters with large L2 distances
    regimes = [
        np.array([0.85, 0.05, 0.03]),    # Extreme positive, lag 1 dominant
        np.array([0.5, -0.25, 0.15]),    # Mixed with negative
        np.array([-0.05, 0.75, 0.08])    # Oscillatory, lag 2 dominant
    ]

    # Verify all are stationary (should pass now)
    for i, phi in enumerate(regimes):
        if not is_stationary(phi):
            print(f"Warning: Further scenario regime {i} not stationary, adjusting...")
            # If somehow not stationary, use a safe alternative
            if i == 0:
                regimes[i] = np.array([0.8, 0.1, 0.05])
            elif i == 1:
                regimes[i] = np.array([0.4, -0.2, 0.2])
            else:
                regimes[i] = np.array([0.05, 0.7, 0.1])

            # Verify again
            if not is_stationary(regimes[i]):
                regimes[i] = project_to_stationary(regimes[i])

    return regimes


def generate_random_scenario(seed: int, max_attempts: int = 100) -> List[np.ndarray]:
    """
    Generate random stationary AR(3) parameters via smart sampling.

    Uses a mixture strategy: sample from smaller region then perturb.

    Parameters
    ----------
    seed : int
        Random seed
    max_attempts : int
        Maximum sampling attempts before using fallback

    Returns
    -------
    List[np.ndarray]
        Three regime parameters sampled from stationary region
    """
    rng = np.random.default_rng(seed)
    regimes = []

    # Known stationary templates to perturb
    templates = [
        np.array([0.8, 0.1, 0.05]),   # Strong first lag
        np.array([0.3, 0.6, 0.05]),   # Strong second lag
        np.array([0.2, 0.2, 0.5]),    # Strong third lag
        np.array([0.5, 0.3, 0.15]),   # Balanced
        np.array([0.6, -0.2, 0.1]),   # With negative
    ]

    for regime_idx in range(3):
        attempt = 0
        stationary = False
        phi = None

        while not stationary and attempt < max_attempts:
            if attempt < max_attempts // 2:
                # First half: try uniform sampling from smaller range
                phi = rng.uniform(-0.5, 0.8, size=3)
            else:
                # Second half: perturb a known stationary template
                template = templates[regime_idx % len(templates)]
                perturbation = rng.normal(0, 0.2, size=3)
                phi = template + perturbation

            stationary = is_stationary(phi)
            attempt += 1

        if not stationary:
            print(f"Warning: Random scenario seed={seed} regime {regime_idx} "
                  f"failed after {max_attempts} attempts, using template")
            phi = templates[regime_idx % len(templates)]
            # Add small random perturbation
            phi = phi + rng.normal(0, 0.05, size=3)
            # Ensure it's still stationary
            if not is_stationary(phi):
                phi = templates[regime_idx % len(templates)]

        regimes.append(phi)

    # Verify regimes are distinct
    min_distance = 0.3
    max_regen_attempts = 5
    for i in range(len(regimes)):
        for j in range(i+1, len(regimes)):
            dist = compute_l2_distance(regimes[i], regimes[j])
            if dist < min_distance:
                # Try to increase distance
                for regen_attempt in range(max_regen_attempts):
                    # Perturb a different template
                    template_idx = (j + regen_attempt + 1) % len(templates)
                    new_phi = templates[template_idx] + rng.normal(0, 0.15, size=3)

                    if is_stationary(new_phi):
                        new_dist = compute_l2_distance(regimes[i], new_phi)
                        if new_dist >= min_distance:
                            regimes[j] = new_phi
                            break

    return regimes


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def simulate_piecewise_ar3(regimes: List[np.ndarray],
                          regime_lengths: Tuple[int, int, int] = (500, 500, 500),
                          noise_sigma: float = 1.0,
                          seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate piecewise AR(3) time series with regime-specific parameters.

    Parameters
    ----------
    regimes : List[np.ndarray]
        List of AR(3) parameter arrays
    regime_lengths : Tuple[int, int, int]
        Length of each regime
    noise_sigma : float
        Standard deviation of noise
    seed : int
        Random seed

    Returns
    -------
    X : np.ndarray
        Simulated time series
    true_imp : np.ndarray
        True feature importances (T, 3)
    """
    rng = np.random.default_rng(seed)
    T = sum(regime_lengths)

    # Build regime index
    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, length in enumerate(regime_lengths):
        reg_idx[start:start+length] = k
        start += length

    # Simulate
    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    for t in range(T):
        k = reg_idx[t]
        phi = regimes[k]

        ar_part = 0.0
        for j in range(1, 4):
            if t - j >= 0:
                ar_part += phi[j-1] * X[t-j]

        X[t] = ar_part + eps[t]

    # Compute true importances
    true_imp = np.zeros((T, 3))
    for t in range(T):
        k = reg_idx[t]
        phi = regimes[k]
        abs_phi = np.abs(phi)
        true_imp[t] = abs_phi / abs_phi.sum() if abs_phi.sum() > 0 else abs_phi

    return X, true_imp


def save_scenario_data(scenario_name: str,
                       X: np.ndarray,
                       true_imp: np.ndarray,
                       regimes: List[np.ndarray],
                       regime_lengths: Tuple[int, int, int],
                       output_dir: Path,
                       scenario_type: str,
                       baseline_regimes: Optional[List[np.ndarray]] = None,
                       seed: Optional[int] = None):
    """
    Save scenario data and metadata to disk.

    Parameters
    ----------
    scenario_name : str
        Name of scenario (e.g., 'closer', 'further', 'random_1')
    X : np.ndarray
        Time series data
    true_imp : np.ndarray
        True feature importances
    regimes : List[np.ndarray]
        Regime parameters used
    regime_lengths : Tuple[int, int, int]
        Length of each regime
    output_dir : Path
        Parent output directory
    scenario_type : str
        Type of scenario for metadata
    baseline_regimes : Optional[List[np.ndarray]]
        Baseline parameters for L2 distance computation
    seed : Optional[int]
        Random seed used
    """
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    data_df = pd.DataFrame({'N': X})
    data_df.to_csv(scenario_dir / 'data.csv', index=False)

    # Save true importances
    imp_cols = {f'true_imp_{i}': true_imp[:, i] for i in range(true_imp.shape[1])}
    imp_df = pd.DataFrame(imp_cols)
    imp_df.to_csv(scenario_dir / 'true_importances.csv', index=False)

    # Compute L2 distances
    pairwise_l2 = {}
    for i in range(len(regimes)):
        for j in range(i+1, len(regimes)):
            key = f"regime_{i}_vs_{j}"
            pairwise_l2[key] = float(compute_l2_distance(regimes[i], regimes[j]))

    max_pairwise_l2 = max(pairwise_l2.values()) if pairwise_l2 else 0.0

    # Compute L2 distances from baseline if provided
    l2_from_baseline = []
    if baseline_regimes is not None:
        for i, phi in enumerate(regimes):
            # Distance to corresponding baseline regime
            dist = compute_l2_distance(phi, baseline_regimes[i])
            l2_from_baseline.append(float(dist))

    # Create metadata
    breakpoints = [regime_lengths[0], regime_lengths[0] + regime_lengths[1]]

    metadata = {
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "max_pairwise_l2": max_pairwise_l2,
        "baseline_reference": "piecewise_ar3",
        "regimes": []
    }

    start = 0
    for k, length in enumerate(regime_lengths):
        regime_meta = {
            "regime_id": k,
            "time_range": [start, start + length],
            "parameters": {
                "phi_1": float(regimes[k][0]),
                "phi_2": float(regimes[k][1]),
                "phi_3": float(regimes[k][2])
            }
        }

        if l2_from_baseline:
            regime_meta["l2_distance_from_baseline"] = l2_from_baseline[k]

        metadata["regimes"].append(regime_meta)
        start += length

    metadata["generation_params"] = {
        "T": len(X),
        "noise_sigma": 1.0,
        "regime_lengths": list(regime_lengths)
    }

    if seed is not None:
        metadata["generation_params"]["seed"] = seed

    metadata["stationarity_check"] = {
        f"regime_{k}_stationary": bool(is_stationary(regimes[k]))
        for k in range(len(regimes))
    }

    metadata["pairwise_l2_distances"] = pairwise_l2

    # Save metadata
    with open(scenario_dir / 'scenario_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {scenario_name}: L2_max={max_pairwise_l2:.3f}")


# ============================================================================
# LPA DETECTION
# ============================================================================

def run_lpa_detection(data_path: Path,
                     output_dir: Path,
                     N0: int = 75,
                     alpha: float = 0.95,
                     num_bootstrap: int = 50,
                     jump: int = 1,
                     growth: str = 'geometric',
                     growth_base: float = 1.41421356237,
                     verbose: bool = False) -> Path:
    """
    Run LPA window detection on a dataset.

    Parameters
    ----------
    data_path : Path
        Path to data.csv
    output_dir : Path
        Directory to save windows.csv
    N0 : int
        Initial window size
    alpha : float
        Confidence level
    num_bootstrap : int
        Number of bootstrap iterations
    jump : int
        Step size
    growth : str
        Window growth strategy
    growth_base : float
        Base for geometric growth
    verbose : bool
        Print progress

    Returns
    -------
    Path
        Path to saved windows.csv
    """
    import torch
    import torch.nn as nn
    from adaptivewinshap import AdaptiveModel, ChangeDetector, store_init_kwargs

    # Define LSTM model class (same as in lstm_simulation.py)
    class AdaptiveLSTM(AdaptiveModel):
        @store_init_kwargs
        def __init__(self, device, seq_length=3, input_size=1, hidden=16, layers=1,
                     dropout=0.2, batch_size=512, lr=1e-12, epochs=50,
                     type_precision=np.float32):
            super().__init__(device=device, batch_size=batch_size, lr=lr, epochs=epochs,
                           type_precision=type_precision)
            self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.fc = nn.Linear(hidden, 1)
            self.seq_length = seq_length
            self.input_size = input_size
            self.hidden = hidden
            self.layers = layers
            self.dropout = dropout

        def forward(self, x):
            out, _ = self.lstm(x)
            yhat = self.fc(out[:, -1, :])
            return yhat.squeeze(-1)

        def prepare_data(self, window, start_abs_idx):
            L = self.seq_length
            F = window.shape[1] if window.ndim == 2 else 1
            n = len(window)

            if n <= L:
                return None, None, None

            if window.ndim == 1:
                window = window[:, None]

            X_list = []
            y_list = []
            for i in range(L, n):
                X_list.append(window[i-L:i])
                y_list.append(window[i, 0])

            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)

            t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)

            X_tensor = torch.from_numpy(X)
            y_tensor = torch.from_numpy(y)
            return X_tensor, y_tensor, t_abs

        @torch.no_grad()
        def predict(self, x: np.ndarray) -> np.ndarray:
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            preds = self(xt)
            return preds.detach().cpu().numpy().reshape(-1, 1)

    if verbose:
        print(f"    Running LPA detection: N0={N0}, alpha={alpha}, "
              f"bootstrap={num_bootstrap}, jump={jump}")

    # Load data
    df = pd.read_csv(data_path)
    target = df["N"].to_numpy(dtype=np.float64)

    # Check for covariates
    cov_cols = [c for c in df.columns if c.startswith('Z_')]
    if cov_cols:
        covariates = df[cov_cols].to_numpy(dtype=np.float64)
        data = np.column_stack([target, covariates])
        input_size = 1 + len(cov_cols)
    else:
        data = target
        input_size = 1

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    # Initialize model (AR(3) by default for piecewise_ar3)
    model = AdaptiveLSTM(
        device, seq_length=3, input_size=input_size,
        hidden=16, layers=1, dropout=0.0,
        batch_size=64, lr=1e-2, epochs=15,
        type_precision=np.float64
    )

    # Create change detector
    cd = ChangeDetector(model, data, debug=False, force_cpu=True)

    # Run detection
    min_seg = 4
    results = cd.detect(
        min_window=min_seg,
        n_0=N0,
        jump=jump,
        search_step=1,
        alpha=alpha,
        num_bootstrap=num_bootstrap,
        t_workers=10,
        b_workers=10,
        one_b_threads=1,
        growth=growth,
        growth_base=growth_base
    )

    # Save results
    windows_df = pd.DataFrame(results)
    windows_df = windows_df[['windows']].rename(columns={'windows': 'window_mean'})

    windows_path = output_dir / 'windows.csv'
    windows_df.to_csv(windows_path, index=False)

    if verbose:
        print(f"    Saved windows: {windows_path}")

    return windows_path


# ============================================================================
# SHAP BENCHMARKING
# ============================================================================

def run_shap_benchmark(data_path: Path,
                      windows_path: Path,
                      output_dir: Path,
                      verbose: bool = False) -> Path:
    """
    Run SHAP benchmark on a dataset with precomputed windows.

    Parameters
    ----------
    data_path : Path
        Path to data.csv
    windows_path : Path
        Path to windows.csv
    output_dir : Path
        Directory to save benchmark results
    verbose : bool
        Print progress

    Returns
    -------
    Path
        Path to benchmark directory
    """
    from benchmark import run_benchmark

    benchmark_dir = output_dir / 'benchmark'
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"    Running SHAP benchmark")

    # Run benchmark
    summary_df = run_benchmark(
        dataset_path=str(data_path),
        output_dir=str(benchmark_dir),
        dataset_type='simulated',
        column_name='N',
        precomputed_windows_path=str(windows_path),
        verbose=verbose
    )

    if verbose:
        print(f"    Saved benchmark results: {benchmark_dir}")

    return benchmark_dir


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_detection_metrics(windows_df: pd.DataFrame,
                              true_breakpoints: List[int],
                              tolerance: int = 50) -> Dict:
    """
    Compute window detection accuracy metrics.

    Parameters
    ----------
    windows_df : pd.DataFrame
        Detected windows
    true_breakpoints : List[int]
        True breakpoint locations
    tolerance : int
        Tolerance window for detection success

    Returns
    -------
    Dict
        Detection metrics
    """
    # Get window sizes
    if 'window_mean' in windows_df.columns:
        windows = windows_df['window_mean'].values
    elif 'windows' in windows_df.columns:
        windows = windows_df['windows'].values
    else:
        # Try to find any window column
        window_cols = [c for c in windows_df.columns if 'window' in c.lower()]
        if window_cols:
            windows = windows_df[window_cols[0]].values
        else:
            return {
                'detection_lag_mean': np.nan,
                'detection_lag_std': np.nan,
                'detection_success_rate': np.nan,
                'false_positive_rate': np.nan
            }

    # Detect changepoints (significant changes in window size)
    window_diff = np.abs(np.diff(windows))
    threshold = np.percentile(window_diff, 90)
    detected_breakpoints = np.where(window_diff > threshold)[0] + 1

    # Compute detection lags
    lags = []
    detected_flags = []

    for true_bp in true_breakpoints:
        # Find nearest detected breakpoint
        if len(detected_breakpoints) > 0:
            distances = np.abs(detected_breakpoints - true_bp)
            min_dist = np.min(distances)

            if min_dist <= tolerance:
                lags.append(min_dist)
                detected_flags.append(True)
            else:
                detected_flags.append(False)
        else:
            detected_flags.append(False)

    # Compute false positives
    false_positives = 0
    for det_bp in detected_breakpoints:
        # Check if near any true breakpoint
        if len(true_breakpoints) > 0:
            min_dist_to_true = np.min(np.abs(np.array(true_breakpoints) - det_bp))
            if min_dist_to_true > tolerance:
                false_positives += 1

    return {
        'detection_lag_mean': np.mean(lags) if lags else np.nan,
        'detection_lag_std': np.std(lags) if lags else np.nan,
        'detection_success_rate': np.mean(detected_flags) if detected_flags else 0.0,
        'false_positive_rate': false_positives / len(windows) if len(windows) > 0 else 0.0
    }


def compute_shap_consistency_metrics(shap_results_path: Path,
                                     true_importances_path: Path,
                                     regime_lengths: Tuple[int, int, int]) -> Dict:
    """
    Compute SHAP consistency with true importances, regime-specific.

    Parameters
    ----------
    shap_results_path : Path
        Path to adaptive_shap_results.csv
    true_importances_path : Path
        Path to true_importances.csv
    regime_lengths : Tuple[int, int, int]
        Length of each regime

    Returns
    -------
    Dict
        SHAP consistency metrics
    """
    if not shap_results_path.exists() or not true_importances_path.exists():
        return {
            'shap_corr_regime0': np.nan,
            'shap_corr_regime1': np.nan,
            'shap_corr_regime2': np.nan,
            'shap_corr_overall': np.nan
        }

    # Load data
    shap_df = pd.read_csv(shap_results_path)
    true_df = pd.read_csv(true_importances_path)

    # Extract SHAP values (look for feature importance columns)
    shap_cols = [c for c in shap_df.columns if c.startswith('feat_') or c.startswith('shap_')]
    true_cols = [c for c in true_df.columns if c.startswith('true_imp_')]

    if not shap_cols or not true_cols:
        return {
            'shap_corr_regime0': np.nan,
            'shap_corr_regime1': np.nan,
            'shap_corr_regime2': np.nan,
            'shap_corr_overall': np.nan
        }

    # Align by taking minimum length
    n = min(len(shap_df), len(true_df))

    shap_values = shap_df[shap_cols].iloc[:n].values
    true_values = true_df[true_cols].iloc[:n].values

    # Compute overall correlation
    shap_flat = shap_values.flatten()
    true_flat = true_values.flatten()

    mask = ~(np.isnan(shap_flat) | np.isnan(true_flat))
    if mask.sum() > 0:
        corr_overall = np.corrcoef(shap_flat[mask], true_flat[mask])[0, 1]
    else:
        corr_overall = np.nan

    # Compute regime-specific correlations
    regime_corrs = []
    breakpoints = [0, regime_lengths[0], regime_lengths[0] + regime_lengths[1], sum(regime_lengths)]

    for i in range(3):
        start = breakpoints[i]
        end = breakpoints[i+1]

        if end > n:
            regime_corrs.append(np.nan)
            continue

        shap_regime = shap_values[start:end].flatten()
        true_regime = true_values[start:end].flatten()

        mask = ~(np.isnan(shap_regime) | np.isnan(true_regime))
        if mask.sum() > 10:  # Need sufficient data
            corr = np.corrcoef(shap_regime[mask], true_regime[mask])[0, 1]
            regime_corrs.append(corr)
        else:
            regime_corrs.append(np.nan)

    return {
        'shap_corr_regime0': regime_corrs[0],
        'shap_corr_regime1': regime_corrs[1],
        'shap_corr_regime2': regime_corrs[2],
        'shap_corr_overall': corr_overall
    }


def compute_window_statistics(windows_df: pd.DataFrame) -> Dict:
    """Compute window size statistics."""
    if 'window_mean' in windows_df.columns:
        windows = windows_df['window_mean'].values
    elif 'windows' in windows_df.columns:
        windows = windows_df['windows'].values
    else:
        window_cols = [c for c in windows_df.columns if 'window' in c.lower()]
        if window_cols:
            windows = windows_df[window_cols[0]].values
        else:
            return {
                'window_mean': np.nan,
                'window_std': np.nan,
                'window_min': np.nan,
                'window_max': np.nan
            }

    return {
        'window_mean': float(np.mean(windows)),
        'window_std': float(np.std(windows)),
        'window_min': float(np.min(windows)),
        'window_max': float(np.max(windows))
    }


def extract_benchmark_metrics(benchmark_summary_path: Path) -> Dict:
    """Extract faithfulness and ablation metrics from benchmark summary."""
    if not benchmark_summary_path.exists():
        return {
            'faithfulness_p50': np.nan,
            'faithfulness_p90': np.nan,
            'ablation_mif_p90': np.nan,
            'ablation_lif_p90': np.nan
        }

    df = pd.read_csv(benchmark_summary_path)

    # Filter to adaptive_shap method
    adaptive_df = df[df['method'] == 'adaptive_shap']

    metrics = {}

    # Extract faithfulness
    faith_df = adaptive_df[adaptive_df['metric_type'] == 'faithfulness']
    for eval_type in ['prtb_p50', 'prtb_p90']:
        row = faith_df[faith_df['evaluation'] == eval_type]
        if len(row) > 0:
            metrics[f'faithfulness_{eval_type.replace("prtb_", "")}'] = float(row['score'].values[0])
        else:
            metrics[f'faithfulness_{eval_type.replace("prtb_", "")}'] = np.nan

    # Extract ablation
    ablation_df = adaptive_df[adaptive_df['metric_type'] == 'ablation']
    for eval_type in ['mif_p90', 'lif_p90']:
        row = ablation_df[ablation_df['evaluation'] == eval_type]
        if len(row) > 0:
            metrics[f'ablation_{eval_type}'] = float(row['score'].values[0])
        else:
            metrics[f'ablation_{eval_type}'] = np.nan

    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_window_evolution_comparison(results_dir: Path,
                                     scenarios: List[str],
                                     figures_dir: Path,
                                     breakpoints: List[int]):
    """Plot window evolution for all scenarios side-by-side."""
    n_scenarios = len(scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    fig.suptitle('Window Size Evolution Across DGP Scenarios',
                fontsize=14, fontweight='bold')

    # Flatten axes
    if n_scenarios == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        windows_path = results_dir / scenario / 'windows.csv'

        if not windows_path.exists():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(scenario)
            continue

        windows_df = pd.read_csv(windows_path)
        if 'window_mean' in windows_df.columns:
            windows = windows_df['window_mean'].values
        else:
            windows = windows_df.iloc[:, 0].values

        time_index = np.arange(len(windows))

        # Plot window sizes
        ax.plot(time_index, windows, linewidth=1.5, alpha=0.8, color='#2ca02c')

        # Add breakpoints
        for bp in breakpoints:
            ax.axvline(x=bp, color='red', linestyle='--', linewidth=1.5,
                      alpha=0.6)

        # Statistics
        mean_window = windows.mean()
        std_window = windows.std()
        ax.axhline(y=mean_window, color='blue', linestyle=':', linewidth=1,
                  alpha=0.5, label=f'Mean: {mean_window:.1f}')

        ax.set_xlabel('Time Index')
        ax.set_ylabel('Window Size')
        ax.set_title(scenario, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_scenarios, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = figures_dir / 'window_evolution_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metrics_vs_l2_distance(summary_df: pd.DataFrame,
                                figures_dir: Path):
    """Plot key metrics vs L2 distance from baseline."""
    # Select key metrics to plot
    metrics = [
        ('detection_lag_mean', 'Detection Lag (steps)', 'lower'),
        ('shap_corr_overall', 'SHAP Correlation', 'higher'),
        ('faithfulness_p90', 'Faithfulness (p90)', 'lower'),
        ('window_std', 'Window Size Std Dev', 'lower')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Performance Metrics vs L2 Distance from Baseline',
                fontsize=14, fontweight='bold')

    for idx, (metric, label, better) in enumerate(metrics):
        ax = axes[idx]

        if metric not in summary_df.columns:
            ax.text(0.5, 0.5, f'{metric} not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        x = summary_df['l2_distance_from_baseline'].values
        y = summary_df[metric].values

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        # Scatter plot with labels
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        ax.scatter(x, y, s=100, alpha=0.7, c=colors, edgecolors='black', linewidths=1.5)

        # Add scenario labels
        for i, (xi, yi) in enumerate(zip(x, y)):
            scenario = summary_df[mask].iloc[i]['scenario_name']
            ax.annotate(scenario, (xi, yi), fontsize=8,
                       xytext=(5, 5), textcoords='offset points')

        # Fit trend line if enough points
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, label='Trend')

        ax.set_xlabel('L2 Distance from Baseline', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} ({better} is better)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if len(x) > 2:
            ax.legend()

    plt.tight_layout()
    save_path = figures_dir / 'metrics_vs_l2_distance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparative_window_overlay(results_dir: Path,
                                    scenarios: List[str],
                                    summary_df: pd.DataFrame,
                                    figures_dir: Path,
                                    breakpoints: List[int]):
    """Overlay all scenario windows on a single plot."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    for idx, scenario in enumerate(scenarios):
        windows_path = results_dir / scenario / 'windows.csv'

        if not windows_path.exists():
            continue

        windows_df = pd.read_csv(windows_path)
        if 'window_mean' in windows_df.columns:
            windows = windows_df['window_mean'].values
        else:
            windows = windows_df.iloc[:, 0].values

        time_index = np.arange(len(windows))

        # Get L2 distance for label
        row = summary_df[summary_df['scenario_name'] == scenario]
        if len(row) > 0:
            l2_dist = row['l2_distance_from_baseline'].values[0]
            label = f'{scenario} (L2={l2_dist:.3f})'
        else:
            label = scenario

        ax.plot(time_index, windows, linewidth=2, alpha=0.7,
               color=colors[idx], label=label)

    # Add breakpoints
    for bp in breakpoints:
        ax.axvline(x=bp, color='red', linestyle='--', linewidth=2,
                  alpha=0.6, label='Breakpoint' if bp == breakpoints[0] else '')

    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('Window Size', fontsize=12)
    ax.set_title('Window Size Evolution: All Scenarios Comparison',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = figures_dir / 'comparative_window_overlay.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_regime_specific_correlations(summary_df: pd.DataFrame,
                                      figures_dir: Path):
    """Plot SHAP correlations for each regime as heatmap."""
    regime_cols = ['shap_corr_regime0', 'shap_corr_regime1', 'shap_corr_regime2']

    # Check if columns exist
    if not all(col in summary_df.columns for col in regime_cols):
        print("  Warning: Regime-specific correlation columns not found, skipping")
        return

    # Extract data
    scenarios = summary_df['scenario_name'].values
    data = summary_df[regime_cols].values

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create heatmap
    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(regime_cols)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_yticklabels(['Regime 0', 'Regime 1', 'Regime 2'])

    # Add values as text
    for i in range(len(regime_cols)):
        for j in range(len(scenarios)):
            val = data[j, i]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center", color="black", fontsize=10)

    ax.set_title('SHAP Correlation with True Importances by Regime',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Regime', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()
    save_path = figures_dir / 'regime_specific_correlations.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DGP Parameter Robustness Test for Adaptive WIN-SHAP'
    )
    parser.add_argument('--dataset', type=str, default='piecewise_ar3',
                       help='Base dataset name (default: piecewise_ar3)')
    parser.add_argument('--output-dir', type=str,
                       default='examples/results/dgp_robustness',
                       help='Output directory')
    parser.add_argument('--seeds', type=str, default='42,43,44',
                       help='Random seeds for 3 random scenarios (comma-separated)')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip data generation, use existing datasets')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip LPA detection, use existing windows')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip SHAP benchmark, use existing results')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only generate visualizations from existing results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')

    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(seeds) != 3:
        raise ValueError("Must provide exactly 3 seeds")

    # Setup paths
    dataset_name = args.dataset
    datasets_base = Path('examples/datasets/simulated')
    baseline_path = datasets_base / dataset_name
    robustness_data_dir = datasets_base / f'{dataset_name}_dgp_robustness'
    results_dir = Path(args.output_dir) / dataset_name

    print("="*80)
    print("DGP Parameter Robustness Test")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Data directory: {robustness_data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Random seeds: {seeds}")
    print("="*80)

    # Load baseline parameters
    baseline_regimes = [
        np.array([0.9, 0.01, 0.01]),
        np.array([0.01, 0.9, 0.01]),
        np.array([0.01, 0.01, 0.9])
    ]
    regime_lengths = (500, 500, 500)
    true_breakpoints = [500, 1000]

    # ========================================================================
    # STEP 1: GENERATE DGP SCENARIOS
    # ========================================================================

    if not args.skip_generation and not args.visualize_only:
        print("\n" + "="*80)
        print("STEP 1: Generating DGP Scenarios")
        print("="*80)

        robustness_data_dir.mkdir(parents=True, exist_ok=True)

        # Generate closer scenario
        print("\n1. Generating 'closer' scenario...")
        closer_regimes = generate_closer_scenario(baseline_regimes, target_reduction=0.5)
        X_closer, true_imp_closer = simulate_piecewise_ar3(
            closer_regimes, regime_lengths, noise_sigma=1.0, seed=100
        )
        save_scenario_data(
            'closer', X_closer, true_imp_closer, closer_regimes, regime_lengths,
            robustness_data_dir, 'l2_distance_reduction', baseline_regimes, seed=100
        )

        # Generate further scenario
        print("\n2. Generating 'further' scenario...")
        further_regimes = generate_further_scenario()
        X_further, true_imp_further = simulate_piecewise_ar3(
            further_regimes, regime_lengths, noise_sigma=1.0, seed=101
        )
        save_scenario_data(
            'further', X_further, true_imp_further, further_regimes, regime_lengths,
            robustness_data_dir, 'l2_distance_increase', baseline_regimes, seed=101
        )

        # Generate random scenarios
        for i, seed in enumerate(seeds, start=1):
            print(f"\n3.{i} Generating 'random_{i}' scenario (seed={seed})...")
            random_regimes = generate_random_scenario(seed)
            X_random, true_imp_random = simulate_piecewise_ar3(
                random_regimes, regime_lengths, noise_sigma=1.0, seed=seed
            )
            save_scenario_data(
                f'random_{i}', X_random, true_imp_random, random_regimes, regime_lengths,
                robustness_data_dir, 'random_sampling', baseline_regimes, seed=seed
            )

        # Create parameter summary table
        print("\n4. Creating parameter summary table...")
        summary_rows = []

        for scenario_name in ['closer', 'further', 'random_1', 'random_2', 'random_3']:
            config_path = robustness_data_dir / scenario_name / 'scenario_config.json'
            with open(config_path) as f:
                config = json.load(f)

            row = {
                'scenario_name': scenario_name,
                'scenario_type': config['scenario_type'],
                'max_pairwise_l2': config['max_pairwise_l2']
            }

            for regime in config['regimes']:
                rid = regime['regime_id']
                row[f'regime{rid}_phi1'] = regime['parameters']['phi_1']
                row[f'regime{rid}_phi2'] = regime['parameters']['phi_2']
                row[f'regime{rid}_phi3'] = regime['parameters']['phi_3']

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(robustness_data_dir / 'dgp_parameters.csv', index=False)
        print(f"  Saved: {robustness_data_dir / 'dgp_parameters.csv'}")

        print("\n" + "="*80)
        print("Data generation complete!")
        print("="*80)

    # ========================================================================
    # STEP 2: RUN LPA DETECTION
    # ========================================================================

    if not args.skip_detection and not args.visualize_only:
        print("\n" + "="*80)
        print("STEP 2: Running LPA Window Detection")
        print("="*80)

        lpa_config = {
            'N0': 75,
            'alpha': 0.95,
            'num_bootstrap': 50,
            'jump': 1,
            'growth': 'geometric'
        }

        print(f"LPA Configuration: {lpa_config}")

        scenarios = ['baseline', 'closer', 'further', 'random_1', 'random_2', 'random_3']

        for scenario_name in scenarios:
            print(f"\n  Processing {scenario_name}...")

            if scenario_name == 'baseline':
                data_path = baseline_path / 'data.csv'
            else:
                data_path = robustness_data_dir / scenario_name / 'data.csv'

            scenario_results_dir = results_dir / scenario_name
            scenario_results_dir.mkdir(parents=True, exist_ok=True)

            if not data_path.exists():
                print(f"    Warning: Data not found at {data_path}, skipping")
                continue

            try:
                windows_path = run_lpa_detection(
                    data_path, scenario_results_dir,
                    verbose=args.verbose, **lpa_config
                )
                print(f"    ✓ Detection complete")
            except Exception as e:
                print(f"    ✗ Detection failed: {e}")
                import traceback
                if args.verbose:
                    traceback.print_exc()

        print("\n" + "="*80)
        print("LPA detection complete!")
        print("="*80)

    # ========================================================================
    # STEP 3: RUN SHAP BENCHMARK
    # ========================================================================

    if not args.skip_benchmark and not args.visualize_only:
        print("\n" + "="*80)
        print("STEP 3: Running SHAP Benchmarks")
        print("="*80)

        scenarios = ['baseline', 'closer', 'further', 'random_1', 'random_2', 'random_3']

        for scenario_name in scenarios:
            print(f"\n  Processing {scenario_name}...")

            if scenario_name == 'baseline':
                data_path = baseline_path / 'data.csv'
            else:
                data_path = robustness_data_dir / scenario_name / 'data.csv'

            scenario_results_dir = results_dir / scenario_name
            windows_path = scenario_results_dir / 'windows.csv'

            if not windows_path.exists():
                print(f"    Warning: Windows not found at {windows_path}, skipping")
                continue

            try:
                benchmark_dir = run_shap_benchmark(
                    data_path, windows_path, scenario_results_dir,
                    verbose=args.verbose
                )
                print(f"    ✓ Benchmark complete")
            except Exception as e:
                print(f"    ✗ Benchmark failed: {e}")
                import traceback
                if args.verbose:
                    traceback.print_exc()

        print("\n" + "="*80)
        print("SHAP benchmarking complete!")
        print("="*80)

    # ========================================================================
    # STEP 4: COMPUTE METRICS & AGGREGATE RESULTS
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 4: Computing Metrics and Aggregating Results")
    print("="*80)

    scenarios = ['baseline', 'closer', 'further', 'random_1', 'random_2', 'random_3']
    summary_rows = []

    for scenario_name in scenarios:
        print(f"\n  Processing {scenario_name}...")

        scenario_results_dir = results_dir / scenario_name

        # Load scenario config for L2 distance
        if scenario_name == 'baseline':
            l2_from_baseline = 0.0
            max_pairwise_l2 = 1.26  # Known baseline value
            scenario_type = 'original'
        else:
            config_path = robustness_data_dir / scenario_name / 'scenario_config.json'
            with open(config_path) as f:
                config = json.load(f)
            l2_from_baseline = np.mean([r.get('l2_distance_from_baseline', 0)
                                       for r in config['regimes']])
            max_pairwise_l2 = config['max_pairwise_l2']
            scenario_type = config['scenario_type']

        row = {
            'scenario_name': scenario_name,
            'scenario_type': scenario_type,
            'l2_distance_from_baseline': l2_from_baseline,
            'max_pairwise_l2': max_pairwise_l2
        }

        # Window detection metrics
        windows_path = scenario_results_dir / 'windows.csv'
        if windows_path.exists():
            windows_df = pd.read_csv(windows_path)
            detection_metrics = compute_detection_metrics(
                windows_df, true_breakpoints, tolerance=50
            )
            row.update(detection_metrics)

            # Window statistics
            window_stats = compute_window_statistics(windows_df)
            row.update(window_stats)
        else:
            print(f"    Warning: windows.csv not found")

        # SHAP consistency metrics
        if scenario_name == 'baseline':
            true_imp_path = baseline_path / 'true_importances.csv'
        else:
            true_imp_path = robustness_data_dir / scenario_name / 'true_importances.csv'

        shap_results_path = scenario_results_dir / 'benchmark' / 'adaptive_shap_results.csv'

        if shap_results_path.exists() and true_imp_path.exists():
            shap_metrics = compute_shap_consistency_metrics(
                shap_results_path, true_imp_path, regime_lengths
            )
            row.update(shap_metrics)
        else:
            print(f"    Warning: SHAP or true importance files not found")

        # Benchmark metrics
        benchmark_summary_path = scenario_results_dir / 'benchmark' / 'benchmark_summary.csv'
        if benchmark_summary_path.exists():
            benchmark_metrics = extract_benchmark_metrics(benchmark_summary_path)
            row.update(benchmark_metrics)
        else:
            print(f"    Warning: benchmark_summary.csv not found")

        summary_rows.append(row)
        print(f"    ✓ Metrics computed")

    # Save aggregated results
    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / 'summary_all_scenarios.csv'
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "="*80)
    print("Results Summary:")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")
    print("="*80)

    # ========================================================================
    # STEP 5: GENERATE VISUALIZATIONS
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 5: Generating Visualizations")
    print("="*80)

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("\n1. Window evolution comparison...")
        plot_window_evolution_comparison(
            results_dir, scenarios, figures_dir, true_breakpoints
        )
    except Exception as e:
        print(f"  Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    try:
        print("\n2. Metrics vs L2 distance...")
        plot_metrics_vs_l2_distance(summary_df, figures_dir)
    except Exception as e:
        print(f"  Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    try:
        print("\n3. Comparative window overlay...")
        plot_comparative_window_overlay(
            results_dir, scenarios, summary_df, figures_dir, true_breakpoints
        )
    except Exception as e:
        print(f"  Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    try:
        print("\n4. Regime-specific correlations...")
        plot_regime_specific_correlations(summary_df, figures_dir)
    except Exception as e:
        print(f"  Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print(f"Results saved to: {results_dir}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {figures_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
