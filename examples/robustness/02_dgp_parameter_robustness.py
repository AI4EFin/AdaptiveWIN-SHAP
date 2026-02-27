"""
DGP Parameter Robustness Test for Adaptive WIN-SHAP

Tests how the method degrades as regime contrast decreases. Uses centroid
interpolation to reduce pairwise L2 distance between AR(3) coefficient
vectors by 0%, 50%, 75%, and 90%.

Usage:
    python examples/robustness/02_dgp_parameter_robustness.py [options]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory and project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptivewinshap import AdaptiveLSTM, ChangeDetector
from benchmark import run_benchmark

np.random.seed(42)


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

BASELINE_REGIMES = [
    np.array([0.9, 0.01, 0.01]),   # lag-1 dominant
    np.array([0.01, 0.9, 0.01]),   # lag-2 dominant
    np.array([0.01, 0.01, 0.9]),   # lag-3 dominant
]

CENTROID = np.mean(BASELINE_REGIMES, axis=0)  # ~(0.3067, 0.3067, 0.3067)

REGIME_LENGTHS = (500, 500, 500)
TRUE_BREAKPOINTS = [500, 1000]

# L2 reduction factors: 0% (baseline), 50%, 75%, 90%
REDUCTION_FACTORS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 0.90]

# LPA config matching lstm_simulation.py defaults
LPA_CONFIG = {
    'N0': 100,
    'alpha': 0.95,
    'mc_reps': 100,
    'penalty_factor': 0.0,
    'growth': 'geometric',
    'growth_base': 1.414,
    'search_step': 1,
    'min_seg': 4,
}


# ============================================================================
# PARAMETER GENERATION
# ============================================================================

def is_stationary(phi: np.ndarray) -> bool:
    """Check if AR(3) parameters are stationary (all characteristic roots outside unit circle)."""
    if np.allclose(phi, 0, atol=1e-8):
        return False
    try:
        coefficients = [-phi[2], -phi[1], -phi[0], 1]
        roots = np.roots(coefficients)
        return bool(np.all(np.abs(roots) > 1.0))
    except Exception:
        return False


def interpolate_regimes(t: float) -> List[np.ndarray]:
    """
    Generate regime parameters by interpolating toward centroid.

    Parameters
    ----------
    t : float
        Interpolation factor (0 = baseline, 1 = all regimes equal centroid).
        Reduces pairwise L2 distance by fraction t.

    Returns
    -------
    List[np.ndarray]
        Three regime parameter vectors.
    """
    regimes = []
    for phi in BASELINE_REGIMES:
        new_phi = (1 - t) * phi + t * CENTROID
        regimes.append(new_phi)
    return regimes


def compute_max_pairwise_l2(regimes: List[np.ndarray]) -> float:
    """Compute the maximum pairwise L2 distance between regime parameters."""
    max_dist = 0.0
    for i in range(len(regimes)):
        for j in range(i + 1, len(regimes)):
            dist = np.linalg.norm(regimes[i] - regimes[j])
            max_dist = max(max_dist, dist)
    return max_dist


def get_scenario_name(t: float, regimes: List[np.ndarray]) -> str:
    """Generate scenario directory name from reduction factor."""
    if t == 0.0:
        return "baseline"
    return f"l2_{int(t * 100)}"


def get_scenario_label(t: float, regimes: List[np.ndarray]) -> str:
    """Generate a label with the actual L2 distance for plot legends."""
    l2 = compute_max_pairwise_l2(regimes)
    if t == 0.0:
        return f"Baseline (L2={l2:.2f})"
    return f"{int(t*100)}% reduction (L2={l2:.2f})"


# ============================================================================
# DATA GENERATION
# ============================================================================

def simulate_piecewise_ar3(regimes: List[np.ndarray],
                           noise_sigma: float = 1.0,
                           seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate piecewise AR(3) time series with regime-specific parameters.

    Returns
    -------
    X : np.ndarray
        Simulated time series (T,)
    true_imp : np.ndarray
        True feature importances (T, 3)
    """
    rng = np.random.default_rng(seed)
    T = sum(REGIME_LENGTHS)

    # Build regime index
    reg_idx = np.zeros(T, dtype=int)
    start = 0
    for k, length in enumerate(REGIME_LENGTHS):
        reg_idx[start:start + length] = k
        start += length

    # Simulate
    X = np.zeros(T)
    eps = rng.normal(0, noise_sigma, size=T)

    for t in range(T):
        phi = regimes[reg_idx[t]]
        ar_part = 0.0
        for j in range(1, 4):
            if t - j >= 0:
                ar_part += phi[j - 1] * X[t - j]
        X[t] = ar_part + eps[t]

    # True importances: normalized absolute coefficients
    true_imp = np.zeros((T, 3))
    for t in range(T):
        phi = regimes[reg_idx[t]]
        abs_phi = np.abs(phi)
        total = abs_phi.sum()
        true_imp[t] = abs_phi / total if total > 0 else abs_phi

    return X, true_imp


def save_scenario_data(scenario_name: str,
                       X: np.ndarray,
                       true_imp: np.ndarray,
                       regimes: List[np.ndarray],
                       t: float,
                       output_dir: Path):
    """Save scenario data and slim metadata to disk."""
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Save time series
    pd.DataFrame({'N': X}).to_csv(scenario_dir / 'data.csv', index=False)

    # Save true importances
    imp_df = pd.DataFrame({
        f'true_imp_{i}': true_imp[:, i] for i in range(true_imp.shape[1])
    })
    imp_df.to_csv(scenario_dir / 'true_importances.csv', index=False)

    # Slim metadata
    max_l2 = compute_max_pairwise_l2(regimes)
    metadata = {
        "scenario_name": scenario_name,
        "l2_reduction_factor": t,
        "max_pairwise_l2": float(max_l2),
        "regime_lengths": list(REGIME_LENGTHS),
        "regimes": [
            {
                "regime_id": k,
                "parameters": {
                    "phi_1": float(regimes[k][0]),
                    "phi_2": float(regimes[k][1]),
                    "phi_3": float(regimes[k][2]),
                }
            }
            for k in range(len(regimes))
        ]
    }

    with open(scenario_dir / 'scenario_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {scenario_name}: L2_max={max_l2:.3f}")


# ============================================================================
# LPA DETECTION
# ============================================================================

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_lpa_detection(data_path: Path,
                      output_dir: Path,
                      verbose: bool = True) -> Path:
    """Run LPA window detection on a dataset."""
    if verbose:
        print(f"    Running LPA detection (penalty={LPA_CONFIG['penalty_factor']}, "
              f"step={LPA_CONFIG['search_step']})")

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

    device = _get_device()

    model = AdaptiveLSTM(
        device, seq_length=3, input_size=input_size,
        hidden=16, layers=1, dropout=0.0,
        batch_size=64, lr=1e-2, epochs=15,
        type_precision=np.float64
    )

    cd = ChangeDetector(model, data, debug=False, force_cpu=True)

    # Compute critical values
    cv_path = output_dir / 'critical_values.csv'
    if cv_path.exists():
        if verbose:
            print(f"    Loading critical values from: {cv_path}")
        cd.load_critical_values(str(cv_path))
    else:
        if verbose:
            print(f"    Computing Monte Carlo critical values (mc_reps={LPA_CONFIG['mc_reps']})...")
        cd.precompute_critical_values(
            data=data,
            n_0=LPA_CONFIG['N0'],
            mc_reps=LPA_CONFIG['mc_reps'],
            alpha=LPA_CONFIG['alpha'],
            search_step=LPA_CONFIG['search_step'],
            min_seg=LPA_CONFIG['min_seg'],
            penalty_factor=LPA_CONFIG['penalty_factor'],
            growth_base=LPA_CONFIG['growth_base'],
            verbose=verbose
        )
        cd.save_critical_values(str(cv_path))

    # Run detection
    start_time = time.time()
    results = cd.detect(
        min_window=LPA_CONFIG['min_seg'],
        n_0=LPA_CONFIG['N0'],
        jump=1,
        search_step=LPA_CONFIG['search_step'],
        alpha=LPA_CONFIG['alpha'],
        t_workers=10,
        debug_anim=False,
        save_path=None,
        growth=LPA_CONFIG['growth'],
        growth_base=LPA_CONFIG['growth_base']
    )
    detection_time = time.time() - start_time

    results.to_csv(output_dir / 'run_0.csv', index=False)

    # Extract and save windows
    windows_df = results[['windows']].copy()
    windows_df = windows_df.rename(columns={'windows': 'window_mean'})

    # Pad to match original data length
    expected_length = len(target)
    if len(windows_df) < expected_length:
        n_pad = expected_length - len(windows_df)
        first_window = windows_df.iloc[0]
        pad_df = pd.DataFrame([first_window] * n_pad, columns=windows_df.columns)
        windows_df = pd.concat([pad_df, windows_df], ignore_index=True)

    windows_path = output_dir / 'windows.csv'
    windows_df.to_csv(windows_path, index=False)

    if verbose:
        valid = windows_df['window_mean'].dropna()
        print(f"    Detection time: {detection_time:.1f}s")
        print(f"    Windows: mean={valid.mean():.1f}, std={valid.std():.1f}")

    return windows_path


# ============================================================================
# SHAP BENCHMARKING
# ============================================================================

def run_shap_benchmark(data_path: Path,
                       windows_path: Path,
                       output_dir: Path,
                       verbose: bool = True) -> Path:
    """Run SHAP benchmark on a dataset with precomputed windows."""
    benchmark_dir = output_dir / 'benchmark'
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"    Running SHAP benchmark")

    device = _get_device()

    run_benchmark(
        dataset_path=str(data_path),
        output_dir=str(benchmark_dir),
        device=device,
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
                              tolerance: int = 50) -> Dict:
    """Compute window detection accuracy metrics at true breakpoints."""
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
                'detection_lag_mean': np.nan,
                'detection_lag_std': np.nan,
                'detection_success_rate': np.nan,
            }

    # Detect changepoints via window size jumps
    window_diff = np.abs(np.diff(windows))
    threshold = np.percentile(window_diff, 90)
    detected_breakpoints = np.where(window_diff > threshold)[0] + 1

    lags = []
    detected_flags = []

    for true_bp in TRUE_BREAKPOINTS:
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

    return {
        'detection_lag_mean': float(np.mean(lags)) if lags else np.nan,
        'detection_lag_std': float(np.std(lags)) if lags else np.nan,
        'detection_success_rate': float(np.mean(detected_flags)) if detected_flags else 0.0,
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
            return {'window_mean': np.nan, 'window_std': np.nan,
                    'window_min': np.nan, 'window_max': np.nan}

    return {
        'window_mean': float(np.mean(windows)),
        'window_std': float(np.std(windows)),
        'window_min': float(np.min(windows)),
        'window_max': float(np.max(windows)),
    }


def compute_shap_correlation(shap_results_path: Path,
                              true_importances_path: Path) -> Dict:
    """Compute Pearson correlation between SHAP values and true importances, overall and per regime."""
    nan_result = {
        'shap_corr_regime0': np.nan,
        'shap_corr_regime1': np.nan,
        'shap_corr_regime2': np.nan,
        'shap_corr_overall': np.nan,
    }

    if not shap_results_path.exists() or not true_importances_path.exists():
        return nan_result

    shap_df = pd.read_csv(shap_results_path)
    true_df = pd.read_csv(true_importances_path)

    shap_cols = [c for c in shap_df.columns if c.startswith('feat_') or c.startswith('shap_')]
    true_cols = [c for c in true_df.columns if c.startswith('true_imp_')]

    if not shap_cols or not true_cols:
        return nan_result

    n = min(len(shap_df), len(true_df))
    shap_values = shap_df[shap_cols].iloc[:n].values
    true_values = true_df[true_cols].iloc[:n].values

    # Overall correlation
    shap_flat = shap_values.flatten()
    true_flat = true_values.flatten()
    mask = ~(np.isnan(shap_flat) | np.isnan(true_flat))
    corr_overall = np.corrcoef(shap_flat[mask], true_flat[mask])[0, 1] if mask.sum() > 0 else np.nan

    # Per-regime correlations
    breakpoints = [0, REGIME_LENGTHS[0], REGIME_LENGTHS[0] + REGIME_LENGTHS[1], sum(REGIME_LENGTHS)]
    regime_corrs = []

    for i in range(3):
        start, end = breakpoints[i], breakpoints[i + 1]
        if end > n:
            regime_corrs.append(np.nan)
            continue

        s = shap_values[start:end].flatten()
        t = true_values[start:end].flatten()
        mask = ~(np.isnan(s) | np.isnan(t))
        if mask.sum() > 10:
            regime_corrs.append(float(np.corrcoef(s[mask], t[mask])[0, 1]))
        else:
            regime_corrs.append(np.nan)

    return {
        'shap_corr_regime0': regime_corrs[0],
        'shap_corr_regime1': regime_corrs[1],
        'shap_corr_regime2': regime_corrs[2],
        'shap_corr_overall': float(corr_overall),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_window_evolution_comparison(results_dir: Path,
                                     scenario_info: List[dict],
                                     figures_dir: Path):
    """Plot window evolution for all scenarios in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    fig.suptitle('Window Size Evolution Across DGP Scenarios',
                 fontsize=14, fontweight='bold')

    for idx, info in enumerate(scenario_info):
        ax = axes[idx]
        windows_path = results_dir / info['name'] / 'windows.csv'

        if not windows_path.exists():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(info['label'])
            continue

        windows_df = pd.read_csv(windows_path)
        windows = windows_df['window_mean'].values if 'window_mean' in windows_df.columns else windows_df.iloc[:, 0].values
        time_index = np.arange(len(windows))

        ax.plot(time_index, windows, linewidth=1.5, alpha=0.8, color='#2ca02c')

        for bp in TRUE_BREAKPOINTS:
            ax.axvline(x=bp, color='red', linestyle='--', linewidth=1.5, alpha=0.6)

        mean_w = windows.mean()
        ax.axhline(y=mean_w, color='blue', linestyle=':', linewidth=1,
                    alpha=0.5, label=f'Mean: {mean_w:.1f}')

        ax.set_xlabel('Time Index')
        ax.set_ylabel('Window Size')
        ax.set_title(info['label'], fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    save_path = figures_dir / 'window_evolution_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metrics_vs_l2_distance(summary_df: pd.DataFrame, figures_dir: Path):
    """Plot key metrics vs max pairwise L2 distance."""
    metrics = [
        ('detection_lag_mean', 'Detection Lag (steps)', 'lower'),
        ('shap_corr_overall', 'SHAP Correlation', 'higher'),
        ('detection_success_rate', 'Detection Success Rate', 'higher'),
        ('window_std', 'Window Size Std Dev', 'lower'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Performance Metrics vs Max Pairwise L2 Distance',
                 fontsize=14, fontweight='bold')

    for idx, (metric, label, better) in enumerate(metrics):
        ax = axes[idx]

        if metric not in summary_df.columns:
            ax.text(0.5, 0.5, f'{metric} not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        x = summary_df['max_pairwise_l2'].values
        y = summary_df[metric].values

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) == 0:
            continue

        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        ax.scatter(x, y, s=120, alpha=0.8, c=colors, edgecolors='black', linewidths=1.5)

        for i, (xi, yi) in enumerate(zip(x, y)):
            scenario = summary_df[mask].iloc[i]['scenario_name']
            ax.annotate(scenario, (xi, yi), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')

        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, label='Trend')
            ax.legend()

        ax.set_xlabel('Max Pairwise L2 Distance', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} ({better} is better)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = figures_dir / 'metrics_vs_l2_distance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparative_window_overlay(results_dir: Path,
                                     scenario_info: List[dict],
                                     figures_dir: Path):
    """Overlay all scenario windows on a single plot."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_info)))

    for idx, info in enumerate(scenario_info):
        windows_path = results_dir / info['name'] / 'windows.csv'
        if not windows_path.exists():
            continue

        windows_df = pd.read_csv(windows_path)
        windows = windows_df['window_mean'].values if 'window_mean' in windows_df.columns else windows_df.iloc[:, 0].values
        time_index = np.arange(len(windows))

        ax.plot(time_index, windows, linewidth=2, alpha=0.7,
                color=colors[idx], label=info['label'])

    for i, bp in enumerate(TRUE_BREAKPOINTS):
        ax.axvline(x=bp, color='red', linestyle='--', linewidth=2,
                   alpha=0.6, label='Breakpoint' if i == 0 else '')

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


def plot_regime_specific_correlations(summary_df: pd.DataFrame, figures_dir: Path):
    """Plot SHAP correlations for each regime as heatmap."""
    regime_cols = ['shap_corr_regime0', 'shap_corr_regime1', 'shap_corr_regime2']

    if not all(col in summary_df.columns for col in regime_cols):
        print("  Warning: Regime-specific correlation columns not found, skipping")
        return

    scenarios = summary_df['scenario_name'].values
    data = summary_df[regime_cols].values

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(regime_cols)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_yticklabels(['Regime 0', 'Regime 1', 'Regime 2'])

    for i in range(len(regime_cols)):
        for j in range(len(scenarios)):
            val = data[j, i]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha="center", va="center",
                        color="black", fontsize=10)

    ax.set_title('SHAP Correlation with True Importances by Regime',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Regime', fontsize=12)

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
    parser.add_argument('--output-dir', type=str,
                        default='examples/results/robustness/dgp_robustness',
                        help='Output directory')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Only generate visualizations from existing results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')

    args = parser.parse_args()

    # Setup paths
    datasets_base = Path('examples/datasets/simulated')
    baseline_data_dir = datasets_base / 'piecewise_ar3'
    robustness_data_dir = datasets_base / 'piecewise_ar3_dgp_robustness'
    results_dir = Path(args.output_dir) / 'piecewise_ar3'

    # Build scenario definitions
    scenarios = []
    for t in REDUCTION_FACTORS:
        regimes = interpolate_regimes(t)
        name = get_scenario_name(t, regimes)
        label = get_scenario_label(t, regimes)
        max_l2 = compute_max_pairwise_l2(regimes)
        scenarios.append({
            'name': name,
            'label': label,
            't': t,
            'regimes': regimes,
            'max_l2': max_l2,
        })

    print("=" * 80)
    print("DGP Parameter Robustness Test")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Scenarios: {len(scenarios)}")
    for s in scenarios:
        r0 = s['regimes'][0]
        print(f"  {s['name']:12s}  t={s['t']:.2f}  L2={s['max_l2']:.3f}  "
              f"R0=({r0[0]:.3f}, {r0[1]:.3f}, {r0[2]:.3f})")
    print("=" * 80)

    # Verify stationarity of all scenarios
    for s in scenarios:
        for k, phi in enumerate(s['regimes']):
            assert is_stationary(phi), (
                f"Regime {k} of scenario {s['name']} is NOT stationary: {phi}"
            )
    print("All scenarios verified stationary.\n")

    # ==================================================================
    # STEP 1: GENERATE DATA
    # ==================================================================

    if not args.visualize_only:
        print("=" * 80)
        print("STEP 1: Generating DGP Scenarios")
        print("=" * 80)

        robustness_data_dir.mkdir(parents=True, exist_ok=True)

        for s in scenarios:
            print(f"\n  Generating '{s['name']}'...")

            if s['t'] == 0.0:
                # Baseline: use existing dataset or generate it
                data_path = baseline_data_dir / 'data.csv'
                if data_path.exists():
                    print(f"  Baseline data exists at {data_path}")
                else:
                    print(f"  Generating baseline data...")
                    X, true_imp = simulate_piecewise_ar3(s['regimes'], seed=123)
                    save_scenario_data(
                        s['name'], X, true_imp, s['regimes'], s['t'],
                        robustness_data_dir
                    )
                continue
            else:
                # Use same seed offset per scenario for reproducibility
                seed = 100 + int(s['t'] * 100)
                X, true_imp = simulate_piecewise_ar3(s['regimes'], seed=seed)
                save_scenario_data(
                    s['name'], X, true_imp, s['regimes'], s['t'],
                    robustness_data_dir
                )

        print("\nData generation complete!")

    # ==================================================================
    # STEP 2: RUN LPA DETECTION
    # ==================================================================

    if not args.visualize_only:
        print("\n" + "=" * 80)
        print("STEP 2: Running LPA Window Detection")
        print("=" * 80)
        print(f"LPA Config: {LPA_CONFIG}\n")

        for s in scenarios:
            print(f"  Processing {s['name']}...")

            if s['t'] == 0.0:
                data_path = baseline_data_dir / 'data.csv'
            else:
                data_path = robustness_data_dir / s['name'] / 'data.csv'

            scenario_results_dir = results_dir / s['name']
            scenario_results_dir.mkdir(parents=True, exist_ok=True)

            if not data_path.exists():
                print(f"    Warning: Data not found at {data_path}, skipping")
                continue

            try:
                run_lpa_detection(data_path, scenario_results_dir, verbose=args.verbose)
                print(f"    Done")
            except Exception as e:
                print(f"    FAILED: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        print("\nLPA detection complete!")

    # ==================================================================
    # STEP 3: RUN SHAP BENCHMARK
    # ==================================================================

    if not args.visualize_only:
        print("\n" + "=" * 80)
        print("STEP 3: Running SHAP Benchmarks")
        print("=" * 80)

        for s in scenarios:
            print(f"\n  Processing {s['name']}...")

            if s['t'] == 0.0:
                data_path = baseline_data_dir / 'data.csv'
            else:
                data_path = robustness_data_dir / s['name'] / 'data.csv'

            scenario_results_dir = results_dir / s['name']
            windows_path = scenario_results_dir / 'windows.csv'

            if not windows_path.exists():
                print(f"    Warning: Windows not found at {windows_path}, skipping")
                continue

            try:
                run_shap_benchmark(data_path, windows_path, scenario_results_dir,
                                   verbose=args.verbose)
                print(f"    Done")
            except Exception as e:
                print(f"    FAILED: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        print("\nSHAP benchmarking complete!")

    # ==================================================================
    # STEP 4: COMPUTE METRICS & AGGREGATE
    # ==================================================================

    print("\n" + "=" * 80)
    print("STEP 4: Computing Metrics and Aggregating Results")
    print("=" * 80)

    summary_rows = []

    for s in scenarios:
        print(f"\n  Processing {s['name']}...")

        scenario_results_dir = results_dir / s['name']

        row = {
            'scenario_name': s['name'],
            'l2_reduction_factor': s['t'],
            'max_pairwise_l2': s['max_l2'],
        }

        # Window detection metrics
        windows_path = scenario_results_dir / 'windows.csv'
        if windows_path.exists():
            windows_df = pd.read_csv(windows_path)
            row.update(compute_detection_metrics(windows_df))
            row.update(compute_window_statistics(windows_df))
        else:
            print(f"    Warning: windows.csv not found")

        # SHAP correlation
        if s['t'] == 0.0:
            true_imp_path = baseline_data_dir / 'true_importances.csv'
        else:
            true_imp_path = robustness_data_dir / s['name'] / 'true_importances.csv'

        shap_results_path = scenario_results_dir / 'benchmark' / 'adaptive_shap_results.csv'
        row.update(compute_shap_correlation(shap_results_path, true_imp_path))

        summary_rows.append(row)
        print(f"    Done")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / 'summary_all_scenarios.csv'
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")

    # ==================================================================
    # STEP 5: VISUALIZATIONS
    # ==================================================================

    print("\n" + "=" * 80)
    print("STEP 5: Generating Visualizations")
    print("=" * 80)

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenario_info = [{'name': s['name'], 'label': s['label']} for s in scenarios]

    for plot_name, plot_fn, plot_args in [
        ("Window evolution comparison",
         plot_window_evolution_comparison,
         (results_dir, scenario_info, figures_dir)),
        ("Metrics vs L2 distance",
         plot_metrics_vs_l2_distance,
         (summary_df, figures_dir)),
        ("Comparative window overlay",
         plot_comparative_window_overlay,
         (results_dir, scenario_info, figures_dir)),
        ("Regime-specific correlations",
         plot_regime_specific_correlations,
         (summary_df, figures_dir)),
    ]:
        try:
            print(f"\n  {plot_name}...")
            plot_fn(*plot_args)
        except Exception as e:
            print(f"  Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"Results: {results_dir}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {figures_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()