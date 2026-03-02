"""
Visualize LPA Sensitivity Analysis Results

Scans config directories under the results folder to reconstruct metrics
from the actual runs, then creates parameter sensitivity plots for:
  1. Correlation with true feature importances vs N0, B (mc_reps), penalty_factor
  2. Window mean vs N0, B (mc_reps), penalty_factor

Usage:
    python examples/robustness/visualize_lpa_sensitivity.py --dataset piecewise_ar3
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

DATASET_BREAKPOINTS = {
    'piecewise_ar3': [500, 1000],
    'arx_rotating': [500, 1000],
    'piecewise_ar3_long': [500, 900, 1200, 1800, 2600, 3600],
    'arx_rotating_long': [500, 900, 1200, 1800, 2600, 3600],
}


# ---------------------------------------------------------------------------
# Data loading: scan config directories as the single source of truth
# ---------------------------------------------------------------------------

def _parse_config_dir(dirname: str) -> dict | None:
    """Extract parameters from a temp_* directory name."""
    m = re.match(
        r'temp_N0(\d+)_alpha([\d.]+)_mc_reps(\d+)_penalty_factor([\d.]+)_growth_base([\d.]+)',
        dirname,
    )
    if not m:
        return None
    return {
        'N0': int(m.group(1)),
        'alpha': float(m.group(2)),
        'mc_reps': int(m.group(3)),
        'penalty_factor': float(m.group(4)),
    }


def _compute_correlation(config_dir: Path, true_imp_df: pd.DataFrame) -> float | None:
    """Compute mean per-feature correlation between |SHAP| and true importances."""
    shap_path = config_dir / 'benchmark' / 'adaptive_shap_results.csv'
    if not shap_path.exists():
        return None

    shap_df = pd.read_csv(shap_path)
    shap_cols = [c for c in shap_df.columns if c.startswith('shap_')]
    true_cols = list(true_imp_df.columns)

    if len(shap_cols) != len(true_cols):
        return None

    correlations = []
    for shap_col, true_col in zip(shap_cols, true_cols):
        shap_vals = np.abs(shap_df[shap_col].values)
        end_indices = shap_df['end_index'].astype(int).values
        end_indices = np.clip(end_indices, 0, len(true_imp_df) - 1)
        true_vals = true_imp_df[true_col].iloc[end_indices].values

        mask = ~(np.isnan(shap_vals) | np.isnan(true_vals))
        if mask.sum() > 10:
            corr = np.corrcoef(shap_vals[mask], true_vals[mask])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return float(np.mean(correlations)) if correlations else None


def _extract_benchmark_metrics(config_dir: Path, method: str = 'adaptive_shap') -> dict:
    """Extract metrics for a given method from benchmark_summary.csv."""
    summary_path = config_dir / 'benchmark' / 'benchmark_summary.csv'
    if not summary_path.exists():
        return {}

    df = pd.read_csv(summary_path)
    df = df[df['method'] == method]

    metrics = {}
    for _, row in df.iterrows():
        key = f"{row['metric_type']}_{row['evaluation']}"
        metrics[key] = row['score']
    return metrics


def _compute_oracle_window(n_timepoints: int, breakpoints: list[int]) -> np.ndarray:
    """
    Compute the oracle window at each timepoint.

    The oracle window at time t is the number of past observations belonging
    to the same regime, i.e. t - last_breakpoint (or t+1 for the first regime).
    """
    oracle = np.zeros(n_timepoints)
    # Sorted breakpoints + boundaries
    boundaries = [0] + sorted(breakpoints) + [n_timepoints]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        for t in range(start, end):
            oracle[t] = t - start + 1
    return oracle


def _compute_window_metrics(
    config_dir: Path, breakpoints: list[int]
) -> dict:
    """Compute window_mean and mean absolute difference to oracle window."""
    windows_path = config_dir / 'windows.csv'
    if not windows_path.exists():
        return {}

    wdf = pd.read_csv(windows_path)
    col = 'window_mean' if 'window_mean' in wdf.columns else wdf.columns[0]
    lpa_windows = wdf[col].values

    # Drop leading NaNs (burn-in period where LPA hasn't started)
    valid_mask = ~np.isnan(lpa_windows)
    if valid_mask.sum() == 0:
        return {}

    metrics = {'window_mean': float(np.nanmean(lpa_windows))}

    # Oracle window for the full series length
    n = len(lpa_windows)
    oracle = _compute_oracle_window(n, breakpoints)

    # Mean absolute difference (only where LPA has values)
    diff = np.abs(lpa_windows[valid_mask] - oracle[valid_mask])
    metrics['oracle_mae'] = float(np.mean(diff))

    return metrics


def load_results_from_configs(
    results_dir: Path,
    dataset_name: str,
    true_imp_path: Path | None = None,
    breakpoints: list[int] | None = None,
) -> pd.DataFrame:
    """
    Scan all temp_* config directories and build a results DataFrame.

    Parameters
    ----------
    results_dir : Path
        Top-level results directory (contains dataset subdirectories).
    dataset_name : str
        Name of the dataset subdirectory to scan.
    true_imp_path : Path or None
        Path to true_importances.csv for computing correlation.
    breakpoints : list of int or None
        Known breakpoint positions for oracle window computation.

    Returns
    -------
    pd.DataFrame with one row per config, columns for parameters + metrics.
    """
    dataset_dir = results_dir / dataset_name
    if breakpoints is None:
        breakpoints = DATASET_BREAKPOINTS.get(dataset_name, [])

    true_imp_df = None
    if true_imp_path and true_imp_path.exists():
        true_imp_df = pd.read_csv(true_imp_path)

    rows = []
    for config_dir in sorted(dataset_dir.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith('temp_'):
            continue

        params = _parse_config_dir(config_dir.name)
        if params is None:
            continue

        # Window mean + oracle MAE
        win_metrics = _compute_window_metrics(config_dir, breakpoints)
        params.update(win_metrics)

        # Benchmark metrics for adaptive_shap
        bench = _extract_benchmark_metrics(config_dir, method='adaptive_shap')
        params.update(bench)

        # Correlation with true importances
        if true_imp_df is not None:
            corr = _compute_correlation(config_dir, true_imp_df)
            if corr is not None:
                params['correlation_true_imp_mean'] = corr

        rows.append(params)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Parameters to plot (skip alpha since only one value was tested)
PARAM_COLS = ['N0', 'mc_reps', 'penalty_factor']
PARAM_LABELS = {'N0': r'$I_0$', 'mc_reps': r'$B$', 'penalty_factor': r'$\lambda$'}


def _setup_clean_ax(ax):
    """Apply clean/transparent aesthetic to an axis."""
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(width=0.8)


def plot_param_sensitivity(
    results_df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    ylabel: str = '',
    title: str = '',
    color: str = '#4A74AA',
):
    """
    Create a 1x3 figure showing metric vs N0, B, penalty_factor.

    Each subplot groups by one parameter and averages over the others,
    showing mean +/- standard error.
    """
    available = [p for p in PARAM_COLS if p in results_df.columns]
    if not available or metric_col not in results_df.columns:
        print(f"  Skipping {metric_col}: missing columns")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.patch.set_alpha(0)
    if n == 1:
        axes = [axes]

    for ax, param in zip(axes, available):
        _setup_clean_ax(ax)

        grouped = results_df.groupby(param)[metric_col].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

        ax.errorbar(
            grouped.index, grouped['mean'],
            yerr=1.96 * grouped['se'],
            marker='o', capsize=4, capthick=1.5,
            linewidth=1.8, markersize=6,
            color=color, ecolor=color, alpha=0.9,
        )

        ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=13)
        if ax is axes[0]:
            ax.set_ylabel(ylabel, fontsize=13)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_dataset(
    dataset_name: str,
    results_dir: Path,
    output_dir: Path,
    datasets_root: Path,
):
    """Create all plots for one dataset."""
    print(f"\n{'='*70}")
    print(f"  LPA Sensitivity Visualization: {dataset_name}")
    print(f"{'='*70}\n")

    # Locate true importances
    true_imp_path = datasets_root / dataset_name / 'true_importances.csv'
    if not true_imp_path.exists():
        print(f"  Warning: true_importances.csv not found at {true_imp_path}")
        true_imp_path = None

    # Load results from config directories
    print("  Loading results from config directories...")
    results_df = load_results_from_configs(results_dir, dataset_name, true_imp_path)

    if results_df.empty:
        print("  No results found. Exiting.")
        return

    print(f"  Loaded {len(results_df)} configurations")
    print(f"  Parameters: N0={sorted(results_df['N0'].unique())}, "
          f"mc_reps={sorted(results_df['mc_reps'].unique())}, "
          f"penalty_factor={sorted(results_df['penalty_factor'].unique())}")
    print(f"  Metrics: {[c for c in results_df.columns if c not in PARAM_COLS + ['alpha']]}")

    out = output_dir / dataset_name

    # 1. Correlation with true importances vs parameters
    if 'correlation_true_imp_mean' in results_df.columns:
        print("\n  1. Correlation with true importances vs parameters")
        plot_param_sensitivity(
            results_df, 'correlation_true_imp_mean',
            output_path=out / 'correlation_vs_params.png',
            ylabel='Correlation with true importances',
            title=f'{dataset_name}: Correlation vs LPA parameters',
            color='#4A74AA',
        )
    else:
        print("\n  1. Skipping correlation plot (no true importances data)")

    # 2. Window mean vs parameters
    if 'window_mean' in results_df.columns:
        print("\n  2. Window mean vs parameters")
        plot_param_sensitivity(
            results_df, 'window_mean',
            output_path=out / 'window_mean_vs_params.png',
            ylabel='Mean window size',
            title=f'{dataset_name}: Window mean vs LPA parameters',
            color='#DB3549',
        )
    else:
        print("\n  2. Skipping window_mean plot (no window data)")

    # 3. Mean absolute difference to oracle window vs parameters
    if 'oracle_mae' in results_df.columns:
        print("\n  3. Oracle window MAE vs parameters")
        plot_param_sensitivity(
            results_df, 'oracle_mae',
            output_path=out / 'oracle_mae_vs_params.png',
            ylabel='MAE to oracle window',
            title=f'{dataset_name}: Window accuracy vs LPA parameters',
            color='#2E8B57',
        )
    else:
        print("\n  3. Skipping oracle MAE plot (no window data)")

    print(f"\n  All plots saved to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize LPA sensitivity analysis results'
    )
    parser.add_argument(
        '--dataset', type=str, default='piecewise_ar3',
        help='Dataset name to visualize (default: piecewise_ar3)',
    )
    parser.add_argument(
        '--results-dir', type=str,
        default='examples/results/robustness/lpa_sensitivity',
        help='Directory containing LPA sensitivity results',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='examples/results/robustness/figures/lpa_sensitivity',
        help='Output directory for figures',
    )
    parser.add_argument(
        '--datasets-root', type=str,
        default='examples/datasets/simulated',
        help='Root directory for dataset files (containing true_importances.csv)',
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    datasets_root = Path(args.datasets_root)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    visualize_dataset(args.dataset, results_dir, output_dir, datasets_root)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
