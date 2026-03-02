"""
Visualization script for DGP Parameter Robustness results.

Generates 4 publication-quality figures:
  1. Window size evolution across scenarios (stacked panels)
  2. SHAP correlation with true importances per regime (grouped bars)
  3. Method comparison: faithfulness + ablation across L2 distances (line plots)
  4. Oracle window MAE + SHAP correlation per regime vs L2 distance (line plots)

Usage:
    python examples/robustness/visualize_dgp_robustness.py
    python examples/robustness/visualize_dgp_robustness.py --results-dir path/to/results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style matching existing robustness visualizations
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Scenario definitions (must match 02_dgp_parameter_robustness.py)
BASELINE_REGIMES = [
    np.array([0.9, 0.01, 0.01]),
    np.array([0.01, 0.9, 0.01]),
    np.array([0.01, 0.01, 0.9]),
]
CENTROID = np.mean(BASELINE_REGIMES, axis=0)
REDUCTION_FACTORS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 0.90]
REGIME_LENGTHS = (500, 500, 500)
TRUE_BREAKPOINTS = [500, 1000]

SCENARIO_DIRS = ['baseline', 'l2_10', 'l2_20', 'l2_30', 'l2_40', 'l2_50', 'l2_75', 'l2_90']

# Methods to compare in Figure 3
CORE_METHODS = ['adaptive_shap', 'global_shap', 'rolling_shap', 'timeshap']
METHOD_LABELS = {
    'adaptive_shap': 'Adaptive SHAP (ours)',
    'global_shap': 'Global SHAP',
    'rolling_shap': 'Rolling SHAP',
    'timeshap': 'TimeSHAP',
}


def compute_max_pairwise_l2(regimes):
    max_dist = 0.0
    for i in range(len(regimes)):
        for j in range(i + 1, len(regimes)):
            max_dist = max(max_dist, np.linalg.norm(regimes[i] - regimes[j]))
    return max_dist


def get_scenario_info():
    """Build scenario metadata."""
    info = []
    for t in REDUCTION_FACTORS:
        regimes = [(1 - t) * phi + t * CENTROID for phi in BASELINE_REGIMES]
        max_l2 = compute_max_pairwise_l2(regimes)
        name = 'baseline' if t == 0.0 else f'l2_{int(t * 100)}'
        if t == 0.0:
            label = f'Baseline ($L_2$={max_l2:.2f})'
        else:
            label = f'{int(t*100)}% reduction ($L_2$={max_l2:.2f})'
        info.append({'name': name, 'label': label, 't': t, 'max_l2': max_l2})
    return info


def compute_oracle_window(n_timepoints, breakpoints):
    """
    Compute the oracle window at each timepoint.

    The oracle window at time t is the number of past observations belonging
    to the same regime, i.e. t - last_breakpoint (or t+1 for the first regime).
    """
    oracle = np.zeros(n_timepoints)
    boundaries = [0] + sorted(breakpoints) + [n_timepoints]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        for t in range(start, end):
            oracle[t] = t - start + 1
    return oracle


def load_windows(results_dir, scenario_name):
    """Load window data from run_0.csv (which has the actual values)."""
    run_path = results_dir / scenario_name / 'run_0.csv'
    if not run_path.exists():
        return None
    df = pd.read_csv(run_path)
    return df['windows'].values


def load_benchmark_summary(results_dir, scenario_name):
    """Load benchmark summary for a scenario."""
    path = results_dir / scenario_name / 'benchmark' / 'benchmark_summary.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_shap_correlation(results_dir, scenario_name, datasets_base):
    """Compute SHAP-vs-true-importance Pearson correlation per regime."""
    shap_path = results_dir / scenario_name / 'benchmark' / 'adaptive_shap_results.csv'

    if scenario_name == 'baseline':
        true_path = datasets_base / 'piecewise_ar3' / 'true_importances.csv'
    else:
        true_path = datasets_base / 'piecewise_ar3_dgp_robustness' / scenario_name / 'true_importances.csv'

    if not shap_path.exists() or not true_path.exists():
        return [np.nan] * 3

    shap_df = pd.read_csv(shap_path)
    true_df = pd.read_csv(true_path)

    shap_cols = [c for c in shap_df.columns if c.startswith('shap_')]
    true_cols = [c for c in true_df.columns if c.startswith('true_imp_')]

    if not shap_cols or not true_cols:
        return [np.nan] * 3

    # Use end_index to align SHAP rows to the correct time points
    end_indices = shap_df['end_index'].astype(int).values
    shap_values = shap_df[shap_cols].values
    true_values = true_df[true_cols].values

    breakpoints = [0, REGIME_LENGTHS[0], REGIME_LENGTHS[0] + REGIME_LENGTHS[1], sum(REGIME_LENGTHS)]
    regime_corrs = []

    for i in range(3):
        start, end = breakpoints[i], breakpoints[i + 1]
        # Select SHAP rows whose end_index falls within this regime
        regime_mask = (end_indices >= start) & (end_indices < end)
        if regime_mask.sum() == 0:
            regime_corrs.append(np.nan)
            continue

        s = shap_values[regime_mask].flatten()
        # Look up true importances at the corresponding time indices
        regime_indices = np.clip(end_indices[regime_mask], 0, len(true_values) - 1)
        t = true_values[regime_indices].flatten()

        valid = ~(np.isnan(s) | np.isnan(t))
        if valid.sum() > 10:
            regime_corrs.append(float(np.corrcoef(s[valid], t[valid])[0, 1]))
        else:
            regime_corrs.append(np.nan)

    return regime_corrs


# ============================================================================
# FIGURE 1: Window Size Evolution (4 stacked panels)
# ============================================================================

def plot_window_evolution(results_dir, figures_dir, rolling_window=10):
    """
    4 vertically stacked panels showing window size evolution.
    Raw trace faint (alpha=0.15), rolling mean bold on top.
    """
    scenario_info = get_scenario_info()
    colors = sns.color_palette("husl", len(scenario_info))

    fig, axes = plt.subplots(len(SCENARIO_DIRS), 1, figsize=(10, 10), sharex=True, sharey=True)

    for idx, (info, ax, color) in enumerate(zip(scenario_info, axes, colors)):
        windows = load_windows(results_dir, info['name'])
        if windows is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_ylabel('Window Size')
            ax.set_title(info['label'], loc='left', fontweight='bold')
            continue

        time_index = np.arange(len(windows))

        # Raw trace (faint)
        ax.plot(time_index, windows, color=color, alpha=0.15, linewidth=0.8)

        # Rolling mean (bold)
        windows_series = pd.Series(windows)
        rolling_mean = windows_series.rolling(window=rolling_window, min_periods=1, center=True).mean()
        ax.plot(time_index, rolling_mean, color=color, linewidth=2.0, alpha=0.9)

        # Breakpoints
        for bp in TRUE_BREAKPOINTS:
            ax.axvline(x=bp, color='#333333', linestyle='--', linewidth=1.2, alpha=0.5)

        ax.set_title(info['label'], loc='left', fontweight='bold')

    fig.text(-0.01, 0.5, 'Window Size', va='center', rotation='vertical')
    axes[-1].set_xlabel('Time Index')

    # Add breakpoint labels once at the top
    for bp in TRUE_BREAKPOINTS:
        axes[0].text(bp, axes[0].get_ylim()[1], f' t={bp}',
                     fontsize=8, color='#333333', va='bottom', ha='left')

    plt.tight_layout()
    save_path = figures_dir / 'fig1_window_evolution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# FIGURE 2: SHAP Correlation per Regime (grouped bars)
# ============================================================================

def plot_shap_correlation_bars(results_dir, datasets_base, figures_dir):
    """
    Grouped bar chart: x-axis = scenario, bars = regime 0, 1, 2.
    Shows how SHAP fidelity degrades per regime as L2 decreases.
    """
    scenario_info = get_scenario_info()
    palette = sns.color_palette("husl", 3)

    regime_data = {'Scenario': [], 'Regime': [], 'Correlation': [], 'L2': []}

    for info in scenario_info:
        corrs = load_shap_correlation(results_dir, info['name'], datasets_base)
        for r_idx, corr in enumerate(corrs):
            regime_data['Scenario'].append(info['label'])
            regime_data['Regime'].append(f'Regime {r_idx}')
            regime_data['Correlation'].append(corr)
            regime_data['L2'].append(info['max_l2'])

    df = pd.DataFrame(regime_data)

    # Drop NaN rows for cleaner plot
    df_valid = df.dropna(subset=['Correlation'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Get unique scenarios in order
    scenario_labels = [info['label'] for info in scenario_info]
    tick_labels = ['Baseline' if info['t'] == 0.0 else f"{int(info['t']*100)}%"
                   for info in scenario_info]
    regimes_present = sorted(df_valid['Regime'].unique())

    x = np.arange(len(scenario_labels))
    n_regimes = len(regimes_present)
    bar_width = 0.25
    offsets = np.linspace(-(n_regimes - 1) / 2 * bar_width, (n_regimes - 1) / 2 * bar_width, n_regimes)

    for r_idx, (regime, offset) in enumerate(zip(regimes_present, offsets)):
        regime_df = df[df['Regime'] == regime]
        vals = []
        for label in scenario_labels:
            row = regime_df[regime_df['Scenario'] == label]
            if len(row) > 0 and not np.isnan(row['Correlation'].values[0]):
                vals.append(row['Correlation'].values[0])
            else:
                vals.append(0)

        bars = ax.bar(x + offset, vals, bar_width, label=regime,
                      color=palette[r_idx], alpha=0.85, edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for xi, v in zip(x + offset, vals):
            if v > 0.02:
                ax.text(xi, v + 0.015, f'{v:.2f}', ha='center', va='bottom',
                        fontsize=8, color=palette[r_idx], fontweight='bold')

    ax.set_xlabel('$L_2$ Reduction')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('SHAP Fidelity vs True Feature Importances by Regime',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.legend(title='', framealpha=0.9, loc='upper right')
    ax.set_ylim(0, max(df_valid['Correlation'].max() * 1.15, 0.6))
    ax.grid(True, axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ['png']:
        save_path = figures_dir / f'fig2_shap_correlation.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# FIGURE 3: Method Comparison - Faithfulness + Ablation (line plots)
# ============================================================================

def plot_method_comparison(results_dir, figures_dir):
    """
    Two subplots: faithfulness (left) and ablation MIF (right) vs L2 distance.
    One line per core method. Shows competitive advantage of adaptive_shap.
    """
    scenario_info = get_scenario_info()
    method_colors = sns.color_palette("husl", len(CORE_METHODS))
    method_markers = ['o', 's', 'D', '^']

    # Collect data
    faith_data = {m: [] for m in CORE_METHODS}
    ablation_data = {m: [] for m in CORE_METHODS}
    l2_values = []

    for info in scenario_info:
        bs = load_benchmark_summary(results_dir, info['name'])
        l2_values.append(info['max_l2'])

        if bs is None:
            for m in CORE_METHODS:
                faith_data[m].append(np.nan)
                ablation_data[m].append(np.nan)
            continue

        for m in CORE_METHODS:
            method_df = bs[bs['method'] == m]

            # Faithfulness prtb_p50
            faith_row = method_df[(method_df['metric_type'] == 'faithfulness') &
                                   (method_df['evaluation'] == 'prtb_p50')]
            faith_data[m].append(float(faith_row['score'].values[0]) if len(faith_row) > 0 else np.nan)

            # Ablation MIF p50
            abl_row = method_df[(method_df['metric_type'] == 'ablation') &
                                 (method_df['evaluation'] == 'mif_p50')]
            ablation_data[m].append(float(abl_row['score'].values[0]) if len(abl_row) > 0 else np.nan)

    l2_arr = np.array(l2_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Faithfulness
    for m_idx, method in enumerate(CORE_METHODS):
        vals = np.array(faith_data[method])
        is_ours = method == 'adaptive_shap'
        lw = 2.5 if is_ours else 1.5
        ms = 10 if is_ours else 7
        zorder = 10 if is_ours else 5

        ax1.plot(l2_arr, vals, marker=method_markers[m_idx], markersize=ms,
                 linewidth=lw, color=method_colors[m_idx],
                 label=METHOD_LABELS[method], zorder=zorder,
                 alpha=1.0 if is_ours else 0.7)

    ax1.set_xlabel('Max Pairwise $L_2$ Distance')
    ax1.set_ylabel('Faithfulness Score (higher = better)')
    ax1.set_title('Faithfulness (Perturbation p50)', fontweight='bold')
    ax1.legend(framealpha=0.9, loc='best')
    ax1.grid(True, alpha=0.2)
    ax1.invert_xaxis()  # Decreasing L2 = harder
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Ablation MIF
    for m_idx, method in enumerate(CORE_METHODS):
        vals = np.array(ablation_data[method])
        is_ours = method == 'adaptive_shap'
        lw = 2.5 if is_ours else 1.5
        ms = 10 if is_ours else 7
        zorder = 10 if is_ours else 5

        ax2.plot(l2_arr, vals, marker=method_markers[m_idx], markersize=ms,
                 linewidth=lw, color=method_colors[m_idx],
                 label=METHOD_LABELS[method], zorder=zorder,
                 alpha=1.0 if is_ours else 0.7)

    ax2.set_xlabel('Max Pairwise $L_2$ Distance')
    ax2.set_ylabel('Ablation MIF Score (higher = better)')
    ax2.set_title('Ablation: Most Important First (p50)', fontweight='bold')
    ax2.legend(framealpha=0.9, loc='best')
    ax2.grid(True, alpha=0.2)
    ax2.invert_xaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ['png']:
        save_path = figures_dir / f'fig3_method_comparison.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# FIGURE 4: Oracle Window MAE + SHAP Correlation per Regime (line plots)
# ============================================================================

def plot_l2_vs_oracle_and_shap(results_dir, datasets_base, figures_dir):
    """
    Two subplots vs L2 distance:
      Left:  MAE to oracle window (single line with annotations).
      Right: SHAP correlation per regime (one line per regime, Fig 3 style).
    """
    scenario_info = get_scenario_info()
    regime_palette = sns.color_palette("husl", 3)
    regime_markers = ['o', 's', 'D']

    l2_values = []
    mae_values = []
    regime_corrs = {0: [], 1: [], 2: []}

    for info in scenario_info:
        # --- Oracle MAE ---
        windows = load_windows(results_dir, info['name'])
        if windows is not None:
            n = len(windows)
            oracle = compute_oracle_window(n, TRUE_BREAKPOINTS)
            valid_mask = ~np.isnan(windows)
            if valid_mask.sum() > 0:
                mae = float(np.mean(np.abs(windows[valid_mask] - oracle[valid_mask])))
            else:
                mae = np.nan
        else:
            mae = np.nan

        # --- SHAP correlation per regime ---
        corrs = load_shap_correlation(results_dir, info['name'], datasets_base)

        l2_values.append(info['max_l2'])
        mae_values.append(mae)
        for r in range(3):
            regime_corrs[r].append(corrs[r])

    l2_arr = np.array(l2_values)
    mae_arr = np.array(mae_values)

    if np.all(np.isnan(mae_arr)):
        print("  No data found for oracle MAE / SHAP correlation plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: Oracle MAE vs L2 ----
    ax1.plot(l2_arr, mae_arr, marker='o', markersize=8, linewidth=2.0,
             color='#2E8B57', zorder=5)
    for x, y, info in zip(l2_arr, mae_arr, scenario_info):
        if np.isnan(y):
            continue
        label = 'BL' if info['t'] == 0.0 else f"{int(info['t']*100)}%"
        ax1.annotate(label, (x, y), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8, color='#555555')

    ax1.set_xlabel('Max Pairwise $L_2$ Distance')
    ax1.set_ylabel('MAE to Oracle Window')
    ax1.set_title('Window Accuracy vs Regime Separation', fontweight='bold')
    ax1.invert_xaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- Right: SHAP Correlation per Regime vs L2 (line plot) ----
    for r in range(3):
        vals = np.array(regime_corrs[r])
        ax2.plot(l2_arr, vals, marker=regime_markers[r], markersize=8,
                 linewidth=2.0, color=regime_palette[r],
                 label=f'Regime {r}', zorder=5)

    ax2.set_xlabel('Max Pairwise $L_2$ Distance')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('SHAP Fidelity per Regime vs Regime Separation', fontweight='bold')
    ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2.invert_xaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = figures_dir / 'fig4_oracle_and_shap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize DGP Parameter Robustness results'
    )
    parser.add_argument('--results-dir', type=str,
                        default='examples/results/robustness/dgp_robustness/piecewise_ar3',
                        help='Path to results directory')
    parser.add_argument('--datasets-base', type=str,
                        default='examples/datasets/simulated',
                        help='Path to datasets base directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    datasets_base = Path(args.datasets_base)
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DGP Robustness Visualization")
    print("=" * 60)
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    print()

    print("Figure 1: Window Size Evolution...")
    plot_window_evolution(results_dir, figures_dir)

    print("\nFigure 2: SHAP Correlation per Regime (bars)...")
    plot_shap_correlation_bars(results_dir, datasets_base, figures_dir)

    print("\nFigure 3: Method Comparison...")
    plot_method_comparison(results_dir, figures_dir)

    print("\nFigure 4: Oracle Window MAE + SHAP Correlation...")
    plot_l2_vs_oracle_and_shap(results_dir, datasets_base, figures_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
