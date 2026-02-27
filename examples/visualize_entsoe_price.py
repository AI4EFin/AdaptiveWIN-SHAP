"""
Visualization for ENTSO-E Romania Day-Ahead Price Case Study

Univariate: Day-ahead price (EUR/MWh)

Generates publication-quality figures:
  1. Raw data overview (price time series)
  2. Window size evolution over time
  3. Grouped lag importance (5 price-profile groups)
  4. Benchmark comparison (faithfulness/ablation bar chart)
  5. Full SHAP heatmap (supplementary)

Usage:
    python examples/visualize_entsoe_price.py
    python examples/visualize_entsoe_price.py --no-show
    python examples/visualize_entsoe_price.py --n0 168 --penalty-factor 0.1
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Publication-quality defaults
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
sns.set_palette("colorblind")

# Lag group definitions (price-dynamics aligned, 1-indexed)
# For electricity prices the interpretation differs slightly from load:
#   - "Recent" lags capture intraday price momentum
#   - "Afternoon" captures the previous trading session
#   - "Midday peak" captures peak-hour pricing memory
#   - "Morning ramp" captures the demand ramp-up effect on price
#   - "Night/previous day" captures the day-ahead cycle
LAG_GROUPS = {
    'Previous day (19-24h)': list(range(19, 25)),
    'Morning ramp (13-18h)': list(range(13, 19)),
    'Midday peak (10-12h)': list(range(10, 13)),
    'Afternoon (5-9h)': list(range(5, 10)),
    'Recent (1-4h)': list(range(1, 5)),
}

GROUP_COLORS = {
    'Previous day (19-24h)': '#1f77b4',
    'Morning ramp (13-18h)': '#ff7f0e',
    'Midday peak (10-12h)': '#2ca02c',
    'Afternoon (5-9h)': '#d62728',
    'Recent (1-4h)': '#9467bd',
}


def load_paths(n0, penalty_factor):
    """Build all result paths given LPA parameters."""
    data_dir = "examples/datasets/empirical/entsoe_ro_price_test"
    detection_dir = f"examples/results/LSTM/entsoe_ro_price_test/Jump_1_N0_{n0}_lambda_{penalty_factor}"
    benchmark_dir = f"examples/results/benchmark_entsoe_ro_price_test/N0_{n0}_lambda_{penalty_factor}"
    figures_dir = os.path.join(benchmark_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    return {
        'data_csv': os.path.join(data_dir, "data.csv"),
        'raw_price_csv': os.path.join(data_dir, "raw_price.csv"),
        'norm_params': os.path.join(data_dir, "normalization_params.json"),
        'windows_csv': os.path.join(detection_dir, "windows.csv"),
        'adaptive_shap_csv': os.path.join(benchmark_dir, "adaptive_shap_results.csv"),
        'adaptive_shap_rm_csv': os.path.join(benchmark_dir, "adaptive_shap_rolling_mean_results.csv"),
        'global_shap_csv': os.path.join(benchmark_dir, "global_shap_results.csv"),
        'timeshap_csv': os.path.join(benchmark_dir, "timeshap_results.csv"),
        'rolling_shap_csv': os.path.join(benchmark_dir, "rolling_shap_results.csv"),
        'benchmark_summary_csv': os.path.join(benchmark_dir, "benchmark_summary.csv"),
        'figures_dir': figures_dir,
    }


# ============================================================
# Plot 1: Raw Data Overview
# ============================================================
def plot_data_overview(paths):
    """Time series of day-ahead price."""
    norm_params_path = paths['norm_params']
    data_path = paths['data_csv']

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Try to denormalize if params exist
    if os.path.exists(norm_params_path):
        with open(norm_params_path) as f:
            params = json.load(f)
        price = df['N'] * params['price_std'] + params['price_mean']
        ylabel = 'Day-Ahead Price (EUR/MWh)'
    else:
        price = df['N']
        ylabel = 'Z-scored Price'

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(df.index, price.values, linewidth=0.4, alpha=0.8, color='#1f77b4',
            label='Day-Ahead Price')

    # Year boundaries
    for year in [2023, 2024, 2025, 2026]:
        boundary = pd.Timestamp(f'{year}-01-01', tz='Europe/Bucharest')
        if df.index.min() < boundary < df.index.max():
            ax.axvline(boundary, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(boundary, ax.get_ylim()[1] * 0.98, str(year),
                    ha='left', va='top', fontsize=10, color='gray')

    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.set_title('ENTSO-E Romania: Day-Ahead Electricity Price (2022-2026)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # fig.autofmt_xdate(rotation=45)

    fig.tight_layout()

    save_path = os.path.join(paths['figures_dir'], 'data_overview.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {save_path}")
    return fig


# ============================================================
# Plot 2: Window Size Evolution
# ============================================================
def plot_window_evolution(paths, rolling_mean_size=168):
    """Window size evolution over time."""
    windows_path = paths['windows_csv']
    if not os.path.exists(windows_path):
        print(f"Windows file not found: {windows_path}")
        return None

    windows_df = pd.read_csv(windows_path, index_col=0)

    # Load datetime index from data CSV
    data_df = pd.read_csv(paths['data_csv'], index_col=0, parse_dates=True)
    time_idx = data_df.index[:len(windows_df)]

    fig, ax = plt.subplots(figsize=(16, 5))

    if 'window_mean' in windows_df.columns:
        vals = windows_df['window_mean'].values
    else:
        wcols = [c for c in windows_df.columns if c.startswith('windows')]
        vals = windows_df[wcols[0]].values

    ax.plot(time_idx, vals, linewidth=0.8, alpha=0.6, color='#3B75AF', label='Window Size')

    rolling = pd.Series(vals).rolling(window=rolling_mean_size, center=True, min_periods=1).mean()
    ax.plot(time_idx, rolling, linewidth=2, color='red', label=f'Rolling Mean ({rolling_mean_size})')

    ax.set_xlabel('Date')
    ax.set_ylabel('Adaptive Window Size (hours)')
    ax.set_title('ENTSO-E Romania Price: Adaptive Window Size Evolution')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # fig.autofmt_xdate(rotation=45)

    fig.tight_layout()

    save_path = os.path.join(paths['figures_dir'], 'window_sizes.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {save_path}")
    return fig


# ============================================================
# Plot 3: Grouped Lag Importance Over Time
# ============================================================
def plot_grouped_lag_importance(paths, smoothing_window=168):
    """
    Aggregate 24 lags into 5 price-profile groups and plot over time.
    """
    shap_path = paths['adaptive_shap_rm_csv']
    if not os.path.exists(shap_path):
        shap_path = paths['adaptive_shap_csv']
    if not os.path.exists(shap_path):
        print("No adaptive SHAP results found.")
        return None

    df = pd.read_csv(shap_path)

    lag_cols = sorted(
        [c for c in df.columns if c.startswith('shap_lag_t-')],
        key=lambda c: int(c.split('-')[1])
    )

    if not lag_cols:
        print("No shap_lag_t-* columns found in results.")
        return None

    # Build group time series
    group_series = {}
    for group_name, lags in LAG_GROUPS.items():
        cols = [f'shap_lag_t-{lag}' for lag in lags if f'shap_lag_t-{lag}' in df.columns]
        if cols:
            group_series[group_name] = df[cols].sum(axis=1).values

    if not group_series:
        print("Could not map any lags to groups.")
        return None

    # Normalize to proportions
    group_df = pd.DataFrame(group_series)
    row_sums = group_df.sum(axis=1).replace(0, np.nan)
    group_props = group_df.div(row_sums, axis=0).fillna(0)

    # Smooth
    group_smooth = group_props.rolling(window=smoothing_window, center=True, min_periods=1).mean()

    if 'end_index' in df.columns:
        x = df['end_index'].values
    else:
        x = np.arange(len(group_smooth))

    # Stacked area chart
    fig, ax = plt.subplots(figsize=(16, 6))

    bottom = np.zeros(len(group_smooth))
    for group_name in LAG_GROUPS:
        if group_name in group_smooth.columns:
            vals = group_smooth[group_name].values
            ax.fill_between(x, bottom, bottom + vals,
                            label=group_name, color=GROUP_COLORS[group_name], alpha=0.8)
            bottom += vals

    ax.set_xlabel('Timepoint (hourly)')
    ax.set_ylabel('Relative Importance (proportion)')
    ax.set_title('ENTSO-E Romania Price: Grouped Lag Importance Over Time')
    ax.set_ylim(0, 1)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True)
    ax.grid(True, alpha=0.2, linestyle='--')
    fig.tight_layout()

    save_path = os.path.join(paths['figures_dir'], 'grouped_lag_importance.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {save_path}")
    return fig


# ============================================================
# Plot 4: Benchmark Comparison
# ============================================================
def plot_benchmark_comparison(paths):
    """Faithfulness and ablation bar chart comparing all methods."""
    summary_path = paths['benchmark_summary_csv']
    if not os.path.exists(summary_path):
        print(f"Benchmark summary not found: {summary_path}")
        return None

    summary = pd.read_csv(summary_path)

    if summary.empty:
        print("Benchmark summary is empty.")
        return None

    faith_df = summary[summary['metric_type'] == 'faithfulness'].copy()
    ablation_df = summary[summary['metric_type'] == 'ablation'].copy()

    method_labels = {
        'global_shap': 'GlobalSHAP',
        'timeshap': 'TimeShap',
        'rolling_shap': 'RollingWindow',
        'adaptive_shap': 'AdaptiveWIN-SHAP',
        'adaptive_shap_rolling_mean': 'AdaptiveWIN-SHAP (RM)',
        'adaptive_shap_max': 'Adaptive (Max)',
        'adaptive_shap_mean': 'Adaptive (Mean)',
    }

    n_plots = sum([len(faith_df) > 0, len(ablation_df) > 0])
    if n_plots == 0:
        print("No metrics found in benchmark summary.")
        return None

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if len(faith_df) > 0:
        ax = axes[plot_idx]
        plot_idx += 1

        pivot = faith_df.pivot(index='method', columns='evaluation', values='score')
        pivot.index = [method_labels.get(m, m) for m in pivot.index]
        pivot = pivot.reindex([v for v in method_labels.values() if v in pivot.index])

        pivot.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Faithfulness Score')
        ax.set_title('Faithfulness Comparison')
        ax.set_xlabel('')
        ax.legend(title='Evaluation', fontsize=8, title_fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    if len(ablation_df) > 0:
        ax = axes[plot_idx]

        pivot = ablation_df.pivot(index='method', columns='evaluation', values='score')
        pivot.index = [method_labels.get(m, m) for m in pivot.index]
        pivot = pivot.reindex([v for v in method_labels.values() if v in pivot.index])

        pivot.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Ablation Score')
        ax.set_title('Ablation Comparison (MIF/LIF)')
        ax.set_xlabel('')
        ax.legend(title='Evaluation', fontsize=8, title_fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    fig.suptitle('ENTSO-E Romania Price: Benchmark Comparison', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    save_path = os.path.join(paths['figures_dir'], 'benchmark_comparison.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {save_path}")
    return fig


# ============================================================
# Plot 5: SHAP Heatmap (supplementary)
# ============================================================
def plot_shap_heatmap(paths, max_timepoints=5000):
    """Full 24-lag heatmap: rows = lags, columns = time."""
    shap_path = paths['adaptive_shap_rm_csv']
    if not os.path.exists(shap_path):
        shap_path = paths['adaptive_shap_csv']
    if not os.path.exists(shap_path):
        print("No adaptive SHAP results for heatmap.")
        return None

    df = pd.read_csv(shap_path)

    lag_cols = sorted(
        [c for c in df.columns if c.startswith('shap_lag_t-')],
        key=lambda c: int(c.split('-')[1])
    )

    if not lag_cols:
        print("No SHAP lag columns for heatmap.")
        return None

    shap_matrix = df[lag_cols].values.T  # (n_lags, n_timepoints)

    if shap_matrix.shape[1] > max_timepoints:
        step = shap_matrix.shape[1] // max_timepoints
        shap_matrix = shap_matrix[:, ::step]
        time_label = f'Timepoint (subsampled 1:{step})'
    else:
        time_label = 'Timepoint'

    fig, ax = plt.subplots(figsize=(18, 6))

    im = ax.imshow(
        shap_matrix,
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest',
        origin='lower'
    )

    ax.set_yticks(range(len(lag_cols)))
    ax.set_yticklabels([c.replace('shap_lag_', '') for c in lag_cols], fontsize=8)
    ax.set_xlabel(time_label)
    ax.set_ylabel('Lag')
    ax.set_title('ENTSO-E Romania Price: SHAP Importance Heatmap (all 24 lags)')

    fig.colorbar(im, ax=ax, shrink=0.8, label='|SHAP value|')
    fig.tight_layout()

    save_path = os.path.join(paths['figures_dir'], 'shap_heatmap.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {save_path}")
    return fig


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Visualize ENTSO-E Romania Price case study results'
    )
    parser.add_argument('--n0', type=int, default=168,
                        help='N0 value used in detection (default: 168)')
    parser.add_argument('--penalty-factor', type=float, default=0.15,
                        help='Penalty factor lambda (default: 0.15)')
    parser.add_argument('--smoothing', type=int, default=168,
                        help='Smoothing window for lag importance (default: 168 = 1 week)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save)')
    args = parser.parse_args()

    paths = load_paths(args.n0, args.penalty_factor)

    print("="*60)
    print("ENTSO-E Romania Price - Visualization")
    print("="*60)
    print(f"N0={args.n0}, Lambda={args.penalty_factor}")
    print(f"Figures directory: {paths['figures_dir']}")
    print("="*60)

    print("\n[1/5] Data overview...")
    try:
        plot_data_overview(paths)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[2/5] Window size evolution...")
    try:
        plot_window_evolution(paths, rolling_mean_size=args.smoothing)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[3/5] Grouped lag importance...")
    try:
        plot_grouped_lag_importance(paths, smoothing_window=args.smoothing)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[4/5] Benchmark comparison...")
    try:
        plot_benchmark_comparison(paths)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[5/5] SHAP heatmap...")
    try:
        plot_shap_heatmap(paths)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"All figures saved to: {paths['figures_dir']}")
    print("="*60)

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
