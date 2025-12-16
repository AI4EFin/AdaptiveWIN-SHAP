"""
Visualization script for robustness benchmarking results.

Adapts benchmark_viz.py for robustness test results, showing:
- Faithfulness/ablation metrics across parameter configurations
- Window size evolution over time for each configuration
- Comparison of performance vs parameter settings
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set style
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_robustness_benchmark_data(results_dir, dataset_name):
    """
    Load robustness benchmark results.

    Parameters
    ----------
    results_dir : Path
        Root robustness results directory (e.g., lpa_sensitivity/ or bootstrap_ci/)
    dataset_name : str
        Name of dataset

    Returns
    -------
    dict with keys:
        - 'summary': aggregated results DataFrame
        - 'configs': list of parameter configurations
        - 'windows': dict mapping config to windows DataFrame
        - 'shap': dict mapping config to SHAP results DataFrame
    """
    data = {}

    dataset_dir = results_dir / dataset_name

    # Load summary results
    summary_file = dataset_dir / 'sensitivity_summary.csv'
    if summary_file.exists():
        data['summary'] = pd.read_csv(summary_file)
    else:
        # Try global results.csv filtered by dataset
        global_results = results_dir / 'results.csv'
        if global_results.exists():
            df = pd.read_csv(global_results)
            data['summary'] = df[df['dataset'] == dataset_name].copy()

    # Find all parameter configuration directories
    data['configs'] = []
    data['windows'] = {}
    data['shap'] = {}
    data['benchmark_results'] = {}

    # Look for temp directories
    for temp_dir in dataset_dir.glob('temp_*'):
        if not temp_dir.is_dir():
            continue

        config_name = temp_dir.name.replace('temp_', '')
        data['configs'].append(config_name)

        # Load windows if available
        windows_csv = temp_dir / 'windows.csv'
        if windows_csv.exists():
            data['windows'][config_name] = pd.read_csv(windows_csv)

        # Load SHAP results if available
        benchmark_dir = temp_dir / 'benchmark'
        if benchmark_dir.exists():
            shap_results = benchmark_dir / 'adaptive_shap_results.csv'
            if shap_results.exists():
                data['shap'][config_name] = pd.read_csv(shap_results)

            # Load benchmark summary
            summary_csv = benchmark_dir / 'benchmark_summary.csv'
            if summary_csv.exists():
                data['benchmark_results'][config_name] = pd.read_csv(summary_csv)

    # Load true importances for comparison
    true_imp_path = f"examples/datasets/simulated/{dataset_name}/true_importances.csv"
    if os.path.exists(true_imp_path):
        data['true_importances'] = pd.read_csv(true_imp_path)

    return data


def plot_window_evolution_per_config(data, dataset_name, save_dir, breakpoints=None):
    """
    Plot window size evolution over time for each configuration.

    Parameters
    ----------
    data : dict
        Loaded robustness data
    dataset_name : str
        Name of dataset
    save_dir : Path
        Directory to save figures
    breakpoints : list
        List of true breakpoints for regime changes
    """
    if not data['windows']:
        print("No window data available - skipping window evolution plots")
        return

    n_configs = len(data['configs'])
    if n_configs == 0:
        return

    # Create subplot grid
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    fig.suptitle(f'{dataset_name}: Window Size Evolution Across Configurations',
                 fontsize=14, fontweight='bold')

    # Flatten axes for easier indexing
    if n_configs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for idx, config_name in enumerate(sorted(data['configs'])):
        ax = axes[idx]
        windows_df = data['windows'][config_name]

        # Plot window size over time
        if 'window_mean' in windows_df.columns:
            window_col = 'window_mean'
        elif 'windows_run_0' in windows_df.columns:
            window_col = 'windows_run_0'
        else:
            # Find first windows column
            window_cols = [c for c in windows_df.columns if c.startswith('windows')]
            if window_cols:
                window_col = window_cols[0]
            else:
                ax.text(0.5, 0.5, 'No window data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(config_name)
                continue

        time_index = np.arange(len(windows_df))
        window_sizes = windows_df[window_col].values

        # Plot window size
        ax.plot(time_index, window_sizes, linewidth=1.5, alpha=0.8, color='#2ca02c')

        # Add breakpoints if provided
        if breakpoints:
            for bp in breakpoints:
                ax.axvline(x=bp, color='red', linestyle='--', linewidth=1.5,
                          alpha=0.6, label='True Breakpoint' if bp == breakpoints[0] else '')

        # Add statistics
        mean_window = window_sizes.mean()
        std_window = window_sizes.std()
        ax.axhline(y=mean_window, color='blue', linestyle=':', linewidth=1,
                  alpha=0.5, label=f'Mean: {mean_window:.1f}')
        ax.fill_between(time_index,
                       mean_window - std_window,
                       mean_window + std_window,
                       alpha=0.2, color='blue',
                       label=f'±1 SD: {std_window:.1f}')

        ax.set_xlabel('Time Index')
        ax.set_ylabel('Window Size')
        ax.set_title(config_name, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = save_dir / 'window_evolution_all_configs.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_window_statistics_comparison(data, save_dir):
    """
    Compare window statistics (mean, std, min, max) across configurations.
    """
    if 'summary' not in data or len(data['summary']) == 0:
        print("No summary data - skipping window statistics comparison")
        return

    summary = data['summary']

    # Check for window statistics columns
    window_metrics = ['window_mean', 'window_std', 'window_min', 'window_max']
    available_metrics = [m for m in window_metrics if m in summary.columns]

    if not available_metrics:
        print("No window metrics in summary - skipping window statistics comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fig.suptitle('Window Statistics Across Configurations', fontsize=14, fontweight='bold')

    for idx, metric in enumerate(window_metrics):
        ax = axes[idx]

        if metric not in summary.columns:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(metric.replace('_', ' ').title())
            continue

        # Create configuration labels
        if 'N0' in summary.columns:
            config_labels = summary.apply(
                lambda row: f"N{int(row['N0'])}_α{row.get('alpha', 0):.2f}_B{int(row.get('num_bootstrap', 0))}",
                axis=1
            )
        else:
            config_labels = [f"Config {i}" for i in range(len(summary))]

        values = summary[metric].values

        bars = ax.bar(range(len(values)), values, alpha=0.7, edgecolor='black',
                     color='#2ca02c')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = save_dir / 'window_statistics_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_faithfulness_across_configs(data, save_dir):
    """
    Plot faithfulness scores across different configurations.
    """
    if not data['benchmark_results']:
        print("No benchmark results - skipping faithfulness comparison")
        return

    # Extract faithfulness scores for each config
    configs = []
    faithfulness_p50 = []
    faithfulness_p90 = []

    for config_name in sorted(data['configs']):
        if config_name not in data['benchmark_results']:
            continue

        benchmark_df = data['benchmark_results'][config_name]

        # Get faithfulness scores for adaptive_shap
        adaptive_results = benchmark_df[benchmark_df['method'] == 'adaptive_shap']
        faith_results = adaptive_results[adaptive_results['metric_type'] == 'faithfulness']

        # Get p50 and p90
        p50_row = faith_results[faith_results['evaluation'] == 'prtb_p50']
        p90_row = faith_results[faith_results['evaluation'] == 'prtb_p90']

        if len(p50_row) > 0 or len(p90_row) > 0:
            configs.append(config_name)
            faithfulness_p50.append(p50_row['score'].values[0] if len(p50_row) > 0 else np.nan)
            faithfulness_p90.append(p90_row['score'].values[0] if len(p90_row) > 0 else np.nan)

    if not configs:
        print("No faithfulness data found")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, faithfulness_p50, width, label='p50',
                  alpha=0.7, edgecolor='black', color='#2ca02c')
    bars2 = ax.bar(x + width/2, faithfulness_p90, width, label='p90',
                  alpha=0.7, edgecolor='black', color='#ff7f0e')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Faithfulness Score')
    ax.set_title('Faithfulness Comparison Across Configurations', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=7)

    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    save_path = save_dir / 'faithfulness_across_configs.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_ablation_across_configs(data, save_dir):
    """
    Plot MIF/LIF ablation scores across different configurations.
    """
    if not data['benchmark_results']:
        print("No benchmark results - skipping ablation comparison")
        return

    # Extract ablation scores for each config
    configs = []
    mif_p50, mif_p90 = [], []
    lif_p50, lif_p90 = [], []

    for config_name in sorted(data['configs']):
        if config_name not in data['benchmark_results']:
            continue

        benchmark_df = data['benchmark_results'][config_name]

        # Get ablation scores for adaptive_shap
        adaptive_results = benchmark_df[benchmark_df['method'] == 'adaptive_shap']
        ablation_results = adaptive_results[adaptive_results['metric_type'] == 'ablation']

        # Get scores
        mif_p50_row = ablation_results[ablation_results['evaluation'] == 'mif_p50']
        mif_p90_row = ablation_results[ablation_results['evaluation'] == 'mif_p90']
        lif_p50_row = ablation_results[ablation_results['evaluation'] == 'lif_p50']
        lif_p90_row = ablation_results[ablation_results['evaluation'] == 'lif_p90']

        if len(mif_p50_row) > 0 or len(lif_p50_row) > 0:
            configs.append(config_name)
            mif_p50.append(mif_p50_row['score'].values[0] if len(mif_p50_row) > 0 else np.nan)
            mif_p90.append(mif_p90_row['score'].values[0] if len(mif_p90_row) > 0 else np.nan)
            lif_p50.append(lif_p50_row['score'].values[0] if len(lif_p50_row) > 0 else np.nan)
            lif_p90.append(lif_p90_row['score'].values[0] if len(lif_p90_row) > 0 else np.nan)

    if not configs:
        print("No ablation data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('MIF vs LIF Ablation Across Configurations', fontsize=14, fontweight='bold')

    percentiles = [('p50', mif_p50, lif_p50), ('p90', mif_p90, lif_p90)]

    for idx, (pct_name, mif_vals, lif_vals) in enumerate(percentiles):
        ax = axes[idx]

        x = np.arange(len(configs))
        width = 0.35

        bars1 = ax.bar(x - width/2, mif_vals, width, label='MIF',
                      alpha=0.7, edgecolor='black', color='#2ca02c')
        bars2 = ax.bar(x + width/2, lif_vals, width, label='LIF',
                      alpha=0.4, edgecolor='black', color='#2ca02c', hatch='//')

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Ablation Score')
        ax.set_title(f'Percentile: {pct_name.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=7)

        for bar in bars2:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    save_path = save_dir / 'ablation_across_configs.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def compute_true_windows(n_timepoints, breakpoints):
    """
    Compute true window size for each timepoint.

    True window = distance from last breakpoint to current timepoint.
    This represents the size of the stable regime at each point.

    Parameters
    ----------
    n_timepoints : int
        Total number of timepoints
    breakpoints : list
        List of breakpoint indices

    Returns
    -------
    np.ndarray
        True window size for each timepoint
    """
    true_windows = np.zeros(n_timepoints)

    for t in range(n_timepoints):
        # Find last breakpoint before or at t
        past_breakpoints = [bp for bp in breakpoints if bp <= t]

        if past_breakpoints:
            last_bp = max(past_breakpoints)
            true_windows[t] = t - last_bp
        else:
            # No breakpoint yet, distance from start
            true_windows[t] = t

    return true_windows


def plot_true_vs_detected_by_n0(data, dataset_name, save_dir, breakpoints=None):
    """
    Plot true vs detected window sizes, grouped by N0 parameter.

    Creates separate plots for each N0 value to see how initial window
    size affects detection accuracy.
    """
    if not data['windows'] or breakpoints is None:
        print("No window data or breakpoints - skipping true vs detected plots")
        return

    # Parse N0 values from config names
    n0_configs = {}
    for config_name in data['configs']:
        if config_name not in data['windows']:
            continue

        # Extract N0 from config name (format: N{N0}_alpha{alpha}_num_bootstrap{B})
        try:
            parts = config_name.split('_')
            n0_str = [p for p in parts if p.startswith('N')][0]
            n0_val = int(n0_str.replace('N', ''))

            if n0_val not in n0_configs:
                n0_configs[n0_val] = []
            n0_configs[n0_val].append(config_name)
        except:
            continue

    if not n0_configs:
        print("Could not parse N0 values from config names")
        return

    # Sort N0 values
    n0_values = sorted(n0_configs.keys())

    # Create one plot per N0 value
    for n0_val in n0_values:
        configs_for_n0 = n0_configs[n0_val]
        n_configs = len(configs_for_n0)

        # Create subplot grid
        n_cols = min(2, n_configs)
        n_rows = (n_configs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 8*n_rows))
        fig.suptitle(f'{dataset_name}: True vs Detected Windows (N0={n0_val})',
                     fontsize=14, fontweight='bold')

        # Flatten axes
        if n_configs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for idx, config_name in enumerate(sorted(configs_for_n0)):
            ax = axes[idx]
            windows_df = data['windows'][config_name]

            # Get detected windows
            if 'window_mean' in windows_df.columns:
                detected_windows = windows_df['window_mean'].values
            elif 'windows_run_0' in windows_df.columns:
                detected_windows = windows_df['windows_run_0'].values
            else:
                window_cols = [c for c in windows_df.columns if c.startswith('windows')]
                if window_cols:
                    detected_windows = windows_df[window_cols[0]].values
                else:
                    ax.text(0.5, 0.5, 'No window data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(config_name)
                    continue

            # Compute true windows
            n_timepoints = len(detected_windows)
            true_windows = compute_true_windows(n_timepoints, breakpoints)

            # Time series overlay (single panel)
            time_index = np.arange(n_timepoints)
            ax.plot(time_index, true_windows, label='True Window',
                   color='green', linewidth=2, alpha=0.7)
            ax.plot(time_index, detected_windows, label='Detected Window',
                   color='blue', linewidth=1.5, alpha=0.7)

            # Mark breakpoints
            for bp in breakpoints:
                ax.axvline(x=bp, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.5)

            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Window Size')
            ax.set_title(f'{config_name}\nTrue vs Detected Over Time')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

        # Hide unused subplots
        for idx in range(n_configs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        save_path = save_dir / f'true_vs_detected_N{n0_val:03d}.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize robustness benchmarking results'
    )
    parser.add_argument(
        '--test-type',
        type=str,
        required=True,
        choices=['lpa_sensitivity', 'bootstrap_ci'],
        help='Type of robustness test'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., piecewise_ar3)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='examples/results/robustness',
        help='Root robustness results directory'
    )

    args = parser.parse_args()

    # Build paths
    results_dir = Path(args.results_dir) / args.test_type
    save_dir = results_dir / 'figures' / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Robustness Benchmark Visualization")
    print(f"Test Type: {args.test_type}")
    print(f"Dataset: {args.dataset}")
    print("="*80)
    print(f"Loading data from: {results_dir}")

    # Load data
    data = load_robustness_benchmark_data(results_dir, args.dataset)

    print(f"Found {len(data['configs'])} configurations")
    print(f"Saving figures to: {save_dir}")
    print("="*80)

    # Define breakpoints for known datasets
    breakpoints_map = {
        'piecewise_ar3': [500, 1000],
        'arx_rotating': [500, 1000],
        'trend_season': [500, 1000],
        'spike_process': [750],
        'garch_regime': [750]
    }
    breakpoints = breakpoints_map.get(args.dataset, None)

    # Generate plots
    print("\nGenerating visualizations...")

    try:
        plot_window_evolution_per_config(data, args.dataset, save_dir, breakpoints)
    except Exception as e:
        print(f"Error in window evolution plots: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_window_statistics_comparison(data, save_dir)
    except Exception as e:
        print(f"Error in window statistics comparison: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_faithfulness_across_configs(data, save_dir)
    except Exception as e:
        print(f"Error in faithfulness comparison: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_ablation_across_configs(data, save_dir)
    except Exception as e:
        print(f"Error in ablation comparison: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_true_vs_detected_by_n0(data, args.dataset, save_dir, breakpoints)
    except Exception as e:
        print(f"Error in true vs detected window comparison: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Visualization Complete!")
    print(f"All figures saved to: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
