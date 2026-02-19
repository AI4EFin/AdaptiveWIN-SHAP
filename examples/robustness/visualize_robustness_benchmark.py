"""
Visualization script for robustness benchmarking results.

Adapts benchmark_viz.py for robustness test results, showing:
- Faithfulness/ablation metrics across parameter configurations
- Window size evolution over time for each configuration
- Comparison of performance vs parameter settings

Usage:
    # Visualize LPA sensitivity results with geometric growth
    python examples/robustness/visualize_robustness_benchmark.py --test-type lpa_sensitivity --dataset piecewise_ar3 --growth geometric

    # Visualize all growth strategies
    python examples/robustness/visualize_robustness_benchmark.py --test-type lpa_sensitivity --dataset piecewise_ar3

    # Visualize bootstrap CI results
    python examples/robustness/visualize_robustness_benchmark.py --test-type bootstrap_ci --dataset piecewise_ar3
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


def load_robustness_benchmark_data(results_dir, dataset_name, growth_strategy=None):
    """
    Load robustness benchmark results.

    Parameters
    ----------
    results_dir : Path
        Root robustness results directory (e.g., lpa_sensitivity/ or bootstrap_ci/)
    dataset_name : str
        Name of dataset
    growth_strategy : str, optional
        Window growth strategy to filter (geometric only, kept for backwards compatibility)

    Returns
    -------
    dict with keys:
        - 'summary': aggregated results DataFrame
        - 'configs': list of parameter configurations
        - 'windows': dict mapping config to windows DataFrame
        - 'shap': dict mapping config to SHAP results DataFrame
        - 'growth_strategy': str or None
    """
    data = {'growth_strategy': growth_strategy}

    dataset_dir = results_dir / dataset_name

    # Load summary results
    summary_file = dataset_dir / 'sensitivity_summary.csv'
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        # Filter by growth strategy if specified
        if growth_strategy and 'growth' in df.columns:
            df = df[df['growth'] == growth_strategy].copy()
        data['summary'] = df
    else:
        # Try global results.csv filtered by dataset
        global_results = results_dir / 'results.csv'
        if global_results.exists():
            df = pd.read_csv(global_results)
            df = df[df['dataset'] == dataset_name].copy()
            # Filter by growth strategy if specified
            if growth_strategy and 'growth' in df.columns:
                df = df[df['growth'] == growth_strategy].copy()
            data['summary'] = df

    # Find all parameter configuration directories
    data['configs'] = []
    data['windows'] = {}
    data['shap'] = {}
    data['benchmark_results'] = {}

    # Determine search directories based on growth strategy
    if growth_strategy:
        # Look in growth-specific subdirectory
        search_dirs = [dataset_dir / growth_strategy]
    else:
        # Look in both main directory and growth subdirectories
        search_dirs = [dataset_dir]
        for growth_dir in ['geometric']:
            potential_dir = dataset_dir / growth_dir
            if potential_dir.exists():
                search_dirs.append(potential_dir)

    # Look for temp directories in all search locations
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for temp_dir in search_dir.glob('temp_*'):
            if not temp_dir.is_dir():
                continue

            config_name = temp_dir.name.replace('temp_', '')

            # Skip if growth strategy filter doesn't match
            if growth_strategy and 'growth' in config_name:
                if growth_strategy not in config_name:
                    continue

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
            def make_label(row):
                n0 = int(row['N0'])
                alpha = row.get('alpha', 0)
                # Support both mc_reps (new) and num_bootstrap (legacy)
                mc = int(row.get('mc_reps', row.get('num_bootstrap', 0)))
                lam = row.get('penalty_factor', 0.25)
                return f"N{n0}_α{alpha:.2f}_M{mc}_λ{lam:.2f}"
            config_labels = summary.apply(make_label, axis=1)
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
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
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
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
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
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
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


def plot_true_vs_detected_by_n0(data, dataset_name, save_dir, breakpoints=None, rolling_mean_size=20):
    """
    Plot true vs detected window sizes, grouped by N0 parameter.

    Creates one figure per N0 with 3 rows (one per alpha) × 3 columns (one per bootstrap).
    Each row has title $\\alpha = <value>$, each subplot has title $B = <value>$.

    Parameters
    ----------
    rolling_mean_size : int
        Window size for rolling mean calculation (default: 20)
    """
    if not data['windows'] or breakpoints is None:
        print("No window data or breakpoints - skipping true vs detected plots")
        return

    # Parse N0, alpha and num_bootstrap values from config names
    # Structure: n0_configs[n0][alpha] = [(config_name, bootstrap_val), ...]
    n0_configs = {}
    for config_name in data['configs']:
        if config_name not in data['windows']:
            continue

        # Extract N0, alpha and num_bootstrap from config name
        # Format: N{N0}_alpha{alpha}_num_bootstrap{B} or similar
        try:
            parts = config_name.split('_')

            # Find N0 value
            n0_part = [p for p in parts if p.startswith('N') and p[1:].isdigit()]
            if n0_part:
                n0_val = int(n0_part[0].replace('N', ''))
            else:
                continue

            # Find alpha value
            alpha_part = [p for p in parts if p.startswith('alpha')]
            if alpha_part:
                alpha_val = float(alpha_part[0].replace('alpha', ''))
            else:
                continue

            # Find mc_reps or num_bootstrap value
            mc_reps_part = [p for p in parts if 'mcreps' in p.lower() or 'mc_reps' in p.lower()]
            bootstrap_part = [p for p in parts if 'bootstrap' in p.lower()]
            if mc_reps_part:
                bootstrap_val = int(mc_reps_part[0].replace('mcreps', '').replace('mc_reps', ''))
            elif bootstrap_part:
                bootstrap_val = int(bootstrap_part[0].replace('numbootstrap', '').replace('bootstrap', ''))
            else:
                bootstrap_val = 0

            # Find penalty_factor (lambda) value
            lambda_part = [p for p in parts if 'penaltyfactor' in p.lower() or 'penalty_factor' in p.lower()]
            if lambda_part:
                lambda_val = float(lambda_part[0].replace('penaltyfactor', '').replace('penalty_factor', ''))
            else:
                lambda_val = 0.25  # default

            if n0_val not in n0_configs:
                n0_configs[n0_val] = {}
            if alpha_val not in n0_configs[n0_val]:
                n0_configs[n0_val][alpha_val] = []
            n0_configs[n0_val][alpha_val].append((config_name, bootstrap_val, lambda_val))
        except:
            continue

    if not n0_configs:
        print("Could not parse N0/alpha values from config names")
        return

    # Sort N0 values
    n0_values = sorted(n0_configs.keys())

    # Create one plot per N0 value
    for n0_val in n0_values:
        alpha_dict = n0_configs[n0_val]
        alpha_values = sorted(alpha_dict.keys())
        n_rows = len(alpha_values)

        # Determine number of columns (max bootstrap values across alphas)
        n_cols = max(len(configs) for configs in alpha_dict.values())
        n_cols = min(3, n_cols)  # Cap at 3 columns

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), sharey=True)
        fig.suptitle(rf'$n_0 = {n0_val}$', fontsize=16, fontweight='bold')

        # Ensure axes is 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        lines = []
        labels = []

        for row_idx, alpha_val in enumerate(alpha_values):
            configs_for_alpha = alpha_dict[alpha_val]
            # Sort by bootstrap value, then by lambda
            configs_for_alpha.sort(key=lambda x: (x[1], x[2]))

            for col_idx, (config_name, bootstrap_val, lambda_val) in enumerate(configs_for_alpha):
                if col_idx >= n_cols:
                    break

                ax = axes[row_idx, col_idx]
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
                        ax.set_title(rf'$M = {bootstrap_val}$, $\lambda = {lambda_val:.2f}$')
                        continue

                # Compute true windows
                n_timepoints = len(detected_windows)
                true_windows = compute_true_windows(n_timepoints, breakpoints)

                # Time series overlay
                time_index = np.arange(n_timepoints)
                l1, = ax.plot(time_index, true_windows, color='green', linewidth=2, alpha=0.7)
                l2, = ax.plot(time_index, detected_windows, color='#3B75AF', linewidth=1.5, alpha=0.7)

                # Add rolling mean for detected windows
                rolling_mean = pd.Series(detected_windows).rolling(window=rolling_mean_size, center=True).mean()
                l3, = ax.plot(time_index, rolling_mean, linewidth=2, alpha=0.9, color='red')

                # Mark breakpoints
                for bp in breakpoints:
                    ax.axvline(x=bp, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

                # Collect legend handles from first subplot only
                if row_idx == 0 and col_idx == 0:
                    lines = [l1, l2, l3]
                    labels = ['True Window', 'Detected Window', f'Rolling Mean ({rolling_mean_size})']

                # Set labels
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Timepoint')
                if col_idx == 0:
                    ax.set_ylabel(rf'$\alpha = {alpha_val}$', fontsize=12)

                # Set subplot title (mc_reps and lambda)
                ax.set_title(rf'$M = {bootstrap_val}$, $\lambda = {lambda_val:.2f}$')

            # Hide unused subplots in this row
            for col_idx in range(len(configs_for_alpha), n_cols):
                axes[row_idx, col_idx].set_visible(False)

        # Add single centered legend at the bottom
        fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  ncol=3, frameon=False, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # Make room for legend

        save_path = save_dir / f'true_vs_detected_N{n0_val:03d}.png'
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
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
    parser.add_argument(
        '--growth',
        type=str,
        default=None,
        help='Deprecated: growth strategy filter (geometric only supported now)'
    )
    parser.add_argument(
        '--rolling-mean-size',
        type=int,
        default=20,
        help='Window size for rolling mean calculation (default: 20)'
    )

    args = parser.parse_args()

    # Build paths
    results_dir = Path(args.results_dir) / args.test_type

    # Include growth strategy in output path if specified
    if args.growth:
        save_dir = results_dir / 'figures' / args.dataset / args.growth
    else:
        save_dir = results_dir / 'figures' / args.dataset

    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Robustness Benchmark Visualization")
    print(f"Test Type: {args.test_type}")
    print(f"Dataset: {args.dataset}")
    if args.growth:
        print(f"Growth Strategy: {args.growth}")
    print("="*80)
    print(f"Loading data from: {results_dir}")

    # Load data
    data = load_robustness_benchmark_data(results_dir, args.dataset, args.growth)

    print(f"Found {len(data['configs'])} configurations")
    print(f"Saving figures to: {save_dir}")
    print("="*80)

    # Define breakpoints for known datasets
    breakpoints_map = {
        'piecewise_ar3': [500, 1000],
        'arx_rotating': [500, 1000],
        'trend_season': [500, 1000],
        'piecewise_ar3_long': [500, 900, 1200, 1800, 2600, 3600],
        'arx_rotating_long': [500, 900, 1200, 1800, 2600, 3600]
    }
    breakpoints = breakpoints_map.get(args.dataset, None)

    # Create display name with growth strategy if specified
    if args.growth:
        dataset_display = f"{args.dataset} ({args.growth})"
    else:
        dataset_display = args.dataset

    # Generate plots
    print("\nGenerating visualizations...")

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
        plot_true_vs_detected_by_n0(data, dataset_display, save_dir, breakpoints, args.rolling_mean_size)
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
