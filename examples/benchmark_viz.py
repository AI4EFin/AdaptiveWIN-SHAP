"""
Visualization script for benchmarking results.

Creates comprehensive visualizations comparing:
- Vanilla SHAP (vanilla kernel SHAP on global model)
- Rolling Window SHAP (fixed window size)
- Adaptive SHAP (adaptive windows from change detection)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style - clean for presentations
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0


def load_benchmark_data(results_dir):
    """Load all benchmark result files."""
    data = {}

    # Load summary
    summary_path = os.path.join(results_dir, 'benchmark_summary.csv')
    if os.path.exists(summary_path):
        data['summary'] = pd.read_csv(summary_path)

    # Load individual method results
    # Load global, timeshap, and adaptive variants
    for method in ['global_shap', 'timeshap', 'adaptive_shap', 'adaptive_shap_rolling_mean']:
        result_path = os.path.join(results_dir, f'{method}_results.csv')
        if os.path.exists(result_path):
            data[method] = pd.read_csv(result_path)

    # Load rolling window (base only, no max/mean variants)
    result_path = os.path.join(results_dir, 'rolling_shap_results.csv')
    if os.path.exists(result_path):
        data['rolling_shap'] = pd.read_csv(result_path)

    # Load adaptive SHAP max and mean variants
    for suffix in ['_max', '_mean']:
        method_key = f'adaptive_shap{suffix}'
        result_path = os.path.join(results_dir, f'{method_key}_results.csv')
        if os.path.exists(result_path):
            data[method_key] = pd.read_csv(result_path)

    # Load config
    config_path = os.path.join(results_dir, 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)

    # Load windows if available
    windows_path = os.path.join(results_dir, 'temp_windows.csv')
    if os.path.exists(windows_path):
        data['windows'] = pd.read_csv(windows_path)

    # Load true importances if available (from dataset directory)
    if 'config' in data:
        dataset_path = data['config'].get('dataset', '')
        if 'simulated' in dataset_path:
            # Extract dataset name from path
            # e.g., "examples/datasets/simulated/arx_rotating/data.csv" -> "arx_rotating"
            parts = dataset_path.split('/')
            if len(parts) >= 4:
                dataset_name = parts[-2]
                true_imp_path = f"examples/datasets/simulated/{dataset_name}/true_importances.csv"
                if os.path.exists(true_imp_path):
                    data['true_importances'] = pd.read_csv(true_imp_path)
                    print(f"Loaded true importances from: {true_imp_path}")

    return data


def plot_faithfulness_comparison(data, save_dir):
    """Compare faithfulness scores across methods."""
    if 'summary' not in data:
        print("No summary data available - skipping faithfulness comparison")
        return

    summary = data['summary']

    # Filter for faithfulness metrics only
    if 'metric_type' in summary.columns:
        faith_summary = summary[summary['metric_type'] == 'faithfulness'].copy()
    else:
        faith_summary = summary.copy()

    if len(faith_summary) == 0:
        print("No faithfulness metrics found in summary - skipping faithfulness comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Faithfulness Comparison Across Methods', fontsize=14, fontweight='bold')

    eval_types = ['prtb']
    percentiles = ['p90', 'p50']

    for i, eval_type in enumerate(eval_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[j]

            eval_key = f'{eval_type}_{percentile}'
            subset = faith_summary[faith_summary['evaluation'] == eval_key]

            if len(subset) > 0:
                methods = []
                scores = []
                colors = []
                color_map = {
                    'global_shap': '#1f77b4',
                    'timeshap': '#e377c2',
                    'rolling_shap': '#ff7f0e',
                    'adaptive_shap_max': '#9467bd',
                    'adaptive_shap_mean': '#8c564b',
                    'adaptive_shap': '#2ca02c',
                    'adaptive_shap_rolling_mean': '#d62728'
                }

                # Dynamically get all methods in the data
                available_methods = subset['method'].unique()
                method_order = ['global_shap', 'timeshap', 'rolling_shap', 'adaptive_shap_max', 'adaptive_shap_mean', 'adaptive_shap', 'adaptive_shap_rolling_mean']

                for method in method_order:
                    if method in available_methods:
                        method_data = subset[subset['method'] == method]
                        if len(method_data) > 0:
                            # Format method name for display
                            if method == 'adaptive_shap_max':
                                display_name = 'Adaptive (Max)'
                            elif method == 'adaptive_shap_mean':
                                display_name = 'Adaptive (Mean)'
                            elif method == 'adaptive_shap_rolling_mean':
                                display_name = 'Adaptive (Smooth)'
                            else:
                                display_name = method.replace('_', ' ').title()

                            methods.append(display_name)
                            scores.append(method_data['score'].values[0])
                            colors.append(color_map.get(method, '#888888'))

                if methods:
                    bars = ax.bar(range(len(methods)), scores, color=colors, alpha=0.7, edgecolor='black')
                    ax.set_xticks(range(len(methods)))
                    ax.set_xticklabels(methods, rotation=45, ha='right')
                    ax.set_ylabel('Faithfulness Score')
                    ax.set_title(f'{eval_type.upper()} - {percentile.upper()}')
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.4f}',
                               ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'faithfulness_comparison.png'), bbox_inches='tight')
    print(f"Saved: faithfulness_comparison.png")
    plt.close()


def plot_ablation_comparison(data, save_dir):
    """Compare ablation scores across methods."""
    if 'summary' not in data:
        print("No summary data available - skipping ablation comparison")
        return

    summary = data['summary']

    # Filter for ablation metrics only
    if 'metric_type' in summary.columns:
        ablation_summary = summary[summary['metric_type'] == 'ablation'].copy()
    else:
        print("No ablation metrics found in summary - skipping ablation comparison")
        return

    if len(ablation_summary) == 0:
        print("No ablation metrics found in summary - skipping ablation comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ablation Score Comparison Across Methods', fontsize=14, fontweight='bold')

    ablation_types = ['mif', 'lif']
    percentiles = ['p90', 'p50']

    for i, ablation_type in enumerate(ablation_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[i, j]

            eval_key = f'{ablation_type}_{percentile}'
            subset = ablation_summary[ablation_summary['evaluation'] == eval_key]

            if len(subset) > 0:
                methods = []
                scores = []
                colors = []
                color_map = {
                    'global_shap': '#1f77b4',
                    'timeshap': '#e377c2',
                    'rolling_shap': '#ff7f0e',
                    'adaptive_shap_max': '#9467bd',
                    'adaptive_shap_mean': '#8c564b',
                    'adaptive_shap': '#2ca02c',
                    'adaptive_shap_rolling_mean': '#d62728'
                }

                # Dynamically get all methods in the data
                available_methods = subset['method'].unique()
                method_order = ['global_shap', 'timeshap', 'rolling_shap', 'adaptive_shap_max', 'adaptive_shap_mean', 'adaptive_shap', 'adaptive_shap_rolling_mean']

                for method in method_order:
                    if method in available_methods:
                        method_data = subset[subset['method'] == method]
                        if len(method_data) > 0:
                            # Format method name for display
                            if method == 'adaptive_shap_max':
                                display_name = 'Adaptive (Max)'
                            elif method == 'adaptive_shap_mean':
                                display_name = 'Adaptive (Mean)'
                            elif method == 'adaptive_shap_rolling_mean':
                                display_name = 'Adaptive (Smooth)'
                            else:
                                display_name = method.replace('_', ' ').title()

                            methods.append(display_name)
                            scores.append(method_data['score'].values[0])
                            colors.append(color_map.get(method, '#888888'))

                if methods:
                    bars = ax.bar(range(len(methods)), scores, color=colors, alpha=0.7, edgecolor='black')
                    ax.set_xticks(range(len(methods)))
                    ax.set_xticklabels(methods, rotation=45, ha='right')
                    ax.set_ylabel('Ablation Score')
                    ax.set_title(f'{ablation_type.upper()} - {percentile.upper()}')
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.4f}',
                               ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'), bbox_inches='tight')
    print(f"Saved: ablation_comparison.png")
    plt.close()


def plot_ablation_mif_vs_lif(data, save_dir):
    """Compare MIF vs LIF ablation scores for each method."""
    if 'summary' not in data:
        print("No summary data available - skipping MIF vs LIF comparison")
        return

    summary = data['summary']

    if 'metric_type' not in summary.columns:
        print("No metric_type column in summary - skipping MIF vs LIF comparison")
        return

    ablation_summary = summary[summary['metric_type'] == 'ablation'].copy()

    if len(ablation_summary) == 0:
        print("No ablation metrics found - skipping MIF vs LIF comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('MIF vs LIF Ablation Comparison (Higher MIF/LIF ratio = Better)',
                 fontsize=14, fontweight='bold')

    percentiles = ['p90', 'p50']
    color_map = {
        'global_shap': '#1f77b4',
        'timeshap': '#e377c2',
        'rolling_shap': '#ff7f0e',
        'adaptive_shap_max': '#9467bd',
        'adaptive_shap_mean': '#8c564b',
        'adaptive_shap': '#2ca02c',
        'adaptive_shap_rolling_mean': '#d62728'
    }

    # Define method order (all methods)
    method_order = ['global_shap', 'timeshap', 'rolling_shap', 'adaptive_shap_max', 'adaptive_shap_mean',
                    'adaptive_shap', 'adaptive_shap_rolling_mean']

    for j, percentile in enumerate(percentiles):
        ax = axes[j]

        methods = []
        mif_scores = []
        lif_scores = []
        colors = []

        for method in method_order:
            # Get MIF score
            mif_key = f'mif_{percentile}'
            mif_data = ablation_summary[(ablation_summary['method'] == method) &
                                        (ablation_summary['evaluation'] == mif_key)]

            # Get LIF score
            lif_key = f'lif_{percentile}'
            lif_data = ablation_summary[(ablation_summary['method'] == method) &
                                        (ablation_summary['evaluation'] == lif_key)]

            if len(mif_data) > 0 and len(lif_data) > 0:
                # Format method name for display
                if method == 'adaptive_shap_max':
                    display_name = 'Adaptive (Max)'
                elif method == 'adaptive_shap_mean':
                    display_name = 'Adaptive (Mean)'
                elif method == 'adaptive_shap_rolling_mean':
                    display_name = 'Adaptive (Smooth)'
                else:
                    display_name = method.replace('_', ' ').title()

                methods.append(display_name)
                mif_scores.append(mif_data['score'].values[0])
                lif_scores.append(lif_data['score'].values[0])
                colors.append(color_map.get(method, '#888888'))

        if methods:
            x = np.arange(len(methods))
            width = 0.35

            bars1 = ax.bar(x - width/2, mif_scores, width, label='MIF (Most Important First)',
                          color=colors, alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x + width/2, lif_scores, width, label='LIF (Least Important First)',
                          color=colors, alpha=0.4, edgecolor='black', hatch='//')

            ax.set_ylabel('Ablation Score')
            ax.set_title(f'Percentile: {percentile.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=7)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_mif_vs_lif.png'), bbox_inches='tight')
    print(f"Saved: ablation_mif_vs_lif.png")
    plt.close()


def plot_shap_over_time(data, save_dir):
    """Plot SHAP values over time for each method."""
    methods = [
        ('global_shap', 'Vanilla SHAP', 'shap_lag_t'),
        ('timeshap', 'TimeShap', 'shap_lag_t'),
        ('rolling_shap', 'Rolling Window SHAP', 'shap_lag_t'),
        ('adaptive_shap', 'Adaptive SHAP', 'shap_lag_t')
    ]

    # Filter to only available methods
    available_methods = [(k, n, p) for k, n, p in methods if k in data]
    n_methods = len(available_methods)

    if n_methods == 0:
        print("No methods available for SHAP over time plot")
        return

    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 4 * n_methods))
    fig.suptitle('SHAP Values Over Time', fontsize=14, fontweight='bold')

    # Handle case when there's only one method
    if n_methods == 1:
        axes = [axes]

    for idx, (method_key, method_name, shap_prefix) in enumerate(available_methods):
        ax = axes[idx]
        df = data[method_key]

        # Get SHAP columns
        shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]

        if len(shap_cols) == 0:
            ax.text(0.5, 0.5, f'No SHAP data available for {method_name}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot each lag
        for col in shap_cols:
            lag_num = col.split('_')[-1]
            ax.plot(df['end_index'], df[col], label=f'Lag {lag_num}', alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Time Index')
        ax.set_ylabel('|SHAP Value|')
        ax.set_title(method_name)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_over_time.png'), bbox_inches='tight')
    print(f"Saved: shap_over_time.png")
    plt.close()


def plot_shap_heatmaps(data, save_dir):
    """Create heatmaps of SHAP values for each method."""
    methods = [
        ('global_shap', 'Vanilla SHAP', 'shap_lag_t'),
        ('timeshap', 'TimeShap', 'shap_lag_t'),
        ('rolling_shap', 'Rolling (Fixed 100)', 'shap_lag_t'),
        ('adaptive_shap_max', 'Adaptive (Max)', 'shap_lag_t'),
        ('adaptive_shap_mean', 'Adaptive (Mean)', 'shap_lag_t'),
        ('adaptive_shap', 'Adaptive SHAP', 'shap_lag_t'),
        ('adaptive_shap_rolling_mean', 'Adaptive (Smooth)', 'shap_lag_t')
    ]

    # Filter to only available methods
    available_methods = [(k, n, p) for k, n, p in methods if k in data]
    n_methods = len(available_methods)

    if n_methods == 0:
        print("No methods available for heatmap plotting")
        return

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle('SHAP Value Heatmaps (Time × Lags)', fontsize=14, fontweight='bold')

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, (method_key, method_name, shap_prefix) in enumerate(available_methods):
        ax = axes_flat[idx]
        df = data[method_key]

        # Get SHAP columns
        shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]

        if len(shap_cols) == 0:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center')
            ax.set_title(method_name)
            continue

        # Sort columns by lag number (t-1, t-2, t-3, ...) to ensure correct ordering
        # Extract lag numbers and sort
        def extract_lag(col_name):
            # Extract the number after 't-' (e.g., 'shap_lag_t-1' -> 1)
            parts = col_name.split('t-')
            if len(parts) > 1:
                return int(parts[-1])
            return 0

        shap_cols = sorted(shap_cols, key=extract_lag)

        # Create heatmap data
        shap_matrix = df[shap_cols].values.T

        # Plot heatmap
        im = ax.imshow(shap_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Time Index (every 10th shown)')
        ax.set_ylabel('Lag')
        ax.set_title(method_name)

        # Set y-axis labels
        ax.set_yticks(range(len(shap_cols)))
        # ax.set_yticklabels([f't-{i+1}' for i in range(len(shap_cols))])
        ax.set_yticklabels(shap_cols)

        # Set x-axis labels (show every 100th index)
        if len(df) > 10:
            step = max(len(df) // 10, 1)
            ax.set_xticks(range(0, len(df), step))
            ax.set_xticklabels([f'{int(df.iloc[i]["end_index"])}' for i in range(0, len(df), step)],
                              rotation=45)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='|SHAP Value|')

    # Hide any unused subplots
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_heatmaps.png'), bbox_inches='tight')
    print(f"Saved: shap_heatmaps.png")
    plt.close()


def plot_prediction_comparison(data, save_dir):
    """Compare predictions across methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prediction Comparison Across Methods', fontsize=14, fontweight='bold')

    # Time series comparison
    ax = axes[0, 0]
    for method_key, label in [('global_shap', 'Vanilla'),
                               ('rolling_shap', 'Rolling'),
                               ('adaptive_shap', 'Adaptive')]:
        if method_key in data:
            df = data[method_key]
            ax.plot(df['end_index'], df['y_hat'], label=label, alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Time Index')
    ax.set_ylabel('Prediction')
    ax.set_title('Predictions Over Time')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    # Scatter plots
    scatter_pairs = [
        (('global_shap', 'rolling_shap'), 'Vanilla vs Rolling', axes[0, 1]),
        (('global_shap', 'adaptive_shap'), 'Vanilla vs Adaptive', axes[1, 0]),
        (('rolling_shap', 'adaptive_shap'), 'Rolling vs Adaptive', axes[1, 1])
    ]

    for (method1, method2), title, ax in scatter_pairs:
        if method1 in data and method2 in data:
            df1 = data[method1]
            df2 = data[method2]

            # Align by end_index
            merged = pd.merge(df1[['end_index', 'y_hat']],
                            df2[['end_index', 'y_hat']],
                            on='end_index',
                            suffixes=('_1', '_2'))

            ax.scatter(merged['y_hat_1'], merged['y_hat_2'], alpha=0.5, s=20)

            # Add diagonal line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Calculate correlation
            if np.std(merged['y_hat_1']) > 0 and np.std(merged['y_hat_2']) > 0:
                corr = np.corrcoef(merged['y_hat_1'], merged['y_hat_2'])[0, 1]
            else:
                corr = 0.0
            ax.text(0.05, 0.95, f'r = {corr:.4f}',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(method1.replace('_', ' ').title())
            ax.set_ylabel(method2.replace('_', ' ').title())
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'), bbox_inches='tight')
    print(f"Saved: prediction_comparison.png")
    plt.close()


def plot_window_size_analysis(data, save_dir):
    """Analyze adaptive window sizes."""
    if 'windows' not in data or 'adaptive_shap' not in data:
        print("Window data not available for analysis")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Adaptive Window Size Analysis', fontsize=14, fontweight='bold')

    windows_df = data['windows']
    adaptive_df = data['adaptive_shap']

    # Plot window sizes over time
    ax = axes[0]
    if 'window_len' in adaptive_df.columns:
        ax.plot(adaptive_df['end_index'], adaptive_df['window_len'],
               label='Adaptive Window Size', linewidth=2)
    elif len(windows_df) == len(adaptive_df):
        ax.plot(adaptive_df['end_index'], windows_df['windows'].values,
               label='Adaptive Window Size', linewidth=2)

    # Add fixed window size reference
    if 'config' in data:
        fixed_size = data['config'].get('rolling_window_size', 100)
        ax.axhline(y=fixed_size, color='r', linestyle='--',
                  label=f'Fixed Window Size ({fixed_size})', linewidth=2)

    ax.set_xlabel('Time Index')
    ax.set_ylabel('Window Size')
    ax.set_title('Window Size Over Time')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    # Plot window size distribution
    ax = axes[1]
    if 'window_len' in adaptive_df.columns:
        window_sizes = adaptive_df['window_len'].values
    elif len(windows_df) == len(adaptive_df):
        window_sizes = windows_df['windows'].values
    else:
        window_sizes = []

    if len(window_sizes) > 0:
        ax.hist(window_sizes, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(window_sizes), color='r', linestyle='--',
                  label=f'Mean: {np.mean(window_sizes):.1f}', linewidth=2)
        ax.axvline(x=np.median(window_sizes), color='g', linestyle='--',
                  label=f'Median: {np.median(window_sizes):.1f}', linewidth=2)
        ax.set_xlabel('Window Size')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Adaptive Window Sizes')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'window_size_analysis.png'), bbox_inches='tight')
    print(f"Saved: window_size_analysis.png")
    plt.close()


def plot_temporal_faithfulness(data, save_dir):
    """Plot faithfulness scores over time for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Faithfulness Scores Over Time', fontsize=14, fontweight='bold')

    eval_types = ['prtb']
    percentiles = ['p90', 'p50']

    # Store handles and labels for shared legend
    handles, labels = None, None

    for i, eval_type in enumerate(eval_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[j]

            eval_key = f'{eval_type}_{percentile}'
            col_name = f'faithfulness_{eval_key}'

            # Plot for each method
            for method_key, label, color in [
                ('global_shap', 'Vanilla', '#1f77b4'),
                ('rolling_shap', 'Rolling (100)', '#ff7f0e'),
                ('adaptive_shap_max', 'Adaptive (Max)', '#9467bd'),
                ('adaptive_shap_mean', 'Adaptive (Mean)', '#8c564b'),
                ('adaptive_shap', 'Adaptive', '#2ca02c'),
                ('adaptive_shap_rolling_mean', 'Adaptive (Smooth)', '#d62728')
            ]:
                if method_key in data:
                    df = data[method_key]
                    if col_name in df.columns:
                        # Plot rolling mean only (window=20)
                        rolling_mean = df[col_name].rolling(window=20, center=True).mean()
                        line, = ax.plot(df['end_index'], rolling_mean,
                                       linewidth=2, color=color, alpha=0.8, label=label)

                        # Capture handles and labels from first subplot
                        if i == 0 and j == 0:
                            if handles is None:
                                handles = []
                                labels = []
                            handles.append(line)
                            labels.append(label)

            ax.set_xlabel('Time Index')
            ax.set_ylabel('Faithfulness Score')
            ax.set_title(f'{eval_type.upper()} - {percentile.upper()}')

    # Add single legend at the bottom of the figure
    if handles is not None:
        fig.legend(handles, labels, loc='lower center', ncol=6,
                  bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend
    plt.savefig(os.path.join(save_dir, 'temporal_faithfulness.png'), bbox_inches='tight')
    print(f"Saved: temporal_faithfulness.png")
    plt.close()


def plot_temporal_ablation(data, save_dir):
    """Plot ablation scores over time for each method."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ablation Scores Over Time', fontsize=14, fontweight='bold')

    ablation_types = ['mif', 'lif']
    percentiles = ['p90', 'p50']

    # Store handles and labels for shared legend
    handles, labels = None, None

    for i, ablation_type in enumerate(ablation_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[i, j]

            eval_key = f'{ablation_type}_{percentile}'
            col_name = f'ablation_{eval_key}'

            # Plot for each method
            for method_key, label, color in [
                ('global_shap', 'Vanilla', '#1f77b4'),
                ('rolling_shap', 'Rolling (100)', '#ff7f0e'),
                ('adaptive_shap_max', 'Adaptive (Max)', '#9467bd'),
                ('adaptive_shap_mean', 'Adaptive (Mean)', '#8c564b'),
                ('adaptive_shap', 'Adaptive', '#2ca02c'),
                ('adaptive_shap_rolling_mean', 'Adaptive (Smooth)', '#d62728')
            ]:
                if method_key in data:
                    df = data[method_key]
                    if col_name in df.columns:
                        # Plot rolling mean only (window=20)
                        rolling_mean = df[col_name].rolling(window=20, center=True).mean()
                        line, = ax.plot(df['end_index'], rolling_mean,
                                       linewidth=2, color=color, alpha=0.8, label=label)

                        # Capture handles and labels from first subplot
                        if i == 0 and j == 0:
                            if handles is None:
                                handles = []
                                labels = []
                            handles.append(line)
                            labels.append(label)

            ax.set_xlabel('Time Index')
            ax.set_ylabel('Ablation Score')
            ax.set_title(f'{ablation_type.upper()} - {percentile.upper()}')

    # Add single legend at the bottom of the figure
    if handles is not None:
        fig.legend(handles, labels, loc='lower center', ncol=6,
                  bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend
    plt.savefig(os.path.join(save_dir, 'temporal_ablation.png'), bbox_inches='tight')
    print(f"Saved: temporal_ablation.png")
    plt.close()


def create_summary_dashboard(data, save_dir):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Benchmark Summary Dashboard', fontsize=16, fontweight='bold')

    # 1. Faithfulness and Ablation scores comparison
    ax1 = fig.add_subplot(gs[0, :2])
    if 'summary' in data:
        summary = data['summary']

        # Prepare data for both metrics
        if 'metric_type' in summary.columns:
            # New format with metric_type
            faith_data = summary[(summary['metric_type'] == 'faithfulness') &
                                 (summary['evaluation'] == 'prtb_p50')]
            ablation_data = summary[(summary['metric_type'] == 'ablation') &
                                    (summary['evaluation'] == 'mif_p50')]
        else:
            # Old format
            faith_data = summary[summary['evaluation'] == 'prtb_p50']
            ablation_data = pd.DataFrame()

        methods = []
        faith_scores = []
        ablation_scores = []
        colors = {
            'global_shap': '#1f77b4',
        'timeshap': '#e377c2',
            'rolling_shap': '#ff7f0e',
            'adaptive_shap_max': '#9467bd',
            'adaptive_shap_mean': '#8c564b',
            'adaptive_shap': '#2ca02c',
            'adaptive_shap_rolling_mean': '#d62728'
        }

        for method in ['global_shap', 'rolling_shap', 'adaptive_shap_max', 'adaptive_shap_mean', 'adaptive_shap', 'adaptive_shap_rolling_mean']:
            if len(faith_data) > 0:
                m_faith = faith_data[faith_data['method'] == method]
                if len(m_faith) > 0:
                    methods.append(method.replace('_', ' ').title())
                    faith_scores.append(m_faith['score'].values[0] if 'score' in m_faith.columns
                                       else m_faith['faithfulness_score'].values[0])

                    # Get ablation score if available
                    if len(ablation_data) > 0:
                        m_abl = ablation_data[ablation_data['method'] == method]
                        ablation_scores.append(m_abl['score'].values[0] if len(m_abl) > 0 else 0)
                    else:
                        ablation_scores.append(0)

        if methods:
            x = np.arange(len(methods))
            width = 0.35

            # Plot both metrics
            method_colors = [colors.get(m.lower().replace(' ', '_'), '#888888') for m in methods]
            bars1 = ax1.bar(x - width/2, faith_scores, width, label='Faithfulness (PRTB p50)',
                           color=method_colors, alpha=0.7, edgecolor='black')

            if any(ablation_scores):
                bars2 = ax1.bar(x + width/2, ablation_scores, width, label='Ablation (MIF p50)',
                               color=method_colors, alpha=0.5, edgecolor='black', hatch='//')

            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.set_ylabel('Score')
            ax1.set_title('Faithfulness & Ablation Score Comparison')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', fontsize=8)
            if any(ablation_scores):
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}',
                                ha='center', va='bottom', fontsize=8)

    # 2. Method statistics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    stats_data = []
    for method_key, method_name in [('global_shap', 'Vanilla'),
                                     ('rolling_shap', 'Rolling'),
                                     ('adaptive_shap', 'Adaptive')]:
        if method_key in data:
            df = data[method_key]
            shap_cols = [c for c in df.columns if 'shap' in c.lower()]
            if len(shap_cols) > 0:
                mean_shap = df[shap_cols].values.mean()
                stats_data.append([method_name, len(df), f'{mean_shap:.4f}'])

    if stats_data:
        table = ax2.table(cellText=stats_data,
                         colLabels=['Method', 'N Points', 'Mean |SHAP|'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.3, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax2.set_title('Method Statistics', pad=20)

    # 3. SHAP values over time (mini version)
    ax3 = fig.add_subplot(gs[1, :])
    for method_key, label, shap_prefix in [('global_shap', 'Vanilla', 'shap_lag_t'),
                                            ('rolling_shap', 'Rolling', 'shap_lag_t'),
                                            ('adaptive_shap', 'Adaptive', 'shap_lag_t')]:
        if method_key in data:
            df = data[method_key]
            shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]
            if len(shap_cols) > 0:
                total_shap = df[shap_cols].sum(axis=1)
                ax3.plot(df['end_index'], total_shap, label=label, alpha=0.7, linewidth=1.5)

    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Total |SHAP|')
    ax3.set_title('Total SHAP Values Over Time')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    # 4. Prediction scatter (Global vs others)
    ax4 = fig.add_subplot(gs[2, 0])
    if 'global_shap' in data and 'rolling_shap' in data:
        df1 = data['global_shap']
        df2 = data['rolling_shap']
        merged = pd.merge(df1[['end_index', 'y_hat']],
                         df2[['end_index', 'y_hat']],
                         on='end_index', suffixes=('_g', '_r'))
        ax4.scatter(merged['y_hat_g'], merged['y_hat_r'], alpha=0.3, s=10)
        lims = [np.min([ax4.get_xlim(), ax4.get_ylim()]),
                np.max([ax4.get_xlim(), ax4.get_ylim()])]
        ax4.plot(lims, lims, 'r--', alpha=0.75)
        ax4.set_xlabel('Global')
        ax4.set_ylabel('Rolling')
        ax4.set_title('Global vs Rolling Predictions')
    ax5 = fig.add_subplot(gs[2, 1])
    if 'global_shap' in data and 'adaptive_shap' in data:
        df1 = data['global_shap']
        df2 = data['adaptive_shap']
        merged = pd.merge(df1[['end_index', 'y_hat']],
                         df2[['end_index', 'y_hat']],
                         on='end_index', suffixes=('_g', '_a'))
        ax5.scatter(merged['y_hat_g'], merged['y_hat_a'], alpha=0.3, s=10)
        lims = [np.min([ax5.get_xlim(), ax5.get_ylim()]),
                np.max([ax5.get_xlim(), ax5.get_ylim()])]
        ax5.plot(lims, lims, 'r--', alpha=0.75)
        ax5.set_xlabel('Global')
        ax5.set_ylabel('Adaptive')
        ax5.set_title('Global vs Adaptive Predictions')
    # 5. Window size distribution (if adaptive)
    ax6 = fig.add_subplot(gs[2, 2])
    if 'adaptive_shap' in data:
        df = data['adaptive_shap']
        if 'window_len' in df.columns:
            ax6.hist(df['window_len'], bins=20, alpha=0.7, edgecolor='black', color='#2ca02c')
            ax6.axvline(x=df['window_len'].mean(), color='r', linestyle='--',
                       label=f'Mean: {df["window_len"].mean():.1f}')
            ax6.set_xlabel('Window Size')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Adaptive Window Sizes')
            ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.savefig(os.path.join(save_dir, 'summary_dashboard.png'), bbox_inches='tight')
    print(f"Saved: summary_dashboard.png")
    plt.close()


def plot_true_importance_heatmap(data, save_dir):
    """Plot heatmap of true feature importances over time."""
    if 'true_importances' not in data:
        print("No true importances available - skipping true importance heatmap")
        return

    true_imp_df = data['true_importances']

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Ground Truth Feature Importances Over Time', fontsize=14, fontweight='bold')

    # Get importance columns
    imp_cols = [c for c in true_imp_df.columns if c.startswith('true_imp_')]

    if len(imp_cols) == 0:
        print("No importance columns found")
        return

    # Create heatmap data
    true_imp_matrix = true_imp_df[imp_cols].values.T

    # Pad true importances to match SHAP feature count if needed
    # Get number of SHAP features from any available method
    n_shap_features = 0
    for method_key in ['global_shap', 'rolling_shap', 'adaptive_shap']:
        if method_key in data:
            df = data[method_key]
            shap_cols = [c for c in df.columns if c.startswith('shap_') and
                        ('t-' in c or '_Z' in c or 'lag' in c or 'lstm' in c)]
            n_shap_features = len(shap_cols)
            break

    n_true_features = len(imp_cols)
    if n_shap_features > 0 and n_true_features < n_shap_features:
        # Pad with zeros at the beginning (for lag features)
        n_missing = n_shap_features - n_true_features
        padding = np.zeros((n_missing, true_imp_matrix.shape[1]))
        true_imp_matrix = np.vstack([padding, true_imp_matrix])

    # Plot heatmap with a diverging colormap
    im = ax.imshow(true_imp_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xlabel('Time Index (every 100th shown)')
    ax.set_ylabel('Feature')
    ax.set_title('True Feature Importances (Ground Truth)')

    # Set y-axis labels (feature names, matching padded size)
    ax.set_yticks(range(true_imp_matrix.shape[0]))
    feature_names = [f'Feature {i}' for i in range(true_imp_matrix.shape[0])]
    ax.set_yticklabels(feature_names)

    # Set x-axis labels (show every 100th index)
    if len(true_imp_df) > 10:
        step = max(len(true_imp_df) // 10, 1)
        ax.set_xticks(range(0, len(true_imp_df), step))
        ax.set_xticklabels([f'{i}' for i in range(0, len(true_imp_df), step)],
                          rotation=45)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Importance Value')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'true_importance_heatmap.png'), bbox_inches='tight')
    print(f"Saved: true_importance_heatmap.png")
    plt.close()


def plot_shap_vs_true_importance_heatmaps(data, save_dir):
    """Create side-by-side heatmaps comparing SHAP values with true importances."""
    if 'true_importances' not in data:
        print("No true importances available - skipping SHAP vs true importance comparison")
        return

    true_imp_df = data['true_importances']
    imp_cols = [c for c in true_imp_df.columns if c.startswith('true_imp_')]
    n_features = len(imp_cols)

    # Create comparison for each method - include all rolling variants
    methods = [
        ('global_shap', 'Vanilla SHAP'),
        ('timeshap', 'TimeShap'),
        ('rolling_shap', 'Rolling (Fixed 100)'),
        ('adaptive_shap_max', 'Adaptive (Max)'),
        ('adaptive_shap_mean', 'Adaptive (Mean)'),
        ('adaptive_shap', 'Adaptive SHAP'),
        ('adaptive_shap_rolling_mean', 'Adaptive SHAP (Smooth)')
    ]

    for method_key, method_name in methods:
        if method_key not in data:
            continue

        df = data[method_key]

        # Find ALL SHAP columns (lags and covariates)
        # Pattern: shap_*_t-* for lags, shap_*_Z* for covariates
        shap_cols = [c for c in df.columns if c.startswith('shap_') and
                     ('t-' in c or '_Z' in c or 'lag' in c or 'lstm' in c)]

        if len(shap_cols) == 0:
            continue

        # Separate lags and covariates
        lag_cols = [c for c in shap_cols if 't-' in c]
        cov_cols = [c for c in shap_cols if '_Z' in c or 'covariate' in c.lower()]

        # Sort lag columns by lag number
        def extract_lag(col_name):
            parts = col_name.split('t-')
            if len(parts) > 1:
                try:
                    return int(parts[-1])
                except:
                    return 0
            return 0

        lag_cols = sorted(lag_cols, key=extract_lag)

        # Sort covariate columns by number
        def extract_cov_num(col_name):
            parts = col_name.split('Z')
            if len(parts) > 1:
                try:
                    # Extract number after Z (e.g., "shap_Z0" -> 0)
                    return int(''.join(filter(str.isdigit, parts[-1])))
                except:
                    return 999
            return 999

        cov_cols = sorted(cov_cols, key=extract_cov_num)

        # Combine: lags first, then covariates
        shap_cols = lag_cols + cov_cols

        if len(shap_cols) == 0:
            continue

        # Create figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f'{method_name} vs Ground Truth Importances', fontsize=14, fontweight='bold')

        # Left: SHAP values
        ax = axes[0]
        shap_matrix = df[shap_cols].values.T

        # Normalize SHAP values row-wise for better comparison (using absolute values)
        shap_matrix_norm = np.abs(shap_matrix) / (np.abs(shap_matrix).sum(axis=0, keepdims=True) + 1e-10)

        im1 = ax.imshow(shap_matrix_norm, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                       vmin=0, vmax=1)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Feature')
        ax.set_title(f'{method_name} (Normalized)')

        # Set y-axis labels
        ax.set_yticks(range(len(shap_cols)))
        ax.set_yticklabels([f'Feature {i}' for i in range(len(shap_cols))])

        # Set x-axis labels
        if len(df) > 10:
            step = max(len(df) // 10, 1)
            ax.set_xticks(range(0, len(df), step))
            ax.set_xticklabels([f'{int(df.iloc[i]["end_index"])}' for i in range(0, len(df), step)],
                              rotation=45)

        plt.colorbar(im1, ax=ax, label='Normalized Importance')

        # Right: True importances (aligned with SHAP timepoints)
        ax = axes[1]

        # Align true importances with SHAP end indices
        if 'end_index' in df.columns:
            # Sample true importances at SHAP timepoints
            end_indices = df['end_index'].values.astype(int)
            # Ensure indices are within bounds
            end_indices = np.clip(end_indices, 0, len(true_imp_df) - 1)
            true_imp_sampled = true_imp_df.iloc[end_indices][imp_cols].values
        else:
            # If no end_index, just take first N rows
            true_imp_sampled = true_imp_df[imp_cols].iloc[:len(df)].values

        # Handle mismatch: if SHAP has more features than true importances (e.g., includes lags)
        # Pad with zeros for missing lag features (lags come first in SHAP feature order)
        n_shap_features = len(shap_cols)
        n_true_features = len(imp_cols)

        if n_true_features < n_shap_features:
            # Number of missing features (assumed to be lags)
            n_missing = n_shap_features - n_true_features
            # Pad with zeros at the beginning (for lag features)
            padding = np.zeros((len(true_imp_sampled), n_missing))
            true_imp_sampled = np.hstack([padding, true_imp_sampled])

        true_imp_aligned = true_imp_sampled.T

        im2 = ax.imshow(true_imp_aligned, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                       vmin=0, vmax=1)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Feature')
        ax.set_title('Ground Truth Importances')

        # Set y-axis labels
        ax.set_yticks(range(len(shap_cols)))
        ax.set_yticklabels([f'Feature {i}' for i in range(len(shap_cols))])

        # Set x-axis labels (same as left plot)
        if len(df) > 10:
            step = max(len(df) // 10, 1)
            ax.set_xticks(range(0, len(df), step))
            ax.set_xticklabels([f'{int(df.iloc[i]["end_index"])}' for i in range(0, len(df), step)],
                              rotation=45)

        plt.colorbar(im2, ax=ax, label='True Importance')

        plt.tight_layout()
        filename = f'shap_vs_true_{method_key}.png'
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


def plot_correlation_with_true_importance(data, save_dir):
    """Plot correlation between SHAP values and true importances for each method."""
    if 'true_importances' not in data:
        print("No true importances available - skipping correlation analysis")
        return

    true_imp_df = data['true_importances']
    imp_cols = [c for c in true_imp_df.columns if c.startswith('true_imp_')]

    # Include all rolling window variants
    methods = [
        ('global_shap', 'Vanilla SHAP', 'shap_lag_t', '#1f77b4'),
        ('timeshap', 'TimeShap', 'shap_lag_t', '#e377c2'),
        ('rolling_shap', 'Rolling (Fixed 100)', 'shap_lag_t', '#ff7f0e'),
        ('adaptive_shap_max', 'Adaptive (Max)', 'shap_lag_t', '#9467bd'),
        ('adaptive_shap_mean', 'Adaptive (Mean)', 'shap_lag_t', '#8c564b'),
        ('adaptive_shap', 'Adaptive SHAP', 'shap_lag_t', '#2ca02c'),
        ('adaptive_shap_rolling_mean', 'Adaptive SHAP (Smooth)', 'shap_lag_t', '#d62728')
    ]

    # Filter to only include available methods
    available_methods = [(k, n, p, c) for k, n, p, c in methods if k in data]
    n_methods = len(available_methods)

    if n_methods == 0:
        print("No methods available for correlation plot")
        return

    # Create grid layout (dynamically adjust based on number of methods)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

    # Handle different subplot configurations
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1:
        axes_flat = axes.flatten()
    elif n_cols == 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()

    fig.suptitle('Correlation: SHAP Values vs Ground Truth Importances',
                 fontsize=14, fontweight='bold')

    for idx, (method_key, method_name, shap_prefix, color) in enumerate(available_methods):
        if method_key not in data:
            continue

        ax = axes_flat[idx]
        df = data[method_key]

        # Find ALL SHAP columns (lags and covariates) - same logic as other functions
        shap_cols = [c for c in df.columns if c.startswith('shap_') and
                     ('t-' in c or '_Z' in c or 'lag' in c or 'lstm' in c)]

        if len(shap_cols) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(method_name)
            continue

        # Separate lags and covariates
        lag_cols = [c for c in shap_cols if 't-' in c]
        cov_cols = [c for c in shap_cols if '_Z' in c or 'covariate' in c.lower()]

        # Sort lag columns by lag number
        def extract_lag(col_name):
            parts = col_name.split('t-')
            if len(parts) > 1:
                try:
                    return int(parts[-1])
                except:
                    return 0
            return 0

        lag_cols = sorted(lag_cols, key=extract_lag)

        # Sort covariate columns by number
        def extract_cov_num(col_name):
            parts = col_name.split('Z')
            if len(parts) > 1:
                try:
                    return int(''.join(filter(str.isdigit, parts[-1])))
                except:
                    return 999
            return 999

        cov_cols = sorted(cov_cols, key=extract_cov_num)

        # Combine: lags first, then covariates
        shap_cols = lag_cols + cov_cols

        # Handle mismatch: if SHAP has more features than true importances
        n_shap_features = len(shap_cols)
        n_true_features = len(imp_cols)

        # Calculate correlation for each feature
        correlations = []
        feature_labels = []

        for i in range(n_shap_features):
            feature_labels.append(f'Feat {i}')

            # For lag features that don't have true importance, skip correlation
            if i < (n_shap_features - n_true_features):
                # This is a lag feature with no true importance (zero importance)
                correlations.append(0.0)  # No correlation with zero
            else:
                # This is a covariate with true importance
                true_idx = i - (n_shap_features - n_true_features)

                # Align data
                if 'end_index' in df.columns:
                    end_indices = df['end_index'].values.astype(int)
                    end_indices = np.clip(end_indices, 0, len(true_imp_df) - 1)
                    true_vals = true_imp_df.iloc[end_indices][imp_cols[true_idx]].values
                else:
                    true_vals = true_imp_df[imp_cols[true_idx]].iloc[:len(df)].values

                # Normalize SHAP values
                shap_vals = df[shap_cols[i]].values
                shap_total = df[shap_cols].sum(axis=1).values + 1e-10
                shap_vals_norm = shap_vals / shap_total

                # Calculate correlation
                if len(true_vals) == len(shap_vals_norm) and len(true_vals) > 0:
                    # Check for zero variance before computing correlation
                    if np.std(true_vals) == 0 or np.std(shap_vals_norm) == 0:
                        # One or both arrays are constant - correlation is undefined
                        correlations.append(0.0)
                    else:
                        corr = np.corrcoef(true_vals, shap_vals_norm)[0, 1]
                        # Handle NaN correlations (can happen if variance is zero)
                        if np.isnan(corr):
                            correlations.append(0.0)
                        else:
                            correlations.append(corr)
                else:
                    # Length mismatch - append 0 to maintain alignment
                    correlations.append(0.0)

        if correlations:
            bars = ax.bar(range(len(correlations)), correlations, color=color, alpha=0.7,
                         edgecolor='black')
            ax.set_xticks(range(len(correlations)))
            ax.set_xticklabels(feature_labels, rotation=45, ha='right')
            ax.set_ylabel('Correlation')
            ax.set_title(method_name)
            ax.set_ylim([-1, 1])
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8)

    # Hide any unused subplots
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_with_true_importance.png'),
                bbox_inches='tight')
    print(f"Saved: correlation_with_true_importance.png")
    plt.close()


def plot_all_methods_with_true_importance(data, save_dir):
    """Create comprehensive heatmap comparison of all methods with ground truth."""
    if 'true_importances' not in data:
        print("No true importances available - skipping all methods + true importance plot")
        return

    true_imp_df = data['true_importances']
    imp_cols = [c for c in true_imp_df.columns if c.startswith('true_imp_')]

    # Define all methods
    methods = [
        ('global_shap', 'Vanilla SHAP', 'shap_lag_t'),
        ('timeshap', 'TimeShap', 'shap_lag_t'),
        ('rolling_shap', 'Rolling (Fixed 100)', 'shap_lag_t'),
        ('adaptive_shap_max', 'Adaptive (Max)', 'shap_lag_t'),
        ('adaptive_shap_mean', 'Adaptive (Mean)', 'shap_lag_t'),
        ('adaptive_shap', 'Adaptive SHAP', 'shap_lag_t'),
        ('adaptive_shap_rolling_mean', 'Adaptive (Smooth)', 'shap_lag_t')
    ]

    # Filter to only available methods
    available_methods = [(k, n, p) for k, n, p in methods if k in data]
    n_methods = len(available_methods)

    if n_methods == 0:
        print("No methods available for comparison with true importance")
        return

    # Calculate grid dimensions (methods + 1 for true importance)
    total_plots = n_methods + 1
    n_cols = 3
    n_rows = (total_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle('All Methods vs Ground Truth Importances', fontsize=14, fontweight='bold')

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Plot each method
    for idx, (method_key, method_name, shap_prefix) in enumerate(available_methods):
        ax = axes_flat[idx]
        df = data[method_key]

        # Find ALL SHAP columns (lags and covariates) - same logic as individual comparison
        shap_cols = [c for c in df.columns if c.startswith('shap_') and
                     ('t-' in c or '_Z' in c or 'lag' in c or 'lstm' in c)]

        if len(shap_cols) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(method_name)
            continue

        # Separate lags and covariates
        lag_cols = [c for c in shap_cols if 't-' in c]
        cov_cols = [c for c in shap_cols if '_Z' in c or 'covariate' in c.lower()]

        # Sort lag columns by lag number
        def extract_lag(col_name):
            parts = col_name.split('t-')
            if len(parts) > 1:
                try:
                    return int(parts[-1])
                except:
                    return 0
            return 0

        lag_cols = sorted(lag_cols, key=extract_lag)

        # Sort covariate columns by number
        def extract_cov_num(col_name):
            parts = col_name.split('Z')
            if len(parts) > 1:
                try:
                    return int(''.join(filter(str.isdigit, parts[-1])))
                except:
                    return 999
            return 999

        cov_cols = sorted(cov_cols, key=extract_cov_num)

        # Combine: lags first, then covariates
        shap_cols = lag_cols + cov_cols

        if len(shap_cols) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(method_name)
            continue

        # Create and normalize SHAP matrix (using absolute values)
        shap_matrix = df[shap_cols].values.T
        shap_matrix_norm = np.abs(shap_matrix) / (np.abs(shap_matrix).sum(axis=0, keepdims=True) + 1e-10)

        # Plot heatmap
        im = ax.imshow(shap_matrix_norm, aspect='auto', cmap='RdYlGn',
                      interpolation='nearest', vmin=0, vmax=1)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Feature')
        ax.set_title(method_name)

        # Set y-axis labels
        ax.set_yticks(range(len(shap_cols)))
        ax.set_yticklabels([f'F{i}' for i in range(len(shap_cols))], fontsize=8)

        # Set x-axis labels
        if len(df) > 10:
            step = max(len(df) // 5, 1)
            ax.set_xticks(range(0, len(df), step))
            ax.set_xticklabels([f'{int(df.iloc[i]["end_index"])}' for i in range(0, len(df), step)],
                              rotation=45, fontsize=8)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Normalized Importance', fraction=0.046, pad=0.04)

    # Plot true importance in the last subplot
    ax = axes_flat[n_methods]
    true_imp_matrix = true_imp_df[imp_cols].values.T

    # Pad true importances to match SHAP feature count if needed
    # Get number of SHAP features from first available method
    n_shap_features = 0
    if n_methods > 0:
        first_method_key, _, _ = available_methods[0]
        df = data[first_method_key]
        shap_cols_first = [c for c in df.columns if c.startswith('shap_') and
                          ('t-' in c or '_Z' in c or 'lag' in c or 'lstm' in c)]
        n_shap_features = len(shap_cols_first)

    n_true_features = len(imp_cols)
    if n_true_features < n_shap_features:
        # Pad with zeros at the beginning (for lag features)
        n_missing = n_shap_features - n_true_features
        padding = np.zeros((n_missing, true_imp_matrix.shape[1]))
        true_imp_matrix = np.vstack([padding, true_imp_matrix])

    im = ax.imshow(true_imp_matrix, aspect='auto', cmap='RdYlGn',
                  interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Feature')
    ax.set_title('Ground Truth', fontweight='bold', color='darkred')

    # Set y-axis labels (matching SHAP feature count)
    ax.set_yticks(range(true_imp_matrix.shape[0]))
    ax.set_yticklabels([f'F{i}' for i in range(true_imp_matrix.shape[0])], fontsize=8)

    # Set x-axis labels
    if len(true_imp_df) > 10:
        step = max(len(true_imp_df) // 5, 1)
        ax.set_xticks(range(0, len(true_imp_df), step))
        ax.set_xticklabels([f'{i}' for i in range(0, len(true_imp_df), step)],
                          rotation=45, fontsize=8)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='True Importance', fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for idx in range(total_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_methods_with_true_importance.png'), bbox_inches='tight')
    print(f"Saved: all_methods_with_true_importance.png")
    plt.close()


def plot_rolling_window_comparison(data, save_dir):
    """Compare different rolling window size variants."""
    # Check which rolling window variants are available
    rolling_variants = []
    for suffix, name in [('', 'Fixed (100)'), ('_max', 'Max Adaptive'), ('_mean', 'Mean Adaptive')]:
        method_key = f'rolling_shap{suffix}'
        if method_key in data:
            rolling_variants.append((method_key, name))

    if len(rolling_variants) < 2:
        print("Not enough rolling window variants for comparison - skipping")
        return

    # Create comprehensive comparison
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Rolling Window Size Comparison', fontsize=16, fontweight='bold')

    colors = ['#ff7f0e', '#9467bd', '#8c564b']  # Different shades for rolling variants

    # 1. SHAP values over time
    ax1 = fig.add_subplot(gs[0, :])
    for idx, (method_key, name) in enumerate(rolling_variants):
        df = data[method_key]
        shap_cols = [c for c in df.columns if c.startswith('shap_lag_t')]
        if shap_cols:
            total_shap = df[shap_cols].sum(axis=1)
            ax1.plot(df['end_index'], total_shap, label=f'Rolling {name}',
                    alpha=0.7, linewidth=1.5, color=colors[idx % len(colors)])

    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Total |SHAP|')
    ax1.set_title('Total SHAP Values Over Time')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    # 2. Faithfulness comparison (if available in summary)
    if 'summary' in data:
        summary = data['summary']
        faith_summary = summary[summary['metric_type'] == 'faithfulness']

        ax2 = fig.add_subplot(gs[1, 0])
        methods = []
        scores = []
        method_colors = []

        eval_key = 'prtb_p50'
        subset = faith_summary[faith_summary['evaluation'] == eval_key]

        for idx, (method_key, name) in enumerate(rolling_variants):
            method_data = subset[subset['method'] == method_key]
            if len(method_data) > 0:
                methods.append(f'Rolling\n{name}')
                scores.append(method_data['score'].values[0])
                method_colors.append(colors[idx % len(colors)])

        if methods:
            bars = ax2.bar(range(len(methods)), scores, color=method_colors, alpha=0.7, edgecolor='black')
            ax2.set_xticks(range(len(methods)))
            ax2.set_xticklabels(methods, rotation=0, ha='center', fontsize=9)
            ax2.set_ylabel('Faithfulness Score')
            ax2.set_title('Faithfulness (PRTB P50)')
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        # 3. Ablation comparison
        ablation_summary = summary[summary['metric_type'] == 'ablation']

        ax3 = fig.add_subplot(gs[1, 1])
        methods = []
        scores = []
        method_colors = []

        eval_key = 'mif_p50'
        subset = ablation_summary[ablation_summary['evaluation'] == eval_key]

        for idx, (method_key, name) in enumerate(rolling_variants):
            method_data = subset[subset['method'] == method_key]
            if len(method_data) > 0:
                methods.append(f'Rolling\n{name}')
                scores.append(method_data['score'].values[0])
                method_colors.append(colors[idx % len(colors)])

        if methods:
            bars = ax3.bar(range(len(methods)), scores, color=method_colors, alpha=0.7, edgecolor='black')
            ax3.set_xticks(range(len(methods)))
            ax3.set_xticklabels(methods, rotation=0, ha='center', fontsize=9)
            ax3.set_ylabel('Ablation Score')
            ax3.set_title('Ablation (MIF P50)')
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # 4. Number of windows comparison
    ax4 = fig.add_subplot(gs[1, 2])
    methods = []
    n_windows = []
    method_colors = []

    for idx, (method_key, name) in enumerate(rolling_variants):
        df = data[method_key]
        methods.append(f'Rolling\n{name}')
        n_windows.append(len(df))
        method_colors.append(colors[idx % len(colors)])

    bars = ax4.bar(range(len(methods)), n_windows, color=method_colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=0, ha='center', fontsize=9)
    ax4.set_ylabel('Number of Windows')
    ax4.set_title('Windows Computed')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

    # 5-7. Prediction scatter plots (compare variants pairwise)
    if len(rolling_variants) >= 2:
        ax5 = fig.add_subplot(gs[2, 0])
        method1_key, method1_name = rolling_variants[0]
        method2_key, method2_name = rolling_variants[1]

        df1 = data[method1_key]
        df2 = data[method2_key]

        merged = pd.merge(df1[['end_index', 'y_hat']],
                         df2[['end_index', 'y_hat']],
                         on='end_index', suffixes=('_1', '_2'))

        ax5.scatter(merged['y_hat_1'], merged['y_hat_2'], alpha=0.5, s=20, color=colors[0])

        lims = [np.min([ax5.get_xlim(), ax5.get_ylim()]),
                np.max([ax5.get_xlim(), ax5.get_ylim()])]
        ax5.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2)
        ax5.set_aspect('equal')
        ax5.set_xlim(lims)
        ax5.set_ylim(lims)

        if np.std(merged['y_hat_1']) > 0 and np.std(merged['y_hat_2']) > 0:
            corr = np.corrcoef(merged['y_hat_1'], merged['y_hat_2'])[0, 1]
        else:
            corr = 0.0
        ax5.text(0.05, 0.95, f'r = {corr:.4f}',
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax5.set_xlabel(f'{method1_name}')
        ax5.set_ylabel(f'{method2_name}')
        ax5.set_title(f'{method1_name} vs {method2_name}')
    if len(rolling_variants) >= 3:
        # Compare first and third
        ax6 = fig.add_subplot(gs[2, 1])
        method1_key, method1_name = rolling_variants[0]
        method3_key, method3_name = rolling_variants[2]

        df1 = data[method1_key]
        df3 = data[method3_key]

        merged = pd.merge(df1[['end_index', 'y_hat']],
                         df3[['end_index', 'y_hat']],
                         on='end_index', suffixes=('_1', '_3'))

        ax6.scatter(merged['y_hat_1'], merged['y_hat_3'], alpha=0.5, s=20, color=colors[1])

        lims = [np.min([ax6.get_xlim(), ax6.get_ylim()]),
                np.max([ax6.get_xlim(), ax6.get_ylim()])]
        ax6.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2)
        ax6.set_aspect('equal')
        ax6.set_xlim(lims)
        ax6.set_ylim(lims)

        if np.std(merged['y_hat_1']) > 0 and np.std(merged['y_hat_3']) > 0:
            corr = np.corrcoef(merged['y_hat_1'], merged['y_hat_3'])[0, 1]
        else:
            corr = 0.0
        ax6.text(0.05, 0.95, f'r = {corr:.4f}',
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax6.set_xlabel(f'{method1_name}')
        ax6.set_ylabel(f'{method3_name}')
        ax6.set_title(f'{method1_name} vs {method3_name}')
        # Compare second and third
        ax7 = fig.add_subplot(gs[2, 2])
        method2_key, method2_name = rolling_variants[1]
        method3_key, method3_name = rolling_variants[2]

        df2 = data[method2_key]
        df3 = data[method3_key]

        merged = pd.merge(df2[['end_index', 'y_hat']],
                         df3[['end_index', 'y_hat']],
                         on='end_index', suffixes=('_2', '_3'))

        ax7.scatter(merged['y_hat_2'], merged['y_hat_3'], alpha=0.5, s=20, color=colors[2])

        lims = [np.min([ax7.get_xlim(), ax7.get_ylim()]),
                np.max([ax7.get_xlim(), ax7.get_ylim()])]
        ax7.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2)
        ax7.set_aspect('equal')
        ax7.set_xlim(lims)
        ax7.set_ylim(lims)

        if np.std(merged['y_hat_2']) > 0 and np.std(merged['y_hat_3']) > 0:
            corr = np.corrcoef(merged['y_hat_2'], merged['y_hat_3'])[0, 1]
        else:
            corr = 0.0
        ax7.text(0.05, 0.95, f'r = {corr:.4f}',
                transform=ax7.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax7.set_xlabel(f'{method2_name}')
        ax7.set_ylabel(f'{method3_name}')
        ax7.set_title(f'{method2_name} vs {method3_name}')
    plt.savefig(os.path.join(save_dir, 'rolling_window_comparison.png'), bbox_inches='tight')
    print(f"Saved: rolling_window_comparison.png")
    plt.close()


def plot_dataset_with_regimes(data, save_dir):
    """Plot the raw dataset with color-coded stationarity regions."""
    if 'config' not in data:
        print("No config available - skipping dataset plot")
        return

    # Load the dataset
    dataset_path = data['config'].get('dataset', '')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    df_data = pd.read_csv(dataset_path)

    # Detect regime changes from true importances
    change_points = []
    if 'true_importances' in data:
        true_imp_df = data['true_importances']
        imp_cols = [c for c in true_imp_df.columns if c.startswith('true_imp_')]

        if len(imp_cols) > 0:
            # Detect changes by looking at differences in importance patterns
            for i in range(1, len(true_imp_df)):
                # Calculate L2 distance between consecutive importance vectors
                prev = true_imp_df.iloc[i-1][imp_cols].values
                curr = true_imp_df.iloc[i][imp_cols].values
                distance = np.sqrt(np.sum((curr - prev)**2))

                # If distance is large, it's a change point
                if distance > 0.01:  # threshold for detecting change
                    change_points.append(i)

    # Define colors for regimes (user specified)
    regime_colors = ['#4A74AA', '#D84835', '#49792F']

    # Create figure with subplots for target and each covariate
    n_covariates = len([c for c in df_data.columns if c.startswith('Z_')])
    n_plots = 1 + n_covariates  # target + covariates

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Dataset with Color-Coded Stationarity Regions',
                 fontsize=14, fontweight='bold')

    # Add 0 and T as boundaries
    boundaries = [0] + change_points + [len(df_data)]
    n_regimes = len(boundaries) - 1

    # Plot target variable
    ax = axes[0]
    target = df_data['N'].values
    time_index = np.arange(len(target))

    for regime_idx in range(n_regimes):
        start = boundaries[regime_idx]
        end = boundaries[regime_idx + 1]
        color = regime_colors[regime_idx % len(regime_colors)]

        ax.plot(time_index[start:end], target[start:end],
               color=color, linewidth=1.5, alpha=0.8,
               label=f'Regime {regime_idx + 1}' if regime_idx == 0 else '')

    # Mark change points with vertical lines
    for cp in change_points:
        ax.axvline(x=cp, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Time Index')
    ax.set_ylabel('Target (N)')
    ax.set_title('Target Variable')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    # Plot covariates
    for cov_idx in range(n_covariates):
        ax = axes[cov_idx + 1]
        col_name = f'Z_{cov_idx}'

        if col_name not in df_data.columns:
            continue

        covariate = df_data[col_name].values

        for regime_idx in range(n_regimes):
            start = boundaries[regime_idx]
            end = boundaries[regime_idx + 1]
            color = regime_colors[regime_idx % len(regime_colors)]

            ax.plot(time_index[start:end], covariate[start:end],
                   color=color, linewidth=1.5, alpha=0.8)

        # Mark change points
        for cp in change_points:
            ax.axvline(x=cp, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.set_xlabel('Time Index')
        ax.set_ylabel(f'Covariate {col_name}')
        ax.set_title(f'Covariate {col_name}')
    # Add regime information as text
    info_text = f"Detected {n_regimes} regimes"
    if change_points:
        info_text += f" with change points at: {', '.join(map(str, change_points))}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(save_dir, 'dataset_with_regimes.png'), bbox_inches='tight')
    print(f"Saved: dataset_with_regimes.png")
    plt.close()


def plot_residual_analysis(data, save_dir):
    """Analyze prediction residuals comparing each method against y_true."""
    # Check if y_true is available in at least one method
    has_y_true = False
    for method_key in ['global_shap', 'rolling_shap', 'adaptive_shap_max', 'adaptive_shap_mean', 'adaptive_shap']:
        if method_key in data and 'y_true' in data[method_key].columns:
            has_y_true = True
            break

    if not has_y_true:
        print("No y_true data available - skipping residual analysis")
        return

    # Define methods with colors
    methods = [
        ('global_shap', 'Vanilla SHAP', '#1f77b4'),
        ('timeshap', 'TimeShap', '#e377c2'),
        ('rolling_shap', 'Rolling (Fixed 100)', '#ff7f0e'),
        ('adaptive_shap_max', 'Adaptive (Max)', '#9467bd'),
        ('adaptive_shap_mean', 'Adaptive (Mean)', '#8c564b'),
        ('adaptive_shap', 'Adaptive SHAP', '#2ca02c')
    ]

    # Filter to only available methods with y_true
    available_methods = [(k, n, c) for k, n, c in methods
                        if k in data and 'y_true' in data[k].columns and 'y_hat' in data[k].columns]

    if len(available_methods) == 0:
        print("No methods with both y_true and y_hat - skipping residual analysis")
        return

    n_methods = len(available_methods)

    # Create comprehensive residual analysis figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, n_methods, hspace=0.3, wspace=0.3)
    fig.suptitle('Residual Analysis: Predictions vs Ground Truth', fontsize=16, fontweight='bold')

    # Row 1: Residuals over time
    for idx, (method_key, method_name, color) in enumerate(available_methods):
        ax = fig.add_subplot(gs[0, idx])
        df = data[method_key]

        residuals = df['y_true'] - df['y_hat']

        ax.plot(df['end_index'], residuals, alpha=0.6, linewidth=1, color=color)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Residual (y_true - y_hat)')
        ax.set_title(f'{method_name}\nResiduals Over Time')
        # Add RMSE as text
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9)

    # Row 2: Residual distributions
    for idx, (method_key, method_name, color) in enumerate(available_methods):
        ax = fig.add_subplot(gs[1, idx])
        df = data[method_key]

        residuals = df['y_true'] - df['y_hat']

        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color=color)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(x=np.mean(residuals), color='blue', linestyle='--',
                  linewidth=1.5, label=f'Mean: {np.mean(residuals):.4f}')
        ax.set_xlabel('Residual (y_true - y_hat)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{method_name}\nResidual Distribution')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    # Row 3: Predictions vs y_true scatter plots
    for idx, (method_key, method_name, color) in enumerate(available_methods):
        ax = fig.add_subplot(gs[2, idx])
        df = data[method_key]

        ax.scatter(df['y_true'], df['y_hat'], alpha=0.5, s=20, color=color)

        # Add diagonal line (perfect predictions)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Prediction')
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Calculate metrics
        if np.std(df['y_true']) > 0 and np.std(df['y_hat']) > 0:
            corr = np.corrcoef(df['y_true'], df['y_hat'])[0, 1]
        else:
            corr = 0.0
        r2 = 1 - np.sum((df['y_true'] - df['y_hat'])**2) / np.sum((df['y_true'] - np.mean(df['y_true']))**2)

        ax.text(0.05, 0.95, f'r = {corr:.4f}\nR² = {r2:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9)

        ax.set_xlabel('y_true')
        ax.set_ylabel('y_hat')
        ax.set_title(f'{method_name}\nPredictions vs Truth')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)

    plt.savefig(os.path.join(save_dir, 'residual_analysis.png'), bbox_inches='tight')
    print(f"Saved: residual_analysis.png")
    plt.close()


def main():
    """Main visualization function."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize SHAP benchmark results')
    parser.add_argument('--dataset', type=str, default='piecewise_ar3',
                        help='Dataset name (default: piecewise_ar3)')
    parser.add_argument('--data-type', type=str, default='simulated',
                        choices=['simulated', 'empirical'],
                        help='Dataset type (default: simulated)')
    args = parser.parse_args()

    if args.data_type == "simulated":
        # Simulated dataset results directory
        results_dir = f'examples/results/benchmark_{args.dataset}'
    elif args.data_type == "empirical":
        # Empirical dataset results directory
        results_dir = 'examples/results/benchmark_empirical'

    save_dir = os.path.join(results_dir, 'figures')

    # Create figures directory
    os.makedirs(save_dir, exist_ok=True)

    print("="*60)
    print(f"Benchmark Visualization - {args.data_type.title()} Data")
    print(f"Dataset: {args.dataset if args.data_type == 'simulated' else 'Empirical'}")
    print("="*60)
    print(f"Loading data from: {results_dir}")

    # Load data
    data = load_benchmark_data(results_dir)

    print(f"Loaded {len(data)} data files")
    print(f"Saving figures to: {save_dir}")
    print("="*60)

    # Generate plots
    print("\nGenerating visualizations...")

    # Plot dataset first
    try:
        plot_dataset_with_regimes(data, save_dir)
    except Exception as e:
        print(f"Error in dataset visualization: {e}")

    try:
        plot_faithfulness_comparison(data, save_dir)
    except Exception as e:
        print(f"Error in faithfulness comparison: {e}")

    try:
        plot_ablation_comparison(data, save_dir)
    except Exception as e:
        print(f"Error in ablation comparison: {e}")

    try:
        plot_ablation_mif_vs_lif(data, save_dir)
    except Exception as e:
        print(f"Error in MIF vs LIF comparison: {e}")

    try:
        plot_shap_over_time(data, save_dir)
    except Exception as e:
        print(f"Error in SHAP over time: {e}")

    try:
        plot_shap_heatmaps(data, save_dir)
    except Exception as e:
        print(f"Error in SHAP heatmaps: {e}")

    try:
        plot_prediction_comparison(data, save_dir)
    except Exception as e:
        print(f"Error in prediction comparison: {e}")

    try:
        plot_window_size_analysis(data, save_dir)
    except Exception as e:
        print(f"Error in window size analysis: {e}")

    try:
        plot_temporal_faithfulness(data, save_dir)
    except Exception as e:
        print(f"Error in temporal faithfulness: {e}")

    try:
        plot_temporal_ablation(data, save_dir)
    except Exception as e:
        print(f"Error in temporal ablation: {e}")

    try:
        create_summary_dashboard(data, save_dir)
    except Exception as e:
        print(f"Error in summary dashboard: {e}")

    # True importance visualizations (if available)
    try:
        plot_true_importance_heatmap(data, save_dir)
    except Exception as e:
        print(f"Error in true importance heatmap: {e}")

    try:
        plot_shap_vs_true_importance_heatmaps(data, save_dir)
    except Exception as e:
        print(f"Error in SHAP vs true importance heatmaps: {e}")

    try:
        plot_correlation_with_true_importance(data, save_dir)
    except Exception as e:
        print(f"Error in correlation with true importance: {e}")

    try:
        plot_all_methods_with_true_importance(data, save_dir)
    except Exception as e:
        print(f"Error in all methods with true importance: {e}")

    try:
        plot_rolling_window_comparison(data, save_dir)
    except Exception as e:
        print(f"Error in rolling window comparison: {e}")

    try:
        plot_residual_analysis(data, save_dir)
    except Exception as e:
        print(f"Error in residual analysis: {e}")

    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"All figures saved to: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
