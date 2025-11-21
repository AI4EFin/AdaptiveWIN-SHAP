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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_benchmark_data(results_dir):
    """Load all benchmark result files."""
    data = {}

    # Load summary
    summary_path = os.path.join(results_dir, 'benchmark_summary.csv')
    if os.path.exists(summary_path):
        data['summary'] = pd.read_csv(summary_path)

    # Load individual method results
    for method in ['global_shap', 'rolling_shap', 'adaptive_shap']:
        result_path = os.path.join(results_dir, f'{method}_results.csv')
        if os.path.exists(result_path):
            data[method] = pd.read_csv(result_path)

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Faithfulness Comparison Across Methods', fontsize=14, fontweight='bold')

    eval_types = ['prtb', 'sqnc']
    percentiles = ['p90', 'p70', 'p50']

    for i, eval_type in enumerate(eval_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[i, j]

            eval_key = f'{eval_type}_{percentile}'
            subset = faith_summary[faith_summary['evaluation'] == eval_key]

            if len(subset) > 0:
                methods = []
                scores = []
                colors = []
                color_map = {'global_shap': '#1f77b4', 'rolling_shap': '#ff7f0e', 'adaptive_shap': '#2ca02c'}

                for method in ['global_shap', 'rolling_shap', 'adaptive_shap']:
                    method_data = subset[subset['method'] == method]
                    if len(method_data) > 0:
                        methods.append(method.replace('_', ' ').title())
                        scores.append(method_data['score'].values[0])
                        colors.append(color_map.get(method, '#888888'))

                if methods:
                    bars = ax.bar(range(len(methods)), scores, color=colors, alpha=0.7, edgecolor='black')
                    ax.set_xticks(range(len(methods)))
                    ax.set_xticklabels(methods, rotation=45, ha='right')
                    ax.set_ylabel('Faithfulness Score')
                    ax.set_title(f'{eval_type.upper()} - {percentile.upper()}')
                    ax.grid(True, alpha=0.3, axis='y')

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Ablation Score Comparison Across Methods', fontsize=14, fontweight='bold')

    ablation_types = ['mif', 'lif']
    percentiles = ['p90', 'p70', 'p50']

    for i, ablation_type in enumerate(ablation_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[i, j]

            eval_key = f'{ablation_type}_{percentile}'
            subset = ablation_summary[ablation_summary['evaluation'] == eval_key]

            if len(subset) > 0:
                methods = []
                scores = []
                colors = []
                color_map = {'global_shap': '#1f77b4', 'rolling_shap': '#ff7f0e', 'adaptive_shap': '#2ca02c'}

                for method in ['global_shap', 'rolling_shap', 'adaptive_shap']:
                    method_data = subset[subset['method'] == method]
                    if len(method_data) > 0:
                        methods.append(method.replace('_', ' ').title())
                        scores.append(method_data['score'].values[0])
                        colors.append(color_map.get(method, '#888888'))

                if methods:
                    bars = ax.bar(range(len(methods)), scores, color=colors, alpha=0.7, edgecolor='black')
                    ax.set_xticks(range(len(methods)))
                    ax.set_xticklabels(methods, rotation=45, ha='right')
                    ax.set_ylabel('Ablation Score')
                    ax.set_title(f'{ablation_type.upper()} - {percentile.upper()}')
                    ax.grid(True, alpha=0.3, axis='y')

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('MIF vs LIF Ablation Comparison (Higher MIF/LIF ratio = Better)',
                 fontsize=14, fontweight='bold')

    percentiles = ['p90', 'p70', 'p50']
    color_map = {'global_shap': '#1f77b4', 'rolling_shap': '#ff7f0e', 'adaptive_shap': '#2ca02c'}

    for j, percentile in enumerate(percentiles):
        ax = axes[j]

        methods = []
        mif_scores = []
        lif_scores = []
        colors = []

        for method in ['global_shap', 'rolling_shap', 'adaptive_shap']:
            # Get MIF score
            mif_key = f'mif_{percentile}'
            mif_data = ablation_summary[(ablation_summary['method'] == method) &
                                        (ablation_summary['evaluation'] == mif_key)]

            # Get LIF score
            lif_key = f'lif_{percentile}'
            lif_data = ablation_summary[(ablation_summary['method'] == method) &
                                        (ablation_summary['evaluation'] == lif_key)]

            if len(mif_data) > 0 and len(lif_data) > 0:
                methods.append(method.replace('_', ' ').title())
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

            ax.set_xlabel('Method')
            ax.set_ylabel('Ablation Score')
            ax.set_title(f'Percentile: {percentile.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

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
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('SHAP Values Over Time', fontsize=14, fontweight='bold')

    methods = [
        ('global_shap', 'Vanilla SHAP', 'shap_lag_t'),
        ('rolling_shap', 'Rolling Window SHAP', 'shap_lag_t'),
        ('adaptive_shap', 'Adaptive SHAP', 'shap_lstm_t')
    ]

    for idx, (method_key, method_name, shap_prefix) in enumerate(methods):
        if method_key not in data:
            continue

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
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_over_time.png'), bbox_inches='tight')
    print(f"Saved: shap_over_time.png")
    plt.close()


def plot_shap_heatmaps(data, save_dir):
    """Create heatmaps of SHAP values for each method."""
    methods = [
        ('global_shap', 'Vanilla SHAP', 'shap_lag_t'),
        ('rolling_shap', 'Rolling Window SHAP', 'shap_lag_t'),
        ('adaptive_shap', 'Adaptive SHAP', 'shap_lstm_t')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('SHAP Value Heatmaps (Time × Lags)', fontsize=14, fontweight='bold')

    for idx, (method_key, method_name, shap_prefix) in enumerate(methods):
        if method_key not in data:
            continue

        ax = axes[idx]
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
    for method_key, label in [('global_shap', 'Global'),
                               ('rolling_shap', 'Rolling'),
                               ('adaptive_shap', 'Adaptive')]:
        if method_key in data:
            df = data[method_key]
            ax.plot(df['end_index'], df['y_hat'], label=label, alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Time Index')
    ax.set_ylabel('Prediction')
    ax.set_title('Predictions Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plots
    scatter_pairs = [
        (('global_shap', 'rolling_shap'), 'Global vs Rolling', axes[0, 1]),
        (('global_shap', 'adaptive_shap'), 'Global vs Adaptive', axes[1, 0]),
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
            corr = np.corrcoef(merged['y_hat_1'], merged['y_hat_2'])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.4f}',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(method1.replace('_', ' ').title())
            ax.set_ylabel(method2.replace('_', ' ').title())
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

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
    ax.legend()
    ax.grid(True, alpha=0.3)

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
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'window_size_analysis.png'), bbox_inches='tight')
    print(f"Saved: window_size_analysis.png")
    plt.close()


def plot_temporal_faithfulness(data, save_dir):
    """Plot faithfulness scores over time for each method."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Faithfulness Scores Over Time', fontsize=14, fontweight='bold')

    eval_types = ['prtb', 'sqnc']
    percentiles = ['p90', 'p70', 'p50']

    for i, eval_type in enumerate(eval_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[i, j]

            eval_key = f'{eval_type}_{percentile}'
            col_name = f'faithfulness_{eval_key}'

            # Plot for each method
            for method_key, label, color in [
                ('global_shap', 'Global', '#1f77b4'),
                ('rolling_shap', 'Rolling', '#ff7f0e'),
                ('adaptive_shap', 'Adaptive', '#2ca02c')
            ]:
                if method_key in data:
                    df = data[method_key]
                    if col_name in df.columns:
                        # Plot with transparency and smoothing
                        ax.plot(df['end_index'], df[col_name], label=label,
                               alpha=0.6, linewidth=1.5, color=color)

                        # Add rolling mean for trend
                        if len(df) > 20:
                            window = min(50, len(df) // 10)
                            rolling_mean = df[col_name].rolling(window=window, center=True).mean()
                            ax.plot(df['end_index'], rolling_mean,
                                   linestyle='--', linewidth=2, color=color, alpha=0.8, label=label)

            ax.set_xlabel('Time Index')
            ax.set_ylabel('Faithfulness Score')
            ax.set_title(f'{eval_type.upper()} - {percentile.upper()}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temporal_faithfulness.png'), bbox_inches='tight')
    print(f"Saved: temporal_faithfulness.png")
    plt.close()


def plot_temporal_ablation(data, save_dir):
    """Plot ablation scores over time for each method."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ablation Scores Over Time', fontsize=14, fontweight='bold')

    ablation_types = ['mif', 'lif']
    percentiles = ['p90', 'p70', 'p50']

    for i, ablation_type in enumerate(ablation_types):
        for j, percentile in enumerate(percentiles):
            ax = axes[i, j]

            eval_key = f'{ablation_type}_{percentile}'
            col_name = f'ablation_{eval_key}'

            # Plot for each method
            for method_key, label, color in [
                ('global_shap', 'Global', '#1f77b4'),
                ('rolling_shap', 'Rolling', '#ff7f0e'),
                ('adaptive_shap', 'Adaptive', '#2ca02c')
            ]:
                if method_key in data:
                    df = data[method_key]
                    if col_name in df.columns:
                        # Plot with transparency and smoothing
                        ax.plot(df['end_index'], df[col_name], label=label,
                               alpha=0.6, linewidth=1.5, color=color)

                        # Add rolling mean for trend
                        if len(df) > 20:
                            window = min(50, len(df) // 10)
                            rolling_mean = df[col_name].rolling(window=window, center=True).mean()
                            ax.plot(df['end_index'], rolling_mean,
                                   linestyle='--', linewidth=2, color=color, alpha=0.8)

            ax.set_xlabel('Time Index')
            ax.set_ylabel('Ablation Score')
            ax.set_title(f'{ablation_type.upper()} - {percentile.upper()}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
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
        colors = {'global_shap': '#1f77b4', 'rolling_shap': '#ff7f0e', 'adaptive_shap': '#2ca02c'}

        for method in ['global_shap', 'rolling_shap', 'adaptive_shap']:
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
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

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
    for method_key, method_name in [('global_shap', 'Global'),
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
    for method_key, label, shap_prefix in [('global_shap', 'Global', 'shap_lag_t'),
                                            ('rolling_shap', 'Rolling', 'shap_lag_t'),
                                            ('adaptive_shap', 'Adaptive', 'shap_lstm_t')]:
        if method_key in data:
            df = data[method_key]
            shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]
            if len(shap_cols) > 0:
                total_shap = df[shap_cols].sum(axis=1)
                ax3.plot(df['end_index'], total_shap, label=label, alpha=0.7, linewidth=1.5)

    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Total |SHAP|')
    ax3.set_title('Total SHAP Values Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

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
        ax4.grid(True, alpha=0.3)

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
        ax5.grid(True, alpha=0.3)

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
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')

    plt.savefig(os.path.join(save_dir, 'summary_dashboard.png'), bbox_inches='tight')
    print(f"Saved: summary_dashboard.png")
    plt.close()


def main():
    """Main visualization function."""
    # ============================================================
    # CHOOSE DATASET TYPE: 'simulated' or 'empirical'
    # ============================================================
    RUN_TYPE = "empirical"  # Change to "empirical" for empirical data

    if RUN_TYPE == "simulated":
        # Simulated dataset results directory
        dataset_type = "ar"
        order = "3"
        results_dir = f'examples/results/benchmark_{dataset_type}_{order}'
    elif RUN_TYPE == "empirical":
        # Empirical dataset results directory
        results_dir = 'examples/results/benchmark_empirical'
    else:
        raise ValueError(f"Invalid RUN_TYPE: {RUN_TYPE}. Must be 'simulated' or 'empirical'")

    save_dir = os.path.join(results_dir, 'figures')

    # Create figures directory
    os.makedirs(save_dir, exist_ok=True)

    print("="*60)
    print(f"Benchmark Visualization - {RUN_TYPE.title()} Data")
    print("="*60)
    print(f"Loading data from: {results_dir}")

    # Load data
    data = load_benchmark_data(results_dir)

    print(f"Loaded {len(data)} data files")
    print(f"Saving figures to: {save_dir}")
    print("="*60)

    # Generate plots
    print("\nGenerating visualizations...")

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

    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"All figures saved to: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
