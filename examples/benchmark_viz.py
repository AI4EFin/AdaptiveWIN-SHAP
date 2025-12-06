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
        plot_ablation_mif_vs_lif(data, save_dir)
    except Exception as e:
        print(f"Error in MIF vs LIF comparison: {e}")

    try:
        plot_shap_over_time(data, save_dir)
    except Exception as e:
        print(f"Error in SHAP over time: {e}")

    try:
        plot_correlation_with_true_importance(data, save_dir)
    except Exception as e:
        print(f"Error in correlation with true importance: {e}")

    try:
        plot_all_methods_with_true_importance(data, save_dir)
    except Exception as e:
        print(f"Error in all methods with true importance: {e}")

    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"All figures saved to: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
