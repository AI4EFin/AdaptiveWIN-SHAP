"""
Aggregate benchmark results across all datasets.

Creates:
1. CSV tables with mean metrics across all datasets
2. Comparison plots for all methods across all datasets
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Set style
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Dataset configurations
DATASETS = [
    'piecewise_ar3',
    'arx_rotating',
    'trend_season',
    'spike_process',
    'garch_regime'
]

# Method configurations with display names
METHODS = {
    'global_shap': 'Vanilla SHAP',
    'timeshap': 'TimeShap',
    'rolling_shap': 'Rolling SHAP',
    'adaptive_shap_max': 'Adaptive (Max)',
    'adaptive_shap_mean': 'Adaptive (Mean)',
    'adaptive_shap': 'Adaptive SHAP',
    'adaptive_shap_rolling_mean': 'Adaptive (Smoothed)'
}

# Metric configurations
PERCENTILE = 'p90'  # Use p90 for all metrics


def compute_correlation_with_true_importance(method_df, true_imp_df, dataset_name):
    """
    Compute correlation between SHAP values and true importances.

    Returns dict with pearson and spearman correlations (overall and per-feature).
    """
    # Find ALL SHAP columns: lag features (shap_lag_*) AND covariate features (shap_Z_*)
    shap_lag_cols = sorted([c for c in method_df.columns if c.startswith('shap_lag_')])
    shap_cov_cols = sorted([c for c in method_df.columns if c.startswith('shap_Z_')])
    shap_cols = shap_lag_cols + shap_cov_cols

    # Find true importance columns
    true_imp_cols = sorted([c for c in true_imp_df.columns if c.startswith('true_imp_')])

    if len(shap_cols) == 0 or len(true_imp_cols) == 0:
        return None

    if len(shap_cols) != len(true_imp_cols):
        print(f"  Warning: SHAP columns ({len(shap_cols)}) != true importance columns ({len(true_imp_cols)}) for {dataset_name}")
        print(f"    SHAP cols: {shap_cols}")
        print(f"    True imp cols: {true_imp_cols}")
        # Match the minimum length
        n_features = min(len(shap_cols), len(true_imp_cols))
        shap_cols = shap_cols[:n_features]
        true_imp_cols = true_imp_cols[:n_features]

    # Align indices - true importances may be longer than SHAP results
    min_len = min(len(method_df), len(true_imp_df))

    # Extract SHAP values and true importances
    shap_values = method_df[shap_cols].iloc[:min_len].values  # [T, n_features]
    true_values = true_imp_df[true_imp_cols].iloc[:min_len].values  # [T, n_features]

    # Compute per-feature correlations
    per_feature_corr = {}
    for i, (shap_col, true_col) in enumerate(zip(shap_cols, true_imp_cols)):
        shap_feat = shap_values[:, i]
        true_feat = true_values[:, i]

        # Remove NaN values
        valid_mask = ~(np.isnan(shap_feat) | np.isnan(true_feat))
        shap_feat = shap_feat[valid_mask]
        true_feat = true_feat[valid_mask]

        if len(shap_feat) >= 2:
            pearson_feat, _ = pearsonr(shap_feat, true_feat)
            spearman_feat, _ = spearmanr(shap_feat, true_feat)

            # Extract feature name from column
            # e.g., "shap_lag_t-1" -> "lag_t-1", "shap_Z_0" -> "Z_0"
            if shap_col.startswith('shap_lag_'):
                feature_name = shap_col.replace('shap_lag_', 'lag_')
            elif shap_col.startswith('shap_Z_'):
                feature_name = shap_col.replace('shap_', '')
            else:
                feature_name = f'feature_{i}'

            per_feature_corr[feature_name] = {
                'pearson': pearson_feat,
                'spearman': spearman_feat
            }

    # Flatten for overall correlation computation
    shap_flat = shap_values.flatten()
    true_flat = true_values.flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(shap_flat) | np.isnan(true_flat))
    shap_flat = shap_flat[valid_mask]
    true_flat = true_flat[valid_mask]

    if len(shap_flat) < 2:
        return None

    # Compute overall correlations
    pearson_corr, _ = pearsonr(shap_flat, true_flat)
    spearman_corr, _ = spearmanr(shap_flat, true_flat)

    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'per_feature': per_feature_corr
    }


def load_dataset_results(dataset_name):
    """Load all method results for a given dataset."""
    results_dir = f"examples/results/benchmark_{dataset_name}"

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None

    # Load summary
    summary_path = os.path.join(results_dir, 'benchmark_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return None

    summary_df = pd.read_csv(summary_path)

    # Load true importances if available
    true_imp_path = f"examples/datasets/simulated/{dataset_name}/true_importances.csv"
    true_imp_df = None
    if os.path.exists(true_imp_path):
        true_imp_df = pd.read_csv(true_imp_path)

    # Load individual method results for correlation computation
    method_dfs = {}
    for method_key in METHODS.keys():
        result_path = os.path.join(results_dir, f'{method_key}_results.csv')
        if os.path.exists(result_path):
            method_dfs[method_key] = pd.read_csv(result_path)

    return {
        'summary': summary_df,
        'true_importances': true_imp_df,
        'method_dfs': method_dfs
    }


def aggregate_metrics_across_datasets():
    """
    Aggregate metrics across all datasets.

    Returns:
    - faithfulness_df: DataFrame with faithfulness metrics
    - ablation_df: DataFrame with ablation metrics
    - correlation_df: DataFrame with correlation metrics
    """

    faithfulness_data = []
    ablation_data = []
    correlation_data = []
    per_feature_correlation_data = []  # NEW: Store per-feature correlations

    for dataset_name in DATASETS:
        print(f"\nProcessing {dataset_name}...")

        data = load_dataset_results(dataset_name)
        if data is None:
            print(f"  Skipping {dataset_name}")
            continue

        summary_df = data['summary']
        true_imp_df = data['true_importances']
        method_dfs = data['method_dfs']

        # Process each method
        for method_key, method_name in METHODS.items():
            # Handle different percentile naming conventions
            # TimeShap uses "90" while others use "p90"
            percentile_variants = [f'prtb_{PERCENTILE}', 'prtb_90']  # Try both

            # Get faithfulness metrics
            faith_prtb = summary_df[
                (summary_df['method'] == method_key) &
                (summary_df['metric_type'] == 'faithfulness') &
                (summary_df['evaluation'].isin(percentile_variants))
            ]

            if len(faith_prtb) > 0:
                faithfulness_data.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'faithfulness_prtb': faith_prtb['score'].values[0]
                })

            # Get ablation metrics (MIF and LIF)
            mif_variants = [f'mif_{PERCENTILE}', 'mif_90']
            ablation_mif = summary_df[
                (summary_df['method'] == method_key) &
                (summary_df['metric_type'] == 'ablation') &
                (summary_df['evaluation'].isin(mif_variants))
            ]

            lif_variants = [f'lif_{PERCENTILE}', 'lif_90']
            ablation_lif = summary_df[
                (summary_df['method'] == method_key) &
                (summary_df['metric_type'] == 'ablation') &
                (summary_df['evaluation'].isin(lif_variants))
            ]

            if len(ablation_mif) > 0:
                mif_val = ablation_mif['score'].values[0]
                lif_val = ablation_lif['score'].values[0] if len(ablation_lif) > 0 else np.nan
                # Compute MIF/LIF ratio - higher is better (MIF should be >> LIF)
                ratio = mif_val / lif_val if not np.isnan(lif_val) and lif_val > 0 else np.nan
                ablation_data.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'ablation_mif': mif_val,
                    'ablation_lif': lif_val,
                    'mif_lif_ratio': ratio
                })

            # Compute correlation with true importance if available
            if true_imp_df is not None and method_key in method_dfs:
                corr_result = compute_correlation_with_true_importance(
                    method_dfs[method_key],
                    true_imp_df,
                    dataset_name
                )

                if corr_result is not None:
                    # Overall correlation - use absolute values
                    correlation_data.append({
                        'dataset': dataset_name,
                        'method': method_name,
                        'pearson': abs(corr_result['pearson']),
                        'spearman': abs(corr_result['spearman'])
                    })
                    print(f"  {method_name}: |Pearson|={abs(corr_result['pearson']):.4f}, |Spearman|={abs(corr_result['spearman']):.4f}")

                    # Per-feature correlations - use absolute values
                    for feature_name, feature_corr in corr_result['per_feature'].items():
                        per_feature_correlation_data.append({
                            'dataset': dataset_name,
                            'method': method_name,
                            'feature': feature_name,
                            'pearson': abs(feature_corr['pearson']),
                            'spearman': abs(feature_corr['spearman'])
                        })

    # Create DataFrames
    faithfulness_df = pd.DataFrame(faithfulness_data)
    ablation_df = pd.DataFrame(ablation_data)
    correlation_df = pd.DataFrame(correlation_data)
    per_feature_correlation_df = pd.DataFrame(per_feature_correlation_data)

    return faithfulness_df, ablation_df, correlation_df, per_feature_correlation_df


def create_aggregated_tables(faithfulness_df, ablation_df, correlation_df, output_dir):
    """
    Create aggregated tables with mean metrics across datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data
    faithfulness_df.to_csv(os.path.join(output_dir, 'faithfulness_all_datasets.csv'), index=False)
    ablation_df.to_csv(os.path.join(output_dir, 'ablation_all_datasets.csv'), index=False)
    correlation_df.to_csv(os.path.join(output_dir, 'correlation_all_datasets.csv'), index=False)

    print("\n" + "="*60)
    print("Creating aggregated tables...")
    print("="*60)

    # Aggregate by method (mean across datasets)
    faith_agg = faithfulness_df.groupby('method')[['faithfulness_prtb']].agg(['mean', 'std'])
    ablation_agg = ablation_df.groupby('method')[['ablation_mif', 'ablation_lif', 'mif_lif_ratio']].agg(['mean', 'std'])
    correlation_agg = correlation_df.groupby('method')[['pearson', 'spearman']].agg(['mean', 'std'])

    # Flatten column names
    faith_agg.columns = ['_'.join(col).strip() for col in faith_agg.columns.values]
    ablation_agg.columns = ['_'.join(col).strip() for col in ablation_agg.columns.values]
    correlation_agg.columns = ['_'.join(col).strip() for col in correlation_agg.columns.values]

    # Combine all metrics into one table
    combined_agg = pd.concat([
        faith_agg,
        ablation_agg,
        correlation_agg
    ], axis=1)

    # Save aggregated tables
    combined_agg.to_csv(os.path.join(output_dir, 'aggregated_metrics_mean_std.csv'))

    print("\nAggregated Metrics (Mean ± Std across datasets):")
    print(combined_agg.to_string())

    # Create a simpler table for presentation (mean only)
    faith_mean = faithfulness_df.groupby('method')[['faithfulness_prtb']].mean()
    ablation_mean = ablation_df.groupby('method')[['ablation_mif', 'ablation_lif', 'mif_lif_ratio']].mean()
    correlation_mean = correlation_df.groupby('method')[['pearson', 'spearman']].mean()

    combined_mean = pd.concat([
        faith_mean,
        ablation_mean,
        correlation_mean
    ], axis=1)

    # Round for presentation
    combined_mean = combined_mean.round(4)

    combined_mean.to_csv(os.path.join(output_dir, 'aggregated_metrics_mean.csv'))

    print("\n\nAggregated Metrics (Mean only, for presentation):")
    print(combined_mean.to_string())

    # Print MIF/LIF ratio analysis
    print("\n" + "="*60)
    print("MIF/LIF Ratio Analysis (Higher = Better discrimination)")
    print("="*60)
    ratio_sorted = ablation_mean['mif_lif_ratio'].sort_values(ascending=False)
    for method, ratio in ratio_sorted.items():
        print(f"  {method:25s}: {ratio:.4f}")
    print("\nNote: MIF/LIF ratio > 1 means the method ablates more error when")
    print("      removing important features first (MIF) vs unimportant first (LIF).")

    return combined_agg, combined_mean


def plot_aggregated_metrics(faithfulness_df, ablation_df, correlation_df, output_dir):
    """
    Create comparison plots for all methods across all datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    # 1. Faithfulness comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Faithfulness PRTB
    faith_prtb_pivot = faithfulness_df.pivot(index='dataset', columns='method', values='faithfulness_prtb')
    faith_prtb_mean = faith_prtb_pivot.mean().sort_values()

    ax.barh(range(len(faith_prtb_mean)), faith_prtb_mean.values)
    ax.set_yticks(range(len(faith_prtb_mean)))
    ax.set_yticklabels(faith_prtb_mean.index)
    ax.set_xlabel('Mean Faithfulness (PRTB)')
    ax.set_title('Faithfulness - Perturbation')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'faithfulness_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: faithfulness_comparison.png")

    # 2. Ablation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MIF
    ablation_mif_pivot = ablation_df.pivot(index='dataset', columns='method', values='ablation_mif')
    ablation_mif_mean = ablation_mif_pivot.mean().sort_values()

    axes[0].barh(range(len(ablation_mif_mean)), ablation_mif_mean.values)
    axes[0].set_yticks(range(len(ablation_mif_mean)))
    axes[0].set_yticklabels(ablation_mif_mean.index)
    axes[0].set_xlabel('Mean Ablation (MIF)')
    axes[0].set_title('Most Important First (MIF)')
    axes[0].grid(axis='x', alpha=0.3)

    # LIF
    ablation_lif_pivot = ablation_df.pivot(index='dataset', columns='method', values='ablation_lif')
    ablation_lif_mean = ablation_lif_pivot.mean().sort_values()

    axes[1].barh(range(len(ablation_lif_mean)), ablation_lif_mean.values)
    axes[1].set_yticks(range(len(ablation_lif_mean)))
    axes[1].set_yticklabels(ablation_lif_mean.index)
    axes[1].set_xlabel('Mean Ablation (LIF)')
    axes[1].set_title('Least Important First (LIF)')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: ablation_comparison.png")

    # 3. Correlation comparison
    if len(correlation_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pearson
        corr_pearson_pivot = correlation_df.pivot(index='dataset', columns='method', values='pearson')
        corr_pearson_mean = corr_pearson_pivot.mean().sort_values(ascending=False)

        axes[0].barh(range(len(corr_pearson_mean)), corr_pearson_mean.values)
        axes[0].set_yticks(range(len(corr_pearson_mean)))
        axes[0].set_yticklabels(corr_pearson_mean.index)
        axes[0].set_xlabel('Mean |Pearson Correlation|')
        axes[0].set_title('Absolute Correlation with True Importance (Pearson)')
        axes[0].grid(axis='x', alpha=0.3)

        # Spearman
        corr_spearman_pivot = correlation_df.pivot(index='dataset', columns='method', values='spearman')
        corr_spearman_mean = corr_spearman_pivot.mean().sort_values(ascending=False)

        axes[1].barh(range(len(corr_spearman_mean)), corr_spearman_mean.values)
        axes[1].set_yticks(range(len(corr_spearman_mean)))
        axes[1].set_yticklabels(corr_spearman_mean.index)
        axes[1].set_xlabel('Mean |Spearman Correlation|')
        axes[1].set_title('Absolute Correlation with True Importance (Spearman)')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_comparison.png'), bbox_inches='tight', dpi=300)
        plt.close()
        print("  Saved: correlation_comparison.png")

    # 4. Combined dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Faithfulness
    ax1 = fig.add_subplot(gs[0, 0])
    faith_mean = faithfulness_df.groupby('method')[['faithfulness_prtb']].mean()
    faith_mean.plot(kind='barh', ax=ax1)
    ax1.set_xlabel('Mean Score')
    ax1.set_title('Faithfulness Metrics')
    ax1.legend(['PRTB'], loc='best')
    ax1.grid(axis='x', alpha=0.3)

    # Ablation
    ax2 = fig.add_subplot(gs[0, 1])
    ablation_mean = ablation_df.groupby('method')[['ablation_mif', 'ablation_lif']].mean()
    ablation_mean.plot(kind='barh', ax=ax2)
    ax2.set_xlabel('Mean Score')
    ax2.set_title('Ablation Metrics')
    ax2.legend(['MIF', 'LIF'], loc='best')
    ax2.grid(axis='x', alpha=0.3)

    # Correlation
    if len(correlation_df) > 0:
        ax3 = fig.add_subplot(gs[1, :])
        corr_mean = correlation_df.groupby('method')[['pearson', 'spearman']].mean()
        corr_mean.plot(kind='barh', ax=ax3)
        ax3.set_xlabel('Mean |Correlation|')
        ax3.set_title('Absolute Correlation with True Importance')
        ax3.legend(['|Pearson|', '|Spearman|'], loc='best')
        ax3.grid(axis='x', alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'combined_dashboard.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: combined_dashboard.png")

    # 5. Heatmaps for each metric across datasets and methods
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Faithfulness PRTB heatmap
    faith_prtb_pivot = faithfulness_df.pivot(index='dataset', columns='method', values='faithfulness_prtb')
    sns.heatmap(faith_prtb_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[0], cbar_kws={'label': 'Score'})
    axes[0].set_title('Faithfulness (PRTB) - Lower is Better')
    axes[0].set_xlabel('')

    # Ablation MIF heatmap
    ablation_mif_pivot = ablation_df.pivot(index='dataset', columns='method', values='ablation_mif')
    sns.heatmap(ablation_mif_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[1], cbar_kws={'label': 'Score'})
    axes[1].set_title('Ablation (MIF) - Lower is Better')
    axes[1].set_xlabel('')

    # Ablation LIF heatmap
    ablation_lif_pivot = ablation_df.pivot(index='dataset', columns='method', values='ablation_lif')
    sns.heatmap(ablation_lif_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[2], cbar_kws={'label': 'Score'})
    axes[2].set_title('Ablation (LIF) - Lower is Better')
    axes[2].set_xlabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: metrics_heatmap.png")

    # 6. Correlation heatmap (if available)
    if len(correlation_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        corr_pearson_pivot = correlation_df.pivot(index='dataset', columns='method', values='pearson')
        sns.heatmap(corr_pearson_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0],
                   cbar_kws={'label': '|Correlation|'}, vmin=0, vmax=1)
        axes[0].set_title('Absolute Pearson Correlation with True Importance - Higher is Better')
        axes[0].set_xlabel('')

        corr_spearman_pivot = correlation_df.pivot(index='dataset', columns='method', values='spearman')
        sns.heatmap(corr_spearman_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
                   cbar_kws={'label': '|Correlation|'}, vmin=0, vmax=1)
        axes[1].set_title('Absolute Spearman Correlation with True Importance - Higher is Better')
        axes[1].set_xlabel('')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight', dpi=300)
        plt.close()
        print("  Saved: correlation_heatmap.png")

    # 7. MIF/LIF Ratio comparison - CRITICAL METRIC
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot
    ratio_pivot = ablation_df.pivot(index='dataset', columns='method', values='mif_lif_ratio')
    ratio_mean = ratio_pivot.mean().sort_values(ascending=False)

    axes[0].barh(range(len(ratio_mean)), ratio_mean.values)
    axes[0].set_yticks(range(len(ratio_mean)))
    axes[0].set_yticklabels(ratio_mean.index)
    axes[0].set_xlabel('Mean MIF/LIF Ratio')
    axes[0].set_title('MIF/LIF Ratio - Higher is Better')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Baseline (ratio=1)')
    axes[0].legend()

    # Heatmap
    sns.heatmap(ratio_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
               cbar_kws={'label': 'MIF/LIF Ratio'}, vmin=0.8, vmax=2.5, center=1.0)
    axes[1].set_title('MIF/LIF Ratio by Dataset - Higher is Better')
    axes[1].set_xlabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mif_lif_ratio_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: mif_lif_ratio_comparison.png")


def main():
    """Main aggregation pipeline."""
    print("="*60)
    print("Aggregating Results Across All Datasets")
    print("="*60)

    # Aggregate metrics
    faithfulness_df, ablation_df, correlation_df, per_feature_corr_df = aggregate_metrics_across_datasets()

    # Output directory
    output_dir = "examples/results/aggregated"
    os.makedirs(output_dir, exist_ok=True)

    # Save per-feature correlations
    if len(per_feature_corr_df) > 0:
        per_feature_corr_df.to_csv(os.path.join(output_dir, 'correlation_per_feature.csv'), index=False)
        print(f"\nSaved per-feature correlations: {len(per_feature_corr_df)} entries")

        # Print per-feature summary
        print("\n" + "="*60)
        print("Per-Feature Correlation Summary")
        print("="*60)
        feature_summary = per_feature_corr_df.groupby(['feature', 'method'])['pearson'].mean().unstack()
        print(feature_summary.round(3).to_string())

    # Create tables
    create_aggregated_tables(faithfulness_df, ablation_df, correlation_df, output_dir)

    # Create plots
    plot_aggregated_metrics(faithfulness_df, ablation_df, correlation_df, output_dir)

    print("\n" + "="*60)
    print(f"Aggregation complete! Results saved to: {output_dir}")
    print("="*60)

    print("\nGenerated files:")
    print("  CSV Tables:")
    print("    - faithfulness_all_datasets.csv")
    print("    - ablation_all_datasets.csv (includes MIF/LIF ratio)")
    print("    - correlation_all_datasets.csv (overall)")
    print("    - correlation_per_feature.csv (PER-FEATURE breakdown!)")
    print("    - aggregated_metrics_mean_std.csv (includes MIF/LIF ratio)")
    print("    - aggregated_metrics_mean.csv (includes MIF/LIF ratio)")
    print("\n  Plots:")
    print("    - faithfulness_comparison.png")
    print("    - ablation_comparison.png")
    print("    - correlation_comparison.png")
    print("    - combined_dashboard.png")
    print("    - metrics_heatmap.png")
    print("    - correlation_heatmap.png")
    print("    - mif_lif_ratio_comparison.png (CRITICAL METRIC)")


if __name__ == "__main__":
    main()