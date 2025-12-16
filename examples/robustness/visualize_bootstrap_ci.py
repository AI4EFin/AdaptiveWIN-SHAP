"""
Visualize Bootstrap Confidence Interval Results

This script creates comprehensive visualizations for bootstrap CI analysis,
including distribution plots, confidence intervals, and stability assessments.

Usage:
    python examples/robustness/visualize_bootstrap_ci.py --dataset piecewise_ar3
    python examples/robustness/visualize_bootstrap_ci.py --all-datasets
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from visualize_robustness import RobustnessVisualizer, DATASET_BREAKPOINTS


def visualize_single_dataset(
    dataset_name: str,
    results_dir: Path,
    output_dir: Path,
    mif_lif_mode: str = 'all',
    window_analysis: bool = False
):
    """
    Create visualizations for a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of dataset
    results_dir : Path
        Directory with bootstrap CI results
    output_dir : Path
        Output directory for figures
    mif_lif_mode : str
        MIF/LIF display mode: 'ratio', 'all', or 'individual'
    window_analysis : bool
        Whether to create window analysis plots
    """
    print(f"\n{'='*80}")
    print(f"Visualizing Bootstrap CI: {dataset_name}")
    print(f"{'='*80}\n")

    # Load results
    dataset_dir = results_dir / dataset_name
    summary_file = dataset_dir / 'ci_summary.csv'
    realizations_file = dataset_dir / 'all_realizations.csv'

    if not summary_file.exists():
        print(f"Warning: No CI summary found for {dataset_name}")
        return

    summary_df = pd.read_csv(summary_file)
    print(f"Loaded CI summary with {len(summary_df)} metrics")

    # Initialize visualizer
    viz = RobustnessVisualizer(output_dir=output_dir / dataset_name)

    # Load realizations if available
    realizations_df = None
    if realizations_file.exists():
        realizations_df = pd.read_csv(realizations_file)
        print(f"Loaded {len(realizations_df)} realizations")

        # Compute MIF/LIF ratios
        realizations_df = viz.compute_mif_lif_ratios(realizations_df, percentiles=[50, 90])

    # Determine metric list based on mif_lif_mode
    if mif_lif_mode == 'ratio':
        metric_list = ['faithfulness', 'mif_lif_ratio_p50', 'mif_lif_ratio_p90']
    elif mif_lif_mode == 'all':
        metric_list = ['faithfulness', 'mif_lif_ratio_p50', 'mif_lif_ratio_p90',
                      'ablation_mif_p50', 'ablation_mif_p90',
                      'ablation_lif_p50', 'ablation_lif_p90']
    else:  # individual
        metric_list = ['faithfulness', 'ablation_mif_p50', 'ablation_mif_p90',
                      'ablation_lif_p50', 'ablation_lif_p90']

    # 1. Distribution plots for each metric
    print("\n1. Creating distribution plots...")
    for _, row in summary_df.iterrows():
        metric_name = row.get('metric', 'unknown')

        # Get values from realizations
        if realizations_df is not None and metric_name in realizations_df.columns:
            values = realizations_df[metric_name].values

            viz.plot_metric_distribution(
                values=values,
                metric_name=metric_name,
                ci=(row.get('ci_lower'), row.get('ci_upper')),
                show_ci=True,
                title=f'{dataset_name}: {metric_name} Bootstrap Distribution',
                save_name=f'{metric_name}_distribution'
            )
            plt.close()

    # 2. Confidence interval summary plot
    print("2. Creating CI summary plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = summary_df['metric'].values
    means = summary_df['mean'].values
    ci_lowers = summary_df['ci_lower'].values
    ci_uppers = summary_df['ci_upper'].values

    # Plot means with error bars
    y_pos = range(len(metrics))
    ax.errorbar(means, y_pos, xerr=[means - ci_lowers, ci_uppers - means],
               fmt='o', markersize=10, capsize=8, linewidth=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Value', fontsize=11)
    ax.set_title(f'{dataset_name}: 95% Confidence Intervals', fontsize=12, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    # Color-code by stability
    if 'stable' in summary_df.columns:
        for i, stable in enumerate(summary_df['stable']):
            color = 'green' if stable else 'red'
            ax.plot(means[i], y_pos[i], 'o', markersize=10, color=color, alpha=0.6)

    fig.tight_layout()
    save_path = output_dir / dataset_name / 'ci_summary_plot.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # 3. CI width comparison
    print("3. Creating CI width comparison...")
    if 'relative_ci_width' in summary_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        rel_widths = summary_df['relative_ci_width'].values
        colors = ['green' if w < 0.2 else 'orange' if w < 0.3 else 'red'
                 for w in rel_widths]

        ax.barh(y_pos, rel_widths, color=colors, alpha=0.7)
        ax.axvline(0.2, color='black', linestyle='--', linewidth=2,
                  label='Stability threshold (20%)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Relative CI Width (CI width / mean)', fontsize=11)
        ax.set_title(f'{dataset_name}: CI Width Assessment', fontsize=12, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        fig.tight_layout()
        save_path = output_dir / dataset_name / 'ci_width_comparison.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    # 4. Stability summary (if realizations available)
    if realizations_df is not None:
        print("4. Creating stability summary...")

        available_metrics = [m for m in metric_list if m in realizations_df.columns]
        if available_metrics:
            viz.plot_stability_summary(
                results_df=realizations_df,
                metric_cols=available_metrics,
                stability_threshold=0.2,
                title=f'{dataset_name}: Stability Assessment',
                save_name='stability_summary'
            )
            plt.close()

    # 5. Convergence plot (if seed column exists)
    if realizations_df is not None and 'seed' in realizations_df.columns:
        print("5. Creating convergence plots...")

        for metric in ['faithfulness', 'mif_score', 'lif_score']:
            if metric in realizations_df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))

                values = realizations_df[metric].values
                cumulative_mean = np.cumsum(values) / np.arange(1, len(values) + 1)
                cumulative_std = pd.Series(values).expanding().std().values

                ax.plot(range(1, len(values) + 1), cumulative_mean,
                       linewidth=2, label='Cumulative Mean')
                ax.fill_between(range(1, len(values) + 1),
                               cumulative_mean - cumulative_std,
                               cumulative_mean + cumulative_std,
                               alpha=0.3, label='±1 Std Dev')

                # Add final CI
                final_ci_row = summary_df[summary_df['metric'] == metric]
                if len(final_ci_row) > 0:
                    ci_lower = final_ci_row['ci_lower'].values[0]
                    ci_upper = final_ci_row['ci_upper'].values[0]
                    ax.axhline(ci_lower, color='red', linestyle='--',
                             label='95% CI', alpha=0.7)
                    ax.axhline(ci_upper, color='red', linestyle='--', alpha=0.7)

                ax.set_xlabel('Number of Realizations', fontsize=11)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
                ax.set_title(f'{dataset_name}: {metric} Convergence',
                           fontsize=12, pad=15)
                ax.legend()
                ax.grid(True, alpha=0.3, linestyle='--')

                fig.tight_layout()
                save_path = output_dir / dataset_name / f'{metric}_convergence.png'
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
                plt.close()

    # 6. Window analysis (if enabled)
    if window_analysis and realizations_df is not None:
        print("\n6. Creating window analysis plots...")

        # Get breakpoints for this dataset
        breakpoints = DATASET_BREAKPOINTS.get(dataset_name, None)
        if breakpoints is None:
            print(f"  Warning: No breakpoints defined for {dataset_name}, skipping window analysis")
        else:
            # Window stability (from realizations)
            print("  a. Window stability across realizations...")
            fig = viz.plot_window_stability(
                results_df=realizations_df,
                dataset_name=dataset_name,
                save_name='window_stability'
            )
            if fig:
                plt.close(fig)

            # Load first seed's windows for detailed analysis
            if 'seed' in realizations_df.columns:
                first_seed = realizations_df['seed'].iloc[0]
                windows_csv = dataset_dir / f"temp_seed_{first_seed}/windows.csv"

                if windows_csv.exists():
                    print(f"  b. Loading windows from seed {first_seed}...")
                    windows_df = pd.read_csv(windows_csv)

                    # Window evolution
                    print("  c. Window evolution over time...")
                    fig = viz.plot_window_evolution(
                        windows_df=windows_df,
                        dataset_name=dataset_name,
                        breakpoints=breakpoints,
                        save_name=f'window_evolution_seed_{first_seed}',
                        show_statistics=True
                    )
                    if fig:
                        plt.close(fig)

                    # True vs detected windows
                    print("  d. True vs detected window comparison...")
                    fig = viz.plot_true_vs_detected_windows(
                        windows_df=windows_df,
                        dataset_name=dataset_name,
                        breakpoints=breakpoints,
                        save_name='true_vs_detected_windows'
                    )
                    if fig:
                        plt.close(fig)

                # Aggregate window distribution across first 10 seeds
                print("  e. Aggregated window distribution...")
                all_windows = []
                for seed in realizations_df['seed'].iloc[:10]:
                    w_csv = dataset_dir / f"temp_seed_{seed}/windows.csv"
                    if w_csv.exists():
                        try:
                            w_df = pd.read_csv(w_csv)
                            if 'window_mean' in w_df.columns:
                                all_windows.extend(w_df['window_mean'].dropna().values)
                            elif 'windows' in w_df.columns:
                                all_windows.extend(w_df['windows'].dropna().values)
                        except:
                            pass

                if len(all_windows) > 0:
                    fig = viz.plot_window_distribution(
                        windows_df=pd.DataFrame({'window_mean': all_windows}),
                        dataset_name=dataset_name,
                        breakpoints=breakpoints,
                        save_name='window_distribution_aggregated'
                    )
                    if fig:
                        plt.close(fig)

    print(f"\nAll visualizations saved to: {output_dir / dataset_name}")


def create_cross_dataset_summary(
    results_dir: Path,
    output_dir: Path,
    datasets: list
):
    """
    Create summary comparing results across all datasets.

    Parameters
    ----------
    results_dir : Path
        Directory with bootstrap CI results
    output_dir : Path
        Output directory for figures
    datasets : list
        List of dataset names
    """
    print(f"\n{'='*80}")
    print("Creating Cross-Dataset Summary")
    print(f"{'='*80}\n")

    # Load all results
    all_summaries = {}
    all_realizations = {}

    for dataset_name in datasets:
        summary_file = results_dir / dataset_name / 'ci_summary.csv'
        realizations_file = results_dir / dataset_name / 'all_realizations.csv'

        if summary_file.exists():
            all_summaries[dataset_name] = pd.read_csv(summary_file)

        if realizations_file.exists():
            all_realizations[dataset_name] = pd.read_csv(realizations_file)

    if not all_summaries:
        print("No results found for cross-dataset summary")
        return

    # Initialize visualizer
    viz = RobustnessVisualizer(output_dir=output_dir / 'cross_dataset')

    # 1. Multi-metric comparison across datasets (using realizations)
    if all_realizations:
        print("1. Creating cross-dataset metric comparison...")
        metric_cols = ['faithfulness', 'mif_score', 'lif_score']

        viz.plot_multi_metric_comparison(
            results_dict=all_realizations,
            metric_cols=metric_cols,
            title='Bootstrap CI: Cross-Dataset Comparison',
            save_name='cross_dataset_metrics'
        )
        plt.close()

    # 2. CI width comparison across datasets
    print("2. Creating cross-dataset CI width comparison...")

    fig, ax = plt.subplots(figsize=(12, 8))

    dataset_names = list(all_summaries.keys())
    metrics = all_summaries[dataset_names[0]]['metric'].unique()

    # Create grouped bar chart
    x = np.arange(len(metrics))
    width = 0.8 / len(dataset_names)

    for i, dataset_name in enumerate(dataset_names):
        df = all_summaries[dataset_name]
        rel_widths = df['relative_ci_width'].values if 'relative_ci_width' in df.columns else []

        if len(rel_widths) > 0:
            ax.bar(x + i * width, rel_widths, width, label=dataset_name, alpha=0.7)

    ax.axhline(0.2, color='black', linestyle='--', linewidth=2,
              label='Stability threshold (20%)')
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Relative CI Width', fontsize=11)
    ax.set_title('Bootstrap CI: Relative CI Width Across Datasets',
               fontsize=12, pad=15)
    ax.set_xticks(x + width * (len(dataset_names) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    fig.tight_layout()
    save_path = output_dir / 'cross_dataset' / 'ci_width_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # 3. Create summary report
    print("3. Creating summary report...")

    test_results = {}
    for dataset_name, summary_df in all_summaries.items():
        metrics_stats = {}

        for _, row in summary_df.iterrows():
            metric_name = row.get('metric', 'unknown')
            metrics_stats[metric_name] = {
                'mean': row.get('mean', np.nan),
                'std': row.get('std', np.nan),
                'min': row.get('ci_lower', np.nan),
                'max': row.get('ci_upper', np.nan)
            }

        # Findings
        stable_count = summary_df['stable'].sum() if 'stable' in summary_df.columns else 0
        total_metrics = len(summary_df)
        findings = [
            f"Bootstrap realizations: {len(all_realizations.get(dataset_name, []))}",
            f"Stable metrics: {stable_count}/{total_metrics} ({stable_count/total_metrics*100:.0f}%)",
            f"Mean CI width: {summary_df['relative_ci_width'].mean():.2%}" if 'relative_ci_width' in summary_df.columns else ""
        ]

        test_results[dataset_name] = {
            'description': 'Bootstrap confidence interval analysis with randomized DGP parameters',
            'metrics': metrics_stats,
            'datasets': [dataset_name],
            'n_experiments': len(all_realizations.get(dataset_name, [])),
            'findings': [f for f in findings if f]
        }

    viz.create_summary_report(
        test_results=test_results,
        output_name='bootstrap_ci_summary'
    )

    print(f"\nCross-dataset summary saved to: {output_dir / 'cross_dataset'}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize bootstrap CI results'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name to visualize (e.g., piecewise_ar3)'
    )
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Visualize all datasets'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='examples/results/robustness/bootstrap_ci',
        help='Directory containing bootstrap CI results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='examples/results/robustness/figures/bootstrap_ci',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--mif-lif-mode',
        type=str,
        choices=['ratio', 'all', 'individual'],
        default='all',
        help='MIF/LIF display mode: "ratio" (show only ratios), "all" (ratios + individual scores), "individual" (individual scores only)'
    )
    parser.add_argument(
        '--window-analysis',
        action='store_true',
        help='Enable window size analysis plots'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("\nPlease run bootstrap CI analysis first:")
        print("  python examples/robustness/06_bootstrap_ci.py --quick-test")
        sys.exit(1)

    # Determine datasets to process
    if args.all_datasets:
        datasets = [d.name for d in results_dir.iterdir() if d.is_dir()]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Error: Please specify --dataset or --all-datasets")
        sys.exit(1)

    # Process each dataset
    for dataset_name in datasets:
        visualize_single_dataset(dataset_name, results_dir, output_dir,
                                mif_lif_mode=args.mif_lif_mode,
                                window_analysis=args.window_analysis)

    # Create cross-dataset summary if multiple datasets
    if len(datasets) > 1:
        create_cross_dataset_summary(results_dir, output_dir, datasets)

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
