"""
Visualize LPA Sensitivity Analysis Results

This script creates comprehensive visualizations for LPA parameter sensitivity
analysis, including individual parameter effects and aggregated summaries.

Usage:
    # Visualize single dataset
    python examples/robustness/visualize_lpa_sensitivity.py --dataset piecewise_ar3

    # Visualize all datasets
    python examples/robustness/visualize_lpa_sensitivity.py --all-datasets

    # Enable window analysis
    python examples/robustness/visualize_lpa_sensitivity.py --dataset piecewise_ar3 --window-analysis
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
    window_analysis: bool = False,
    growth_strategy: str = None
):
    """
    Create visualizations for a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of dataset
    results_dir : Path
        Directory with LPA sensitivity results
    output_dir : Path
        Output directory for figures
    mif_lif_mode : str
        MIF/LIF display mode: 'ratio', 'all', or 'individual'
    window_analysis : bool
        Whether to create window analysis plots
    growth_strategy : str
        Window growth strategy to visualize (kept for backwards compatibility)
    """
    print(f"\n{'='*80}")
    print(f"Visualizing LPA Sensitivity: {dataset_name}")
    if growth_strategy:
        print(f"Growth Strategy: {growth_strategy}")
    print(f"{'='*80}\n")

    # Load results - try from growth subdirectory first, then fall back to main directory
    dataset_dir = results_dir / dataset_name

    # If growth strategy is specified, look in that subdirectory
    if growth_strategy:
        growth_dir = dataset_dir / growth_strategy
        if growth_dir.exists():
            # Check if summary exists in growth subdirectory
            summary_file = growth_dir.parent / 'sensitivity_summary.csv'
            if not summary_file.exists():
                # Try loading from parent and filtering
                summary_file = dataset_dir / 'sensitivity_summary.csv'
            dataset_dir = growth_dir
        else:
            print(f"Warning: Growth directory not found: {growth_dir}")
            print(f"Trying to filter results from main directory...")
            summary_file = dataset_dir / 'sensitivity_summary.csv'
    else:
        summary_file = dataset_dir / 'sensitivity_summary.csv'

    if not summary_file.exists():
        print(f"Warning: No results found for {dataset_name}")
        return

    results_df = pd.read_csv(summary_file)

    # Filter by growth strategy if specified
    if growth_strategy and 'growth' in results_df.columns:
        results_df = results_df[results_df['growth'] == growth_strategy].copy()
        print(f"Filtered to {growth_strategy} growth strategy")

    print(f"Loaded {len(results_df)} parameter combinations")

    # Initialize visualizer with growth-specific output directory
    if growth_strategy:
        viz_output_dir = output_dir / dataset_name / growth_strategy
        title_prefix = f"{dataset_name} ({growth_strategy})"
    else:
        viz_output_dir = output_dir / dataset_name
        title_prefix = dataset_name

    viz = RobustnessVisualizer(output_dir=viz_output_dir)

    # Compute MIF/LIF ratios
    results_df = viz.compute_mif_lif_ratios(results_df, percentiles=[50, 90])

    # Determine metric columns based on mif_lif_mode
    if mif_lif_mode == 'ratio':
        metric_cols = ['faithfulness', 'mif_lif_ratio_p50', 'mif_lif_ratio_p90', 'window_mean']
    elif mif_lif_mode == 'all':
        metric_cols = ['faithfulness', 'mif_lif_ratio_p50', 'mif_lif_ratio_p90',
                      'ablation_mif_p50', 'ablation_mif_p90',
                      'ablation_lif_p50', 'ablation_lif_p90', 'window_mean']
    else:  # individual
        metric_cols = ['faithfulness', 'ablation_mif_p50', 'ablation_mif_p90',
                      'ablation_lif_p50', 'ablation_lif_p90', 'window_mean']

    # Filter metric_cols to only include columns that exist in the data
    metric_cols = [m for m in metric_cols if m in results_df.columns]

    if len(metric_cols) == 0:
        # Fallback to any numeric columns
        numeric_cols = results_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        metric_cols = [c for c in numeric_cols if c not in ['N0', 'alpha', 'num_bootstrap']]

    print(f"Available metrics: {metric_cols}")

    # 1. N0 sensitivity
    print("\n1. Creating N0 sensitivity plots...")
    for metric in metric_cols:
        if metric in results_df.columns:
            viz.plot_parameter_sensitivity(
                results_df=results_df,
                param_col='N0',
                metric_cols=metric,
                dataset_name=dataset_name,
                title=f'{title_prefix}: {metric} vs N0',
                save_name=f'n0_sensitivity_{metric}',
                ylabel=metric.replace('_', ' ').title(),
                show_error_bars=True
            )
            plt.close()

    # 2. Alpha sensitivity
    print("2. Creating alpha sensitivity plots...")
    for metric in metric_cols:
        if metric in results_df.columns:
            viz.plot_parameter_sensitivity(
                results_df=results_df,
                param_col='alpha',
                metric_cols=metric,
                dataset_name=dataset_name,
                title=f'{title_prefix}: {metric} vs alpha',
                save_name=f'alpha_sensitivity_{metric}',
                ylabel=metric.replace('_', ' ').title(),
                show_error_bars=True
            )
            plt.close()

    # 3. num_bootstrap sensitivity
    print("3. Creating num_bootstrap sensitivity plots...")
    for metric in metric_cols:
        if metric in results_df.columns:
            viz.plot_parameter_sensitivity(
                results_df=results_df,
                param_col='num_bootstrap',
                metric_cols=metric,
                dataset_name=dataset_name,
                title=f'{title_prefix}: {metric} vs num_bootstrap',
                save_name=f'num_bootstrap_sensitivity_{metric}',
                ylabel=metric.replace('_', ' ').title(),
                show_error_bars=True
            )
            plt.close()

    # 4. Multi-metric comparison for N0
    print("4. Creating multi-metric N0 comparison...")
    # Use first 3 available metrics
    if len(metric_cols) >= 3:
        multi_metrics = metric_cols[:3]
    else:
        multi_metrics = metric_cols

    if len(multi_metrics) > 0:
        viz.plot_parameter_sensitivity(
            results_df=results_df,
            param_col='N0',
            metric_cols=multi_metrics,
            dataset_name=dataset_name,
            title=f'{title_prefix}: All Metrics vs N0',
            save_name='n0_sensitivity_all_metrics',
            ylabel='Score',
            show_error_bars=False
        )
        plt.close()

    # 5. Heatmap: N0 vs alpha for first available metric
    print("5. Creating parameter combination heatmap...")
    if 'N0' in results_df.columns and 'alpha' in results_df.columns and len(metric_cols) > 0:
        # Use first available metric
        heatmap_metric = metric_cols[0]
        try:
            # Aggregate over num_bootstrap
            heatmap_data = results_df.groupby(['N0', 'alpha'])[heatmap_metric].mean().unstack()

            viz.plot_heatmap(
                data=heatmap_data,
                title=f'{title_prefix}: {heatmap_metric} (N0 vs alpha)',
                save_name=f'n0_alpha_{heatmap_metric}_heatmap',
                cmap='RdYlGn',
                fmt='.3f'
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not create heatmap: {e}")

    # 6. Stability summary
    print("6. Creating stability summary...")
    if len(metric_cols) > 0:
        try:
            viz.plot_stability_summary(
                results_df=results_df,
                metric_cols=metric_cols,
                stability_threshold=0.2,
                title=f'{title_prefix}: Stability Assessment',
                save_name='stability_summary'
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not create stability summary: {e}")

    # 7. Window analysis (if enabled)
    if window_analysis:
        print("\n7. Creating window analysis plots...")

        # Get breakpoints for this dataset
        breakpoints = DATASET_BREAKPOINTS.get(dataset_name, None)
        if breakpoints is None:
            print(f"  Warning: No breakpoints defined for {dataset_name}, skipping window analysis")
        else:
            # Window vs parameters
            print("  a. Window size vs parameters...")
            for param in ['N0', 'alpha', 'num_bootstrap']:
                fig = viz.plot_window_vs_parameters(
                    results_df=results_df,
                    param_col=param,
                    dataset_name=dataset_name,
                    save_name=f'window_mean_vs_{param.lower()}',
                    window_stat='mean'
                )
                if fig:
                    plt.close(fig)

            # Find best configuration (highest value of first available metric)
            if len(metric_cols) > 0:
                best_idx = results_df[metric_cols[0]].idxmax()
            else:
                best_idx = results_df['window_mean'].idxmax()  # Fallback
            best_config = results_df.loc[best_idx]

            # Build path to windows.csv for best config
            N0_int = int(best_config['N0'])
            alpha_val = best_config['alpha']
            num_boot = int(best_config['num_bootstrap'])

            # Get growth info for best config
            if 'growth' in best_config and 'growth_base' in best_config:
                growth_val = best_config['growth']
                growth_base_val = best_config['growth_base']
                param_str = f"temp_N{N0_int}_alpha{alpha_val}_num_bootstrap{num_boot}_growth{growth_val}_growth_base{growth_base_val}"
            else:
                param_str = f"temp_N{N0_int}_alpha{alpha_val}_num_bootstrap{num_boot}"

            # Try different path combinations
            possible_paths = [
                dataset_dir / param_str / "windows.csv",  # With growth in param string
                dataset_dir / f"temp_N{N0_int:03d}_alpha{alpha_val:.2f}_num_bootstrap{num_boot:03d}/windows.csv",  # Old format
                dataset_dir / f"temp_N{N0_int}_alpha{alpha_val}_num_bootstrap{num_boot}/windows.csv"  # Alternative
            ]

            windows_csv = None
            for path in possible_paths:
                if path.exists():
                    windows_csv = path
                    break

            if windows_csv and windows_csv.exists():
                print(f"  b. Loading windows from best config (N0={N0_int}, alpha={alpha_val}, num_bootstrap={num_boot})...")
                windows_df = pd.read_csv(windows_csv)

                # Window evolution
                print("  c. Window evolution over time...")
                fig = viz.plot_window_evolution(
                    windows_df=windows_df,
                    dataset_name=dataset_name,
                    breakpoints=breakpoints,
                    save_name='window_evolution_best',
                    show_statistics=True
                )
                if fig:
                    plt.close(fig)

                # Window distribution
                print("  d. Window size distribution...")
                fig = viz.plot_window_distribution(
                    windows_df=windows_df,
                    dataset_name=dataset_name,
                    breakpoints=breakpoints,
                    save_name='window_distribution_best'
                )
                if fig:
                    plt.close(fig)

                # True vs detected windows
                print("  e. True vs detected window comparison...")
                fig = viz.plot_true_vs_detected_windows(
                    windows_df=windows_df,
                    dataset_name=dataset_name,
                    breakpoints=breakpoints,
                    save_name='true_vs_detected_windows'
                )
                if fig:
                    plt.close(fig)
            else:
                if windows_csv:
                    print(f"  Warning: Windows file not found: {windows_csv}")
                else:
                    print(f"  Warning: Could not find windows.csv in any expected location")

    print(f"\nAll visualizations saved to: {viz_output_dir}")


def create_cross_dataset_summary(
    results_dir: Path,
    output_dir: Path,
    datasets: list,
    growth_strategy: str = None
):
    """
    Create summary comparing results across all datasets.

    Parameters
    ----------
    results_dir : Path
        Directory with LPA sensitivity results
    output_dir : Path
        Output directory for figures
    datasets : list
        List of dataset names
    growth_strategy : str
        Window growth strategy to visualize (kept for backwards compatibility)
    """
    print(f"\n{'='*80}")
    print("Creating Cross-Dataset Summary")
    if growth_strategy:
        print(f"Growth Strategy: {growth_strategy}")
    print(f"{'='*80}\n")

    # Load all results
    all_results = {}
    for dataset_name in datasets:
        summary_file = results_dir / dataset_name / 'sensitivity_summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            # Filter by growth strategy if specified
            if growth_strategy and 'growth' in df.columns:
                df = df[df['growth'] == growth_strategy].copy()
            if len(df) > 0:  # Only include if data remains after filtering
                all_results[dataset_name] = df

    if not all_results:
        print("No results found for cross-dataset summary")
        return

    # Initialize visualizer with growth-specific output directory
    if growth_strategy:
        viz_output_dir = output_dir / 'cross_dataset' / growth_strategy
        title_suffix = f" ({growth_strategy})"
    else:
        viz_output_dir = output_dir / 'cross_dataset'
        title_suffix = ""

    viz = RobustnessVisualizer(output_dir=viz_output_dir)

    # Find common metrics across all datasets
    all_metrics_sets = [set(df.columns) for df in all_results.values()]
    common_metrics = set.intersection(*all_metrics_sets) if all_metrics_sets else set()

    # Define preferred metrics in order of importance
    preferred_metrics = ['mif_lif_ratio_p50', 'mif_lif_ratio_p90', 'faithfulness',
                        'ablation_mif_p50', 'ablation_mif_p90',
                        'ablation_lif_p50', 'ablation_lif_p90',
                        'window_mean', 'window_std']

    # Use common metrics that are in preferred list, or just window_mean as fallback
    metric_cols = [m for m in preferred_metrics if m in common_metrics]
    if not metric_cols:
        metric_cols = ['window_mean'] if 'window_mean' in common_metrics else []

    # Limit to first 3 metrics for cleaner plots
    metric_cols = metric_cols[:3]

    print(f"Common metrics across datasets: {metric_cols}")

    # 1. Multi-metric comparison across datasets
    if len(metric_cols) > 0:
        print("1. Creating cross-dataset metric comparison...")
        try:
            viz.plot_multi_metric_comparison(
                results_dict=all_results,
                metric_cols=metric_cols,
                title=f'LPA Sensitivity: Cross-Dataset Comparison{title_suffix}',
                save_name='cross_dataset_metrics'
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not create cross-dataset comparison: {e}")

    # 2. Create summary report
    print("2. Creating summary report...")

    test_results = {}
    for dataset_name, df in all_results.items():
        # Use all numeric columns for this dataset
        dataset_metrics = [col for col in df.columns
                          if df[col].dtype in ['float64', 'int64'] and
                          col not in ['N0', 'alpha', 'num_bootstrap']]

        metrics_stats = {}
        for metric in dataset_metrics:
            if metric in df.columns:
                values = df[metric].values
                metrics_stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max()
                }

        # Find best configuration
        if 'faithfulness' in df.columns:
            best_idx = df['faithfulness'].idxmax()
            best_config = df.loc[best_idx]
            findings = [
                f"Best faithfulness: {best_config['faithfulness']:.4f}",
                f"Optimal N0: {best_config['N0']:.0f}",
                f"Optimal alpha: {best_config['alpha']:.3f}",
                f"Optimal num_bootstrap: {best_config['num_bootstrap']:.0f}"
            ]
        else:
            findings = []

        test_results[dataset_name] = {
            'description': 'LPA parameter sensitivity analysis',
            'metrics': metrics_stats,
            'datasets': [dataset_name],
            'n_experiments': len(df),
            'findings': findings
        }

    viz.create_summary_report(
        test_results=test_results,
        output_name='lpa_sensitivity_summary'
    )

    print(f"\nCross-dataset summary saved to: {viz_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize LPA sensitivity analysis results'
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
        default='examples/results/robustness/lpa_sensitivity',
        help='Directory containing LPA sensitivity results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='examples/results/robustness/figures/lpa_sensitivity',
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
    parser.add_argument(
        '--growth',
        type=str,
        choices=['geometric'],
        default='geometric',
        help='Window growth strategy (geometric only, kept for backwards compatibility)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("\nPlease run LPA sensitivity analysis first:")
        print("  python examples/robustness/01_lpa_sensitivity.py --quick-test")
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
                                window_analysis=args.window_analysis,
                                growth_strategy=args.growth)

    # Create cross-dataset summary if multiple datasets
    if len(datasets) > 1:
        create_cross_dataset_summary(results_dir, output_dir, datasets,
                                    growth_strategy=args.growth)

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()