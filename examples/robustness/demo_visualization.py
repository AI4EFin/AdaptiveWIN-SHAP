"""
Demo Script for Robustness Visualization Framework

This script demonstrates the visualization framework with synthetic data.
Run this to verify the framework is working correctly before using with real results.

Usage:
    python examples/robustness/demo_visualization.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

from visualize_robustness import RobustnessVisualizer


def generate_synthetic_lpa_results(n_samples=50):
    """Generate synthetic LPA sensitivity results."""

    np.random.seed(42)

    results = []

    # Parameter grid
    N0_values = [50, 75, 100, 150, 200, 250]
    alpha_values = [0.90, 0.95, 0.99]
    num_bootstrap_values = [50, 100, 200, 500]

    for N0 in N0_values:
        for alpha in alpha_values:
            for num_bootstrap in num_bootstrap_values:
                # Simulate metrics with some parameter dependencies
                base_faithfulness = 0.80 + 0.10 * (N0 / 250)
                base_faithfulness += 0.05 * ((alpha - 0.90) / 0.09)
                base_faithfulness -= 0.02 * (num_bootstrap > 200)

                faithfulness = base_faithfulness + np.random.normal(0, 0.02)
                mif_score = faithfulness + np.random.normal(0, 0.03)
                lif_score = faithfulness * 0.9 + np.random.normal(0, 0.03)
                avg_window = 50 + 30 * (N0 / 250) + np.random.normal(0, 5)

                results.append({
                    'N0': N0,
                    'alpha': alpha,
                    'num_bootstrap': num_bootstrap,
                    'faithfulness': np.clip(faithfulness, 0, 1),
                    'mif_score': np.clip(mif_score, 0, 1),
                    'lif_score': np.clip(lif_score, 0, 1),
                    'avg_window_size': max(10, avg_window)
                })

    return pd.DataFrame(results)


def generate_synthetic_bootstrap_results(n_realizations=100):
    """Generate synthetic bootstrap results."""

    np.random.seed(42)

    # Generate realizations with slight skew
    faithfulness_vals = np.random.beta(80, 20, n_realizations)  # Mean ~0.8
    mif_vals = np.random.beta(90, 10, n_realizations)  # Mean ~0.9
    lif_vals = np.random.beta(75, 25, n_realizations)  # Mean ~0.75
    window_vals = np.random.normal(80, 15, n_realizations)

    realizations_df = pd.DataFrame({
        'seed': range(1000, 1000 + n_realizations),
        'faithfulness': faithfulness_vals,
        'mif_score': mif_vals,
        'lif_score': lif_vals,
        'avg_window_size': window_vals
    })

    # Compute CI summary
    from scipy import stats

    def compute_ci(values, alpha=0.05):
        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        ci_width = ci_upper - ci_lower
        rel_ci_width = ci_width / mean if mean != 0 else np.inf
        stable = rel_ci_width < 0.2

        return {
            'mean': mean,
            'std': std,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'relative_ci_width': rel_ci_width,
            'stable': stable
        }

    summary_rows = []
    for metric in ['faithfulness', 'mif_score', 'lif_score', 'avg_window_size']:
        ci_stats = compute_ci(realizations_df[metric].values)
        ci_stats['metric'] = metric
        summary_rows.append(ci_stats)

    summary_df = pd.DataFrame(summary_rows)

    return realizations_df, summary_df


def demo_lpa_visualizations():
    """Demonstrate LPA sensitivity visualizations."""

    print("\n" + "="*80)
    print("DEMO: LPA Sensitivity Visualizations")
    print("="*80)

    # Generate synthetic data
    print("\nGenerating synthetic LPA sensitivity results...")
    results_df = generate_synthetic_lpa_results()
    print(f"Generated {len(results_df)} parameter combinations")

    # Initialize visualizer
    output_dir = Path('examples/results/robustness/demo/lpa_sensitivity')
    viz = RobustnessVisualizer(output_dir=output_dir)

    # 1. N0 sensitivity
    print("\n1. Creating N0 sensitivity plot...")
    viz.plot_parameter_sensitivity(
        results_df=results_df,
        param_col='N0',
        metric_cols=['faithfulness', 'mif_score', 'lif_score'],
        dataset_name='demo_dataset',
        title='Demo: Metrics vs N0',
        save_name='n0_sensitivity',
        show_error_bars=True
    )

    # 2. Alpha sensitivity
    print("2. Creating alpha sensitivity plot...")
    viz.plot_parameter_sensitivity(
        results_df=results_df,
        param_col='alpha',
        metric_cols='faithfulness',
        dataset_name='demo_dataset',
        save_name='alpha_sensitivity_faithfulness',
        ylabel='Faithfulness Score',
        show_error_bars=True
    )

    # 3. Heatmap
    print("3. Creating parameter combination heatmap...")
    heatmap_data = results_df.groupby(['N0', 'alpha'])['faithfulness'].mean().unstack()
    viz.plot_heatmap(
        data=heatmap_data,
        title='Demo: Faithfulness (N0 vs alpha)',
        save_name='n0_alpha_heatmap',
        cmap='RdYlGn',
        fmt='.3f'
    )

    # 4. Stability summary
    print("4. Creating stability summary...")
    viz.plot_stability_summary(
        results_df=results_df,
        metric_cols=['faithfulness', 'mif_score', 'lif_score'],
        stability_threshold=0.2,
        title='Demo: Stability Assessment',
        save_name='stability_summary'
    )

    print(f"\n✓ LPA visualizations saved to: {output_dir}")


def demo_bootstrap_visualizations():
    """Demonstrate bootstrap CI visualizations."""

    print("\n" + "="*80)
    print("DEMO: Bootstrap CI Visualizations")
    print("="*80)

    # Generate synthetic data
    print("\nGenerating synthetic bootstrap results...")
    realizations_df, summary_df = generate_synthetic_bootstrap_results(n_realizations=100)
    print(f"Generated {len(realizations_df)} realizations")

    # Initialize visualizer
    output_dir = Path('examples/results/robustness/demo/bootstrap_ci')
    viz = RobustnessVisualizer(output_dir=output_dir)

    # 1. Distribution plots
    print("\n1. Creating distribution plots...")
    for _, row in summary_df.iterrows():
        metric = row['metric']
        values = realizations_df[metric].values

        viz.plot_metric_distribution(
            values=values,
            metric_name=metric,
            ci=(row['ci_lower'], row['ci_upper']),
            show_ci=True,
            title=f'Demo: {metric} Bootstrap Distribution',
            save_name=f'{metric}_distribution'
        )

    # 2. Stability summary
    print("2. Creating stability summary...")
    viz.plot_stability_summary(
        results_df=realizations_df,
        metric_cols=['faithfulness', 'mif_score', 'lif_score'],
        stability_threshold=0.2,
        title='Demo: Bootstrap Stability',
        save_name='stability_summary'
    )

    print(f"\n✓ Bootstrap CI visualizations saved to: {output_dir}")


def demo_cross_test_comparison():
    """Demonstrate cross-test comparison."""

    print("\n" + "="*80)
    print("DEMO: Cross-Test Comparison")
    print("="*80)

    # Generate synthetic results for two tests
    print("\nGenerating synthetic results for comparison...")
    test_a_results = generate_synthetic_lpa_results()
    test_b_results = generate_synthetic_bootstrap_results()[0]

    # Initialize visualizer
    output_dir = Path('examples/results/robustness/demo/cross_test')
    viz = RobustnessVisualizer(output_dir=output_dir)

    # Multi-metric comparison
    print("\nCreating cross-test comparison...")
    viz.plot_multi_metric_comparison(
        results_dict={
            'LPA_Sensitivity': test_a_results,
            'Bootstrap_CI': test_b_results
        },
        metric_cols=['faithfulness', 'mif_score', 'lif_score'],
        title='Demo: Cross-Test Metric Comparison',
        save_name='cross_test_comparison'
    )

    # Summary report
    print("\nCreating summary report...")

    test_results = {
        'LPA_Sensitivity': {
            'description': 'Parameter sensitivity analysis',
            'metrics': {
                'faithfulness': {
                    'mean': test_a_results['faithfulness'].mean(),
                    'std': test_a_results['faithfulness'].std(),
                    'min': test_a_results['faithfulness'].min(),
                    'max': test_a_results['faithfulness'].max()
                },
                'mif_score': {
                    'mean': test_a_results['mif_score'].mean(),
                    'std': test_a_results['mif_score'].std(),
                    'min': test_a_results['mif_score'].min(),
                    'max': test_a_results['mif_score'].max()
                }
            },
            'datasets': ['demo_dataset'],
            'n_experiments': len(test_a_results),
            'findings': [
                'Optimal N0: 250',
                'Optimal alpha: 0.99',
                'Faithfulness improves with larger N0'
            ]
        },
        'Bootstrap_CI': {
            'description': 'Bootstrap confidence intervals',
            'metrics': {
                'faithfulness': {
                    'mean': test_b_results['faithfulness'].mean(),
                    'std': test_b_results['faithfulness'].std(),
                    'min': test_b_results['faithfulness'].min(),
                    'max': test_b_results['faithfulness'].max()
                },
                'mif_score': {
                    'mean': test_b_results['mif_score'].mean(),
                    'std': test_b_results['mif_score'].std(),
                    'min': test_b_results['mif_score'].min(),
                    'max': test_b_results['mif_score'].max()
                }
            },
            'datasets': ['demo_dataset'],
            'n_experiments': len(test_b_results),
            'findings': [
                'Narrow confidence intervals (< 10% of mean)',
                'All metrics stable across realizations',
                'Method demonstrates strong robustness'
            ]
        }
    }

    viz.create_summary_report(
        test_results=test_results,
        output_name='demo_robustness_summary'
    )

    print(f"\n✓ Cross-test comparison saved to: {output_dir}")


def main():
    """Run all demos."""

    print("="*80)
    print("ROBUSTNESS VISUALIZATION FRAMEWORK - DEMO")
    print("="*80)
    print("\nThis demo creates synthetic data and generates example visualizations")
    print("to verify the visualization framework is working correctly.")

    try:
        # Run demos
        demo_lpa_visualizations()
        demo_bootstrap_visualizations()
        demo_cross_test_comparison()

        print("\n" + "="*80)
        print("DEMO COMPLETE")
        print("="*80)
        print("\nAll demo visualizations created successfully!")
        print("\nCheck the following directories:")
        print("  - examples/results/robustness/demo/lpa_sensitivity/")
        print("  - examples/results/robustness/demo/bootstrap_ci/")
        print("  - examples/results/robustness/demo/cross_test/")
        print("\nThe framework is ready to use with your actual results.")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
