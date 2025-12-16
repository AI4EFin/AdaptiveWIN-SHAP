"""
Analyze Bootstrap Confidence Interval Results

Computes 95% confidence intervals and generates visualizations for:
- Faithfulness scores
- MIF/LIF ratios
- Correlation with true importance
- Window size statistics
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


class BootstrapAnalyzer:
    """Analyze bootstrap confidence interval results."""

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_df = None
        self.load_results()

    def load_results(self):
        """Load results from CSV."""
        results_path = self.results_dir / 'results.csv'
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        self.results_df = pd.read_csv(results_path)
        print(f"Loaded {len(self.results_df)} results")
        print(f"Datasets: {self.results_df['dataset'].unique()}")
        print(f"Success rate: {self.results_df['success'].mean():.1%}")

    def compute_confidence_intervals(self, alpha=0.05):
        """
        Compute confidence intervals for all metrics.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI)

        Returns
        -------
        pd.DataFrame
            Summary statistics with confidence intervals
        """
        # Metrics to analyze
        metric_cols = [
            'window_mean', 'window_std', 'window_min', 'window_max',
            'faithfulness_prtb_p90', 'faithfulness_prtb_p50',
            'faithfulness_prtb_p10',
            'ablation_mif_p90', 'ablation_mif_p50', 'ablation_mif_p10',
            'ablation_lif_p90', 'ablation_lif_p50', 'ablation_lif_p10',
            'correlation_true_imp_mean',
            'detection_time'
        ]

        # Filter successful runs
        df = self.results_df[self.results_df['success'] == True].copy()

        summary_list = []

        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]

            for metric in metric_cols:
                if metric not in dataset_df.columns:
                    continue

                values = dataset_df[metric].dropna()

                if len(values) == 0:
                    continue

                # Compute statistics
                mean = values.mean()
                std = values.std()
                median = values.median()

                # 95% CI using t-distribution (appropriate for small samples)
                n = len(values)
                se = std / np.sqrt(n)

                # t-distribution critical value
                t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

                ci_lower = mean - t_crit * se
                ci_upper = mean + t_crit * se

                # CI width and coefficient of variation
                ci_width = ci_upper - ci_lower
                cv = std / mean if mean != 0 else np.nan

                summary_list.append({
                    'dataset': dataset,
                    'metric': metric,
                    'n_samples': n,
                    'mean': mean,
                    'std': std,
                    'median': median,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_width,
                    'cv': cv,  # Coefficient of variation
                    'ci_width_pct': (ci_width / mean * 100) if mean != 0 else np.nan
                })

        summary_df = pd.DataFrame(summary_list)

        # Save to CSV
        output_path = self.results_dir / 'confidence_intervals.csv'
        summary_df.to_csv(output_path, index=False)
        print(f"\nSaved confidence intervals to {output_path}")

        return summary_df

    def plot_confidence_intervals(self, ci_df):
        """
        Generate confidence interval plots for key metrics.

        Parameters
        ----------
        ci_df : pd.DataFrame
            Confidence interval dataframe from compute_confidence_intervals()
        """
        # Key metrics for main plot
        key_metrics = [
            'faithfulness_prtb_p90',
            'ablation_mif_p90',
            'ablation_lif_p90',
            'correlation_true_imp_mean',
            'window_mean'
        ]

        # Friendly names for plotting
        metric_names = {
            'faithfulness_prtb_p90': 'Faithfulness (p90)',
            'ablation_mif_p90': 'MIF Score (p90)',
            'ablation_lif_p90': 'LIF Score (p90)',
            'correlation_true_imp_mean': 'Correlation with True Importance',
            'window_mean': 'Mean Window Size'
        }

        # Filter to key metrics
        plot_df = ci_df[ci_df['metric'].isin(key_metrics)].copy()

        if len(plot_df) == 0:
            print("Warning: No key metrics found for plotting")
            return

        # Create figure with subplots
        datasets = plot_df['dataset'].unique()
        n_datasets = len(datasets)

        fig, axes = plt.subplots(n_datasets, 1, figsize=(12, 4*n_datasets))

        if n_datasets == 1:
            axes = [axes]

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            dataset_df = plot_df[plot_df['dataset'] == dataset].copy()

            # Sort by metric order
            dataset_df['metric_order'] = dataset_df['metric'].apply(
                lambda x: key_metrics.index(x) if x in key_metrics else 999
            )
            dataset_df = dataset_df.sort_values('metric_order')

            # Create labels
            labels = [metric_names.get(m, m) for m in dataset_df['metric']]
            y_pos = np.arange(len(labels))

            # Plot confidence intervals as error bars
            ax.errorbar(
                dataset_df['mean'].values,
                y_pos,
                xerr=np.array([
                    dataset_df['mean'].values - dataset_df['ci_lower'].values,
                    dataset_df['ci_upper'].values - dataset_df['mean'].values
                ]),
                fmt='o',
                markersize=10,
                capsize=8,
                capthick=2,
                linewidth=2,
                color='steelblue',
                ecolor='steelblue',
                label='95% CI'
            )

            # Add vertical line at mean
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Metric Value', fontsize=12)
            ax.set_title(f'Bootstrap Confidence Intervals - {dataset}',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            ax.legend()

            # Add sample size annotation
            for i, row in enumerate(dataset_df.itertuples()):
                ax.text(
                    row.ci_upper + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                    i,
                    f'n={int(row.n_samples)}',
                    va='center',
                    fontsize=9,
                    color='gray'
                )

        plt.tight_layout()
        output_path = self.results_dir / 'confidence_intervals_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CI plot to {output_path}")

    def plot_distributions(self):
        """Plot distributions of metrics across bootstrap samples."""
        df = self.results_df[self.results_df['success'] == True].copy()

        # Key metrics to visualize
        metrics = [
            'faithfulness_prtb_p90',
            'ablation_mif_p90',
            'correlation_true_imp_mean',
            'window_mean'
        ]

        for metric in metrics:
            if metric not in df.columns:
                continue

            datasets = df['dataset'].unique()
            n_datasets = len(datasets)

            fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4))

            if n_datasets == 1:
                axes = [axes]

            for idx, dataset in enumerate(datasets):
                ax = axes[idx]
                values = df[df['dataset'] == dataset][metric].dropna()

                if len(values) == 0:
                    continue

                # Histogram + KDE
                ax.hist(values, bins=15, alpha=0.6, color='steelblue',
                       edgecolor='black', density=True, label='Histogram')

                # KDE overlay
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 100)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                except:
                    pass

                # Add mean and median lines
                mean_val = values.mean()
                median_val = values.median()
                ax.axvline(mean_val, color='darkblue', linestyle='--',
                          linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='orange', linestyle='-.',
                          linewidth=2, label=f'Median: {median_val:.3f}')

                ax.set_xlabel(metric, fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = self.results_dir / f'distribution_{metric}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved distribution plot: {output_path}")

    def generate_summary_table(self, ci_df):
        """
        Generate a summary table highlighting key findings.

        Parameters
        ----------
        ci_df : pd.DataFrame
            Confidence interval dataframe
        """
        # Filter to key metrics
        key_metrics = [
            'faithfulness_prtb_p90',
            'ablation_mif_p90',
            'correlation_true_imp_mean',
            'window_mean'
        ]

        summary = ci_df[ci_df['metric'].isin(key_metrics)].copy()

        # Round values for readability
        summary['mean_str'] = summary['mean'].apply(lambda x: f"{x:.3f}")
        summary['ci_str'] = summary.apply(
            lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
            axis=1
        )
        summary['cv_str'] = summary['cv'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

        # Select columns for display
        display_df = summary[[
            'dataset', 'metric', 'n_samples', 'mean_str', 'ci_str', 'cv_str'
        ]]
        display_df.columns = ['Dataset', 'Metric', 'N', 'Mean', '95% CI', 'CV']

        # Save to text file
        output_path = self.results_dir / 'summary_table.txt'
        with open(output_path, 'w') as f:
            f.write("Bootstrap Confidence Intervals - Summary\n")
            f.write("="*80 + "\n\n")
            f.write(display_df.to_string(index=False))
            f.write("\n\n")
            f.write("Notes:\n")
            f.write("- N: Number of bootstrap samples\n")
            f.write("- CV: Coefficient of Variation (std/mean)\n")
            f.write("- CI width < 20% of mean indicates good stability\n")

        print(f"\nSaved summary table to {output_path}")
        print("\n" + "="*80)
        print("Summary Table Preview:")
        print("="*80)
        print(display_df.to_string(index=False))
        print("="*80)

    def assess_stability(self, ci_df):
        """
        Assess stability based on confidence interval width.

        According to success criteria in the plan:
        - CI width < 20% of mean indicates good stability
        """
        print("\n" + "="*80)
        print("Stability Assessment")
        print("="*80)

        # Key metrics for stability assessment
        key_metrics = [
            'faithfulness_prtb_p90',
            'ablation_mif_p90',
            'correlation_true_imp_mean'
        ]

        for dataset in ci_df['dataset'].unique():
            print(f"\n{dataset}:")
            print("-" * 40)

            dataset_df = ci_df[
                (ci_df['dataset'] == dataset) &
                (ci_df['metric'].isin(key_metrics))
            ]

            for _, row in dataset_df.iterrows():
                metric = row['metric']
                ci_width_pct = row['ci_width_pct']

                # Stability check: CI width < 20% of mean
                is_stable = ci_width_pct < 20 if pd.notna(ci_width_pct) else False
                status = "✓ STABLE" if is_stable else "✗ UNSTABLE"

                print(f"  {metric:30s}: {ci_width_pct:6.2f}% {status}")

        print("="*80)

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*80)
        print("Bootstrap Confidence Interval Analysis")
        print("="*80)

        # 1. Compute confidence intervals
        print("\n1. Computing confidence intervals...")
        ci_df = self.compute_confidence_intervals()

        # 2. Generate summary table
        print("\n2. Generating summary table...")
        self.generate_summary_table(ci_df)

        # 3. Assess stability
        print("\n3. Assessing stability...")
        self.assess_stability(ci_df)

        # 4. Plot confidence intervals
        print("\n4. Generating confidence interval plots...")
        self.plot_confidence_intervals(ci_df)

        # 5. Plot distributions
        print("\n5. Generating distribution plots...")
        self.plot_distributions()

        print("\n" + "="*80)
        print("Analysis Complete!")
        print(f"All outputs saved to: {self.results_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze bootstrap confidence interval results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing bootstrap results')

    args = parser.parse_args()

    analyzer = BootstrapAnalyzer(args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()