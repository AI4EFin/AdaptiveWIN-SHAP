"""
Analyze LPA sensitivity results and generate summary reports.

Computes sensitivity indices, generates plots, and creates summary tables.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SensitivityAnalyzer:
    """Analyze LPA sensitivity experiment results."""

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

    def compute_sensitivity_indices(self):
        """
        Compute Sobol-style first-order sensitivity indices.

        For each metric, compute variance explained by each parameter.
        """
        # Filter to only successful runs
        df = self.results_df[self.results_df['success'] == True].copy()

        # Metrics to analyze
        metric_cols = [
            'window_mean', 'window_std',
            'faithfulness_prtb_p90', 'faithfulness_prtb_p50',
            'ablation_mif_p90', 'ablation_lif_p90',
            'correlation_true_imp_mean',
            'breakpoint_detection_lag_mean',
            'detection_time'
        ]

        # Parameters to analyze (support both new mc_reps and legacy num_bootstrap)
        if 'mc_reps' in df.columns:
            param_cols = ['N0', 'alpha', 'mc_reps', 'penalty_factor']
        else:
            param_cols = ['N0', 'alpha', 'num_bootstrap']
        param_cols = [p for p in param_cols if p in df.columns]

        sensitivity_indices = {}

        for metric in metric_cols:
            if metric not in df.columns:
                continue

            # Remove NaN values
            df_clean = df[[metric] + param_cols].dropna()

            if len(df_clean) == 0:
                continue

            total_variance = df_clean[metric].var()

            if total_variance == 0:
                continue

            indices = {}
            for param in param_cols:
                # Compute variance between groups
                group_means = df_clean.groupby(param)[metric].mean()
                between_group_var = group_means.var()

                # First-order Sobol index
                indices[param] = between_group_var / total_variance

            sensitivity_indices[metric] = indices

        # Convert to DataFrame
        si_df = pd.DataFrame(sensitivity_indices).T
        si_df.to_csv(self.results_dir / 'sensitivity_indices.csv')

        print("\nSensitivity Indices (higher = more sensitive to parameter):")
        print("="*60)
        print(si_df.round(3))
        print("="*60)

        return si_df

    def generate_summary_statistics(self):
        """Generate summary statistics per parameter."""
        df = self.results_df[self.results_df['success'] == True].copy()

        if 'mc_reps' in df.columns:
            param_cols = ['N0', 'alpha', 'mc_reps', 'penalty_factor']
        else:
            param_cols = ['N0', 'alpha', 'num_bootstrap']
        param_cols = [p for p in param_cols if p in df.columns]

        metric_cols = [
            'window_mean', 'faithfulness_prtb_p90',
            'correlation_true_imp_mean', 'detection_time'
        ]
        metric_cols = [m for m in metric_cols if m in df.columns]

        summaries = {}

        for param in param_cols:
            param_summary = df.groupby(param)[metric_cols].agg(['mean', 'std', 'count'])
            summaries[param] = param_summary

            # Save to CSV
            output_path = self.results_dir / f'summary_{param}.csv'
            param_summary.to_csv(output_path)
            print(f"\nSaved summary for {param} to {output_path}")

        return summaries

    def plot_sensitivity_heatmaps(self):
        """Generate heatmaps for each parameter."""
        df = self.results_df[self.results_df['success'] == True].copy()

        if 'mc_reps' in df.columns:
            param_cols = ['N0', 'alpha', 'mc_reps', 'penalty_factor']
        else:
            param_cols = ['N0', 'alpha', 'num_bootstrap']
        param_cols = [p for p in param_cols if p in df.columns]

        metric_cols = [
            'window_mean', 'faithfulness_prtb_p90',
            'correlation_true_imp_mean', 'detection_time'
        ]
        metric_cols = [m for m in metric_cols if m in df.columns]

        for param in param_cols:
            # Create aggregated dataframe
            agg_df = df.groupby([param, 'dataset'])[metric_cols].mean().reset_index()

            # Pivot for heatmap
            for metric in metric_cols:
                if metric not in agg_df.columns:
                    continue

                pivot_df = agg_df.pivot(index='dataset', columns=param, values=metric)

                if pivot_df.empty:
                    continue

                # Create heatmap
                plt.figure(figsize=(10, 4))
                sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                           center=pivot_df.mean().mean())

                plt.title(f'{metric} vs. {param}')
                plt.tight_layout()

                output_path = self.results_dir / f'heatmap_{param}_{metric}.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved heatmap: {output_path}")

    def plot_parameter_sensitivity(self):
        """Generate line plots for each parameter."""
        df = self.results_df[self.results_df['success'] == True].copy()

        if 'mc_reps' in df.columns:
            param_cols = ['N0', 'alpha', 'mc_reps', 'penalty_factor']
        else:
            param_cols = ['N0', 'alpha', 'num_bootstrap']
        param_cols = [p for p in param_cols if p in df.columns]

        metrics = ['faithfulness_prtb_p90', 'correlation_true_imp_mean', 'window_mean']
        metrics = [m for m in metrics if m in df.columns]

        for metric in metrics:
            if metric not in df.columns:
                continue

            fig, axes = plt.subplots(1, len(param_cols), figsize=(20, 4))

            for ax, param in zip(axes, param_cols):
                # Group by parameter and compute statistics
                grouped = df.groupby(param)[metric].agg(['mean', 'std', 'count'])
                grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

                # Plot with error bars
                ax.errorbar(grouped.index, grouped['mean'],
                           yerr=1.96 * grouped['se'],  # 95% CI
                           marker='o', capsize=5, capthick=2,
                           linewidth=2, markersize=8,
                           color='steelblue')

                ax.set_xlabel(param, fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'{metric} vs. {param}', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = self.results_dir / f'sensitivity_lines_{metric}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved line plot: {output_path}")

    def generate_recommendations(self):
        """Generate recommended parameter ranges based on results."""
        df = self.results_df[self.results_df['success'] == True].copy()

        # For each parameter, find range that gives best performance
        recommendations = {}

        # N0: Maximize faithfulness while minimizing computation time
        if 'faithfulness_prtb_p90' in df.columns and 'detection_time' in df.columns:
            # Normalize both metrics to [0, 1]
            df['faith_norm'] = (df['faithfulness_prtb_p90'] - df['faithfulness_prtb_p90'].min()) / \
                              (df['faithfulness_prtb_p90'].max() - df['faithfulness_prtb_p90'].min())

            df['time_norm'] = (df['detection_time'] - df['detection_time'].min()) / \
                             (df['detection_time'].max() - df['detection_time'].min())

            # Combined score (maximize faithfulness, minimize time)
            df['combined_score'] = df['faith_norm'] - 0.3 * df['time_norm']

            # Find best N0
            best_n0 = df.groupby('N0')['combined_score'].mean().idxmax()
            recommendations['N0'] = {
                'recommended': int(best_n0),
                'range': f"[{int(best_n0 * 0.7)}, {int(best_n0 * 1.3)}]",
                'reasoning': 'Balances faithfulness and computational cost'
            }

        # Alpha: Standard 0.95 is usually good, but check stability
        if 'window_std' in df.columns:
            best_alpha = df.groupby('alpha')['window_std'].mean().idxmin()
            recommendations['alpha'] = {
                'recommended': float(best_alpha),
                'reasoning': 'Minimizes window size variance'
            }

        # mc_reps / num_bootstrap: More is better but diminishing returns
        mc_col = 'mc_reps' if 'mc_reps' in df.columns else 'num_bootstrap'
        if mc_col in df.columns and 'faithfulness_prtb_p90' in df.columns:
            mc_perf = df.groupby(mc_col)['faithfulness_prtb_p90'].mean()
            # Find elbow point (where improvement < 1%)
            improvements = mc_perf.pct_change()
            min_sufficient = mc_perf.index[improvements < 0.01][0] if any(improvements < 0.01) else mc_perf.index[-1]

            recommendations[mc_col] = {
                'recommended': int(min_sufficient),
                'reasoning': 'Sufficient MC replications for stable critical values'
            }

        # Penalty factor: Find value that minimizes window size variance
        if 'penalty_factor' in df.columns and 'window_std' in df.columns:
            best_penalty = df.groupby('penalty_factor')['window_std'].mean().idxmin()
            recommendations['penalty_factor'] = {
                'recommended': float(best_penalty),
                'reasoning': 'Minimizes window size variance'
            }

        # Save recommendations
        with open(self.results_dir / 'recommendations.txt', 'w') as f:
            f.write("LPA Hyperparameter Recommendations\n")
            f.write("="*60 + "\n\n")

            for param, rec in recommendations.items():
                f.write(f"{param}:\n")
                f.write(f"  Recommended: {rec['recommended']}\n")
                if 'range' in rec:
                    f.write(f"  Safe range: {rec['range']}\n")
                f.write(f"  Reasoning: {rec['reasoning']}\n\n")

        print("\n" + "="*60)
        print("Recommendations:")
        print("="*60)
        for param, rec in recommendations.items():
            print(f"{param}: {rec['recommended']} - {rec['reasoning']}")
        print("="*60)

        return recommendations

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*60)
        print("LPA Sensitivity Analysis")
        print("="*60)

        # 1. Compute sensitivity indices
        print("\n1. Computing sensitivity indices...")
        si_df = self.compute_sensitivity_indices()

        # 2. Generate summary statistics
        print("\n2. Generating summary statistics...")
        summaries = self.generate_summary_statistics()

        # 3. Plot heatmaps
        print("\n3. Generating sensitivity heatmaps...")
        self.plot_sensitivity_heatmaps()

        # 4. Plot line charts
        print("\n4. Generating sensitivity line plots...")
        self.plot_parameter_sensitivity()

        # 5. Generate recommendations
        print("\n5. Generating parameter recommendations...")
        recommendations = self.generate_recommendations()

        print("\n" + "="*60)
        print("Analysis Complete!")
        print(f"All outputs saved to: {self.results_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze LPA sensitivity results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing sensitivity results')

    args = parser.parse_args()

    analyzer = SensitivityAnalyzer(args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()