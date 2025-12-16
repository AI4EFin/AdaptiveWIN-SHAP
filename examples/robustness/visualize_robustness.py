"""
Generic Visualization Framework for Robustness Tests

This module provides reusable visualization functions for different types of
robustness analyses. It supports:
- Individual test result visualizations
- Aggregated cross-test comparisons
- Summary reports suitable for paper figures

Usage:
    from visualize_robustness import RobustnessVisualizer

    viz = RobustnessVisualizer(output_dir='results/figures')
    viz.plot_parameter_sensitivity(results_df, param_col='N0', metric_col='faithfulness')
    viz.plot_metric_distribution(values, metric_name='faithfulness')
    viz.create_summary_report(all_results, test_names=['LPA_sensitivity', 'Bootstrap_CI'])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import warnings

# Set publication-quality plot defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
sns.set_palette("colorblind")

# Dataset breakpoints for simulated datasets
# Maps dataset name to list of breakpoint timepoints
DATASET_BREAKPOINTS = {
    'piecewise_ar3': [500, 1000],
    'arx_rotating': [500, 1000],
    'trend_season': [500, 1000],
    'spike_process': [750],
    'garch_regime': [750]
}


class RobustnessVisualizer:
    """Generic visualizer for robustness test results."""

    def __init__(self, output_dir: Union[str, Path] = 'results/robustness/figures'):
        """
        Initialize visualizer.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_mif_lif_ratios(
        self,
        results_df: pd.DataFrame,
        percentiles: List[int] = [50, 90]
    ) -> pd.DataFrame:
        """
        Compute MIF/LIF ratios from existing MIF and LIF scores.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with ablation_mif_p{X} and ablation_lif_p{X} columns
        percentiles : list of int
            Percentiles to compute ratios for (default: [50, 90])

        Returns
        -------
        results_df : pd.DataFrame
            Same DataFrame with added mif_lif_ratio_p{X} columns
        """
        for p in percentiles:
            mif_col = f'ablation_mif_p{p}'
            lif_col = f'ablation_lif_p{p}'
            ratio_col = f'mif_lif_ratio_p{p}'

            if mif_col in results_df.columns and lif_col in results_df.columns:
                # Compute ratio, handling division by zero
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    results_df[ratio_col] = results_df[mif_col] / results_df[lif_col]

                # Replace inf with NaN
                results_df[ratio_col] = results_df[ratio_col].replace([np.inf, -np.inf], np.nan)

        return results_df

    def plot_parameter_sensitivity(
        self,
        results_df: pd.DataFrame,
        param_col: str,
        metric_cols: Union[str, List[str]],
        dataset_name: Optional[str] = None,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_points: bool = True,
        show_error_bars: bool = False,
        error_col: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot how metrics vary with a parameter.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results with columns for parameter and metrics
        param_col : str
            Column name for the parameter being varied
        metric_cols : str or list of str
            Column name(s) for metric(s) to plot
        dataset_name : str, optional
            Name of dataset being analyzed
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save (without extension)
        ylabel : str, optional
            Y-axis label
        show_points : bool
            Whether to show individual data points
        show_error_bars : bool
            Whether to show error bars
        error_col : str, optional
            Column name for error values (if show_error_bars=True)

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if isinstance(metric_cols, str):
            metric_cols = [metric_cols]

        fig, ax = plt.subplots(figsize=(10, 6))

        for metric_col in metric_cols:
            # Group by parameter and compute mean
            grouped = results_df.groupby(param_col)[metric_col].agg(['mean', 'std', 'count'])

            # Plot line
            ax.plot(grouped.index, grouped['mean'], marker='o',
                   label=metric_col, linewidth=2, markersize=8)

            # Add error bars if requested
            if show_error_bars and error_col in results_df.columns:
                ax.errorbar(grouped.index, grouped['mean'],
                           yerr=results_df.groupby(param_col)[error_col].mean(),
                           fmt='none', capsize=5, alpha=0.5)
            elif show_error_bars:
                # Use std error
                se = grouped['std'] / np.sqrt(grouped['count'])
                ax.fill_between(grouped.index,
                               grouped['mean'] - se,
                               grouped['mean'] + se,
                               alpha=0.2)

            # Add individual points if requested
            if show_points and len(results_df) < 100:
                ax.scatter(results_df[param_col], results_df[metric_col],
                          alpha=0.3, s=30)

        ax.set_xlabel(param_col, fontsize=11)
        ax.set_ylabel(ylabel or 'Metric Value', fontsize=11)

        if title:
            ax.set_title(title, fontsize=12, pad=15)
        elif dataset_name:
            ax.set_title(f'{dataset_name}: {param_col} Sensitivity', fontsize=12, pad=15)

        if len(metric_cols) > 1:
            ax.legend(frameon=True, fancybox=True)

        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_metric_distribution(
        self,
        values: Union[np.ndarray, list],
        metric_name: str = 'Metric',
        ci: Optional[Tuple[float, float]] = None,
        show_ci: bool = True,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        reference_value: Optional[float] = None
    ) -> plt.Figure:
        """
        Plot distribution of a metric (e.g., from bootstrap).

        Parameters
        ----------
        values : array-like
            Metric values
        metric_name : str
            Name of metric
        ci : tuple of (lower, upper), optional
            Confidence interval bounds
        show_ci : bool
            Whether to show confidence interval lines
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save
        reference_value : float, optional
            Reference value to mark (e.g., original estimate)

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        values = np.array(values)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(values, bins=30, alpha=0.6, edgecolor='black', density=True)

        # KDE
        from scipy import stats
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Mean line
        mean_val = values.mean()
        ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')

        # CI lines
        if ci is not None and show_ci:
            ax.axvline(ci[0], color='green', linestyle=':', linewidth=2,
                      label=f'95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]')
            ax.axvline(ci[1], color='green', linestyle=':', linewidth=2)

        # Reference value
        if reference_value is not None:
            ax.axvline(reference_value, color='orange', linestyle='-', linewidth=2,
                      label=f'Reference: {reference_value:.4f}')

        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title or f'{metric_name} Distribution', fontsize=12, pad=15)
        ax.legend(frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_heatmap(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        cmap: str = 'RdYlGn',
        fmt: str = '.3f',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center: Optional[float] = None
    ) -> plt.Figure:
        """
        Plot heatmap for parameter combinations or cross-test comparisons.

        Parameters
        ----------
        data : DataFrame or array
            2D data to plot
        row_labels : list of str, optional
            Row labels
        col_labels : list of str, optional
            Column labels
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save
        cmap : str
            Colormap name
        fmt : str
            Format string for cell annotations
        vmin, vmax : float, optional
            Color scale limits
        center : float, optional
            Center value for diverging colormap

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data, index=row_labels, columns=col_labels)

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap,
                   vmin=vmin, vmax=vmax, center=center,
                   square=False, linewidths=0.5, cbar_kws={'shrink': 0.8},
                   ax=ax)

        ax.set_title(title or 'Heatmap', fontsize=12, pad=15)

        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_multi_metric_comparison(
        self,
        results_dict: Dict[str, pd.DataFrame],
        metric_cols: List[str],
        test_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare multiple metrics across different tests or configurations.

        Parameters
        ----------
        results_dict : dict
            Dictionary mapping test names to DataFrames with metrics
        metric_cols : list of str
            Metric columns to compare
        test_names : list of str, optional
            Names for tests (defaults to dict keys)
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if test_names is None:
            test_names = list(results_dict.keys())

        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metric_cols):
            ax = axes[idx]

            # Collect data
            data_list = []
            labels_list = []

            for test_name, df in results_dict.items():
                if metric in df.columns:
                    data_list.append(df[metric].values)
                    labels_list.append(test_name)

            # Box plot
            bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True,
                           showmeans=True, meanline=True)

            # Color boxes
            colors = sns.color_palette("colorblind", len(data_list))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            # Rotate labels if needed
            if max(len(l) for l in labels_list) > 10:
                ax.set_xticklabels(labels_list, rotation=45, ha='right')

        if title:
            fig.suptitle(title, fontsize=12, y=1.02)

        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_stability_summary(
        self,
        results_df: pd.DataFrame,
        metric_cols: List[str],
        stability_threshold: float = 0.2,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot stability summary showing CV (coefficient of variation) for metrics.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results with metric columns
        metric_cols : list of str
            Metrics to analyze
        stability_threshold : float
            CV threshold for stability (default 0.2 = 20%)
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Compute statistics
        stats_data = []
        for metric in metric_cols:
            values = results_df[metric].values
            mean_val = values.mean()
            std_val = values.std()
            cv = std_val / mean_val if mean_val != 0 else np.inf

            stats_data.append({
                'Metric': metric,
                'Mean': mean_val,
                'Std': std_val,
                'CV': cv,
                'Stable': cv < stability_threshold
            })

        stats_df = pd.DataFrame(stats_data)

        # Plot 1: CV bar chart
        colors = ['green' if stable else 'red'
                 for stable in stats_df['Stable']]

        ax1.bar(range(len(stats_df)), stats_df['CV'], color=colors, alpha=0.6)
        ax1.axhline(stability_threshold, color='black', linestyle='--',
                   label=f'Threshold ({stability_threshold:.0%})')
        ax1.set_xticks(range(len(stats_df)))
        ax1.set_xticklabels(stats_df['Metric'], rotation=45, ha='right')
        ax1.set_ylabel('Coefficient of Variation (CV)', fontsize=11)
        ax1.set_title('Stability Assessment (CV)', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Plot 2: Mean ± Std
        x = range(len(stats_df))
        ax2.errorbar(x, stats_df['Mean'], yerr=stats_df['Std'],
                    fmt='o', markersize=8, capsize=5, linewidth=2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stats_df['Metric'], rotation=45, ha='right')
        ax2.set_ylabel('Metric Value', fontsize=11)
        ax2.set_title('Mean ± Std Dev', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')

        if title:
            fig.suptitle(title, fontsize=12, y=1.02)

        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_summary_report(
        self,
        test_results: Dict[str, Dict],
        output_name: str = 'robustness_summary_report'
    ) -> None:
        """
        Create comprehensive summary report across multiple robustness tests.

        Parameters
        ----------
        test_results : dict
            Dictionary mapping test names to result dictionaries.
            Each result dict should have:
            - 'description': str
            - 'metrics': dict of metric_name -> statistics
            - 'datasets': list of dataset names tested
            - 'n_experiments': int
        output_name : str
            Base name for output files
        """
        report_dir = self.output_dir / 'summary_reports'
        report_dir.mkdir(exist_ok=True)

        # Generate markdown report
        report_path = report_dir / f"{output_name}.md"

        with open(report_path, 'w') as f:
            f.write("# Robustness Analysis Summary Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            for test_name, results in test_results.items():
                f.write(f"## {test_name}\n\n")
                f.write(f"**Description**: {results.get('description', 'N/A')}\n\n")
                f.write(f"**Datasets**: {', '.join(results.get('datasets', ['N/A']))}\n\n")
                f.write(f"**Number of Experiments**: {results.get('n_experiments', 'N/A')}\n\n")

                # Metrics table
                if 'metrics' in results:
                    f.write("### Key Metrics\n\n")
                    f.write("| Metric | Mean | Std | Min | Max | CV |\n")
                    f.write("|--------|------|-----|-----|-----|----|)\n")

                    for metric_name, stats in results['metrics'].items():
                        mean = stats.get('mean', np.nan)
                        std = stats.get('std', np.nan)
                        min_val = stats.get('min', np.nan)
                        max_val = stats.get('max', np.nan)
                        cv = std / mean if mean != 0 else np.nan

                        f.write(f"| {metric_name} | {mean:.4f} | {std:.4f} | "
                               f"{min_val:.4f} | {max_val:.4f} | {cv:.4f} |\n")

                    f.write("\n")

                # Additional findings
                if 'findings' in results:
                    f.write("### Key Findings\n\n")
                    for finding in results['findings']:
                        f.write(f"- {finding}\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"Summary report saved: {report_path}")

        # Generate summary figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Overall title
        fig.suptitle('Robustness Analysis Summary', fontsize=14, fontweight='bold')

        # Collect all metrics across tests
        all_metrics = {}
        test_labels = []

        for test_name, results in test_results.items():
            test_labels.append(test_name)
            if 'metrics' in results:
                for metric_name, stats in results['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = {'means': [], 'stds': []}
                    all_metrics[metric_name]['means'].append(stats.get('mean', np.nan))
                    all_metrics[metric_name]['stds'].append(stats.get('std', np.nan))

        # Plot up to 6 metrics
        metric_names = list(all_metrics.keys())[:6]

        for idx, metric_name in enumerate(metric_names):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            means = all_metrics[metric_name]['means']
            stds = all_metrics[metric_name]['stds']

            x = range(len(test_labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(test_labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(f'{metric_name} Across Tests', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        fig.tight_layout()

        fig_path = report_dir / f"{output_name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Summary figure saved: {fig_path}")

        plt.close(fig)

    def plot_window_evolution(
        self,
        windows_df: pd.DataFrame,
        dataset_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        breakpoints: Optional[List[int]] = None,
        show_statistics: bool = True
    ) -> plt.Figure:
        """
        Plot window size evolution over time.

        Parameters
        ----------
        windows_df : pd.DataFrame
            DataFrame with 'window_mean' column and optionally individual run columns
        dataset_name : str
            Name of dataset
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save
        breakpoints : list of int, optional
            Timepoints of regime breakpoints to mark with vertical lines
        show_statistics : bool
            Whether to show statistics panel

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Get individual run columns if they exist
        run_cols = [col for col in windows_df.columns if col.startswith('windows_run_')]

        if len(run_cols) > 0 and len(run_cols) <= 10:
            # Plot individual runs
            for col in run_cols:
                ax.plot(windows_df.index, windows_df[col], alpha=0.3, linewidth=0.5, color='gray')

        # Plot mean if available
        if 'window_mean' in windows_df.columns:
            mean_vals = windows_df['window_mean'].values
            ax.plot(windows_df.index, mean_vals, linewidth=2, label='Mean', color='blue')

            # Add ±1 std band if we can compute it
            if len(run_cols) > 1:
                std_vals = windows_df[run_cols].std(axis=1).values
                ax.fill_between(windows_df.index,
                               mean_vals - std_vals,
                               mean_vals + std_vals,
                               alpha=0.2, color='blue', label='±1 Std Dev')
        elif 'windows' in windows_df.columns:
            # Single run
            ax.plot(windows_df.index, windows_df['windows'], linewidth=2, label='Window Size', color='blue')

        # Mark breakpoints
        if breakpoints is not None:
            for i, bp in enumerate(breakpoints):
                ax.axvline(bp, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax.text(bp, ax.get_ylim()[1] * 0.95, f'Regime {i+2}',
                       ha='left', va='top', fontsize=10, color='red',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlabel('Timepoint', fontsize=11)
        ax.set_ylabel('Window Size', fontsize=11)
        ax.set_title(title or f'{dataset_name}: Window Size Evolution', fontsize=12, pad=15)
        ax.legend(frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics panel if requested
        if show_statistics and 'window_mean' in windows_df.columns:
            stats_text = f"Mean: {mean_vals.mean():.1f}\n"
            stats_text += f"Std: {mean_vals.std():.1f}\n"
            stats_text += f"Min: {mean_vals.min():.1f}\n"
            stats_text += f"Max: {mean_vals.max():.1f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_window_vs_parameters(
        self,
        results_df: pd.DataFrame,
        param_col: str,
        dataset_name: Optional[str] = None,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        window_stat: str = 'mean'
    ) -> plt.Figure:
        """
        Plot window statistics vs parameter values.

        Wrapper around plot_parameter_sensitivity() for window metrics.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results with parameter column and window statistics
        param_col : str
            Parameter column name ('N0', 'alpha', 'num_bootstrap')
        dataset_name : str, optional
            Name of dataset
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save
        window_stat : str
            Which window statistic to plot ('mean', 'std', 'min', 'max')

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        metric_col = f'window_{window_stat}'

        if metric_col not in results_df.columns:
            # Try alternative column names
            if window_stat == 'mean' and 'avg_window_size' in results_df.columns:
                metric_col = 'avg_window_size'
            else:
                print(f"Warning: Column '{metric_col}' not found in results")
                return None

        return self.plot_parameter_sensitivity(
            results_df=results_df,
            param_col=param_col,
            metric_cols=metric_col,
            dataset_name=dataset_name,
            title=title or f'{dataset_name}: Window {window_stat.title()} vs {param_col}',
            save_name=save_name,
            ylabel=f'Window {window_stat.title()}',
            show_error_bars=True
        )

    def plot_window_distribution(
        self,
        windows_df: pd.DataFrame,
        dataset_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        show_breakpoints: bool = True,
        breakpoints: Optional[List[int]] = None
    ) -> plt.Figure:
        """
        Plot distribution of window sizes.

        Parameters
        ----------
        windows_df : pd.DataFrame
            DataFrame with window size data
        dataset_name : str
            Name of dataset
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save
        show_breakpoints : bool
            Whether to show regime length markers
        breakpoints : list of int, optional
            Breakpoint locations

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Get window values
        if 'window_mean' in windows_df.columns:
            window_values = windows_df['window_mean'].dropna().values
        elif 'windows' in windows_df.columns:
            window_values = windows_df['windows'].dropna().values
        else:
            print("Warning: No window size column found")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(window_values, bins=50, alpha=0.6, edgecolor='black', density=True, label='Histogram')

        # KDE
        from scipy import stats as scipy_stats
        try:
            kde = scipy_stats.gaussian_kde(window_values)
            x_range = np.linspace(window_values.min(), window_values.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass

        # Statistics
        mean_val = window_values.mean()
        median_val = np.median(window_values)

        ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')

        # Regime length markers
        if show_breakpoints and breakpoints is not None and len(breakpoints) > 0:
            regime_length = breakpoints[0]  # Assuming equal-length regimes
            ax.axvline(regime_length, color='orange', linestyle='-.', linewidth=2,
                      label=f'Regime length: {regime_length}')

        ax.set_xlabel('Window Size', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title or f'{dataset_name}: Window Size Distribution', fontsize=12, pad=15)
        ax.legend(frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_window_stability(
        self,
        results_df: pd.DataFrame,
        dataset_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bootstrap confidence intervals for window statistics.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with window statistics across realizations
        dataset_name : str
            Name of dataset
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Determine which window columns are available
        window_cols = []
        for col in ['window_mean', 'window_std', 'window_min', 'window_max', 'avg_window_size']:
            if col in results_df.columns:
                window_cols.append(col)

        if len(window_cols) == 0:
            print("Warning: No window statistics found in results")
            return None

        # Compute statistics
        from scipy import stats as scipy_stats

        stats_data = []
        for col in window_cols:
            values = results_df[col].dropna().values
            if len(values) > 1:
                mean = values.mean()
                std = values.std(ddof=1)
                se = std / np.sqrt(len(values))
                t_crit = scipy_stats.t.ppf(0.975, df=len(values)-1)
                ci_lower = mean - t_crit * se
                ci_upper = mean + t_crit * se

                stats_data.append({
                    'metric': col.replace('_', ' ').title(),
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })

        if len(stats_data) == 0:
            print("Warning: Could not compute statistics")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot
        metrics = [d['metric'] for d in stats_data]
        means = [d['mean'] for d in stats_data]
        ci_lowers = [d['ci_lower'] for d in stats_data]
        ci_uppers = [d['ci_upper'] for d in stats_data]

        y_pos = range(len(metrics))
        ax.errorbar(means, y_pos,
                   xerr=[np.array(means) - np.array(ci_lowers),
                        np.array(ci_uppers) - np.array(means)],
                   fmt='o', markersize=10, capsize=8, linewidth=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Window Size', fontsize=11)
        ax.set_title(title or f'{dataset_name}: Window Statistics (95% CI)', fontsize=12, pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_true_vs_detected_windows(
        self,
        windows_df: pd.DataFrame,
        dataset_name: str,
        breakpoints: List[int],
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare detected window sizes to true window sizes (distance to next breakpoint).

        Parameters
        ----------
        windows_df : pd.DataFrame
            DataFrame with window size data and timepoint index
        dataset_name : str
            Name of dataset
        breakpoints : list of int
            Breakpoint timepoints
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Get detected windows
        if 'window_mean' in windows_df.columns:
            detected_windows = windows_df['window_mean'].values
        elif 'windows' in windows_df.columns:
            detected_windows = windows_df['windows'].values
        else:
            print("Warning: No window size column found")
            return None

        # Compute true window sizes (distance to next breakpoint)
        timepoints = windows_df.index.values
        true_windows = np.zeros(len(timepoints))

        for t in timepoints:
            # Find next breakpoint
            future_breakpoints = [bp for bp in breakpoints if bp > t]
            if len(future_breakpoints) > 0:
                true_windows[t] = future_breakpoints[0] - t
            else:
                # After last breakpoint, use distance to end of series
                true_windows[t] = len(timepoints) - t

        # Create 2-panel plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Time series overlay
        ax1.plot(timepoints, true_windows, label='True Window Size', linewidth=2, alpha=0.7, color='green')
        ax1.plot(timepoints, detected_windows, label='Detected Window Size', linewidth=2, alpha=0.7, color='blue')

        # Mark breakpoints
        for bp in breakpoints:
            ax1.axvline(bp, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax1.set_xlabel('Timepoint', fontsize=11)
        ax1.set_ylabel('Window Size', fontsize=11)
        ax1.set_title('True vs Detected Window Sizes', fontsize=11)
        ax1.legend(frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Panel 2: Scatter plot with regression
        ax2.scatter(true_windows, detected_windows, alpha=0.5, s=20)

        # Regression line
        from scipy import stats as scipy_stats
        try:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(true_windows, detected_windows)
            line_x = np.array([true_windows.min(), true_windows.max()])
            line_y = slope * line_x + intercept
            ax2.plot(line_x, line_y, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')

            # Compute metrics
            pearson_r, _ = scipy_stats.pearsonr(true_windows, detected_windows)
            spearman_rho, _ = scipy_stats.spearmanr(true_windows, detected_windows)
            mae = np.mean(np.abs(detected_windows - true_windows))
            rmse = np.sqrt(np.mean((detected_windows - true_windows)**2))

            # Add metrics text
            metrics_text = f'Pearson r: {pearson_r:.3f}\n'
            metrics_text += f'Spearman ρ: {spearman_rho:.3f}\n'
            metrics_text += f'MAE: {mae:.1f}\n'
            metrics_text += f'RMSE: {rmse:.1f}'

            ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except:
            pass

        # Perfect agreement line
        max_val = max(true_windows.max(), detected_windows.max())
        ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.3, label='Perfect Agreement')

        ax2.set_xlabel('True Window Size', fontsize=11)
        ax2.set_ylabel('Detected Window Size', fontsize=11)
        ax2.set_title('Scatter Plot with Regression', fontsize=11)
        ax2.legend(frameon=True, fancybox=True)
        ax2.grid(True, alpha=0.3, linestyle='--')

        fig.suptitle(title or f'{dataset_name}: True vs Detected Window Comparison',
                    fontsize=12, y=1.02)
        fig.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig


def load_lpa_sensitivity_results(results_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load LPA sensitivity analysis results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing LPA sensitivity results

    Returns
    -------
    results : dict
        Dictionary mapping dataset names to results DataFrames
    """
    results_dir = Path(results_dir)
    datasets = {}

    for dataset_dir in results_dir.iterdir():
        if dataset_dir.is_dir():
            summary_file = dataset_dir / 'sensitivity_summary.csv'
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                datasets[dataset_dir.name] = df

    return datasets


def load_bootstrap_ci_results(results_dir: Union[str, Path]) -> Dict[str, Dict]:
    """
    Load bootstrap CI results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing bootstrap CI results

    Returns
    -------
    results : dict
        Dictionary mapping dataset names to result dictionaries with:
        - 'summary': DataFrame with CI statistics
        - 'realizations': DataFrame with all realization results
    """
    results_dir = Path(results_dir)
    datasets = {}

    for dataset_dir in results_dir.iterdir():
        if dataset_dir.is_dir():
            summary_file = dataset_dir / 'ci_summary.csv'
            realizations_file = dataset_dir / 'all_realizations.csv'

            if summary_file.exists():
                datasets[dataset_dir.name] = {
                    'summary': pd.read_csv(summary_file)
                }

                if realizations_file.exists():
                    datasets[dataset_dir.name]['realizations'] = pd.read_csv(realizations_file)

    return datasets


if __name__ == "__main__":
    # Example usage
    print("Robustness Visualization Framework")
    print("=" * 80)
    print("\nThis module provides visualization tools for robustness analysis.")
    print("\nExample usage:")
    print("""
    from visualize_robustness import RobustnessVisualizer

    # Initialize visualizer
    viz = RobustnessVisualizer(output_dir='results/figures')

    # Plot parameter sensitivity
    viz.plot_parameter_sensitivity(
        results_df=lpa_results,
        param_col='N0',
        metric_cols=['faithfulness', 'mif_score'],
        dataset_name='piecewise_ar3',
        save_name='lpa_N0_sensitivity'
    )

    # Plot metric distribution
    viz.plot_metric_distribution(
        values=bootstrap_faithfulness,
        metric_name='Faithfulness',
        ci=(ci_lower, ci_upper),
        save_name='bootstrap_faithfulness_dist'
    )

    # Create summary report
    viz.create_summary_report(
        test_results={
            'LPA_Sensitivity': {...},
            'Bootstrap_CI': {...}
        },
        output_name='full_robustness_summary'
    )
    """)