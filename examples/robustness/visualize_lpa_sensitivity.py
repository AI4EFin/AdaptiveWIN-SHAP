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


def _load_windows_for_config(row, dataset_dir, results_dir):
    """Load a windows.csv for a given results row, trying multiple path strategies."""
    # 1. Use window_csv column directly
    if 'window_csv' in row.index and pd.notna(row.get('window_csv')):
        p = Path(row['window_csv'])
        if p.exists():
            return pd.read_csv(p)

    # 2. Build expected path from parameters
    parts = [f"N0{int(row['N0'])}", f"alpha{row['alpha']}"]
    if 'mc_reps' in row.index and pd.notna(row.get('mc_reps')):
        parts.append(f"mc_reps{int(row['mc_reps'])}")
    if 'penalty_factor' in row.index and pd.notna(row.get('penalty_factor')):
        parts.append(f"penalty_factor{row['penalty_factor']}")
    if 'growth_base' in row.index and pd.notna(row.get('growth_base')):
        parts.append(f"growth_base{row['growth_base']}")
    param_str = "temp_" + "_".join(parts)

    for base in [dataset_dir, results_dir]:
        candidate = base / param_str / "windows.csv"
        if candidate.exists():
            return pd.read_csv(candidate)

    return None


def _plot_window_over_time_grid(
    results_df: pd.DataFrame,
    dataset_dir: Path,
    results_dir: Path,
    dataset_name: str,
    output_dir: Path,
):
    """
    Plot window size over time for every configuration, laid out as a grid.

    Creates two figures:
    - Grid by N0 (subplots), lines colored by penalty_factor (mc_reps fixed to first value)
    - Grid by penalty_factor (subplots), lines colored by N0 (mc_reps fixed to first value)

    Breakpoints are shown as vertical dashed lines.
    Figures are saved into *output_dir*.
    """
    breakpoints = DATASET_BREAKPOINTS.get(dataset_name, [])

    # Fix mc_reps to first available value (they produce identical windows)
    mc_col = 'mc_reps' if 'mc_reps' in results_df.columns else None
    if mc_col and results_df[mc_col].nunique() > 1:
        fixed_mc = sorted(results_df[mc_col].unique())[0]
        df = results_df[results_df[mc_col] == fixed_mc].copy()
    else:
        df = results_df.copy()
        fixed_mc = None

    n0_values = sorted(df['N0'].unique())
    pf_col = 'penalty_factor'
    has_pf = pf_col in df.columns and df[pf_col].nunique() > 1
    pf_values = sorted(df[pf_col].unique()) if has_pf else [None]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Figure 1: one subplot per N0, lines by penalty_factor ----------
    n_cols = min(3, len(n0_values))
    n_rows = int(np.ceil(len(n0_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    cmap = plt.cm.viridis
    pf_norm = plt.Normalize(min(pf_values) if has_pf else 0,
                            max(pf_values) if has_pf else 1)

    for idx, n0 in enumerate(n0_values):
        ax = axes[idx // n_cols][idx % n_cols]
        subset = df[df['N0'] == n0]

        for _, row in subset.iterrows():
            pf_val = row[pf_col] if has_pf else 0
            wdf = _load_windows_for_config(row, dataset_dir, results_dir)
            if wdf is None:
                continue
            col = 'window_mean' if 'window_mean' in wdf.columns else wdf.columns[0]
            color = cmap(pf_norm(pf_val)) if has_pf else 'steelblue'
            label = f"pf={pf_val}" if has_pf else None
            ax.plot(wdf.index, wdf[col], linewidth=1.0, alpha=0.8, color=color, label=label)

        for bp in breakpoints:
            ax.axvline(bp, color='red', linestyle='--', linewidth=1.2, alpha=0.7)

        ax.set_title(f"N0 = {int(n0)}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Window size')
        ax.grid(True, alpha=0.25)

    # Hide unused subplots
    for idx in range(len(n0_values), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # Colour bar for penalty_factor
    if has_pf:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=pf_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_label('penalty_factor', fontsize=11)

    mc_note = f" (mc_reps={int(fixed_mc)})" if fixed_mc is not None else ""
    fig.suptitle(f"{dataset_name}: Window over time by N0{mc_note}",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = output_dir / 'window_over_time_by_n0.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---------- Figure 2: one subplot per penalty_factor, lines by N0 ----------
    if has_pf:
        n_cols2 = min(4, len(pf_values))
        n_rows2 = int(np.ceil(len(pf_values) / n_cols2))
        fig2, axes2 = plt.subplots(n_rows2, n_cols2, figsize=(6 * n_cols2, 4.5 * n_rows2),
                                   sharex=True, sharey=True, squeeze=False)

        n0_cmap = plt.cm.plasma
        n0_norm = plt.Normalize(min(n0_values), max(n0_values))

        for idx, pf_val in enumerate(pf_values):
            ax = axes2[idx // n_cols2][idx % n_cols2]
            subset = df[df[pf_col] == pf_val]

            for _, row in subset.iterrows():
                wdf = _load_windows_for_config(row, dataset_dir, results_dir)
                if wdf is None:
                    continue
                col = 'window_mean' if 'window_mean' in wdf.columns else wdf.columns[0]
                ax.plot(wdf.index, wdf[col], linewidth=1.0, alpha=0.8,
                        color=n0_cmap(n0_norm(row['N0'])))

            for bp in breakpoints:
                ax.axvline(bp, color='red', linestyle='--', linewidth=1.2, alpha=0.7)

            ax.set_title(f"pf = {pf_val}", fontsize=12, fontweight='bold')
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Window size')
            ax.grid(True, alpha=0.25)

        for idx in range(len(pf_values), n_rows2 * n_cols2):
            axes2[idx // n_cols2][idx % n_cols2].set_visible(False)

        sm2 = plt.cm.ScalarMappable(cmap=n0_cmap, norm=n0_norm)
        sm2.set_array([])
        cbar2 = fig2.colorbar(sm2, ax=axes2, shrink=0.6, pad=0.02)
        cbar2.set_label('N0', fontsize=11)

        fig2.suptitle(f"{dataset_name}: Window over time by penalty_factor{mc_note}",
                      fontsize=14, fontweight='bold', y=1.01)
        fig2.tight_layout()
        out2 = output_dir / 'window_over_time_by_penalty_factor.png'
        fig2.savefig(out2, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {out2}")


def _plot_individual_window_runs(
    results_df: pd.DataFrame,
    dataset_dir: Path,
    results_dir: Path,
    dataset_name: str,
):
    """
    Save one window-over-time plot per configuration, next to its windows.csv.

    Each figure shows the window evolution for that single run with breakpoints
    marked, making it easy to inspect individual configurations.
    """
    breakpoints = DATASET_BREAKPOINTS.get(dataset_name, [])
    count = 0

    for _, row in results_df.iterrows():
        # Resolve the directory that holds windows.csv
        config_dir = None
        if 'window_csv' in row.index and pd.notna(row.get('window_csv')):
            p = Path(row['window_csv'])
            if p.exists():
                config_dir = p.parent

        if config_dir is None:
            parts = [f"N0{int(row['N0'])}", f"alpha{row['alpha']}"]
            if 'mc_reps' in row.index and pd.notna(row.get('mc_reps')):
                parts.append(f"mc_reps{int(row['mc_reps'])}")
            if 'penalty_factor' in row.index and pd.notna(row.get('penalty_factor')):
                parts.append(f"penalty_factor{row['penalty_factor']}")
            if 'growth_base' in row.index and pd.notna(row.get('growth_base')):
                parts.append(f"growth_base{row['growth_base']}")
            param_str = "temp_" + "_".join(parts)
            for base in [dataset_dir, results_dir]:
                candidate = base / param_str
                if (candidate / "windows.csv").exists():
                    config_dir = candidate
                    break

        if config_dir is None:
            continue

        wdf = _load_windows_for_config(row, dataset_dir, results_dir)
        if wdf is None:
            continue

        col = 'window_mean' if 'window_mean' in wdf.columns else wdf.columns[0]

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

        ax.plot(wdf.index, wdf[col], linewidth=1.0, color='#4A74AA', alpha=1.0, label='Window size')
        rolling = wdf[col].rolling(window=10, center=True).mean()
        ax.plot(wdf.index, rolling, linewidth=2.0, color='#DB3549', label='MA(10)')

        for bp in breakpoints:
            ax.axvline(bp, color='#DB3549', linestyle='--', linewidth=1.5, alpha=0.7,
                       label='Breakpoint' if bp == breakpoints[0] else None)

        # Build a concise title from the parameters
        title_parts = [f"N0={int(row['N0'])}"]
        if 'alpha' in row.index:
            title_parts.append(f"alpha={row['alpha']}")
        if 'mc_reps' in row.index and pd.notna(row.get('mc_reps')):
            title_parts.append(f"mc_reps={int(row['mc_reps'])}")
        if 'penalty_factor' in row.index and pd.notna(row.get('penalty_factor')):
            title_parts.append(f"pf={row['penalty_factor']}")
        title = f"{dataset_name}  |  {', '.join(title_parts)}"

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Window size')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fontsize=9, ncol=3, frameon=False)

        fig.tight_layout()
        out = config_dir / 'window_over_time.png'
        fig.savefig(out, dpi=150, bbox_inches='tight', transparent=True)
        plt.close(fig)
        count += 1

    print(f"  Saved {count} individual window plots (window_over_time.png in each config dir)")


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
        # Fallback: try results.csv (produced by 01_lpa_sensitivity.py)
        fallback = dataset_dir / 'results.csv'
        if fallback.exists():
            summary_file = fallback
        else:
            # Also try parent results.csv filtered by dataset
            parent_results = results_dir / 'results.csv'
            if parent_results.exists():
                summary_file = parent_results
            else:
                print(f"Warning: No results found for {dataset_name}")
                return

    results_df = pd.read_csv(summary_file)

    # Filter to this dataset if loaded from a global results file
    if 'dataset' in results_df.columns:
        results_df = results_df[results_df['dataset'] == dataset_name].copy()

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
        metric_cols = [c for c in numeric_cols if c not in ['N0', 'alpha', 'mc_reps', 'penalty_factor', 'growth_base', 'num_bootstrap']]

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

    # 3. mc_reps sensitivity
    mc_reps_col = 'mc_reps' if 'mc_reps' in results_df.columns else 'num_bootstrap'
    print(f"3. Creating {mc_reps_col} sensitivity plots...")
    if mc_reps_col in results_df.columns:
        for metric in metric_cols:
            if metric in results_df.columns:
                viz.plot_parameter_sensitivity(
                    results_df=results_df,
                    param_col=mc_reps_col,
                    metric_cols=metric,
                    dataset_name=dataset_name,
                    title=f'{title_prefix}: {metric} vs {mc_reps_col}',
                    save_name=f'{mc_reps_col}_sensitivity_{metric}',
                    ylabel=metric.replace('_', ' ').title(),
                    show_error_bars=True
                )
                plt.close()

    # 3b. penalty_factor sensitivity
    if 'penalty_factor' in results_df.columns and results_df['penalty_factor'].nunique() > 1:
        print("3b. Creating penalty_factor sensitivity plots...")
        for metric in metric_cols:
            if metric in results_df.columns:
                viz.plot_parameter_sensitivity(
                    results_df=results_df,
                    param_col='penalty_factor',
                    metric_cols=metric,
                    dataset_name=dataset_name,
                    title=f'{title_prefix}: {metric} vs penalty_factor',
                    save_name=f'penalty_factor_sensitivity_{metric}',
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
            # Aggregate over mc_reps/penalty_factor
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

    # 7. Window over time for all configs
    print("\n7. Creating window-over-time grid plots...")
    _plot_window_over_time_grid(
        results_df=results_df,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        dataset_name=dataset_name,
        output_dir=viz_output_dir,
    )

    # 7b. Individual window-over-time plot per config (saved next to each windows.csv)
    print("\n7b. Creating individual window-over-time plots per config...")
    _plot_individual_window_runs(
        results_df=results_df,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        dataset_name=dataset_name,
    )

    # 8. Window analysis (if enabled)
    if window_analysis:
        print("\n8. Creating window analysis plots...")

        # Get breakpoints for this dataset
        breakpoints = DATASET_BREAKPOINTS.get(dataset_name, None)
        if breakpoints is None:
            print(f"  Warning: No breakpoints defined for {dataset_name}, skipping window analysis")
        else:
            # Window vs parameters
            print("  a. Window size vs parameters...")
            param_candidates = ['N0', 'alpha']
            if 'mc_reps' in results_df.columns:
                param_candidates.append('mc_reps')
            elif 'num_bootstrap' in results_df.columns:
                param_candidates.append('num_bootstrap')
            if 'penalty_factor' in results_df.columns and results_df['penalty_factor'].nunique() > 1:
                param_candidates.append('penalty_factor')
            for param in param_candidates:
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

            # Build param_str matching the format used by 01_lpa_sensitivity.py
            # Format: temp_N0{val}_alpha{val}_mc_reps{val}_penalty_factor{val}_growth_base{val}
            param_parts = [f"N0{N0_int}", f"alpha{alpha_val}"]
            if 'mc_reps' in best_config:
                param_parts.append(f"mc_reps{int(best_config['mc_reps'])}")
            elif 'num_bootstrap' in best_config:
                param_parts.append(f"num_bootstrap{int(best_config['num_bootstrap'])}")
            if 'penalty_factor' in best_config:
                param_parts.append(f"penalty_factor{best_config['penalty_factor']}")
            if 'growth_base' in best_config:
                param_parts.append(f"growth_base{best_config['growth_base']}")
            param_str = "temp_" + "_".join(param_parts)

            # Try different path combinations
            possible_paths = [
                dataset_dir / param_str / "windows.csv",  # New format
            ]
            # Also try legacy format with num_bootstrap
            if 'num_bootstrap' in best_config:
                num_boot = int(best_config['num_bootstrap'])
                possible_paths.append(dataset_dir / f"temp_N{N0_int}_alpha{alpha_val}_num_bootstrap{num_boot}/windows.csv")

            # Also glob for any matching temp dir
            for temp_dir in dataset_dir.glob(f"temp_*N0{N0_int}*"):
                candidate = temp_dir / "windows.csv"
                if candidate.exists() and candidate not in possible_paths:
                    possible_paths.append(candidate)

            windows_csv = None
            for path in possible_paths:
                if path.exists():
                    windows_csv = path
                    break

            if windows_csv and windows_csv.exists():
                print(f"  b. Loading windows from best config ({param_str})...")
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
        if not summary_file.exists():
            summary_file = results_dir / dataset_name / 'results.csv'
        if not summary_file.exists():
            # Try global results.csv
            global_file = results_dir / 'results.csv'
            if global_file.exists():
                summary_file = global_file
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            if 'dataset' in df.columns:
                df = df[df['dataset'] == dataset_name].copy()
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
        param_exclude = {'N0', 'alpha', 'mc_reps', 'penalty_factor', 'growth_base', 'num_bootstrap'}
        dataset_metrics = [col for col in df.columns
                          if df[col].dtype in ['float64', 'int64'] and
                          col not in param_exclude]

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
            ]
            if 'mc_reps' in best_config:
                findings.append(f"Optimal mc_reps: {best_config['mc_reps']:.0f}")
            if 'penalty_factor' in best_config:
                findings.append(f"Optimal penalty_factor: {best_config['penalty_factor']:.3f}")
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