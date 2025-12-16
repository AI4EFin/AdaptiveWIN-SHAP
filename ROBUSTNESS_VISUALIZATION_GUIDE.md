# Robustness Test Visualization Guide

This guide explains how to use the generic visualization framework for robustness analysis results.

## Overview

The visualization framework provides reusable tools for creating publication-quality figures across different types of robustness tests. It includes:

1. **Core Module**: `visualize_robustness.py` - Generic plotting functions
2. **Test-Specific Scripts**: Ready-to-use scripts for different robustness tests
3. **Customizable**: Easy to extend for new test types

## Quick Start

### Visualize LPA Sensitivity Results

After running LPA sensitivity analysis:

```bash
# Single dataset
python examples/robustness/visualize_lpa_sensitivity.py --dataset piecewise_ar3

# All datasets
python examples/robustness/visualize_lpa_sensitivity.py --all-datasets
```

### Visualize Bootstrap CI Results

After running bootstrap confidence intervals:

```bash
# Single dataset
python examples/robustness/visualize_bootstrap_ci.py --dataset piecewise_ar3

# All datasets
python examples/robustness/visualize_bootstrap_ci.py --all-datasets
```

## Generated Visualizations

### LPA Sensitivity Analysis

For each dataset, the following figures are created:

1. **Parameter Sensitivity Plots**
   - How each metric varies with N0, alpha, and num_bootstrap
   - Shows mean with error bars
   - Individual plots per metric and combined plots

2. **Parameter Combination Heatmaps**
   - 2D heatmap showing faithfulness across N0 × alpha grid
   - Helps identify optimal parameter regions

3. **Stability Summary**
   - Coefficient of variation (CV) for each metric
   - Identifies which metrics are stable (CV < 20%)
   - Mean ± Std Dev comparison

4. **Cross-Dataset Summary** (when visualizing all datasets)
   - Box plots comparing metrics across datasets
   - Comprehensive markdown report with statistics
   - Summary figure with key metrics

**Output Location**: `examples/results/robustness/figures/lpa_sensitivity/`

### Bootstrap CI Analysis

For each dataset, the following figures are created:

1. **Distribution Plots**
   - Histogram + KDE for each metric
   - Shows mean, 95% CI bounds
   - Useful for assessing normality

2. **Confidence Interval Summary**
   - Error bars showing mean with 95% CI
   - Color-coded by stability (green = stable, red = unstable)

3. **CI Width Comparison**
   - Horizontal bar chart of relative CI widths
   - Threshold line at 20% (stability criterion)
   - Color-coded: green (< 20%), orange (20-30%), red (> 30%)

4. **Convergence Plots**
   - Shows cumulative mean as realizations accumulate
   - Demonstrates convergence to final estimate
   - Final CI bounds marked

5. **Stability Summary**
   - Same as LPA sensitivity (CV assessment)

6. **Cross-Dataset Summary** (when visualizing all datasets)
   - Metric distributions compared across datasets
   - CI width comparison
   - Comprehensive report

**Output Location**: `examples/results/robustness/figures/bootstrap_ci/`

## Using the Core Module

For custom visualizations, use `RobustnessVisualizer` directly:

```python
from visualize_robustness import RobustnessVisualizer
import pandas as pd

# Initialize
viz = RobustnessVisualizer(output_dir='my_results/figures')

# Load your results
results_df = pd.read_csv('my_experiment_results.csv')

# 1. Parameter sensitivity
viz.plot_parameter_sensitivity(
    results_df=results_df,
    param_col='N0',
    metric_cols=['faithfulness', 'mif_score'],
    dataset_name='my_dataset',
    save_name='n0_sensitivity',
    show_error_bars=True
)

# 2. Metric distribution
bootstrap_values = results_df['faithfulness'].values
viz.plot_metric_distribution(
    values=bootstrap_values,
    metric_name='Faithfulness',
    ci=(ci_lower, ci_upper),
    save_name='faithfulness_dist'
)

# 3. Heatmap for 2D parameter grid
heatmap_data = results_df.pivot(index='N0', columns='alpha', values='faithfulness')
viz.plot_heatmap(
    data=heatmap_data,
    title='Faithfulness: N0 vs alpha',
    save_name='parameter_heatmap',
    cmap='RdYlGn'
)

# 4. Multi-metric comparison
viz.plot_multi_metric_comparison(
    results_dict={
        'Test_A': results_df_A,
        'Test_B': results_df_B
    },
    metric_cols=['faithfulness', 'mif_score', 'lif_score'],
    save_name='test_comparison'
)

# 5. Stability summary
viz.plot_stability_summary(
    results_df=results_df,
    metric_cols=['faithfulness', 'mif_score', 'lif_score'],
    stability_threshold=0.2,
    save_name='stability'
)

# 6. Summary report
viz.create_summary_report(
    test_results={
        'Test_A': {
            'description': 'Test A description',
            'metrics': {
                'faithfulness': {'mean': 0.85, 'std': 0.02, 'min': 0.80, 'max': 0.90},
                'mif_score': {'mean': 0.92, 'std': 0.03, 'min': 0.85, 'max': 0.95}
            },
            'datasets': ['piecewise_ar3', 'arx_rotating'],
            'n_experiments': 100,
            'findings': ['Finding 1', 'Finding 2']
        },
        'Test_B': { ... }
    },
    output_name='my_robustness_summary'
)
```

## Available Plotting Functions

### `plot_parameter_sensitivity()`
- **Purpose**: Show how metrics vary with a parameter
- **Use cases**: LPA sensitivity, hyperparameter tuning, ablation studies
- **Options**: Error bars, individual points, multiple metrics

### `plot_metric_distribution()`
- **Purpose**: Visualize distribution of a metric
- **Use cases**: Bootstrap distributions, Monte Carlo simulations
- **Options**: Confidence intervals, reference values, KDE overlay

### `plot_heatmap()`
- **Purpose**: 2D visualization of metric across parameter combinations
- **Use cases**: Grid search results, correlation matrices
- **Options**: Custom colormaps, annotations, diverging scales

### `plot_multi_metric_comparison()`
- **Purpose**: Compare multiple metrics across different tests/configurations
- **Use cases**: Cross-test comparisons, method benchmarking
- **Options**: Box plots with outliers and means

### `plot_stability_summary()`
- **Purpose**: Assess stability using coefficient of variation
- **Use cases**: Robustness assessment, consistency checks
- **Options**: Customizable stability threshold

### `create_summary_report()`
- **Purpose**: Generate comprehensive markdown report + summary figure
- **Use cases**: Final robustness reports, paper supplementary material
- **Outputs**: Markdown file + PNG figure

## Extending for New Test Types

To add visualizations for a new robustness test:

1. **Create a new script** (e.g., `visualize_my_test.py`)
2. **Import the visualizer**:
   ```python
   from visualize_robustness import RobustnessVisualizer
   ```
3. **Load your results** (CSV, DataFrame, etc.)
4. **Use appropriate plotting functions**
5. **Add custom plots if needed**

Example template:

```python
"""
Visualize My New Robustness Test Results
"""

import argparse
import pandas as pd
from pathlib import Path
from visualize_robustness import RobustnessVisualizer

def visualize_my_test(dataset_name, results_dir, output_dir):
    """Visualize results for my test."""

    # Load results
    results_df = pd.read_csv(results_dir / dataset_name / 'my_results.csv')

    # Initialize visualizer
    viz = RobustnessVisualizer(output_dir=output_dir / dataset_name)

    # Create visualizations
    viz.plot_parameter_sensitivity(
        results_df=results_df,
        param_col='my_parameter',
        metric_cols='my_metric',
        dataset_name=dataset_name,
        save_name='my_parameter_sensitivity'
    )

    # Add more plots as needed...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--all-datasets', action='store_true')
    args = parser.parse_args()

    # Process datasets...

if __name__ == "__main__":
    main()
```

## Output Organization

All figures are saved to `examples/results/robustness/figures/` with the following structure:

```
figures/
├── lpa_sensitivity/
│   ├── piecewise_ar3/
│   │   ├── n0_sensitivity_faithfulness.png
│   │   ├── alpha_sensitivity_faithfulness.png
│   │   ├── n0_alpha_faithfulness_heatmap.png
│   │   ├── stability_summary.png
│   │   └── ...
│   ├── arx_rotating/
│   │   └── ...
│   └── cross_dataset/
│       ├── cross_dataset_metrics.png
│       └── summary_reports/
│           ├── lpa_sensitivity_summary.md
│           └── lpa_sensitivity_summary.png
│
├── bootstrap_ci/
│   ├── piecewise_ar3/
│   │   ├── faithfulness_distribution.png
│   │   ├── ci_summary_plot.png
│   │   ├── ci_width_comparison.png
│   │   ├── faithfulness_convergence.png
│   │   └── ...
│   └── cross_dataset/
│       └── ...
│
└── my_new_test/
    └── ...
```

## Figure Specifications

All figures are saved with:
- **Format**: PNG
- **DPI**: 300 (publication quality)
- **Size**: Optimized for papers (typically 10×6 inches)
- **Style**: Clean, professional, colorblind-friendly palette

## Tips for Paper Figures

1. **Use cross-dataset summaries** for main paper figures
2. **Use individual dataset plots** for supplementary material
3. **CI plots are ideal** for demonstrating robustness
4. **Heatmaps are great** for showing optimal parameter regions
5. **Stability summaries** provide quantitative robustness metrics

## Common Workflows

### Workflow 1: LPA Parameter Selection
1. Run LPA sensitivity: `python examples/robustness/01_lpa_sensitivity.py --quick-test`
2. Visualize: `python examples/robustness/visualize_lpa_sensitivity.py --all-datasets`
3. Check heatmaps to identify optimal N0 and alpha
4. Use those parameters for subsequent experiments

### Workflow 2: Bootstrap CI for Paper
1. Run bootstrap: `python examples/robustness/06_bootstrap_ci.py --n-realizations 100`
2. Visualize: `python examples/robustness/visualize_bootstrap_ci.py --all-datasets`
3. Use distribution plots + CI summary plot in paper
4. Report statistics from markdown summary in text

### Workflow 3: Full Robustness Report
1. Run all robustness tests
2. Visualize each test individually
3. Combine cross-dataset summaries
4. Use `create_summary_report()` to generate comprehensive report

## Customization

### Changing Color Schemes
```python
import seaborn as sns
sns.set_palette("Set2")  # or "husl", "Paired", etc.
```

### Adjusting Figure Sizes
```python
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)  # width, height in inches
```

### Custom Colormaps for Heatmaps
```python
viz.plot_heatmap(
    data=my_data,
    cmap='viridis',  # or 'plasma', 'coolwarm', 'RdYlGn', etc.
    vmin=0.5,
    vmax=1.0,
    center=0.75  # for diverging colormaps
)
```

## Troubleshooting

**Q: No figures are generated**
- Check that results files exist in the expected directory
- Verify column names match expected names (e.g., 'faithfulness', 'N0')

**Q: Figures look cluttered**
- Reduce number of data points shown
- Use `show_points=False` in parameter sensitivity plots
- Increase figure size in rcParams

**Q: Error bars not showing**
- Ensure `show_error_bars=True`
- Check that std or error columns exist
- For parameter sensitivity, errors are computed from grouped data

**Q: Cross-dataset summary is empty**
- Ensure multiple datasets have results
- Check that dataset directories have summary files
- Verify column names are consistent across datasets

## Next Steps

1. Run your robustness experiments
2. Use the visualization scripts to generate figures
3. Review figures and identify key findings
4. Use figures and summary reports in your paper
5. Customize as needed for your specific use case

For questions or issues, check the example scripts:
- `visualize_lpa_sensitivity.py`
- `visualize_bootstrap_ci.py`

These demonstrate all major use cases of the visualization framework.
