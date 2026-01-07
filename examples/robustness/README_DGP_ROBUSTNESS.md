# DGP Parameter Robustness Test - Quick Start Guide

## Overview

The DGP parameter robustness test evaluates how Adaptive WIN-SHAP performs when the data generating process (DGP) parameters vary. This implementation tests 5 scenarios:

1. **Baseline**: Original piecewise_ar3 dataset (L2 distance = 0)
2. **Closer**: Parameters closer together (L2 reduced to ~50%)
3. **Further**: Parameters further apart with mixed dynamics (L2 increased)
4. **Random 1-3**: Three randomly sampled stationary parameter sets

## Quick Start

### Run the full pipeline:
```bash
python examples/robustness/dgp_parameter_robustness.py --verbose
```

This will:
1. Generate all 5 DGP scenario datasets
2. Run LPA window detection on each scenario
3. Run SHAP benchmarks on each scenario
4. Compute robustness metrics
5. Generate visualizations

### Expected Runtime
- Data generation: ~1 minute
- LPA detection per scenario: ~5-10 minutes (depending on hardware)
- SHAP benchmark per scenario: ~10-15 minutes
- **Total**: ~2-3 hours for full pipeline

## Output Structure

```
examples/
  datasets/
    simulated/
      piecewise_ar3_dgp_robustness/
        closer/
          data.csv
          true_importances.csv
          scenario_config.json
        further/
          [same structure]
        random_1/
        random_2/
        random_3/
        dgp_parameters.csv          # Summary of all parameters

  results/
    dgp_robustness/
      piecewise_ar3/
        baseline/
          windows.csv
          benchmark/
            adaptive_shap_results.csv
            benchmark_summary.csv
        closer/
          [same structure]
        further/
        random_1/
        random_2/
        random_3/
        summary_all_scenarios.csv   # KEY RESULTS FILE
        figures/
          window_evolution_comparison.png
          metrics_vs_l2_distance.png
          comparative_window_overlay.png
          regime_specific_correlations.png
```

## Key Results Files

### summary_all_scenarios.csv
Main results table with one row per scenario containing:
- L2 distances from baseline
- Detection lag and success rate
- SHAP correlations (overall and per-regime)
- Faithfulness and ablation metrics
- Window size statistics

### Figures
1. **window_evolution_comparison.png**: Side-by-side plots of window sizes over time
2. **metrics_vs_l2_distance.png**: Performance metrics vs L2 distance (shows degradation patterns)
3. **comparative_window_overlay.png**: All scenarios overlaid on single plot
4. **regime_specific_correlations.png**: Heatmap of SHAP correlations per regime

## Usage Options

### Skip steps (for faster iteration):
```bash
# Only regenerate visualizations from existing results
python examples/robustness/dgp_parameter_robustness.py --visualize-only

# Skip data generation (use existing datasets)
python examples/robustness/dgp_parameter_robustness.py --skip-generation

# Skip LPA detection (use existing windows)
python examples/robustness/dgp_parameter_robustness.py --skip-detection

# Skip SHAP benchmark (use existing benchmark results)
python examples/robustness/dgp_parameter_robustness.py --skip-benchmark
```

### Custom random seeds:
```bash
python examples/robustness/dgp_parameter_robustness.py --seeds 100,200,300
```

### Different base dataset:
```bash
python examples/robustness/dgp_parameter_robustness.py --dataset arx_rotating
```

## Interpreting Results

### Good Robustness Indicators:
- **Detection lag** stays low across all L2 distances
- **SHAP correlations** remain high (>0.8) even for closer scenarios
- **Faithfulness scores** are consistent across scenarios
- **Window statistics** show similar patterns across scenarios

### Degradation Patterns:
- **Linear degradation**: Performance decreases proportionally with L2 distance
- **Threshold effect**: Performance is stable until a critical L2 distance
- **Regime-specific issues**: Some regimes harder to detect/explain than others

### Using summary_all_scenarios.csv:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('examples/results/dgp_robustness/piecewise_ar3/summary_all_scenarios.csv')

# Plot detection lag vs L2 distance
plt.scatter(df['l2_distance_from_baseline'], df['detection_lag_mean'])
plt.xlabel('L2 Distance from Baseline')
plt.ylabel('Detection Lag (steps)')
plt.title('Robustness: Detection Lag vs Parameter Distance')
plt.show()

# Compare SHAP correlations across regimes
regime_corrs = df[['scenario_name', 'shap_corr_regime0', 'shap_corr_regime1', 'shap_corr_regime2']]
print(regime_corrs)
```

## Troubleshooting

### "Non-stationary parameters" warning
- The script automatically projects non-stationary parameters onto the stationary region
- This is expected for random scenarios and is handled gracefully
- Check `scenario_config.json` to see if fallback was used

### LPA detection fails
- Reduce `num_bootstrap` in the LPA config (line 1264 in script)
- Increase `N0` for more stable initial windows
- Check if data contains NaN or infinite values

### SHAP benchmark fails
- Ensure benchmark.py is working independently first
- Check if windows.csv exists and has correct format
- Verify data.csv has 'N' column for target variable

### Memory issues
- Run scenarios sequentially instead of storing all results
- Reduce `num_bootstrap` in LPA config
- Use `--skip-detection` or `--skip-benchmark` to run in stages

## Customization

### Modify LPA configuration:
Edit lines 1263-1269 in `dgp_parameter_robustness.py`:
```python
lpa_config = {
    'N0': 75,           # Initial window size
    'alpha': 0.95,      # Confidence level
    'num_bootstrap': 50,# Bootstrap iterations
    'jump': 1,          # Step size
    'growth': 'geometric'
}
```

### Modify scenario generation:
Edit parameter generation functions (lines 46-243):
- `generate_closer_scenario()`: Change `target_reduction` parameter
- `generate_further_scenario()`: Specify different extreme parameters
- `generate_random_scenario()`: Adjust sampling strategy

### Add new metrics:
Add computation in `compute_*_metrics()` functions (lines 718-842)
Then add column to summary in main() (lines 1304-1405)

## Next Steps

After running the robustness test:

1. **Analyze summary_all_scenarios.csv** to identify key findings
2. **Review figures** to visualize degradation patterns
3. **Compare regime-specific results** to identify harder regimes
4. **Test on other datasets** (arx_rotating, trend_season, etc.)
5. **Vary LPA parameters** to see if tuning helps with harder scenarios

## Citation

If you use this robustness test in your research, please cite:
- The main Adaptive WIN-SHAP paper
- This specification: SPEC_DGP_ROBUSTNESS.md
