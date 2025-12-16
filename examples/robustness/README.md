# Robustness Analysis Scripts

This directory contains scripts for conducting comprehensive robustness analysis of the Adaptive WIN-SHAP methodology.

## Quick Start

### 1. LPA Sensitivity Analysis (Step 1 of Robustness Plan)

Test how sensitive Adaptive WIN-SHAP is to LPA hyperparameters.

#### Quick Test (10-15 minutes)
```bash
# Run on 1 dataset with reduced parameter grid
python examples/robustness/01_lpa_sensitivity.py \
    --datasets piecewise_ar3 \
    --quick-test \
    --n-runs 2
```

#### Full Analysis (several hours)
```bash
# Run on 2 datasets with full parameter grid
python examples/robustness/01_lpa_sensitivity.py \
    --datasets piecewise_ar3 arx_rotating \
    --n-runs 3
```

#### Analyze Results
```bash
# Generate plots and summary statistics
python examples/robustness/analyze_lpa_sensitivity.py \
    --results-dir examples/results/robustness/lpa_sensitivity
```

## What Gets Computed

### LPA Sensitivity (01_lpa_sensitivity.py)

For each parameter combination:

1. **Window Detection**
   - Runs LPA with specified N0, jump, alpha, num_bootstrap
   - Records window sizes across time
   - Computes window statistics (mean, std, min, max)
   - Measures detection time

2. **Breakpoint Accuracy**
   - Compares detected windows to true regime breakpoints
   - Computes lag/lead from true breakpoints
   - Calculates detection accuracy

3. **Full Pipeline Metrics**
   - Runs SHAP computation with detected windows
   - Computes faithfulness (PRTB P90, P50)
   - Computes ablation ratios (MIF/LIF)
   - Computes correlation with true importance

### Analysis Script (analyze_lpa_sensitivity.py)

Generates:
- **Sensitivity indices** (Sobol-style first-order)
- **Summary statistics** per parameter
- **Heatmaps** showing metric variation
- **Line plots** with confidence intervals
- **Parameter recommendations** based on results

## Output Structure

```
examples/results/robustness/lpa_sensitivity/
├── config.json                           # Experiment configuration
├── results.csv                           # All results
├── sensitivity_indices.csv               # Sensitivity indices
├── summary_N0.csv                        # Summary for N0
├── summary_jump.csv                      # Summary for jump
├── summary_alpha.csv                     # Summary for alpha
├── summary_num_bootstrap.csv             # Summary for num_bootstrap
├── recommendations.txt                   # Parameter recommendations
├── heatmap_N0_faithfulness_prtb_p90.png # Heatmaps
├── heatmap_*.png                        # More heatmaps
├── sensitivity_lines_faithfulness.png   # Line plots
└── piecewise_ar3/                       # Dataset-specific results
    └── temp_N0100_jump1_alpha0.95_num_bootstrap50/
        ├── windows.csv                   # Detected windows
        ├── run_0.csv                     # Detection results
        └── benchmark/                    # Benchmark results
            ├── adaptive_shap_results.csv
            └── ...
```

## Parameter Grids

### Quick Test Mode
- N0: [50, 100]
- Jump: [1, 5]
- Alpha: [0.95]
- Num_bootstrap: [10, 50]
- **Total combinations**: 8

### Full Mode
- N0: [25, 50, 75, 100, 150, 200]
- Jump: [1, 2, 5, 10]
- Alpha: [0.90, 0.95, 0.99]
- Num_bootstrap: [10, 50, 100]
- **Total combinations**: 216

## Customization

### Test Different Datasets

```bash
python examples/robustness/01_lpa_sensitivity.py \
    --datasets piecewise_ar3 trend_season spike_process \
    --n-runs 2
```

### Modify Parameter Grid

Edit the `param_grid` dictionary in `01_lpa_sensitivity.py`:

```python
param_grid = {
    'N0': [50, 75, 100, 125],        # Add/remove values
    'jump': [1, 3, 5],
    'alpha': [0.90, 0.95, 0.99],
    'num_bootstrap': [20, 50, 100]
}
```

### Change Detection Settings

Modify constants in `run_lpa_detection()` method:
- `min_window`: Minimum segment size (default: 4)
- `search_step`: Step size for breakpoint search (default: 5)

## Expected Runtime

Times are approximate on M3 Max with 16 cores:

| Configuration | Time per Combo | Total Time |
|--------------|----------------|------------|
| Quick test (1 dataset) | ~2 min | ~15 min |
| Full grid (1 dataset) | ~3 min | ~10 hours |
| Full grid (2 datasets) | ~6 min | ~20 hours |

**Recommendation**: Start with quick test, then run full grid overnight.

## Troubleshooting

### Out of Memory

Reduce batch processing:
- Decrease `t_workers` and `b_workers` in `run_lpa_detection()`
- Or run with fewer parameter combinations

### Slow Performance

- Use GPU if available (automatically detected)
- Reduce `n_runs` from 3 to 1
- Increase `jump` parameter (less time points)

### Missing Metrics

If benchmark metrics are missing:
- Check that windows CSV was generated correctly
- Verify dataset paths are correct
- Look for error messages in output

## Integration with Paper

Use these results for:

1. **Section 5.X: Robustness Analysis**
   - Table: Parameter sensitivity summary
   - Figure: Sensitivity heatmaps
   - Figure: Line plots with confidence intervals

2. **Appendix**
   - Full sensitivity indices table
   - Complete parameter grid results

3. **Practical Guidelines**
   - Recommended parameter ranges from `recommendations.txt`

## Next Steps

After completing LPA sensitivity:

1. **Step 2**: Architecture Sensitivity
   ```bash
   python examples/robustness/02_architecture_sensitivity.py
   ```

2. **Step 3**: Noise Robustness
   ```bash
   python examples/robustness/04_noise_robustness.py
   ```

3. **Step 4**: Bootstrap Confidence Intervals
   ```bash
   python examples/robustness/06_bootstrap_ci.py
   ```

See `ROBUSTNESS_ANALYSIS_PLAN.md` for complete roadmap.