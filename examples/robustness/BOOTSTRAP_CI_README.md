# Bootstrap Confidence Intervals - Adaptive WIN-SHAP

**Robustness Analysis Plan Section 3.1**

## Overview

This implementation quantifies uncertainty in Adaptive WIN-SHAP performance metrics by running the full pipeline on multiple realizations of each dataset with different random seeds.

## What It Tests

### Statistical Robustness
- **Objective**: Quantify uncertainty in performance metrics
- **Method**: Generate multiple dataset realizations with different random seeds
- **Metrics Tracked**:
  - Faithfulness scores (p90, p50, p10)
  - MIF/LIF ratios (p90, p50, p10)
  - Correlation with true importance
  - Window size statistics (mean, std, min, max)
  - Detection time

### Success Criteria

According to the robustness analysis plan, the method shows good stability if:
- **CI width < 20% of mean** for key metrics
- Consistent performance across multiple random seeds (100 runs recommended)

## Files

### Experiment Scripts
- **`06_bootstrap_ci.py`**: Main script to run bootstrap experiments
  - Generates dataset realizations with different seeds
  - Runs full pipeline: LPA detection → SHAP → metrics
  - Uses fixed LPA hyperparameters (N0=75, alpha=0.95, num_bootstrap=100)
  - Saves results for each realization

- **`analyze_bootstrap_ci.py`**: Analysis and visualization script
  - Computes 95% confidence intervals using t-distribution
  - Generates confidence interval plots
  - Creates distribution plots for key metrics
  - Assesses stability based on CI width
  - Produces summary tables

## Usage

### 1. Run Bootstrap Experiments

#### Quick Test (3 realizations)
```bash
python examples/robustness/06_bootstrap_ci.py \
  --datasets piecewise_ar3 \
  --quick-test
```

#### Single Dataset (10 realizations)
```bash
python examples/robustness/06_bootstrap_ci.py \
  --datasets piecewise_ar3 \
  --n-realizations 10
```

#### Multiple Datasets (Full Analysis)
```bash
python examples/robustness/06_bootstrap_ci.py \
  --datasets piecewise_ar3 arx_rotating trend_season spike_process garch_regime \
  --n-realizations 100 \
  --N0 75 \
  --alpha 0.95 \
  --num-bootstrap 100
```

#### Custom Parameters
```bash
python examples/robustness/06_bootstrap_ci.py \
  --datasets piecewise_ar3 \
  --n-realizations 50 \
  --start-seed 2000 \
  --N0 100 \
  --alpha 0.99 \
  --num-bootstrap 200 \
  --output-dir examples/results/robustness/bootstrap_custom
```

### 2. Analyze Results

```bash
python examples/robustness/analyze_bootstrap_ci.py \
  --results-dir examples/results/robustness/bootstrap_ci
```

## Output Files

```
examples/results/robustness/bootstrap_ci/
├── config.json                          # Experiment configuration
├── results.csv                          # Raw results for all realizations
├── confidence_intervals.csv             # Computed CI statistics
├── summary_table.txt                    # Human-readable summary
├── confidence_intervals_plot.png        # CI visualization
├── distribution_faithfulness_prtb_p90.png
├── distribution_ablation_mif_p90.png
├── distribution_correlation_true_imp_mean.png
├── distribution_window_mean.png
└── {dataset}/                          # Per-dataset temporary files
    └── temp_seed_{seed}/
        ├── dataset/
        │   ├── data.csv
        │   └── true_importances.csv
        ├── windows.csv
        └── benchmark/
```

## Interpretation

### Confidence Interval Width

The CI width as a percentage of the mean indicates stability:
- **< 10%**: Very stable, excellent reproducibility
- **< 20%**: Stable, good reproducibility (success criterion)
- **20-30%**: Moderate stability, some variability
- **> 30%**: High variability, investigate sources of instability

### Coefficient of Variation (CV)

CV = std / mean provides a scale-independent measure of variability:
- **< 0.10**: Low variability
- **0.10-0.20**: Moderate variability
- **> 0.20**: High variability

### Example Interpretation

```
faithfulness_prtb_p90: 6.50% ✓ STABLE
  Mean: 0.853
  95% CI: [0.825, 0.881]
  CV: 8.2%
```

This indicates:
- Mean faithfulness score: 0.853
- 95% confidence: true mean is between 0.825 and 0.881
- CI width is only 6.5% of the mean → excellent stability
- Low coefficient of variation (8.2%) → consistent across realizations

## Implementation Details

### Dataset Generation

Each realization uses a different random seed to generate:
- **piecewise_ar3**: Different noise realizations, same regime structure
- **arx_rotating**: Different covariate and noise realizations
- **trend_season**: Different AR noise on top of deterministic components
- **spike_process**: Different spike locations and magnitudes
- **garch_regime**: Different volatility realizations

### Fixed vs. Variable Parameters

**Fixed** (same across all realizations):
- LPA parameters: N0, alpha, num_bootstrap
- Model architecture: LSTM hidden size, layers, dropout
- Training parameters: epochs, batch size, learning rate
- Regime structure: breakpoint locations, regime lengths

**Variable** (changes with seed):
- Random noise in data generation
- Covariate realizations
- LSTM initialization (via PyTorch's random state)

### Statistical Method

- **Confidence Intervals**: Computed using t-distribution (appropriate for small sample sizes)
- **CI Formula**: `mean ± t_critical * (std / sqrt(n))`
- **t_critical**: Based on (n-1) degrees of freedom and alpha=0.05 for 95% CI

## Computational Cost

Estimated runtime per realization:
- **piecewise_ar3**: ~2-3 minutes
- **arx_rotating**: ~3-4 minutes (3 covariates)
- **trend_season**: ~2-3 minutes
- **spike_process**: ~3-4 minutes (2 covariates)
- **garch_regime**: ~2-3 minutes

**Total for 100 realizations on 5 datasets**: ~20-30 hours on single CPU

**Recommendation**: Use parallel execution or cloud compute for full analysis

## Next Steps

After running bootstrap analysis:

1. **Check stability**: Look for metrics with CI width > 20%
2. **Compare datasets**: Which datasets show more variability?
3. **Investigate failures**: Check failed realizations in results.csv
4. **Increase sample size**: If CV is high, run more realizations
5. **Report findings**: Include CI plots and summary table in paper

## Related Robustness Tests

- **1.1 LPA Sensitivity**: Tests sensitivity to LPA hyperparameters (N0, alpha, num_bootstrap)
- **2.1 Noise Robustness**: Tests sensitivity to noise levels
- **3.1 Bootstrap CI** (this): Tests statistical stability across random realizations

## References

From Robustness Analysis Plan Section 13:
> The robustness analysis is successful if the method shows consistent performance
> (CI width < 20% of mean) across multiple random seeds (100 runs).

## Troubleshooting

### High Memory Usage
- Reduce `--n-realizations` for testing
- Use `--quick-test` mode first

### Long Runtime
- Use `--quick-test` for validation
- Reduce number of datasets
- Consider parallelization (not yet implemented)

### Failed Realizations
- Check `results.csv` for error messages
- Some seeds may produce challenging datasets (expected)
- Success rate > 90% is acceptable

## Contact

For questions about this implementation, refer to:
- Main robustness plan: `ROBUSTNESS_ANALYSIS_PLAN.md`
- Bootstrap CI section: Plan Section 3.1