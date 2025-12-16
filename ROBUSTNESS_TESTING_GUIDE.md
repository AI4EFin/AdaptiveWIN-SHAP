# Robustness Testing Guide

## Overview

This guide explains how to test the robustness of the Adaptive WIN-SHAP methodology against changes in parameter magnitudes. The goal is to assess whether the method's performance is stable when the true importance of features varies.

## Methodology

For each of the 5 data generating processes (DGPs), we:

1. **Scale Parameters**: Modify the magnitude of DGP parameters by various percentages:
   - Reductions: -50%, -20%, -10%
   - Baseline: 0% (original parameters)
   - Increases: +10%, +20%, +50%, +100%, +200%

2. **Generate Datasets**: Create time series data using the scaled parameters

3. **Run Benchmarks**: Execute the full benchmark suite (LPA detection + SHAP explanation + metrics) for each scaled dataset

4. **Analyze Results**: Compare performance across scaling factors to assess robustness

## Parameter Scaling Details

### 1. Piecewise AR(3)
- **Parameters scaled**: AR coefficients (θ₁, θ₂, θ₃)
- **Original**: [0.9, 0.01, 0.01], [0.01, 0.9, 0.01], [0.01, 0.01, 0.9]
- **Effect**: Changes the magnitude of lag importance

### 2. ARX Rotating Covariates
- **Parameters scaled**: Covariate coefficients (β for D, F, R)
- **Fixed**: AR coefficients remain at [0.6, 0.2, 0.1]
- **Effect**: Changes the relative importance of external drivers

### 3. Trend + Seasonality
- **Parameters scaled**: AR coefficients
- **Fixed**: Trend and seasonal components remain constant
- **Effect**: Changes the importance of historical lags

### 4. Spike Process
- **Parameters scaled**: Spike probability coefficients (γ for covariates)
- **Fixed**: AR coefficients remain at [0.7, 0.2, 0.05]
- **Effect**: Changes the strength of covariate influence on spikes

### 5. GARCH Regime
- **Parameters scaled**: Factor loadings (β for market and volatility factors)
- **Fixed**: GARCH parameters (ω, α, β) remain constant
- **Effect**: Changes the relative importance of market vs. volatility factors

## Workflow

### Step 1: Generate Scaled Datasets

Generate all datasets with various scaling factors:

```bash
# Generate all 5 datasets with all scaling factors
python examples/robustness_dgp_scaled.py
```

This creates directories:
```
examples/datasets/robustness/
├── piecewise_ar3/
│   ├── baseline/
│   ├── scale_-50pct/
│   ├── scale_-20pct/
│   ├── scale_-10pct/
│   ├── scale_+10pct/
│   ├── scale_+20pct/
│   ├── scale_+50pct/
│   ├── scale_+100pct/
│   └── scale_+200pct/
├── arx_rotating/
│   └── ... (same structure)
└── ... (other datasets)
```

Each directory contains:
- `data.csv`: Time series data with target (N) and covariates (Z_0, Z_1, ...)
- `true_importances.csv`: Ground truth feature importances

### Step 2: Run Benchmarks

#### Minimal Test (Recommended First)

Test with one dataset and two scaling factors:

```bash
python examples/robustness_benchmark_runner.py --minimal
```

This runs:
- Dataset: `piecewise_ar3`
- Scaling factors: -50%, +50%
- Estimated time: ~15-30 minutes

#### Single Dataset

Run full analysis for one dataset:

```bash
python examples/robustness_benchmark_runner.py --datasets piecewise_ar3
```

#### All Datasets

Run complete robustness analysis (this will take several hours):

```bash
python examples/robustness_benchmark_runner.py
```

#### Custom Configuration

```bash
python examples/robustness_benchmark_runner.py \
  --datasets piecewise_ar3 arx_rotating \
  --scaling-pcts -50 0 50 100 \
  --n0 75 \
  --jump 1 \
  --rolling-mean-window 75
```

### Step 3: Analyze Results

Analyze robustness for a specific dataset:

```bash
python examples/robustness_analyze.py --dataset piecewise_ar3
```

This produces:
1. **Robustness metrics CSV**: Summary statistics (CV, max degradation, etc.)
2. **Robustness curves**: Line plots showing performance vs. scaling factor
3. **Robustness heatmap**: CV (coefficient of variation) across methods

Output location: `examples/results/robustness/{dataset_name}/analysis/`

## Understanding the Results

### Key Metrics

1. **Mean Score**: Average performance across all scaling factors
   - Higher is better for faithfulness and ablation

2. **Standard Deviation**: Variability in performance
   - Lower is better (indicates stability)

3. **Coefficient of Variation (CV)**: `std / mean`
   - Lower is better (indicates robustness)
   - **Most important metric** for robustness assessment

4. **Max Degradation**: Worst performance drop from baseline
   - Lower is better

### Interpreting Results

**Robust Method**:
- Low CV (< 0.1)
- Small max degradation
- Smooth curves in robustness plots

**Fragile Method**:
- High CV (> 0.3)
- Large max degradation
- Erratic curves in robustness plots

### Expected Behavior

- **Adaptive WIN-SHAP variants** should show:
  - Low CV across scaling factors
  - Graceful degradation (if any)
  - Consistent performance relative to baselines

- **Vanilla SHAP** may show:
  - Poor baseline performance
  - High variability

- **TimeSHAP** may show:
  - Moderate robustness
  - Dataset-dependent behavior

## Example: Minimal Test Run

```bash
# Step 1: Generate test datasets
python -c "
from examples.robustness_dgp_scaled import generate_scaled_datasets, sim_piecewise_ar3_rotating_scaled

generate_scaled_datasets(
    dataset_name='piecewise_ar3',
    dgp_func=sim_piecewise_ar3_rotating_scaled,
    T=1500,
    scaling_percentages=[-50, 50],
    seed=123
)
"

# Step 2: Run minimal benchmark
python examples/robustness_benchmark_runner.py --minimal

# Step 3: Analyze results
python examples/robustness_analyze.py --dataset piecewise_ar3 --scaling-pcts -50 0 50
```

## Directory Structure

```
examples/
├── datasets/
│   └── robustness/              # Scaled datasets
│       └── {dataset_name}/
│           └── {scale_dir}/
│               ├── data.csv
│               └── true_importances.csv
│
└── results/
    └── robustness/              # Benchmark results
        ├── {dataset_name}/
        │   └── {scale_dir}/
        │       ├── LPA_N0_75_Jump_1/
        │       │   └── windows.csv
        │       └── benchmark/
        │           ├── benchmark_summary.csv
        │           ├── global_shap_results.csv
        │           ├── rolling_shap_results.csv
        │           ├── adaptive_shap_results.csv
        │           └── ... (other method results)
        │
        └── overall_summary_N0_75_Jump_1.csv  # Aggregated results
```

## Advanced Options

### Custom Scaling Factors

Modify `scaling_percentages` in the scripts:

```python
# In robustness_dgp_scaled.py
scaling_pcts = [-25, -10, 0, 10, 25, 75]  # Custom factors
```

### Different LPA Parameters

Test sensitivity to LPA configuration:

```bash
python examples/robustness_benchmark_runner.py \
  --datasets piecewise_ar3 \
  --n0 50 \
  --jump 2
```

### Force Recomputation

Rerun benchmarks even if results exist:

```bash
python examples/robustness_benchmark_runner.py --force
```

## Troubleshooting

### Issue: "Dataset not found"
- Ensure you ran `robustness_dgp_scaled.py` first
- Check that `examples/datasets/robustness/` exists

### Issue: "LPA detection failed"
- Check data quality (NaNs, infinite values)
- Try different N0 value (larger N0 = more stable, but less adaptive)
- Increase epochs if LSTM training is unstable

### Issue: "Out of memory"
- Reduce `batch_size` in benchmark runner
- Process datasets sequentially instead of in parallel
- Use smaller `shap_nsamples` (default: 500)

### Issue: Results show high variance
- This may indicate:
  1. The method is truly sensitive to parameter magnitude
  2. Random seed effects (run multiple times with different seeds)
  3. LPA detection instability (try different N0/Jump)

## Performance Considerations

### Computational Cost

For **full analysis** (5 datasets × 9 scaling factors):
- **Minimal**: 1 dataset × 2 scales ≈ 15-30 min
- **Single dataset**: 1 dataset × 9 scales ≈ 1-2 hours
- **Full analysis**: 5 datasets × 9 scales ≈ 8-12 hours

Time depends on:
- Hardware (CPU vs GPU)
- LPA parameters (N0, Jump)
- SHAP parameters (nsamples, max_background)

### Optimization Tips

1. **Start minimal**: Test with `--minimal` flag first
2. **Use GPU**: Set device in benchmark runner (requires CUDA)
3. **Reduce SHAP samples**: Lower `shap_nsamples` from 500 to 200
4. **Increase stride**: Use `rolling_stride=5` for faster (but coarser) analysis
5. **Parallel processing**: Run different datasets on different machines

## Citation

If you use this robustness testing framework, please cite:

```bibtex
@article{bag2025adaptivewinshap,
  title={Changepoint-Aware SHAP: Adaptive Windows for Local Explainability},
  author={Bag, Raul Cristian and Lessmann, Stefan and H{\"a}rdle, Wolfgang Karl and Pele, Daniel Traian},
  journal={arXiv preprint},
  year={2025}
}
```

## Questions?

For issues or questions:
1. Check the main paper for methodology details
2. Review ROBUSTNESS_QUICKSTART.md for quick examples
3. Open an issue on the repository