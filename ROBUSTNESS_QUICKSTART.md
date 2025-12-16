# Robustness Testing - Quick Start

## TL;DR - Run This

```bash
# Quick demo (15-30 minutes)
./run_robustness_tests.sh demo

# Or with Python directly
python examples/robustness_quickstart_demo.py
```

## What This Does

Tests how Adaptive WIN-SHAP performance changes when parameter magnitudes in the data generating processes vary by -50% to +200%.

**Goal**: Demonstrate that the methodology is robust to changes in true feature importance magnitudes.

## Quick Commands

### 1. Generate All Scaled Datasets

```bash
python examples/robustness_dgp_scaled.py
```

Creates datasets with parameter scaling: -50%, -20%, -10%, 0%, +10%, +20%, +50%, +100%, +200%

### 2. Run Benchmarks

**Minimal test** (one dataset, 2 scales, ~30 min):
```bash
python examples/robustness_benchmark_runner.py --minimal
```

**Single dataset** (all scales, ~1-2 hours):
```bash
python examples/robustness_benchmark_runner.py --datasets piecewise_ar3
```

**All datasets** (~8-12 hours):
```bash
python examples/robustness_benchmark_runner.py
```

### 3. Analyze Results

```bash
python examples/robustness_analyze.py --dataset piecewise_ar3
```

Generates:
- Robustness metrics CSV
- Performance vs. scaling curves
- Coefficient of variation heatmap

## Using the Shell Script

The `run_robustness_tests.sh` script automates everything:

```bash
# Quick demo
./run_robustness_tests.sh demo

# Single dataset (complete)
./run_robustness_tests.sh single piecewise_ar3

# All datasets (WARNING: very slow)
./run_robustness_tests.sh full

# Analyze existing results only
./run_robustness_tests.sh analyze piecewise_ar3
```

## Understanding Results

### Key Metric: Coefficient of Variation (CV)

```
CV = Standard Deviation / Mean
```

**Lower CV = More Robust**

- CV < 0.1: Very robust
- CV 0.1-0.3: Moderately robust
- CV > 0.3: Fragile

### Example Output

```
Robustness Metrics (Coefficient of Variation)

method                      faithfulness    ablation
adaptive_shap                    0.05          0.08
adaptive_shap_rolling_mean       0.06          0.09
rolling_shap                     0.12          0.15
timeshap                         0.25          0.28
global_shap                      0.45          0.52
```

**Interpretation**: Adaptive methods show lower CV (more robust) than baselines.

## Files Overview

### Scripts

| File | Purpose | Time |
|------|---------|------|
| `robustness_dgp_scaled.py` | Generate scaled datasets | ~2 min |
| `robustness_benchmark_runner.py` | Run all benchmarks | ~1-12 hours |
| `robustness_analyze.py` | Analyze results | ~1 min |
| `robustness_quickstart_demo.py` | Interactive demo | ~30 min |
| `run_robustness_tests.sh` | Automated pipeline | Varies |

### Output Directories

```
examples/
├── datasets/robustness/           # Generated datasets
│   └── {dataset}/
│       └── {scale_dir}/
│           ├── data.csv
│           └── true_importances.csv
│
└── results/robustness/            # Benchmark results
    └── {dataset}/
        ├── {scale_dir}/
        │   ├── LPA_N0_75_Jump_1/
        │   │   └── windows.csv
        │   └── benchmark/
        │       └── benchmark_summary.csv
        │
        └── analysis/               # Analysis outputs
            ├── {dataset}_robustness_metrics.csv
            ├── {dataset}_*_robustness.png
            └── {dataset}_robustness_cv_heatmap.png
```

## Datasets

| Dataset | Parameters Scaled | What Changes |
|---------|-------------------|--------------|
| `piecewise_ar3` | AR coefficients | Lag importance |
| `arx_rotating` | Covariate coefficients | External driver importance |
| `trend_season` | AR coefficients | Historical lag strength |
| `spike_process` | Spike probability coefficients | Covariate influence on spikes |
| `garch_regime` | Factor loadings | Market vs. volatility importance |

## Common Issues

### "Dataset not found"
```bash
# Solution: Generate datasets first
python examples/robustness_dgp_scaled.py
```

### "LPA detection failed"
```bash
# Solution: Try different N0
python examples/robustness_benchmark_runner.py --n0 100
```

### Out of memory
```bash
# Solution: Reduce batch size or SHAP samples
# Edit benchmark_runner.py and reduce:
# - batch_size (default: 64)
# - shap_nsamples (default: 500)
```

## Minimal Working Example

Test the complete pipeline with minimal data:

```python
# 1. Generate test data
from examples.robustness_dgp_scaled import (
    generate_scaled_datasets,
    sim_piecewise_ar3_rotating_scaled
)

generate_scaled_datasets(
    dataset_name='piecewise_ar3',
    dgp_func=sim_piecewise_ar3_rotating_scaled,
    T=1500,
    scaling_percentages=[-50, 50],
    seed=123
)

# 2. Run benchmark
from examples.robustness_benchmark_runner import run_robustness_analysis

results = run_robustness_analysis(
    datasets=['piecewise_ar3'],
    scaling_percentages=[-50, 0, 50],
    n0=75,
    jump=1
)

# 3. Analyze
from examples.robustness_analyze import analyze_dataset_robustness

metrics = analyze_dataset_robustness(
    dataset_name='piecewise_ar3',
    scaling_percentages=[-50, 0, 50]
)
```

## Performance Tips

1. **Start small**: Use `--minimal` flag
2. **Use GPU**: Modify device in benchmark_runner.py
3. **Parallel runs**: Run different datasets on different machines
4. **Reduce samples**: Lower `shap_nsamples` from 500 to 200

## Expected Results

**Adaptive WIN-SHAP should show**:
- ✓ Low CV across all metrics
- ✓ Consistent relative performance
- ✓ Graceful degradation (if any)

**Baseline methods may show**:
- ✗ High CV (especially Vanilla SHAP)
- ✗ Dataset-dependent variability
- ✗ Erratic behavior at extreme scaling

## Next Steps

1. **Run demo**: `./run_robustness_tests.sh demo`
2. **Review plots**: Check `examples/results/robustness/{dataset}/analysis/`
3. **Read full guide**: See `ROBUSTNESS_TESTING_GUIDE.md`
4. **Customize**: Modify scaling factors, LPA parameters, etc.

## Questions?

- **Methodology**: Read the paper (Section 5: Aggregated Results)
- **Technical details**: See `ROBUSTNESS_TESTING_GUIDE.md`
- **Bugs**: Open an issue on GitHub

---

**Remember**: The goal is to show that Adaptive WIN-SHAP maintains good performance even when the true importance magnitudes change significantly!