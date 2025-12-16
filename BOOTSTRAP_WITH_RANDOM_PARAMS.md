# Bootstrap CI with Randomized DGP Parameters

## Summary

The bootstrap confidence interval implementation now supports **randomized data generating process (DGP) parameters** across seeds while maintaining structural properties.

## What Changed

### Before (Original Implementation)
- **Fixed coefficients** across all seeds
- Only noise realizations varied
- Example for piecewise_ar3:
  - Seed 1000: φ = [0.9, 0.01, 0.01]
  - Seed 1001: φ = [0.9, 0.01, 0.01] ← Same!
  - Seed 1002: φ = [0.9, 0.01, 0.01] ← Same!

### After (New Implementation)
- **Randomized coefficients** while maintaining dominance structure
- Both parameters AND noise vary across seeds
- Example for piecewise_ar3:
  - Seed 1000: φ₁ = [0.85, 0.10, 0.05], φ₂ = [0.05, 0.85, 0.10], φ₃ = [0.10, 0.05, 0.85]
  - Seed 1001: φ₁ = [0.92, 0.03, 0.05], φ₂ = [0.05, 0.92, 0.03], φ₃ = [0.03, 0.05, 0.92]
  - Seed 1002: φ₁ = [0.78, 0.15, 0.07], φ₂ = [0.07, 0.78, 0.15], φ₃ = [0.15, 0.07, 0.78]

  **Structure preserved**: Lag 1 dominates in regime 1, Lag 2 in regime 2, Lag 3 in regime 3

## Why This Is Better

This tests robustness across:
1. **Different magnitudes** of regime differences
2. **Varying strengths** of dominant features
3. **Different noise levels**
4. **Multiple parameter configurations** (not just noise)

This is a **much stronger test** of the method's generalizability!

## Implementation Details

### New File: `generate_simulated_datasets_random.py`

Extends the original data generators with `randomize_params` flag:

```python
Y, Z, true_imp, params = sim_piecewise_ar3_rotating(
    T=1500,
    seed=1000,
    randomize_params=True  # ← New parameter
)
```

### Parameter Randomization Rules

**piecewise_ar3**:
- Dominant coefficient: Uniform[0.7, 0.95]
- Non-dominant total: Uniform[0.05, 0.25], split via Dirichlet
- Noise σ: Uniform[0.5, 1.5]
- Structure: Always φ₁ > φ₂, φ₃ in regime 1; φ₂ > φ₁, φ₃ in regime 2; etc.

**arx_rotating**:
- AR(1) coefficient: Uniform[0.5, 0.7]
- Dominant covariate β: Uniform[0.8, 1.5]
- Non-dominant β: Uniform[-0.2, 0.3]
- Noise σ: Uniform[0.3, 0.7]
- Structure: Different covariate dominates in each regime

**Similar for other datasets** (trend_season, spike_process, garch_regime)

### Parameter Documentation

Each realization saves its exact parameters to:
```
temp_seed_1000/dataset/generation_params.json
```

Example content:
```json
{
  "regime_1_coefs": [0.8523, 0.1102, 0.0375],
  "regime_2_coefs": [0.0375, 0.8523, 0.1102],
  "regime_3_coefs": [0.1102, 0.0375, 0.8523],
  "noise_sigma": 1.1234,
  "regime_lengths": [500, 500, 500],
  "randomized": true
}
```

## Usage

### Run with Randomized Parameters (Default)
```bash
python examples/robustness/06_bootstrap_ci.py \
  --datasets piecewise_ar3 \
  --n-realizations 100
```

### Run with Fixed Parameters (Original Behavior)
To disable randomization and use original fixed parameters:
1. Open `06_bootstrap_ci.py`
2. Change line 119: `randomize_params=False`

Or import from original `generate_simulated_datasets.py` instead of `generate_simulated_datasets_random.py`.

## Verification

To verify parameters are actually randomized, check the saved JSON files:

```bash
# View parameters for first two seeds
cat examples/results/robustness/bootstrap_ci/piecewise_ar3/temp_seed_1000/dataset/generation_params.json
cat examples/results/robustness/bootstrap_ci/piecewise_ar3/temp_seed_1001/dataset/generation_params.json
```

They should show different coefficients!

## For Your Paper

### Reporting

You should report:

1. **Bootstrap Design**:
   - "We generate N realizations (N=100) with randomized DGP parameters"
   - "Each realization uses a different random seed to generate both new noise realizations AND new coefficient values"
   - "Coefficient magnitudes are randomized within reasonable ranges while maintaining regime structure"

2. **Parameter Ranges** (example for piecewise_ar3):
   - Dominant AR coefficient: φ_dom ~ Uniform[0.7, 0.95]
   - Non-dominant coefficients: Sum to value ~ Uniform[0.05, 0.25]
   - Noise standard deviation: σ ~ Uniform[0.5, 1.5]

3. **Structural Preservation**:
   - "Despite parameter randomization, the qualitative regime structure is preserved"
   - "E.g., for piecewise_ar3, Lag 1 always dominates in Regime 1 across all realizations"

4. **Why This Matters**:
   - "This tests whether Adaptive WIN-SHAP is robust to the MAGNITUDE of regime differences, not just noise"
   - "A method that only works when φ=0.9 vs 0.01 would fail with φ=0.75 vs 0.15"

### Example Table for Paper

| Dataset | Parameter | Range | Structure Preserved |
|---------|-----------|-------|---------------------|
| piecewise_ar3 | φ_dominant | [0.7, 0.95] | Lag order rotation |
| | φ_non-dominant (sum) | [0.05, 0.25] | |
| | σ | [0.5, 1.5] | |
| arx_rotating | β_dominant | [0.8, 1.5] | Covariate rotation |
| | β_non-dominant | [-0.2, 0.3] | |
| | σ | [0.3, 0.7] | |

### Example Results Interpretation

"Across 100 realizations with randomized parameters, Adaptive WIN-SHAP achieved a mean faithfulness score of 0.85 (95% CI: [0.82, 0.88]). The narrow confidence interval (CI width = 7% of mean) demonstrates robustness not only to noise variation but also to different magnitudes of regime differences."

## Next Steps

1. **Run full bootstrap**: `python examples/robustness/06_bootstrap_ci.py --n-realizations 100`
2. **Analyze results**: `python examples/robustness/analyze_bootstrap_ci.py`
3. **Check parameter files**: Verify coefficients vary across seeds
4. **Report in paper**: Use the documented parameter ranges and structure

This implementation provides **much stronger evidence** of robustness than testing only noise variation!