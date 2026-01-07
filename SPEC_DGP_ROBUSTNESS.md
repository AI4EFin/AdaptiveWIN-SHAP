# DGP Parameter Robustness Test Specification

## 1. Overview

This document specifies a robustness test that evaluates how the Adaptive WIN-SHAP method performs when the data generating process (DGP) parameters vary. Specifically, we test the method's sensitivity to changes in AR(3) coefficient values in the `piecewise_ar3` dataset.

### Objectives
- Test method robustness to parameter variations along L2 distance dimension
- Evaluate performance with parameters **closer** together (harder changepoint detection)
- Evaluate performance with parameters **further** apart (easier detection, mixed dynamics)
- Explore random parameter configurations to test generalizability
- Maintain existing analysis infrastructure and output format compatibility

## 2. Scenario Design

We will generate **5 DGP scenarios** total:

### 2.1 Baseline (Reference)
- Use existing `piecewise_ar3` dataset
- No new data generation needed
- Parameters: `[0.9, 0.01, 0.01]`, `[0.01, 0.9, 0.01]`, `[0.01, 0.01, 0.9]`
- Breakpoints: `[500, 1000]`
- L2 reference distance: ~1.26 (max pairwise distance between regimes)

### 2.2 Closer Scenario
- **Goal**: Reduce L2 distance to 50% of baseline (~0.63)
- **Method**: Linear interpolation toward centroid
- **Regime structure**: 3 regimes, maintain rotating dominant lag structure
- **Breakpoints**: `[500, 1000]` (unchanged)
- **Noise sigma**: 1.0 (unchanged)

### 2.3 Further Scenario
- **Goal**: Increase L2 distance with mixed dynamics
- **Method**: Manual specification with tested parameters
- **Dynamics**: Mix extreme positive coefficients with negative coefficients
- **Structure**: Some regimes with single dominant lag, others with oscillatory patterns
- **Breakpoints**: `[500, 1000]` (unchanged)
- **Noise sigma**: 1.0 (unchanged)

### 2.4-2.6 Random Scenarios (3 total)
- **Goal**: Test robustness across diverse parameter space
- **Method**: Independent random draws with different seeds
- **Constraint**: All parameters must satisfy AR(3) stationarity (characteristic polynomial roots outside unit circle)
- **Sampling**: Reject sampling from uniform `[-1, 1]^3` box
- **Structure**: Mix of dominant-lag, balanced, and intermediate structures across scenarios
- **Breakpoints**: `[500, 1000]` (unchanged)
- **Noise sigma**: 1.0 (unchanged)

## 3. Parameter Generation Algorithms

### 3.1 Closer Scenario Generation

```
Input:
  - baseline_params: original 3 regime parameter sets
  - target_reduction: 0.5 (reduce to 50%)

Algorithm:
  1. Compute centroid of all regimes:
     centroid = mean([phi_1, phi_2, phi_3])

  2. For each regime k:
     phi_k_closer = phi_k + target_reduction * (centroid - phi_k)

  3. Verify stationarity for each phi_k_closer
     - Compute roots of characteristic polynomial: 1 - phi_1*z - phi_2*z^2 - phi_3*z^3
     - Check all roots have modulus > 1

  4. If non-stationary, project onto stationary boundary:
     - Scale coefficients uniformly: phi_k_closer *= scale_factor
     - Binary search for largest scale_factor maintaining stationarity

  5. Verify L2 distance constraint:
     - Compute max pairwise L2 distance between new regimes
     - Should be approximately 50% of baseline
     - If not within 10% tolerance, adjust and retry

Output: Three regime parameter sets with reduced L2 distances
```

### 3.2 Further Scenario Generation

```
Input:
  - baseline_params: original parameters
  - target_multiplier: 2.0 (double the L2 distance)

Algorithm:
  1. Manually specify three regimes with mixed dynamics:

     Regime 1 (extreme positive):
       phi_1 = [0.95, 0.01, 0.01]

     Regime 2 (mixed with negative):
       phi_2 = [0.6, -0.3, 0.2]

     Regime 3 (oscillatory):
       phi_3 = [-0.1, 0.8, 0.1]

  2. Verify each regime is stationary
     - Check characteristic polynomial roots

  3. Verify L2 distances are larger than baseline:
     - Compute L2(phi_1, phi_2), L2(phi_2, phi_3), L2(phi_1, phi_3)
     - Ensure max pairwise distance > 1.5 * baseline

  4. If constraints not met, adjust coefficients:
     - Scale coefficients to increase distance
     - Add/adjust negative components
     - Re-verify stationarity after each adjustment

Output: Three regime parameter sets with increased L2 distances and mixed dynamics
```

### 3.3 Random Scenario Generation

```
Input:
  - seed: unique random seed for each scenario
  - max_attempts: 1000
  - n_regimes: 3

Algorithm:
  1. Initialize random number generator with seed

  2. For each regime k in 1..n_regimes:

     attempt = 0
     stationary = False

     while not stationary and attempt < max_attempts:

       # Sample uniformly from [-1, 1] for each coefficient
       phi_k = uniform(-1, 1, size=3)

       # Check stationarity
       roots = compute_characteristic_roots(phi_k)
       stationary = all(abs(roots) > 1.0)

       attempt += 1

     if not stationary:
       # Fallback: project onto stationary region
       phi_k = project_to_stationary(phi_k)

     regimes[k] = phi_k

  3. Verify regimes are distinct:
     - Compute pairwise L2 distances
     - If any distance < 0.3, regenerate that pair

Output: Three regime parameter sets sampled from stationary region
```

### 3.4 Stationarity Check Helper

```
Function: is_stationary(phi)
  Input: phi = [phi_1, phi_2, phi_3]

  # Characteristic polynomial: 1 - phi_1*z - phi_2*z^2 - phi_3*z^3 = 0
  # For stationarity, roots must be outside unit circle

  coefficients = [1, -phi_1, -phi_2, -phi_3]
  roots = numpy.roots(coefficients)

  return all(abs(roots) > 1.0)
```

### 3.5 Projection to Stationary Region (Fallback)

```
Function: project_to_stationary(phi)
  Input: phi = [phi_1, phi_2, phi_3] (potentially non-stationary)

  # Binary search for largest scaling factor
  scale_min = 0.0
  scale_max = 1.0
  tolerance = 1e-4

  while scale_max - scale_min > tolerance:
    scale = (scale_min + scale_max) / 2
    phi_scaled = phi * scale

    if is_stationary(phi_scaled):
      scale_min = scale
    else:
      scale_max = scale

  return phi * scale_min
```

## 4. Directory Structure

```
examples/
  datasets/
    simulated/
      piecewise_ar3/                    # Original baseline (unchanged)
        data.csv
        true_importances.csv

      piecewise_ar3_dgp_robustness/     # New parent directory
        closer/
          data.csv
          true_importances.csv
          scenario_config.json

        further/
          data.csv
          true_importances.csv
          scenario_config.json

        random_1/
          data.csv
          true_importances.csv
          scenario_config.json

        random_2/
          data.csv
          true_importances.csv
          scenario_config.json

        random_3/
          data.csv
          true_importances.csv
          scenario_config.json

        dgp_parameters.csv              # Summary table of all scenarios

  results/
    dgp_robustness/                     # Robustness test results
      piecewise_ar3/
        baseline/                       # Results for original dataset
          windows.csv
          benchmark/
            adaptive_shap_results.csv
            benchmark_summary.csv
            config.json

        closer/                         # Results for closer scenario
          windows.csv
          benchmark/
            [same structure]

        further/
          [same structure]

        random_1/
          [same structure]

        random_2/
          [same structure]

        random_3/
          [same structure]

        summary_all_scenarios.csv       # Aggregated results across all scenarios

        figures/                        # Visualizations
          window_comparison.png
          metrics_vs_l2_distance.png
          shap_correlation_by_scenario.png
          [additional plots]
```

## 5. Metadata Format

Each scenario will have a `scenario_config.json` file with complete configuration:

```json
{
  "scenario_name": "closer",
  "scenario_type": "l2_distance_manipulation",
  "target_l2_distance": 0.63,
  "actual_l2_distance": 0.65,
  "baseline_reference": "piecewise_ar3",

  "regimes": [
    {
      "regime_id": 0,
      "time_range": [0, 500],
      "parameters": {
        "phi_1": 0.7,
        "phi_2": 0.15,
        "phi_3": 0.15
      },
      "l2_distance_from_baseline": 0.32
    },
    {
      "regime_id": 1,
      "time_range": [500, 1000],
      "parameters": {
        "phi_1": 0.15,
        "phi_2": 0.7,
        "phi_3": 0.15
      },
      "l2_distance_from_baseline": 0.35
    },
    {
      "regime_id": 2,
      "time_range": [1000, 1500],
      "parameters": {
        "phi_1": 0.15,
        "phi_2": 0.15,
        "phi_3": 0.7
      },
      "l2_distance_from_baseline": 0.34
    }
  ],

  "generation_params": {
    "T": 1500,
    "noise_sigma": 1.0,
    "seed": 42,
    "generation_method": "linear_interpolation_to_centroid"
  },

  "stationarity_check": {
    "regime_0_stationary": true,
    "regime_1_stationary": true,
    "regime_2_stationary": true
  },

  "pairwise_l2_distances": {
    "regime_0_vs_1": 0.78,
    "regime_1_vs_2": 0.78,
    "regime_0_vs_2": 0.78
  }
}
```

## 6. Pipeline Steps (End-to-End Script)

The main script will execute these steps sequentially:

```
Step 1: Generate DGP Scenarios
  For each scenario in [closer, further, random_1, random_2, random_3]:
    - Generate AR(3) parameters according to specification
    - Verify stationarity constraints
    - Simulate time series data
    - Compute true importances
    - Save data.csv, true_importances.csv, scenario_config.json

Step 2: Run LPA Window Detection
  For each scenario (including baseline):
    - Load data from CSV
    - Run LPA with fixed parameters: N0=75, alpha=0.95, num_bootstrap=50, jump=1
    - Save windows.csv

Step 3: Run SHAP Benchmark
  For each scenario:
    - Load data and windows
    - Run benchmark.py equivalent with precomputed windows
    - Compute adaptive SHAP values
    - Compute faithfulness and ablation metrics
    - Save benchmark results

Step 4: Compute Robustness Metrics
  For each scenario:
    - Load windows.csv, true breakpoints, SHAP results, true importances
    - Compute:
      * Window detection accuracy (lag to true breakpoints)
      * Regime-specific SHAP-to-truth correlations
      * Aggregate benchmark metrics (faithfulness, ablation)
    - Save to scenario-specific results

Step 5: Aggregate Results
  - Combine all scenario metrics into summary_all_scenarios.csv
  - Compute L2 distances between scenarios
  - Create comparison table

Step 6: Generate Visualizations
  - Window evolution comparison across scenarios
  - Metrics vs L2 distance scatter plots
  - Regime-specific correlation heatmaps
  - Comparative window overlay plots
```

## 7. Key Metrics to Compute

### 7.1 Window Detection Accuracy
For each scenario and each true breakpoint:
```
- Detection lag: t_detected - t_true
- Detection success: whether breakpoint detected within tolerance window (±50 steps)
- Mean absolute detection lag across all breakpoints
- False positive rate: number of detected changepoints with no true breakpoint nearby
```

### 7.2 SHAP Consistency with True Importances
For each scenario and each regime:
```
- Pearson correlation between SHAP values and true importances
- Spearman rank correlation (for ordinal consistency)
- RMSE between normalized SHAP and true importances
- Top-k feature agreement (do top features match?)
```

### 7.3 Benchmark Metrics
For each scenario (aggregate across all timepoints):
```
- Faithfulness: mean perturbation score (prtb_p50, prtb_p90)
- Ablation: mean MIF and LIF scores (mif_p50, mif_p90, lif_p50, lif_p90)
```

### 7.4 Window Statistics
For each scenario:
```
- Mean window size
- Window size standard deviation (stability)
- Min/max window sizes
- Window size variance within each regime (should be low)
- Window size variance near transitions (may be high)
```

## 8. Results Summary CSV Format

File: `summary_all_scenarios.csv`

One row per scenario with aggregated metrics:

```csv
scenario_name,scenario_type,l2_distance_from_baseline,max_pairwise_l2,detection_lag_mean,detection_lag_std,detection_success_rate,false_positive_rate,shap_corr_regime0,shap_corr_regime1,shap_corr_regime2,shap_corr_overall,faithfulness_p50,faithfulness_p90,ablation_mif_p90,ablation_lif_p90,window_mean,window_std,window_min,window_max
baseline,original,0.0,1.26,12.5,8.3,1.0,0.0,0.92,0.89,0.91,0.91,0.0023,0.0045,0.85,0.12,180,45,50,320
closer,l2_reduced,0.65,0.65,25.3,15.2,0.75,0.05,0.88,0.84,0.87,0.86,0.0028,0.0052,0.81,0.15,195,62,48,340
further,l2_increased,1.85,2.10,8.1,5.5,1.0,0.0,0.94,0.92,0.93,0.93,0.0021,0.0041,0.88,0.10,165,38,55,305
...
```

## 9. Visualization Requirements

### 9.1 Window Evolution Comparison
- Multi-panel plot (one panel per scenario)
- Each panel shows window size over time
- Vertical lines at true breakpoints
- Coloring/shading by regime
- Title includes scenario name and L2 distance

### 9.2 Metrics vs L2 Distance
- Scatter plot with L2 distance on x-axis
- Separate subplots or colors for different metrics:
  * Detection lag
  * SHAP correlation
  * Faithfulness score
- Shows whether degradation is monotonic with distance

### 9.3 Comparative Window Overlay
- Single plot with all scenarios overlaid
- Different colors/line styles per scenario
- True breakpoints marked
- Legend with scenario names and L2 distances
- Shows how window detection changes with parameters

### 9.4 Regime-Specific SHAP Correlations
- Grouped bar chart or heatmap
- Scenarios on one axis, regimes on another
- Color intensity = correlation strength
- Identifies which regimes are harder to explain

## 10. LPA Configuration (Fixed Across All Scenarios)

```python
LPA_CONFIG = {
    'N0': 75,           # Initial window size
    'alpha': 0.95,      # Confidence level
    'num_bootstrap': 50, # Bootstrap iterations
    'jump': 1,          # Step size
    'growth': 'geometric' # Window growth strategy
}
```

This balanced configuration provides good accuracy without excessive computation.

## 11. Error Handling Strategy

### Non-Stationary Parameters
```
If parameter generation produces non-stationary coefficients:
  1. Log warning with attempted parameters
  2. Project parameters onto stationary boundary using binary search scaling
  3. Verify projected parameters maintain desired structure (e.g., dominant lag)
  4. Continue execution with fallback parameters
  5. Note in scenario_config.json that fallback was used
```

### Detection Failures
```
If LPA detection fails or produces degenerate windows:
  1. Log error with scenario and parameters
  2. Save partial results (data still generated)
  3. Mark scenario as "detection_failed" in summary CSV
  4. Continue with remaining scenarios
```

### Benchmark Failures
```
If SHAP computation fails:
  1. Log error with scenario and failure point
  2. Save whatever results were computed up to failure
  3. Mark metrics as NaN in summary CSV
  4. Continue with remaining scenarios
```

## 12. Script Interface

### Main Script: `examples/robustness/dgp_parameter_robustness.py`

```
Usage:
  python examples/robustness/dgp_parameter_robustness.py [options]

Options:
  --dataset DATASET       Base dataset name (default: piecewise_ar3)
  --output-dir DIR        Output directory (default: examples/results/dgp_robustness)
  --seeds SEED1,SEED2,SEED3   Random seeds for 3 random scenarios (default: 42,43,44)
  --skip-generation       Skip data generation, use existing datasets
  --skip-detection        Skip LPA detection, use existing windows
  --skip-benchmark        Skip SHAP benchmark, use existing results
  --visualize-only        Only generate visualizations from existing results
  --verbose               Print detailed progress information

Examples:
  # Run full pipeline
  python examples/robustness/dgp_parameter_robustness.py --verbose

  # Regenerate visualizations only
  python examples/robustness/dgp_parameter_robustness.py --visualize-only

  # Use different random seeds
  python examples/robustness/dgp_parameter_robustness.py --seeds 100,200,300
```

## 13. Success Criteria

The robustness test implementation is successful if:

1. All 5 scenarios generate valid, stationary time series data
2. L2 distance constraints are met (closer ≈ 0.63, further > 1.5)
3. LPA detection completes for all scenarios
4. SHAP benchmarks complete for all scenarios
5. Summary CSV contains complete metrics for all scenarios
6. Visualizations clearly show performance differences across scenarios
7. Results directory structure matches specification
8. All metadata files are valid JSON and contain required fields

## 14. Extension Points

Future enhancements could include:

- Testing other baseline datasets (arx_rotating, trend_season, etc.)
- Varying regime lengths in addition to parameters
- Testing interaction effects (DGP × LPA parameters)
- Statistical significance tests between scenarios
- Automated parameter tuning based on DGP characteristics
- Confidence intervals via bootstrap resampling of each scenario