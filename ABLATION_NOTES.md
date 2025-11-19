 1. Added Ablation Metric Function (examples/benchmarking/metrics.py)

  - Created compute_point_ablation() function that implements ablation analysis
  - Supports two ablation strategies:
    - MIF (Most Important First): Removes most important features first - should cause large prediction changes
    - LIF (Least Important First): Removes least important features first - should cause small prediction changes
  - Uses percentile-based evaluation (p90, p70, p50) matching the faithfulness implementation
  - Iteratively removes features and measures cumulative effect on predictions

  2. Integrated into All Baseline Methods

  - GlobalSHAP (examples/benchmarking/baselines.py:275-299): Computes ablation scores alongside faithfulness
  - RollingWindowSHAP (examples/benchmarking/baselines.py:422-441): Computes ablation scores for each rolling window
  - AdaptiveWinShap (src/adaptivewinshap/shap.py:385-418): Computes ablation scores for adaptive windows

  3. Updated Benchmark Summary (examples/benchmark.py)

  - Modified extract_metrics() function to extract both faithfulness and ablation columns
  - CSV results now include columns like: ablation_mif_p90, ablation_mif_p70, ablation_lif_p90, etc.
  - Summary DataFrame includes both metric types with proper aggregation

  4. Exported New Functions (examples/benchmarking/__init__.py)

  - Added compute_point_ablation to module exports

  How Ablation Works

  Ablation is considered one of the most rigorous XAI evaluation metrics because it:

  1. Systematically removes features in order of importance (based on SHAP values)
  2. Measures cumulative effect on model predictions as each feature is removed
  3. Validates explanations by confirming that removing important features causes larger prediction changes

  Key insight: If SHAP correctly identifies important features, then:
  - MIF (removing most important first) should cause large prediction changes
  - LIF (removing least important first) should cause small prediction changes
  - The ratio or difference between MIF and LIF can quantify explanation quality

  Running the Benchmark

  Your existing benchmark script will now automatically compute ablation scores:

  python examples/benchmark.py

  The results will include ablation columns in all three CSV files:
  - global_shap_results.csv
  - rolling_shap_results.csv
  - adaptive_shap_results.csv

  And the benchmark_summary.csv will contain aggregated ablation scores for comparison across methods.