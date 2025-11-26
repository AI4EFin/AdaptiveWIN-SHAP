#!/bin/bash
# Run visualizations for all simulated dataset benchmarks

echo "============================================================"
echo "Running Benchmark Visualizations for All Datasets"
echo "============================================================"
echo ""

# Array of all datasets (matching run_all_benchmarks.sh)
datasets=(
    "piecewise_ar3"
    "arx_rotating"
    "trend_season"
    "spike_process"
    "tvp_arx"
    #"garch_regime"
    #"cointegration"
)

# Track results
declare -a results

# Run visualization for each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Visualizing: $dataset"
    echo "========================================"

    # Check if benchmark results exist
    results_dir="examples/results/benchmark_${dataset}"
    if [ ! -d "$results_dir" ]; then
        echo "⚠  Warning: Benchmark results not found for $dataset"
        echo "   Expected directory: $results_dir"
        echo "   Run benchmarks first with: python examples/benchmark.py --dataset $dataset"
        results+=("$dataset : SKIPPED (no results)")
        continue
    fi

    # Check if required result files exist
    required_files=("global_shap_results.csv" "rolling_shap_results.csv" "adaptive_shap_results.csv")
    missing_files=()

    for file in "${required_files[@]}"; do
        if [ ! -f "$results_dir/$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "⚠  Warning: Missing result files for $dataset:"
        for file in "${missing_files[@]}"; do
            echo "   - $file"
        done
        results+=("$dataset : SKIPPED (incomplete results)")
        continue
    fi

    # Run visualization
    python examples/benchmark_viz.py \
        --dataset "$dataset" \
        --data-type simulated

    if [ $? -eq 0 ]; then
        results+=("$dataset : SUCCESS")
        echo "✓ Successfully visualized: $dataset"

        # Count generated figures
        figures_dir="$results_dir/figures"
        if [ -d "$figures_dir" ]; then
            num_figures=$(ls -1 "$figures_dir"/*.png 2>/dev/null | wc -l)
            echo "  Generated $num_figures visualization(s) in $figures_dir"
        fi
    else
        results+=("$dataset : FAILED")
        echo "✗ Failed: $dataset"
    fi
done

# Print summary
echo ""
echo "============================================================"
echo "VISUALIZATION SUMMARY"
echo "============================================================"
for result in "${results[@]}"; do
    echo "$result"
done

echo ""
echo "============================================================"
echo "All visualizations complete!"
echo "============================================================"
echo ""
echo "View results:"
echo "  examples/results/benchmark_*/figures/"
echo ""
echo "Each dataset includes visualizations for:"
echo "  • Faithfulness & Ablation comparisons"
echo "  • SHAP values over time & heatmaps"
echo "  • True importance comparisons (ground truth)"
echo "  • Correlation analysis"
echo "  • Summary dashboard"
echo "============================================================"