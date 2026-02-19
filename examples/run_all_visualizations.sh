#!/bin/bash
# Run visualizations for all simulated dataset benchmarks

echo "============================================================"
echo "Running Benchmark Visualizations for All Datasets"
echo "============================================================"
echo ""

# Array of all datasets (matching run_all_benchmarks.sh)
datasets=(
    "piecewise_ar3"
#    "arx_rotating"
#    "trend_season"
#    "piecewise_ar3_long"
#    "arx_rotating_long"
)

# Configuration (must match run_all_benchmarks.sh)
N0=100
PENALTY_FACTOR=0.1

# Track results
declare -a results

# Run visualization for each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Visualizing: $dataset"
    echo "========================================"

    # Check if benchmark results directory exists
    results_dir="examples/results/benchmark_${dataset}/N0_${N0}_lambda_${PENALTY_FACTOR}"
    if [ ! -d "$results_dir" ]; then
        echo "Warning: Benchmark results not found for $dataset"
        echo "   Expected directory: $results_dir"
        results+=("$dataset : SKIPPED (no results)")
        continue
    fi

    # Check if benchmark summary exists
    if [ ! -f "$results_dir/benchmark_summary.csv" ]; then
        echo "Warning: benchmark_summary.csv not found for $dataset"
        results+=("$dataset : SKIPPED (no benchmark_summary.csv)")
        continue
    fi

    # Run visualization
    python examples/benchmark_viz.py \
        --dataset "$dataset" \
        --data-type simulated \
        --n0 $N0 \
        --penalty-factor $PENALTY_FACTOR

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
echo "  examples/results/benchmark_*/N0_${N0}_lambda_${PENALTY_FACTOR}/figures/"
echo "============================================================"
