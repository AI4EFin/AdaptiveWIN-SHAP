#!/bin/bash
# Run SHAP benchmarks for all simulated datasets

echo "============================================================"
echo "Running SHAP Benchmarks for All Datasets"
echo "============================================================"
echo ""

# Array of all datasets (matching run_all_simulations.sh)
datasets=(
    "piecewise_ar3"
#    "arx_rotating"
#    "trend_season"
#    "piecewise_ar3_long"
#    "arx_rotating_long"
)

# Configuration (must match run_all_simulations.sh)
N0=100
JUMP=1
ROLLING_MEAN_WINDOW=75
GROWTH="geometric"
PENALTY_FACTOR=0.1

# Track results
declare -a results

# Run benchmark for each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Benchmarking: $dataset"
    echo "========================================"

    python examples/benchmark.py \
        --dataset "$dataset" \
        --data-type simulated \
        --n0 $N0 \
        --jump $JUMP \
        --rolling-mean-window $ROLLING_MEAN_WINDOW \
        --growth $GROWTH \
        --penalty-factor $PENALTY_FACTOR

    if [ $? -eq 0 ]; then
        results+=("$dataset : SUCCESS")
        echo "✓ Successfully completed: $dataset"
    else
        results+=("$dataset : FAILED")
        echo "✗ Failed: $dataset"
    fi
done

# Print summary
echo ""
echo "============================================================"
echo "BENCHMARK SUMMARY"
echo "============================================================"
for result in "${results[@]}"; do
    echo "$result"
done

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Review benchmark results in examples/results/benchmark_*/N0_${N0}_lambda_${PENALTY_FACTOR}/"
echo "2. Run visualization script for each dataset"
echo "============================================================"
