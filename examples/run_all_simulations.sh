#!/bin/bash
# Run window detection for all simulated datasets

echo "============================================================"
echo "Running Window Detection for All Datasets"
echo "============================================================"
echo ""

# Array of all datasets
datasets=(
    "piecewise_ar3"
#    "arx_rotating"
#    "trend_season"
#    "spike_process"
#    "garch_regime"
)

# Configuration
N0=100
NUM_BOOTSTRAP=30
JUMP=1
STEP=2
NUM_RUNS=1
GROWTH="geometric"  # Options: "arithmetic" or "geometric"
GROWTH_BASE=1.41421356237     # Base for geometric growth


# Track results
declare -a results

# Run detection for each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing: $dataset"
    echo "========================================"

    python examples/lstm_simulation.py \
        --dataset "$dataset" \
        --n0 $N0 \
        --jump $JUMP \
        --step $STEP \
        --num-runs $NUM_RUNS \
        --growth $GROWTH \
        --growth-base $GROWTH_BASE \
        --num-bootstrap $NUM_BOOTSTRAP

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
echo "SUMMARY"
echo "============================================================"
for result in "${results[@]}"; do
    echo "$result"
done

echo ""
echo "============================================================"
echo "All simulations complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Review the window detection results in examples/results/LSTM/"
echo "2. Run benchmark.py for each dataset to compare SHAP methods"
echo "============================================================"