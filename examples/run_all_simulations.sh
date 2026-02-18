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
#    "switching_factor"
)

# Configuration (matching lstm_simulation.py defaults and 01_lpa_sensitivity.py)
N0=100
JUMP=1
STEP=2
ALPHA=0.95
NUM_RUNS=1
GROWTH="geometric"
GROWTH_BASE=1.41421356237     # sqrt(2)
MC_REPS=300
PENALTY_FACTOR=0.1

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
        --alpha $ALPHA \
        --num-runs $NUM_RUNS \
        --growth $GROWTH \
        --growth-base $GROWTH_BASE \
        --mc-reps $MC_REPS \
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
