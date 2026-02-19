#!/bin/bash
# Complete pipeline: Window Detection → Benchmarking → Visualization

echo "============================================================"
echo "AdaptiveWIN-SHAP Complete Benchmarking Pipeline"
echo "============================================================"
echo ""
echo "This script will run:"
echo "  1. Window detection (adaptive window sizes)"
echo "  2. SHAP benchmarking (Global, Rolling, Adaptive)"
echo "  3. Visualization generation"
echo ""
echo "This may take several hours to complete."
echo "============================================================"
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# Array of all datasets
datasets=(
#    "piecewise_ar3"
    "arx_rotating"
    "trend_season"
    "piecewise_ar3_long"
    "arx_rotating_long"
)

# Configuration (matching lstm_simulation.py defaults and 01_lpa_sensitivity.py)
N0=100
JUMP=1
STEP=1
ALPHA=0.95
NUM_RUNS=1
GROWTH="geometric"
GROWTH_BASE=1.41421356237     # sqrt(2)
MC_REPS=300
PENALTY_FACTOR=0.0

# Track results
declare -a window_results
declare -a benchmark_results
declare -a viz_results

echo ""
echo "============================================================"
echo "STEP 1: Window Detection"
echo "============================================================"
echo ""

for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Detecting windows: $dataset"
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
        window_results+=("$dataset : SUCCESS")
        echo "✓ Window detection complete: $dataset"
    else
        window_results+=("$dataset : FAILED")
        echo "✗ Window detection failed: $dataset"
    fi
done

echo ""
echo "============================================================"
echo "STEP 2: SHAP Benchmarking"
echo "============================================================"
echo ""

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
        benchmark_results+=("$dataset : SUCCESS")
        echo "✓ Benchmark complete: $dataset"
    else
        benchmark_results+=("$dataset : FAILED")
        echo "✗ Benchmark failed: $dataset"
    fi
done

echo ""
echo "============================================================"
echo "STEP 3: Visualization"
echo "============================================================"
echo ""

for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Visualizing: $dataset"
    echo "========================================"

    python examples/benchmark_viz.py \
        --dataset "$dataset" \
        --data-type simulated \
        --n0 $N0 \
        --penalty-factor $PENALTY_FACTOR

    if [ $? -eq 0 ]; then
        viz_results+=("$dataset : SUCCESS")
        echo "✓ Visualization complete: $dataset"
    else
        viz_results+=("$dataset : FAILED")
        echo "✗ Visualization failed: $dataset"
    fi
done

# Print comprehensive summary
echo ""
echo "============================================================"
echo "COMPLETE PIPELINE SUMMARY"
echo "============================================================"
echo ""

echo "Window Detection Results:"
echo "-------------------------"
for result in "${window_results[@]}"; do
    echo "$result"
done

echo ""
echo "Benchmark Results:"
echo "-------------------------"
for result in "${benchmark_results[@]}"; do
    echo "$result"
done

echo ""
echo "Visualization Results:"
echo "-------------------------"
for result in "${viz_results[@]}"; do
    echo "$result"
done

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo ""
echo "Results are organized in:"
echo "  - Window sizes: examples/results/LSTM/{dataset}/Jump_${JUMP}_N0_${N0}_lambda_${PENALTY_FACTOR}/"
echo "  - Benchmarks:   examples/results/benchmark_{dataset}/N0_${N0}_lambda_${PENALTY_FACTOR}/"
echo "  - Figures:      examples/results/benchmark_{dataset}/N0_${N0}_lambda_${PENALTY_FACTOR}/figures/"
echo ""
echo "Review benchmark_summary.csv in each results directory for metrics."
echo "============================================================"
