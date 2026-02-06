
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
    "piecewise_ar3"
    "arx_rotating"
    "trend_season"
    "spike_process"
    "garch_regime"
)

# Configuration (matching run_all_simulations.sh)
N0=75
JUMP=1
NUM_RUNS=9
GROWTH="geometric"
GROWTH_BASE=2.0     # Base for geometric growth

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
        --num-runs $NUM_RUNS \
        --growth $GROWTH \
        --growth-base $GROWTH_BASE

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
        --growth $GROWTH

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
        --data-type simulated

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
echo "  • Window sizes: examples/results/LSTM/{dataset}/${GROWTH}/Jump_${JUMP}_N0_${N0}/"
echo "  • Benchmarks:   examples/results/benchmark_{dataset}/"
echo "  • Figures:      examples/results/benchmark_{dataset}/figures/"
echo ""
echo "Review benchmark_summary.csv in each results directory for metrics."
echo "============================================================"