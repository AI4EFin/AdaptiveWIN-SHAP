"""
Visualize window sizes from an LSTM simulation run.

Plots window size evolution over time with breakpoints, rolling mean,
and basic statistics.

Usage:
    # Auto-detect latest result for a dataset
    python examples/visualize_lstm_simulation.py --dataset piecewise_ar3

    # Specify exact result directory
    python examples/visualize_lstm_simulation.py --result-dir examples/results/LSTM/piecewise_ar3/Jump_1_N0_100_lambda_0.1

    # Customize rolling mean window
    python examples/visualize_lstm_simulation.py --dataset arx_rotating --rolling-mean 30
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.robustness.visualize_robustness import RobustnessVisualizer, DATASET_BREAKPOINTS

sns.set_palette("colorblind")


def find_result_dir(dataset_name, base_dir='examples/results/LSTM'):
    """Find the result directory for a dataset, picking the first match."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        return None

    # Look for subdirectories with windows.csv
    for entry in sorted(os.listdir(dataset_dir)):
        candidate = os.path.join(dataset_dir, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'windows.csv')):
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize window sizes from LSTM simulation results'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset name (e.g. piecewise_ar3). Auto-detects result directory.'
    )
    parser.add_argument(
        '--result-dir', type=str, default=None,
        help='Explicit path to result directory containing windows.csv'
    )
    parser.add_argument(
        '--rolling-mean', type=int, default=20,
        help='Rolling mean window size (default: 20)'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display the plot (only save)'
    )
    args = parser.parse_args()

    if args.result_dir is None and args.dataset is None:
        parser.error('Provide either --dataset or --result-dir')

    # Resolve result directory
    if args.result_dir:
        result_dir = args.result_dir
        # Infer dataset name from path: .../LSTM/<dataset>/...
        parts = result_dir.replace('\\', '/').split('/')
        if 'LSTM' in parts:
            dataset_name = parts[parts.index('LSTM') + 1]
        else:
            dataset_name = 'unknown'
    else:
        dataset_name = args.dataset
        result_dir = find_result_dir(dataset_name)
        if result_dir is None:
            print(f"Error: No results found for dataset '{dataset_name}' in examples/results/LSTM/")
            sys.exit(1)

    windows_path = os.path.join(result_dir, 'windows.csv')
    if not os.path.exists(windows_path):
        print(f"Error: {windows_path} not found")
        sys.exit(1)

    print(f"Dataset:    {dataset_name}")
    print(f"Result dir: {result_dir}")
    print(f"Windows:    {windows_path}")

    # Load windows
    windows_df = pd.read_csv(windows_path, index_col=0)
    breakpoints = DATASET_BREAKPOINTS.get(dataset_name)

    # Build title from directory name
    config_name = os.path.basename(result_dir)
    title = f'{dataset_name}: Window Size Evolution ({config_name})'

    # Save figures
    fig_dir = os.path.join(result_dir, 'figures')
    viz = RobustnessVisualizer(output_dir=fig_dir)

    viz.plot_window_evolution(
        windows_df,
        dataset_name,
        title=title,
        breakpoints=breakpoints,
        save_name='window_sizes',
        rolling_mean_size=args.rolling_mean,
    )

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
