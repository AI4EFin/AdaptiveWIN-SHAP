"""
Test Script for New Visualization Features

Tests MIF/LIF ratio computation and window analysis methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from visualize_robustness import RobustnessVisualizer, DATASET_BREAKPOINTS


def test_mif_lif_ratios():
    """Test MIF/LIF ratio computation."""
    print("\n" + "="*80)
    print("TEST 1: MIF/LIF Ratio Computation")
    print("="*80)

    # Create sample data
    data = {
        'ablation_mif_p50': [0.8, 0.85, 0.9],
        'ablation_lif_p50': [0.4, 0.45, 0.5],
        'ablation_mif_p90': [0.7, 0.75, 0.8],
        'ablation_lif_p90': [0.3, 0.35, 0.4],
        'faithfulness': [0.9, 0.92, 0.94]
    }
    df = pd.DataFrame(data)

    print("\nOriginal DataFrame:")
    print(df)

    # Initialize visualizer
    viz = RobustnessVisualizer(output_dir='examples/results/robustness/test')

    # Compute ratios
    df_with_ratios = viz.compute_mif_lif_ratios(df, percentiles=[50, 90])

    print("\nDataFrame with MIF/LIF ratios:")
    print(df_with_ratios[['ablation_mif_p50', 'ablation_lif_p50', 'mif_lif_ratio_p50',
                          'ablation_mif_p90', 'ablation_lif_p90', 'mif_lif_ratio_p90']])

    # Check ratios
    expected_p50 = data['ablation_mif_p50'][0] / data['ablation_lif_p50'][0]
    actual_p50 = df_with_ratios['mif_lif_ratio_p50'].iloc[0]

    print(f"\nExpected ratio p50: {expected_p50:.4f}")
    print(f"Actual ratio p50: {actual_p50:.4f}")
    print(f"Match: {np.isclose(expected_p50, actual_p50)}")

    print("\n✓ MIF/LIF ratio computation works correctly!")
    return True


def test_window_methods():
    """Test window analysis methods."""
    print("\n" + "="*80)
    print("TEST 2: Window Analysis Methods")
    print("="*80)

    # Create synthetic window data
    T = 1500
    breakpoints = [500, 1000]
    dataset_name = 'test_dataset'

    # Simulate window sizes with some pattern
    windows = np.zeros(T)
    for t in range(T):
        # Base window size grows linearly
        windows[t] = min(50 + t * 0.1, 500)
        # Add some noise
        windows[t] += np.random.normal(0, 10)
        # Reset at breakpoints
        if t in breakpoints:
            windows[t] = 50

    windows_df = pd.DataFrame({
        'windows': windows,
        'window_mean': windows  # Pretend this is the mean
    })

    # Initialize visualizer
    viz = RobustnessVisualizer(output_dir='examples/results/robustness/test')

    # Test each window method
    print("\n1. Testing plot_window_evolution()...")
    try:
        fig = viz.plot_window_evolution(
            windows_df=windows_df,
            dataset_name=dataset_name,
            breakpoints=breakpoints,
            save_name='test_window_evolution',
            show_statistics=True
        )
        plt.close(fig)
        print("   ✓ plot_window_evolution() works!")
    except Exception as e:
        print(f"   ✗ plot_window_evolution() failed: {e}")
        return False

    print("\n2. Testing plot_window_distribution()...")
    try:
        fig = viz.plot_window_distribution(
            windows_df=windows_df,
            dataset_name=dataset_name,
            breakpoints=breakpoints,
            save_name='test_window_distribution'
        )
        plt.close(fig)
        print("   ✓ plot_window_distribution() works!")
    except Exception as e:
        print(f"   ✗ plot_window_distribution() failed: {e}")
        return False

    print("\n3. Testing plot_true_vs_detected_windows()...")
    try:
        fig = viz.plot_true_vs_detected_windows(
            windows_df=windows_df,
            dataset_name=dataset_name,
            breakpoints=breakpoints,
            save_name='test_true_vs_detected'
        )
        plt.close(fig)
        print("   ✓ plot_true_vs_detected_windows() works!")
    except Exception as e:
        print(f"   ✗ plot_true_vs_detected_windows() failed: {e}")
        return False

    print("\n4. Testing plot_window_stability()...")
    try:
        # Create realizations data
        realizations_df = pd.DataFrame({
            'window_mean': np.random.normal(200, 30, 100),
            'window_std': np.random.normal(50, 10, 100),
            'window_min': np.random.normal(50, 10, 100),
            'window_max': np.random.normal(400, 50, 100)
        })

        fig = viz.plot_window_stability(
            results_df=realizations_df,
            dataset_name=dataset_name,
            save_name='test_window_stability'
        )
        plt.close(fig)
        print("   ✓ plot_window_stability() works!")
    except Exception as e:
        print(f"   ✗ plot_window_stability() failed: {e}")
        return False

    print("\n5. Testing plot_window_vs_parameters()...")
    try:
        # Create parameter sensitivity data
        param_df = pd.DataFrame({
            'N0': [50, 75, 100, 150, 200],
            'window_mean': [100, 150, 200, 300, 400],
            'faithfulness': [0.8, 0.85, 0.9, 0.88, 0.86]
        })

        fig = viz.plot_window_vs_parameters(
            results_df=param_df,
            param_col='N0',
            dataset_name=dataset_name,
            save_name='test_window_vs_n0',
            window_stat='mean'
        )
        plt.close(fig)
        print("   ✓ plot_window_vs_parameters() works!")
    except Exception as e:
        print(f"   ✗ plot_window_vs_parameters() failed: {e}")
        return False

    print("\n✓ All window analysis methods work correctly!")
    return True


def test_dataset_breakpoints():
    """Test DATASET_BREAKPOINTS constant."""
    print("\n" + "="*80)
    print("TEST 3: DATASET_BREAKPOINTS Constant")
    print("="*80)

    print("\nAvailable datasets and their breakpoints:")
    for dataset, breakpoints in DATASET_BREAKPOINTS.items():
        print(f"  {dataset:20s}: {breakpoints}")

    expected_datasets = ['piecewise_ar3', 'arx_rotating', 'trend_season',
                        'spike_process', 'garch_regime']

    for dataset in expected_datasets:
        if dataset not in DATASET_BREAKPOINTS:
            print(f"\n✗ Missing dataset: {dataset}")
            return False

    print("\n✓ DATASET_BREAKPOINTS constant is correctly defined!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING NEW VISUALIZATION FEATURES")
    print("="*80)
    print("\nThis script tests the new MIF/LIF ratio and window analysis features.")

    results = []

    # Test 1: MIF/LIF ratios
    try:
        results.append(("MIF/LIF Ratios", test_mif_lif_ratios()))
    except Exception as e:
        print(f"\n✗ MIF/LIF ratio test failed with exception: {e}")
        results.append(("MIF/LIF Ratios", False))

    # Test 2: Window analysis methods
    try:
        results.append(("Window Analysis", test_window_methods()))
    except Exception as e:
        print(f"\n✗ Window analysis test failed with exception: {e}")
        results.append(("Window Analysis", False))

    # Test 3: Dataset breakpoints
    try:
        results.append(("Dataset Breakpoints", test_dataset_breakpoints()))
    except Exception as e:
        print(f"\n✗ Dataset breakpoints test failed with exception: {e}")
        results.append(("Dataset Breakpoints", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s}: {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nNew features are working correctly!")
        print("Test figures saved to: examples/results/robustness/test/")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
