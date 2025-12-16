"""
Quick test to verify robustness analysis setup.

Runs a minimal LPA sensitivity test to ensure all dependencies
and integrations work correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from adaptivewinshap import AdaptiveModel, ChangeDetector


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        import torch
        import numpy as np
        import pandas as pd
        from adaptivewinshap import AdaptiveModel, ChangeDetector
        from itertools import product
        from tqdm import tqdm
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_dataset_access():
    """Test that datasets are accessible."""
    print("\nTesting dataset access...")

    datasets = ['piecewise_ar3', 'arx_rotating']
    all_found = True

    for dataset in datasets:
        path = f"examples/datasets/simulated/{dataset}/data.csv"
        true_imp_path = f"examples/datasets/simulated/{dataset}/true_importances.csv"

        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ {dataset}: {len(df)} time points, {len(df.columns)} columns")
        else:
            print(f"✗ {dataset}: not found at {path}")
            all_found = False

        if os.path.exists(true_imp_path):
            true_df = pd.read_csv(true_imp_path)
            print(f"  True importances: {len(true_df.columns)} features")
        else:
            print(f"  ⚠ True importances not found")

    return all_found


def test_lpa_detection():
    """Test minimal LPA detection."""
    print("\nTesting LPA detection...")

    try:
        # Import the LSTM model class inline to avoid circular dependency
        from adaptivewinshap import AdaptiveModel, store_init_kwargs
        import torch.nn as nn

        class AdaptiveLSTM(AdaptiveModel):
            @store_init_kwargs
            def __init__(self, device, seq_length=3, input_size=1, hidden=16, layers=1,
                         dropout=0.2, batch_size=512, lr=1e-12, epochs=50, type_precision=np.float32):
                super().__init__(device=device, batch_size=batch_size, lr=lr, epochs=epochs,
                                 type_precision=type_precision)
                self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                                    dropout=dropout if layers > 1 else 0.0)
                self.fc = nn.Linear(hidden, 1)
                self.seq_length = seq_length
                self.input_size = input_size

            def forward(self, x):
                out, _ = self.lstm(x)
                yhat = self.fc(out[:, -1, :])
                return yhat.squeeze(-1)

            def prepare_data(self, window, start_abs_idx):
                L = self.seq_length
                n = len(window)
                if n <= L:
                    return None, None, None
                if window.ndim == 1:
                    window = window[:, None]
                X_list, y_list = [], []
                for i in range(L, n):
                    X_list.append(window[i-L:i])
                    y_list.append(window[i, 0])
                X = np.array(X_list, dtype=np.float32)
                y = np.array(y_list, dtype=np.float32)
                t_abs = np.arange(start_abs_idx + L, start_abs_idx + n, dtype=np.int64)
                return torch.from_numpy(X), torch.from_numpy(y), t_abs

        # Load a small dataset
        dataset_path = "examples/datasets/simulated/piecewise_ar3/data.csv"
        df = pd.read_csv(dataset_path)
        data = df["N"].to_numpy(dtype=np.float64)[:300]  # Use only first 300 points

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        model = AdaptiveLSTM(
            device=device,
            seq_length=3,
            input_size=1,
            hidden=8,  # Smaller for speed
            layers=1,
            dropout=0.0,
            batch_size=32,
            lr=1e-2,
            epochs=5,  # Fewer epochs for speed
            type_precision=np.float64
        )

        print(f"  Model initialized on {device}")

        # Initialize change detector
        cd = ChangeDetector(model, data, debug=False, force_cpu=True)
        print("  Change detector initialized")

        # Run minimal detection
        print("  Running detection (this may take 1-2 minutes)...")
        results = cd.detect(
            min_window=4,
            n_0=50,  # Small window
            jump=50,  # Large jump for speed
            search_step=10,
            alpha=0.95,
            num_bootstrap=5,  # Very few bootstraps
            t_workers=2,
            b_workers=2,
            one_b_threads=1,
            debug_anim=False,
            save_path=None
        )

        print(f"✓ Detection completed successfully")
        print(f"  Detected windows: {len(results['windows'])} time points")
        print(f"  Mean window size: {np.mean(results['windows']):.1f}")
        print(f"  Window size range: [{np.min(results['windows']):.0f}, {np.max(results['windows']):.0f}]")

        return True

    except Exception as e:
        print(f"✗ LPA detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_directories():
    """Test that output directories can be created."""
    print("\nTesting output directories...")

    test_dir = "examples/results/robustness/test"

    try:
        from pathlib import Path
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {test_dir}")

        # Test CSV writing
        test_csv = f"{test_dir}/test.csv"
        pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}).to_csv(test_csv, index=False)
        print(f"✓ Created test CSV: {test_csv}")

        return True
    except Exception as e:
        print(f"✗ Directory creation failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Robustness Analysis Setup Test")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Dataset Access", test_dataset_access),
        ("Output Directories", test_output_directories),
        ("LPA Detection", test_lpa_detection),  # This one takes time
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"{'='*60}")
        results[test_name] = test_func()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Ready to run robustness analysis.")
        print("\nNext steps:")
        print("1. Quick test (10-15 min):")
        print("   python examples/robustness/01_lpa_sensitivity.py --quick-test --n-runs 2")
        print("\n2. Full analysis (several hours):")
        print("   python examples/robustness/01_lpa_sensitivity.py --n-runs 3")
    else:
        print("\n✗ Some tests failed. Please fix issues before running analysis.")
        print("\nCommon fixes:")
        print("- Missing datasets: Run python examples/generate_simulated_datasets.py")
        print("- Import errors: Check that adaptivewinshap is installed")
        print("- CUDA errors: Set CUDA_VISIBLE_DEVICES=\"\" to force CPU")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test robustness analysis setup')
    parser.add_argument('--skip-lpa', action='store_true',
                       help='Skip LPA detection test (faster)')
    args = parser.parse_args()

    if args.skip_lpa:
        print("Skipping LPA detection test (use --skip-lpa=False to include)")
        # Remove LPA test
        test_imports()
        test_dataset_access()
        test_output_directories()
    else:
        run_all_tests()