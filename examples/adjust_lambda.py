"""
Adjust pre-computed critical values to a different Spokoiny penalty factor (lambda).

The Spokoiny adjustment is:

    CV_adjusted(k) = CV_raw(k) * [1 + lambda * sqrt(log(n_K_max / n_k))]

This script reverses the original adjustment, then re-applies it with a
new lambda -- no Monte Carlo recomputation needed.

Usage:
    # Adjust from lambda=0.1 to lambda=0.2
    python examples/adjust_lambda.py \
        --input examples/results/LSTM/entsoe_ro_price/Jump_1_N0_168_lambda_0.1/critical_values.csv \
        --new-lambda 0.2

    # Custom output path
    python examples/adjust_lambda.py \
        --input path/to/critical_values.csv \
        --new-lambda 0.05 \
        --output path/to/critical_values_lambda_0.05.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def adjust_critical_values(df: pd.DataFrame, new_lambda: float) -> pd.DataFrame:
    """
    Re-adjust critical values from one penalty factor to another.

    Parameters
    ----------
    df : pd.DataFrame
        Critical values table with columns:
        k, n_k, critical_value_95, critical_value_99, adjustment_factor, penalty_factor
    new_lambda : float
        New Spokoiny penalty factor.

    Returns
    -------
    pd.DataFrame
        Adjusted critical values table.
    """
    df = df.copy()

    # Recover raw (unadjusted) critical values
    old_adjustment = df['adjustment_factor'].values
    raw_cv_95 = df['critical_value_95'].values / old_adjustment
    raw_cv_99 = df['critical_value_99'].values / old_adjustment

    # Recompute adjustment with new lambda
    n_k = df['n_k'].values
    n_K_max = n_k.max()
    ratio = n_K_max / n_k
    new_adjustment = 1 + new_lambda * np.sqrt(np.log(ratio))

    # Apply
    df['critical_value_95'] = raw_cv_95 * new_adjustment
    df['critical_value_99'] = raw_cv_99 * new_adjustment
    df['adjustment_factor'] = new_adjustment
    df['penalty_factor'] = new_lambda

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Adjust critical values to a different Spokoiny penalty factor (lambda)'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to existing critical_values.csv')
    parser.add_argument('--new-lambda', type=float, required=True,
                        help='New penalty factor lambda')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: auto-generated from input path)')
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.input)
    old_lambda = df['penalty_factor'].iloc[0] if 'penalty_factor' in df.columns else '?'

    print("="*60)
    print("Adjusting Spokoiny penalty factor")
    print("="*60)
    print(f"Input:      {args.input}")
    print(f"Old lambda: {old_lambda}")
    print(f"New lambda: {args.new_lambda}")
    print(f"Scales:     {len(df)}")

    # Adjust
    df_new = adjust_critical_values(df, args.new_lambda)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        # Replace lambda in the directory name if present, otherwise append
        input_dir = os.path.dirname(args.input)
        if f"lambda_{old_lambda}" in input_dir:
            new_dir = input_dir.replace(f"lambda_{old_lambda}", f"lambda_{args.new_lambda}")
        else:
            new_dir = input_dir + f"_lambda_{args.new_lambda}"
        os.makedirs(new_dir, exist_ok=True)
        out_path = os.path.join(new_dir, "critical_values.csv")

    df_new.to_csv(out_path, index=False)
    print(f"Output:     {out_path}")

    # Summary table
    print()
    print("Critical Values Summary:")
    print("-"*60)
    print(f"{'k':>4} | {'n_k':>6} | {'CV95 (old)':>11} | {'CV95 (new)':>11} | {'adj':>6}")
    print("-"*60)

    df_old = pd.read_csv(args.input)
    for i, row in df_new.iterrows():
        old_cv = df_old.loc[i, 'critical_value_95']
        new_cv = row['critical_value_95']
        print(f"{int(row['k']):4d} | {int(row['n_k']):6d} | {old_cv:11.2f} | {new_cv:11.2f} | {row['adjustment_factor']:6.3f}")

    print("="*60)
    print("Done.")


if __name__ == "__main__":
    main()
