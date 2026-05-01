#!/usr/bin/env python3
"""
Statistical analysis of Health Factor distributions
for a fixed wallet tuple (C, D, L) under different price processes.

Uses:
- scipy.stats for descriptive statistics
- fitter for distribution fitting

Outputs:
- stats_summary.csv
- printed comparison table
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from fitter import Fitter



def describe_distribution(x):
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "std": np.std(x, ddof=1),
        "variance": np.var(x, ddof=1),
        "skew": stats.skew(x),
        "kurtosis": stats.kurtosis(x),
        "q01": np.quantile(x, 0.01),
        "q05": np.quantile(x, 0.05),
        "q10": np.quantile(x, 0.10),
        "min": np.min(x),
        "max": np.max(x),
    }


# -----------------------------
# Compute HF distributions
# -----------------------------

def compute_hf_distributions(csv_file, C, D, L):
    df = pd.read_csv(csv_file)

    if not {"close","twap"}.issubset(df.columns):
        raise ValueError("CSV must contain close and twap columns.")

    # # Benchmark price = TWAP
    # P_bench = df["twap"].dropna().to_numpy()

    # # Oracle price = close (you can modify if needed)
    # P_oracle = df["close"].loc[df["twap"].notna()].to_numpy()

    # HF_true = (C * P_bench * L) / D
    # HF_oracle = (C * P_oracle * L) / D

    # Oracle price = TWAP (oracle data)
    P_oracle = df["twap"].dropna().to_numpy()

    # True/market price = close (aligned to TWAP non-NaN)
    P_true = df["close"].loc[df["twap"].notna()].to_numpy()

    HF_oracle = (C * P_oracle * L) / D
    HF_true   = (C * P_true   * L) / D


    return HF_true, HF_oracle


def required_collateral_buffer(HF_true, HF_oracle):
    hidden = (HF_true < 1.0) & (HF_oracle >= 1.0)

    if not hidden.any():
        return 0.0

    worst_gap = np.max(1.0 - HF_true[hidden])
    alpha = worst_gap
    return alpha



def fit_distribution(data, label):
    f = Fitter(data, distributions=["norm","lognorm","gamma","t","cauchy"])
    f.fit()
    best = f.get_best(method="sumsquare_error")
    print(f"\nBest fit for {label}:")
    print(best)
    return best




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--C", type=float, required=True)
    ap.add_argument("--D", type=float, required=True)
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--out_prefix", default="hf_stats")
    args = ap.parse_args()

    HF_true, HF_oracle = compute_hf_distributions(
        args.csv, args.C, args.D, args.L
    )

    stats_true = describe_distribution(HF_true)
    stats_oracle = describe_distribution(HF_oracle)

    p_liq_true = np.mean(HF_true < 1.0)
    p_liq_oracle = np.mean(HF_oracle < 1.0)
    p_hidden = np.mean((HF_true < 1.0) & (HF_oracle >= 1.0))

    alpha_required = required_collateral_buffer(HF_true, HF_oracle)

    
    print("\n--- Health Factor Statistics ---")
    
    #print("\nBenchmark (True Price):")
    
    print("\nTrue / Market Price (close):")

    for k,v in stats_true.items():
        print(f"{k}: {v:.6f}")

    #print("\nOracle Price:")

    print("\nOracle Price (TWAP):")

    for k,v in stats_oracle.items():
        print(f"{k}: {v:.6f}")

    print("\n--- Liquidation Risk ---")
    print(f"P(liq true):   {p_liq_true:.4%}")
    print(f"P(liq oracle): {p_liq_oracle:.4%}")
    print(f"P(hidden):     {p_hidden:.4%}")

    print("\n--- Collateral Buffer ---")
    print(f"Minimum collateral increase needed (alpha): {alpha_required:.4%}")

    # Fit distributions
    fit_distribution(HF_true, "HF_true")
    fit_distribution(HF_oracle, "HF_oracle")

    # Save summary
    rows = []
    for k,v in stats_true.items():
        rows.append(["true_"+k, v])
    for k,v in stats_oracle.items():
        rows.append(["oracle_"+k, v])

    rows.append(["p_liq_true", p_liq_true])
    rows.append(["p_liq_oracle", p_liq_oracle])
    rows.append(["p_hidden", p_hidden])
    rows.append(["alpha_required", alpha_required])

    out_csv = f"{args.out_prefix}_summary.csv"
    pd.DataFrame(rows, columns=["metric","value"]).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
