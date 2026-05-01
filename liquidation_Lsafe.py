#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="market_hourly_YYYY-..csv")
    ap.add_argument("--L", type=float, default=0.80, help="protocol liquidation threshold")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["time_utc"])
    df = df.sort_values("time").dropna(subset=["twap"]).copy()

    df["abs_dev"] = (df["close"] - df["twap"]).abs() / df["twap"]

    worst = float(df["abs_dev"].max())
    avg   = float(df["abs_dev"].mean())
    best  = float(df["abs_dev"].quantile(0.05))  # 5th percentile calmness

    L_safe = args.L / (1.0 + worst) if worst > 0 else args.L

    print(f"File: {args.csv}")
    print(f"Best case (5th pct): {100*best:.2f}%")
    print(f"Average case (mean): {100*avg:.2f}%")
    print(f"Worst case (max):    {100*worst:.2f}%")
    print(f"Suggested L_safe = L/(1+worst) with L={args.L} -> {L_safe:.3f}")

if __name__ == "__main__":
    main()
