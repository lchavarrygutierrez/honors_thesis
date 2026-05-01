#!/usr/bin/env python3
"""
Reads market_hourly_*.csv (must contain close + twap).
Outputs:
  1) histogram CSV (bins + % time)
  2) tails summary CSV (quantiles + tail probabilities)
  3) histogram PNG
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def histogram_with_edges(eps, edges):
    """
    edges: sorted list like [-0.025,-0.01,-0.005,0,0.005,0.01,0.025]
    Adds -inf and +inf automatically.
    Returns pct_time table.
    """
    eps = np.asarray(eps)
    eps = eps[np.isfinite(eps)]
    if eps.size == 0:
        return pd.DataFrame(columns=["bin_left","bin_right","count","pct_time"])

    edges = np.array(sorted(edges), dtype=float)
    full_edges = np.concatenate(([-np.inf], edges, [np.inf]))

    counts, _ = np.histogram(eps, bins=full_edges)
    pct = counts / counts.sum() * 100.0

    rows = []
    for i in range(len(full_edges)-1):
        rows.append({
            "bin_left": full_edges[i],
            "bin_right": full_edges[i+1],
            "count": int(counts[i]),
            "pct_time": float(pct[i]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="market_hourly_*.csv")
    ap.add_argument("--bin", type=float, default=0.0025, help="hist bin width (e.g. 0.0025 = 0.25%)")
    ap.add_argument("--tail_levels", type=float, nargs="+", default=[0.005, 0.01, 0.02],
                    help="tail thresholds to report (fractions)")
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.99],
                    help="tail quantiles to report")
    ap.add_argument("--out_prefix", default="hist")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["close","twap"]).copy()

    # signed error
    df["eps"] = (df["close"] - df["twap"]) / df["twap"]

    eps = df["eps"].to_numpy()
    eps = eps[np.isfinite(eps)]

    if eps.size == 0:
        raise SystemExit("No valid eps values. Check that twap exists and is not NaN.")

    # ----- Histogram table -----
    bw = float(args.bin)
    max_abs = float(np.max(np.abs(eps)))
    # symmetric range around 0
    lo = -np.ceil(max_abs / bw) * bw
    hi =  np.ceil(max_abs / bw) * bw

    edges = np.arange(lo, hi + bw, bw)
    counts, _ = np.histogram(eps, bins=edges)
    pct = counts / counts.sum() * 100.0

    hist = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
        "pct_time": pct
    })

    out_hist = f"{args.out_prefix}_bins.csv"
    hist.to_csv(out_hist, index=False)

    # ----- Tail stats -----
    eps_pos = eps[eps > 0]
    eps_neg = -eps[eps < 0]   # magnitude of negative tail

    rows = []

    # mean absolute error
    rows.append(["mean_abs_eps", float(np.mean(np.abs(eps)))])

    # worst tails
    rows.append(["worst_eps_pos", float(eps_pos.max()) if eps_pos.size else 0.0])
    rows.append(["worst_eps_neg", float(eps_neg.max()) if eps_neg.size else 0.0])

    # probabilities at tail thresholds
    for t in args.tail_levels:
        t = float(t)
        rows.append([f"P(eps>{t})", float(np.mean(eps > t))])
        rows.append([f"P(eps<-{t})", float(np.mean(eps < -t))])
        rows.append([f"P(|eps|>{t})", float(np.mean(np.abs(eps) > t))])

    # tail quantiles
    for q in args.quantiles:
        q = float(q)
        qp = float(np.quantile(eps_pos, q)) if eps_pos.size else 0.0
        qn = float(np.quantile(eps_neg, q)) if eps_neg.size else 0.0
        rows.append([f"q{int(q*100)}_eps_pos", qp])
        rows.append([f"q{int(q*100)}_eps_neg", qn])

    out_tails = f"{args.out_prefix}_tails.csv"
    pd.DataFrame(rows, columns=["metric","value"]).to_csv(out_tails, index=False)

    # ----- Plot -----
    plt.figure(figsize=(9,4))
    plt.hist(eps, bins=80, density=True, alpha=0.8)
    plt.title("Signed error ε = (close - TWAP)/TWAP")
    plt.xlabel("ε")
    plt.ylabel("density")
    out_png = f"{args.out_prefix}_hist.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)

    print("Saved:", out_hist)
    print("Saved:", out_tails)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()
