
"""
Input:
  - One or more market_hourly_*.csv files (each has close, twap)
  - Fixed wallet tuple (C, D, L)

some notes
- Benchmark price = TWAP
- Oracle price = close
- HF_true = (C * twap * L) / D
- HF_oracle = (C * close * L) / D
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# optional fitter
try:
    from fitter import Fitter
    HAS_FITTER = True
except Exception:
    HAS_FITTER = False


def extract_window_name(path: str) -> str:
    """Try to infer window name from filename market_hourly_START_END.csv."""
    base = os.path.basename(path)
    m = re.search(r"market_hourly_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv", base)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return os.path.splitext(base)[0]


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x)
    out = {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "skew": float(stats.skew(x)) if len(x) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(x)) if len(x) > 3 else 0.0,
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    return out


def collateral_buffer_alpha(HF_true: np.ndarray) -> float:
    """ this is the minimal alpha so that (1+alpha)*HF_true >= 1 for all HF_true < 1. """
    bad = HF_true[HF_true < 1.0]
    if len(bad) == 0:
        return 0.0
    return float(np.max((1.0 / bad) - 1.0))


def load_hf(csv_path: str, C: float, D: float, L: float):
    df = pd.read_csv(csv_path)
    if not {"close", "twap"}.issubset(df.columns):
        raise ValueError(f"{csv_path} missing close/twap columns")

    df = df.dropna(subset=["close", "twap"]).copy()
    # HF_true = (C * df["twap"].to_numpy() * L) / D
    # HF_orac = (C * df["close"].to_numpy() * L) / D
    # return HF_true, HF_orac
    HF_true = (C * df["close"].to_numpy() * L) / D   # true/market
    HF_orac = (C * df["twap"].to_numpy() * L) / D    # oracle
    return HF_true, HF_orac


def best_fit_name(data: np.ndarray):
    """Return best-fit distribution name (optional)."""
    if not HAS_FITTER:
        return ""
    # keep it small and robust
    dists = ["norm", "lognorm", "gamma", "t", "cauchy"]
    f = Fitter(data, distributions=dists)
    f.fit()
    best = f.get_best(method="sumsquare_error")
    # best is dict like {"lognorm": {...params...}}
    return list(best.keys())[0] if best else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="market_hourly_*.csv",
                    help="glob pattern for CSVs (default: market_hourly_*.csv)")
    ap.add_argument("--C", type=float, required=True)
    ap.add_argument("--D", type=float, required=True)
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--out_prefix", default="batch")
    ap.add_argument("--plot_sample", type=int, default=4000,
                    help="max points sampled per window for overlay plots")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {args.pattern}")

    rows = []
    fits_rows = []

    # For plots
    overlay_true = []
    overlay_orac = []
    labels = []

    for fp in files:
        name = extract_window_name(fp)
        HF_true, HF_orac = load_hf(fp, args.C, args.D, args.L)

        s_true = describe(HF_true)
        s_orac = describe(HF_orac)

        p_liq_true = float(np.mean(HF_true < 1.0))
        p_liq_orac = float(np.mean(HF_orac < 1.0))
        p_hidden = float(np.mean((HF_true < 1.0) & (HF_orac >= 1.0)))
        alpha = collateral_buffer_alpha(HF_true)

        row = {
            "window": name,
            "file": os.path.basename(fp),

            **{f"true_{k}": v for k, v in s_true.items()},
            **{f"oracle_{k}": v for k, v in s_orac.items()},

            "p_liq_true": p_liq_true,
            "p_liq_oracle": p_liq_orac,
            "p_hidden": p_hidden,
            "alpha_required": alpha,
        }
        rows.append(row)

        if HAS_FITTER:
            ft = best_fit_name(HF_true)
            fo = best_fit_name(HF_orac)
            fits_rows.append({"window": name, "fit_true": ft, "fit_oracle": fo})

        n = min(len(HF_true), args.plot_sample)
        idx = np.random.default_rng(0).choice(len(HF_true), size=n, replace=False) if len(HF_true) > n else np.arange(len(HF_true))
        overlay_true.append(HF_true[idx])
        overlay_orac.append(HF_orac[idx])
        labels.append(name)

    out_stats = f"{args.out_prefix}_stats.csv"
    pd.DataFrame(rows).sort_values("window").to_csv(out_stats, index=False)
    print(f"Saved: {out_stats}")

    if HAS_FITTER and fits_rows:
        out_fits = f"{args.out_prefix}_fits.csv"
        pd.DataFrame(fits_rows).to_csv(out_fits, index=False)
        print(f"Saved: {out_fits}")
    else:
        print("Note: fitter not installed -> skipping distribution fit labels (install with: pip install fitter)")


    dfm = pd.DataFrame(rows).sort_values("window")
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(dfm["window"], dfm["p_liq_true"], marker="o", label="P(liq true)")
    ax1.plot(dfm["window"], dfm["p_liq_oracle"], marker="o", label="P(liq oracle)")
    ax1.plot(dfm["window"], dfm["p_hidden"], marker="o", label="P(hidden)")
    ax1.set_title("Liquidation risk across time windows")
    ax1.set_ylabel("probability")
    ax1.tick_params(axis='x', rotation=25)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(dfm["window"], dfm["alpha_required"], marker="o")
    ax2.set_title("Required collateral buffer alpha across time windows")
    ax2.set_ylabel("alpha (increase C by 1+alpha)")
    ax2.tick_params(axis='x', rotation=25)

    out_png = f"{args.out_prefix}_plot.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
