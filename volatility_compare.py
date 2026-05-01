import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def load_returns(csv_path: str):
    df = pd.read_csv(csv_path).dropna(subset=["close", "twap"]).copy()

    # benchmark = TWAP
    ret_bench = np.diff(np.log(df["twap"].to_numpy()))
    ret_bench = ret_bench[np.isfinite(ret_bench)]

    # oracle = close
    ret_oracle = np.diff(np.log(df["close"].to_numpy()))
    ret_oracle = ret_oracle[np.isfinite(ret_oracle)]

    # epsilon
    eps = ((df["close"] - df["twap"]) / df["twap"]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    return ret_bench, ret_oracle, eps


def stats_dict(x: np.ndarray, label: str, series_name: str):
    return {
        "window": label,
        "series": series_name,
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "skew": float(skew(x, bias=False)),
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "abs_q95": float(np.quantile(np.abs(x), 0.95)),
        "abs_q99": float(np.quantile(np.abs(x), 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def add_hist(ax, x, bins, title, xlabel, xlim=None, ylim=None, color=None):
    ax.hist(x, bins=bins, density=True, alpha=0.75, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm", required=True)
    ap.add_argument("--medium", required=True)
    ap.add_argument("--shock", required=True)
    ap.add_argument("--out_prefix", default="volatility_compare")
    args = ap.parse_args()

    windows = {
        "Calm": args.calm,
        "Medium": args.medium,
        "Shock": args.shock,
    }

    rows = []
    data = {}

    # -----------------------------
    # First pass: load everything
    # -----------------------------
    for label, path in windows.items():
        ret_bench, ret_oracle, eps = load_returns(path)
        data[label] = {
            "bench": ret_bench,
            "oracle": ret_oracle,
            "eps": eps,
        }

        rows.append(stats_dict(ret_bench, label, "benchmark_twap"))
        rows.append(stats_dict(ret_oracle, label, "oracle_close"))
        rows.append(stats_dict(eps, label, "epsilon"))

    # -----------------------------
    # Global ranges for consistent scaling
    # -----------------------------
    all_bench = np.concatenate([data[k]["bench"] for k in data])
    all_oracle = np.concatenate([data[k]["oracle"] for k in data])
    all_eps = np.concatenate([data[k]["eps"] for k in data])

    # Use symmetric x-limits around zero for easier comparison
    bench_lim = float(np.max(np.abs(all_bench)))
    oracle_lim = float(np.max(np.abs(all_oracle)))
    eps_lim = float(np.max(np.abs(all_eps)))

    bench_xlim = (-bench_lim, bench_lim)
    oracle_xlim = (-oracle_lim, oracle_lim)
    eps_xlim = (-eps_lim, eps_lim)

    # Same bin edges across regimes
    bench_bins = np.linspace(bench_xlim[0], bench_xlim[1], 61)
    oracle_bins = np.linspace(oracle_xlim[0], oracle_xlim[1], 61)
    eps_bins = np.linspace(eps_xlim[0], eps_xlim[1], 61)

    # -----------------------------
    # Compute common y-limits for each column
    # -----------------------------
    bench_ymax = 0
    oracle_ymax = 0
    eps_ymax = 0

    for label in data:
        bench_counts, _ = np.histogram(data[label]["bench"], bins=bench_bins, density=True)
        oracle_counts, _ = np.histogram(data[label]["oracle"], bins=oracle_bins, density=True)
        eps_counts, _ = np.histogram(data[label]["eps"], bins=eps_bins, density=True)

        bench_ymax = max(bench_ymax, bench_counts.max())
        oracle_ymax = max(oracle_ymax, oracle_counts.max())
        eps_ymax = max(eps_ymax, eps_counts.max())

    bench_ylim = (0, bench_ymax * 1.08)
    oracle_ylim = (0, oracle_ymax * 1.08)
    eps_ylim = (0, eps_ymax * 1.08)

    # -----------------------------
    # Plot volatility figure
    # -----------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex='col', sharey='col')

    ordered_labels = ["Calm", "Medium", "Shock"]

    for i, label in enumerate(ordered_labels):
        add_hist(
            axes[i, 0],
            data[label]["bench"],
            bins=bench_bins,
            title=f"{label} - Benchmark Volatility (TWAP)",
            xlabel="log return",
            xlim=bench_xlim,
            ylim=bench_ylim
        )

        add_hist(
            axes[i, 1],
            data[label]["oracle"],
            bins=oracle_bins,
            title=f"{label} - Oracle Volatility (Close)",
            xlabel="log return",
            xlim=oracle_xlim,
            ylim=oracle_ylim
        )

    fig.tight_layout()
    out_png = f"{args.out_prefix}_volatility_histograms.png"
    fig.savefig(out_png, dpi=180)

    # -----------------------------
    # Plot epsilon figure
    # -----------------------------
    fig_eps, axes_eps = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, label in enumerate(ordered_labels):
        add_hist(
            axes_eps[i],
            data[label]["eps"],
            bins=eps_bins,
            title=f"{label} - Epsilon Distribution",
            xlabel="epsilon",
            xlim=eps_xlim,
            ylim=eps_ylim
        )

    fig_eps.tight_layout()
    out_eps_png = f"{args.out_prefix}_epsilon_histograms.png"
    fig_eps.savefig(out_eps_png, dpi=180)

    # -----------------------------
    # Save stats
    # -----------------------------
    out_csv = f"{args.out_prefix}_stats.csv"
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_csv, index=False)

    summary_rows = []
    for label in ordered_labels:
        b = stats_df[(stats_df["window"] == label) & (stats_df["series"] == "benchmark_twap")].iloc[0]
        o = stats_df[(stats_df["window"] == label) & (stats_df["series"] == "oracle_close")].iloc[0]
        e = stats_df[(stats_df["window"] == label) & (stats_df["series"] == "epsilon")].iloc[0]
        summary_rows.append({
            "window": label,
            "benchmark_vol_std": b["std"],
            "oracle_vol_std": o["std"],
            "oracle_minus_benchmark_vol": o["std"] - b["std"],
            "epsilon_kurtosis": e["kurtosis"],
            "epsilon_abs_q99": e["abs_q99"],
        })

    out_summary = f"{args.out_prefix}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_eps_png}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_summary}")

    print("\nVolatility summary:")
    for row in summary_rows:
        print(
            f"{row['window']}: "
            f"benchmark std={row['benchmark_vol_std']:.4%}, "
            f"oracle std={row['oracle_vol_std']:.4%}, "
            f"diff={row['oracle_minus_benchmark_vol']:.4%}, "
            f"epsilon kurtosis={row['epsilon_kurtosis']:.2f}, "
            f"|epsilon| q99={row['epsilon_abs_q99']:.4%}"
        )


if __name__ == "__main__":
    main()