import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def bootstrap_paths(ret: np.ndarray, H: int, N: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(ret), size=(N, H))
    return ret[idx]


def alpha_from_hf_paths(hf_paths: np.ndarray, alpha_stat: str) -> float:
    """
    Compute pathwise alpha = max_t max(0, 1/HF_t - 1),
    then summarize across paths by mean / q95 / q99.
    """
    hf_safe = np.where(hf_paths <= 0, 1e-12, hf_paths)
    alpha_t = np.maximum(0.0, 1.0 / hf_safe - 1.0)
    alpha_path = alpha_t.max(axis=1)

    if alpha_stat == "mean":
        return float(alpha_path.mean())
    elif alpha_stat == "q95":
        return float(np.quantile(alpha_path, 0.95))
    elif alpha_stat == "q99":
        return float(np.quantile(alpha_path, 0.99))
    else:
        raise ValueError(f"Unknown alpha_stat: {alpha_stat}")


def simulate_benchmark_only(HF0: float, ret_bench: np.ndarray, H: int, N: int, seed: int, alpha_stat: str) -> float:
    rng = np.random.default_rng(seed)
    R = bootstrap_paths(ret_bench, H, N, rng)
    growth = np.exp(R)
    hf_bench = np.cumprod(growth, axis=1) * HF0
    return alpha_from_hf_paths(hf_bench, alpha_stat)


def simulate_oracle_direct(HF0: float, ret_oracle: np.ndarray, H: int, N: int, seed: int, alpha_stat: str) -> float:
    rng = np.random.default_rng(seed)
    R = bootstrap_paths(ret_oracle, H, N, rng)
    growth = np.exp(R)
    hf_oracle = np.cumprod(growth, axis=1) * HF0
    return alpha_from_hf_paths(hf_oracle, alpha_stat)


def simulate_benchmark_times_eps(HF0: float, ret_bench: np.ndarray, eps_signed: np.ndarray,
                                 H: int, N: int, seed: int, alpha_stat: str) -> float:
    rng = np.random.default_rng(seed)
    R = bootstrap_paths(ret_bench, H, N, rng)
    growth = np.exp(R)

    idx_e = rng.integers(0, len(eps_signed), size=(N, H))
    E = eps_signed[idx_e]

    hf_bench = np.cumprod(growth, axis=1) * HF0
    hf_oracle_eps = hf_bench * (1.0 + E)

    # avoid zero/negative from pathological draws
    hf_oracle_eps = np.where(hf_oracle_eps <= 0, 1e-12, hf_oracle_eps)

    return alpha_from_hf_paths(hf_oracle_eps, alpha_stat)


def epsilon_stats(df: pd.DataFrame, label: str) -> dict:
    eps = ((df["close"] - df["twap"]) / df["twap"]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    pos = eps[eps > 0]
    neg = -eps[eps < 0]

    return {
        "window": label,
        "n_eps": int(len(eps)),
        "mean_eps": float(np.mean(eps)),
        "std_eps": float(np.std(eps, ddof=1)),
        "skew_eps": float(skew(eps, bias=False)),
        "kurtosis_eps": float(kurtosis(eps, fisher=True, bias=False)),
        "q95_eps": float(np.quantile(eps, 0.95)),
        "q99_eps": float(np.quantile(eps, 0.99)),
        "q95_abs_eps": float(np.quantile(np.abs(eps), 0.95)),
        "q99_abs_eps": float(np.quantile(np.abs(eps), 0.99)),
        "max_eps_pos": float(pos.max()) if len(pos) else 0.0,
        "max_eps_neg_mag": float(neg.max()) if len(neg) else 0.0,
    }


def load_window(csv_path: str):
    df = pd.read_csv(csv_path).dropna(subset=["close", "twap"]).copy()

    # benchmark = TWAP
    twap = df["twap"].to_numpy()
    ret_bench = np.diff(np.log(twap))
    ret_bench = ret_bench[np.isfinite(ret_bench)]

    # oracle direct = close
    close = df["close"].to_numpy()
    ret_oracle = np.diff(np.log(close))
    ret_oracle = ret_oracle[np.isfinite(ret_oracle)]

    # epsilon = (oracle - benchmark)/benchmark
    eps_signed = ((df["close"] - df["twap"]) / df["twap"]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    if len(ret_bench) < 50 or len(ret_oracle) < 50 or len(eps_signed) < 50:
        raise ValueError(f"{csv_path}: not enough samples after filtering")

    return df, ret_bench, ret_oracle, eps_signed


def compute_panel(label: str, csv_path: str, hf_grid, seeds, H, N, alpha_stat):
    df, ret_bench, ret_oracle, eps_signed = load_window(csv_path)

    rows = []
    for hf0 in hf_grid:
        vals_bench = []
        vals_oracle = []
        vals_eps = []

        for seed in seeds:
            vals_bench.append(simulate_benchmark_only(hf0, ret_bench, H, N, seed, alpha_stat))
            vals_oracle.append(simulate_oracle_direct(hf0, ret_oracle, H, N, seed + 1000, alpha_stat))
            vals_eps.append(simulate_benchmark_times_eps(hf0, ret_bench, eps_signed, H, N, seed + 2000, alpha_stat))

        rows.append({
            "window": label,
            "HF0": hf0,

            "bench_mean_across_seeds": float(np.mean(vals_bench)),
            "bench_std_across_seeds": float(np.std(vals_bench, ddof=1)) if len(vals_bench) > 1 else 0.0,

            "oracle_direct_mean_across_seeds": float(np.mean(vals_oracle)),
            "oracle_direct_std_across_seeds": float(np.std(vals_oracle, ddof=1)) if len(vals_oracle) > 1 else 0.0,

            "bench_times_eps_mean_across_seeds": float(np.mean(vals_eps)),
            "bench_times_eps_std_across_seeds": float(np.std(vals_eps, ddof=1)) if len(vals_eps) > 1 else 0.0,
        })

    return pd.DataFrame(rows), epsilon_stats(df, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm", required=True)
    ap.add_argument("--medium", required=True)
    ap.add_argument("--shock", required=True)
    ap.add_argument("--H", type=int, default=72)
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--hf_grid", type=float, nargs="+", default=[1.00, 1.02, 1.04, 1.06, 1.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42, 99, 123])
    ap.add_argument("--alpha_stat", choices=["mean", "q95", "q99"], default="mean")
    ap.add_argument("--out_prefix", default="alpha_vs_hf")
    args = ap.parse_args()

    calm_df, calm_eps = compute_panel("Calm", args.calm, args.hf_grid, args.seeds, args.H, args.N, args.alpha_stat)
    med_df, med_eps = compute_panel("Medium", args.medium, args.hf_grid, args.seeds, args.H, args.N, args.alpha_stat)
    shock_df, shock_eps = compute_panel("Shock", args.shock, args.hf_grid, args.seeds, args.H, args.N, args.alpha_stat)

    all_df = pd.concat([calm_df, med_df, shock_df], ignore_index=True)
    all_df.to_csv(f"{args.out_prefix}_alpha_grid.csv", index=False)

    eps_df = pd.DataFrame([calm_eps, med_eps, shock_eps])
    eps_df.to_csv(f"{args.out_prefix}_epsilon_stats.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    panel_order = [("Calm", calm_df), ("Medium", med_df), ("Shock", shock_df)]
    for ax, (label, d) in zip(axes, panel_order):
        x = d["HF0"].to_numpy()

        y1 = d["bench_mean_across_seeds"].to_numpy() * 100
        e1 = d["bench_std_across_seeds"].to_numpy() * 100

        y2 = d["oracle_direct_mean_across_seeds"].to_numpy() * 100
        e2 = d["oracle_direct_std_across_seeds"].to_numpy() * 100

        y3 = d["bench_times_eps_mean_across_seeds"].to_numpy() * 100
        e3 = d["bench_times_eps_std_across_seeds"].to_numpy() * 100

        ax.errorbar(x, y1, yerr=e1, marker="o", capsize=3, label="Benchmark only")
        ax.errorbar(x, y2, yerr=e2, marker="o", capsize=3, label="Oracle direct")
        ax.errorbar(x, y3, yerr=e3, marker="o", capsize=3, label="Benchmark × (1+ε)")

        ax.set_title(label)
        ax.set_xlabel("Starting HF")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(f"{args.alpha_stat} alpha across paths (%)")
    axes[0].legend()

    fig.suptitle(f"Alpha vs HF | H={args.H}, N={args.N}, seeds={args.seeds}")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_plot.png", dpi=180)
    print(f"Saved: {args.out_prefix}_alpha_grid.csv")
    print(f"Saved: {args.out_prefix}_epsilon_stats.csv")
    print(f"Saved: {args.out_prefix}_plot.png")


if __name__ == "__main__":
    main()