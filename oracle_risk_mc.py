import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_paths(ret: np.ndarray, H: int, N: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(ret), size=(N, H))
    return ret[idx]


def simulate_liq_probs(
    HF0: float,
    ret: np.ndarray,
    eps_signed: np.ndarray,
    H: int,
    N: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    R = bootstrap_paths(ret, H, N, rng)
    growth = np.exp(R)

    idx_e = rng.integers(0, len(eps_signed), size=(N, H))
    E = eps_signed[idx_e]

    HF_true = np.cumprod(growth, axis=1) * HF0
    HF_oracle = HF_true * (1.0 + E)

    liq_true = (HF_true.min(axis=1) < 1.0)
    liq_oracle = (HF_oracle.min(axis=1) < 1.0)
    hidden = liq_true & (~liq_oracle)

    X = (1.0 + E) * E * HF_true
    min_X = X.min(axis=1)

    adj = np.where(E < 0, (1.0 + E), 1.0)
    adj = np.where(adj <= 0, 1e-12, adj)

    alpha_t = np.maximum(0.0, (1.0 / (HF_true * adj)) - 1.0)
    alpha_path = alpha_t.max(axis=1)

    return {
        "p_liq_true": float(liq_true.mean()),
        "p_liq_oracle": float(liq_oracle.mean()),
        "p_hidden": float(hidden.mean()),
        "q95_min_X": float(np.quantile(min_X, 0.95)),
        "q99_min_X": float(np.quantile(min_X, 0.99)),
        "mean_min_X": float(np.mean(min_X)),
        "mean_alpha": float(np.mean(alpha_path)),
        "q95_alpha": float(np.quantile(alpha_path, 0.95)),
        "q99_alpha": float(np.quantile(alpha_path, 0.99)),
    }


def simulate_two_loans(
    HF0_1: float,
    HF0_2: float,
    C1: float,
    C2: float,
    ret: np.ndarray,
    eps_signed: np.ndarray,
    H: int,
    N: int,
    seed: int,
) -> dict:
    """
    Two-loan simulation with the SAME market path and SAME epsilon path.

    IMPORTANT:
    If HF0_1 <= HF0_2, then loan 1 should be the weaker/riskier loan.
    The code reports verification metrics for the ordered-nesting idea.
    """
    rng = np.random.default_rng(seed)

    # shared market path
    R = bootstrap_paths(ret, H, N, rng)
    growth = np.exp(R)

    # shared epsilon path
    idx_e = rng.integers(0, len(eps_signed), size=(N, H))
    E = eps_signed[idx_e]

    # loan-level benchmark HFs
    HF_true_1 = np.cumprod(growth, axis=1) * HF0_1
    HF_true_2 = np.cumprod(growth, axis=1) * HF0_2

    # loan-level oracle HFs
    HF_oracle_1 = HF_true_1 * (1.0 + E)
    HF_oracle_2 = HF_true_2 * (1.0 + E)

    # loan-level liquidations
    liq_true_1 = (HF_true_1.min(axis=1) < 1.0)
    liq_true_2 = (HF_true_2.min(axis=1) < 1.0)

    liq_oracle_1 = (HF_oracle_1.min(axis=1) < 1.0)
    liq_oracle_2 = (HF_oracle_2.min(axis=1) < 1.0)

    hidden_1 = liq_true_1 & (~liq_oracle_1)
    hidden_2 = liq_true_2 & (~liq_oracle_2)

    # three buckets under TRUE HF
    both_healthy_true = (~liq_true_1) & (~liq_true_2)
    only_1_true = liq_true_1 & (~liq_true_2)
    only_2_true = (~liq_true_1) & liq_true_2
    both_liq_true = liq_true_1 & liq_true_2

    # three buckets under ORACLE HF
    both_healthy_oracle = (~liq_oracle_1) & (~liq_oracle_2)
    only_1_oracle = liq_oracle_1 & (~liq_oracle_2)
    only_2_oracle = (~liq_oracle_1) & liq_oracle_2
    both_liq_oracle = liq_oracle_1 & liq_oracle_2

    # unions
    any_liq_true = liq_true_1 | liq_true_2
    any_liq_oracle = liq_oracle_1 | liq_oracle_2
    any_hidden = hidden_1 | hidden_2
    both_hidden = hidden_1 & hidden_2

    # advisor sign rule for alpha
    adj = np.where(E < 0, (1.0 + E), 1.0)
    adj = np.where(adj <= 0, 1e-12, adj)

    alpha_t_1 = np.maximum(0.0, (1.0 / (HF_true_1 * adj)) - 1.0)
    alpha_t_2 = np.maximum(0.0, (1.0 / (HF_true_2 * adj)) - 1.0)

    alpha_path_1 = alpha_t_1.max(axis=1)
    alpha_path_2 = alpha_t_2.max(axis=1)

    # aggregate alpha using advisor correction
    extra_total = alpha_path_1 * C1 + alpha_path_2 * C2
    alpha_path_agg = extra_total / (C1 + C2)

    # verification metrics for ordering logic
    # if loan 1 is weaker, then:
    # L2 subset of L1
    share_paths_liq2_implies_liq1_true = float((~liq_true_2 | liq_true_1).mean())
    share_paths_liq2_implies_liq1_oracle = float((~liq_oracle_2 | liq_oracle_1).mean())
    share_paths_alpha1_ge_alpha2 = float((alpha_path_1 >= alpha_path_2).mean())

    return {
        # loan-level probabilities
        "p_liq_true_1": float(liq_true_1.mean()),
        "p_liq_true_2": float(liq_true_2.mean()),
        "p_liq_oracle_1": float(liq_oracle_1.mean()),
        "p_liq_oracle_2": float(liq_oracle_2.mean()),
        "p_hidden_1": float(hidden_1.mean()),
        "p_hidden_2": float(hidden_2.mean()),

        # unions
        "p_any_liq_true": float(any_liq_true.mean()),
        "p_any_liq_oracle": float(any_liq_oracle.mean()),
        "p_any_hidden": float(any_hidden.mean()),
        "p_both_hidden": float(both_hidden.mean()),

        # three-bucket decomposition (true)
        "p_both_healthy_true": float(both_healthy_true.mean()),
        "p_only_1_true": float(only_1_true.mean()),
        "p_only_2_true": float(only_2_true.mean()),
        "p_both_liq_true": float(both_liq_true.mean()),

        # three-bucket decomposition (oracle)
        "p_both_healthy_oracle": float(both_healthy_oracle.mean()),
        "p_only_1_oracle": float(only_1_oracle.mean()),
        "p_only_2_oracle": float(only_2_oracle.mean()),
        "p_both_liq_oracle": float(both_liq_oracle.mean()),

        # alphas
        "mean_alpha_1": float(np.mean(alpha_path_1)),
        "q95_alpha_1": float(np.quantile(alpha_path_1, 0.95)),
        "q99_alpha_1": float(np.quantile(alpha_path_1, 0.99)),

        "mean_alpha_2": float(np.mean(alpha_path_2)),
        "q95_alpha_2": float(np.quantile(alpha_path_2, 0.95)),
        "q99_alpha_2": float(np.quantile(alpha_path_2, 0.99)),

        "mean_alpha_agg": float(np.mean(alpha_path_agg)),
        "q95_alpha_agg": float(np.quantile(alpha_path_agg, 0.95)),
        "q99_alpha_agg": float(np.quantile(alpha_path_agg, 0.99)),

        # verification metrics
        "share_paths_liq2_implies_liq1_true": share_paths_liq2_implies_liq1_true,
        "share_paths_liq2_implies_liq1_oracle": share_paths_liq2_implies_liq1_oracle,
        "share_paths_alpha1_ge_alpha2": share_paths_alpha1_ge_alpha2,

        # expected equalities to verify numerically
        "p_any_minus_p1_true": float(any_liq_true.mean() - liq_true_1.mean()),
        "p_both_minus_p2_true": float(both_liq_true.mean() - liq_true_2.mean()),
        "p_any_minus_p1_oracle": float(any_liq_oracle.mean() - liq_oracle_1.mean()),
        "p_both_minus_p2_oracle": float(both_liq_oracle.mean() - liq_oracle_2.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="market_hourly_*.csv (needs close, twap)")
    ap.add_argument("--L", type=float, default=0.80, help="protocol liquidation threshold L (reported only)")
    ap.add_argument("--HF0", type=float, default=1.28, help="starting Health Factor for loan 1")
    ap.add_argument("--HF0_2", type=float, default=None, help="starting Health Factor for loan 2 (optional)")
    ap.add_argument("--C1", type=float, default=None, help="collateral amount for loan 1 (required for two-loan mode)")
    ap.add_argument("--C2", type=float, default=None, help="collateral amount for loan 2 (required for two-loan mode)")
    ap.add_argument("--H", type=int, default=24, help="horizon in hours")
    ap.add_argument("--N", type=int, default=10000, help="number of bootstrap paths")
    ap.add_argument("--tail_pcts", type=float, nargs="+", default=[0.90, 0.95, 0.99],
                    help="quantiles to report for tails (eps+ and eps- magnitude)")
    ap.add_argument("--out_prefix", default="oracle_risk", help="output prefix")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.HF0_2 is not None:
        if args.C1 is None or args.C2 is None:
            raise ValueError("For two-loan mode, you must provide --C1 and --C2")
        if args.C1 <= 0 or args.C2 <= 0:
            raise ValueError("C1 and C2 must be positive")
        if args.HF0 > args.HF0_2:
            print("\nWARNING: HF0 > HF0_2. For the ordering verification, loan 1 should be the weaker loan.")
            print("         Recommended: set HF0 <= HF0_2.\n")

    df = pd.read_csv(args.csv)
    if not {"close", "twap"}.issubset(df.columns):
        raise ValueError("CSV must have columns: close, twap")

    df = df.dropna(subset=["close", "twap"]).copy()

    df["eps"] = (df["close"] - df["twap"]) / df["twap"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["eps"]).copy()

    if len(df) < 50:
        raise ValueError("Not enough eps samples after filtering. Use a longer window or smaller TWAP window.")

    df["eps_pos"] = np.where(df["eps"] > 0, df["eps"], 0.0)
    df["eps_neg"] = np.where(df["eps"] < 0, -df["eps"], 0.0)

    eps = df["eps"].to_numpy()
    mean_abs = float(np.mean(np.abs(eps)))

    worst_pos = float(df["eps_pos"].max()) if (df["eps_pos"] > 0).any() else 0.0
    worst_neg = float(df["eps_neg"].max()) if (df["eps_neg"] > 0).any() else 0.0
    L_safe = float(args.L / (1.0 + worst_pos)) if worst_pos > 0 else float(args.L)

    pos = df.loc[df["eps_pos"] > 0, "eps_pos"].to_numpy()
    neg = df.loc[df["eps_neg"] > 0, "eps_neg"].to_numpy()

    tails = []
    for p in args.tail_pcts:
        qp = float(np.quantile(pos, p)) if len(pos) else 0.0
        qn = float(np.quantile(neg, p)) if len(neg) else 0.0
        tails.append((p, qp, qn))

    print(f"\nWorst oracle underpricing (ε>0) = {worst_pos:.3%} -> L_safe = {L_safe:.3f} (from L={args.L})")

    close = df["close"].to_numpy()
    ret = np.diff(np.log(close))
    ret = ret[np.isfinite(ret)]

    if len(ret) < 50:
        raise ValueError("Not enough return samples for bootstrap. Use a longer window.")

    eps_signed = df["eps"].to_numpy()

    if args.HF0_2 is None:
        probs = simulate_liq_probs(
            HF0=args.HF0,
            ret=ret,
            eps_signed=eps_signed,
            H=args.H,
            N=args.N,
            seed=args.seed,
        )
    else:
        probs = simulate_two_loans(
            HF0_1=args.HF0,
            HF0_2=args.HF0_2,
            C1=args.C1,
            C2=args.C2,
            ret=ret,
            eps_signed=eps_signed,
            H=args.H,
            N=args.N,
            seed=args.seed,
        )

    rows = []
    rows.append(["L", float(args.L)])
    rows.append(["HF0", float(args.HF0)])
    if args.HF0_2 is not None:
        rows.append(["HF0_2", float(args.HF0_2)])
        rows.append(["C1", float(args.C1)])
        rows.append(["C2", float(args.C2)])
    rows.append(["H_hours", float(args.H)])
    rows.append(["N_paths", float(args.N)])
    rows.append(["mean_abs_eps", mean_abs])
    rows.append(["worst_eps_pos", worst_pos])
    rows.append(["worst_eps_neg", worst_neg])
    rows.append(["L_safe_from_worst_pos", L_safe])

    for p, qp, qn in tails:
        rows.append([f"q{int(p*100)}_eps_pos", qp])
        rows.append([f"q{int(p*100)}_eps_neg", qn])

    for k, v in probs.items():
        rows.append([k, float(v)])

    out_csv = f"{args.out_prefix}_summary.csv"
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(eps, bins=60, density=True, alpha=0.7)
    ax.set_title("Oracle signed error ε = (close - TWAP)/TWAP")
    ax.set_xlabel("ε")
    ax.set_ylabel("density")

    if worst_pos > 0:
        ax.axvline(worst_pos, color="red", linewidth=2, label=f"worst ε+ = {worst_pos:.2%}")
    if worst_neg > 0:
        ax.axvline(-worst_neg, color="blue", linewidth=2, label=f"worst ε- = {-worst_neg:.2%}")

    if (worst_pos > 0) or (worst_neg > 0):
        ax.legend()

    out_png = f"{args.out_prefix}_tails.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")

    print("\nTail quantiles (ε⁺ / ε⁻ magnitude):")
    for p, qp, qn in tails:
        print(f"  q{int(p*100)}:  ε⁺={qp:.3%}  ε⁻={qn:.3%}")

    print(f"\nWorst overvaluation ε* = {worst_pos:.3%} -> L_safe = {L_safe:.3f} (from L={args.L})")
    print(f"\nMC Results (H={args.H}h, N={args.N}):")
    for k, v in probs.items():
        print(f"  {k}: {v:.6f}" if abs(v) < 0.01 else f"  {k}: {v:.2%}")


if __name__ == "__main__":
    main()