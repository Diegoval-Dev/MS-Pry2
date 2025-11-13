"""Batch comparison between M/M/1 and M/M/g by sweeping g."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange

try:
    from sim import MM1Params, get_params, mmg_theory, run_mmg
except ModuleNotFoundError:  # pragma: no cover
    from .sim import MM1Params, get_params, mmg_theory, run_mmg


def parse_g_list(spec: str) -> List[int]:
    values = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            g = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid g value '{chunk}'.") from exc
        if g < 1:
            raise argparse.ArgumentTypeError("Every g must be >= 1.")
        values.append(g)
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one server count via --g-list.")
    if 1 not in values:
        values.append(1)
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare M/M/1 vs. M/M/g metrics across g.")
    parser.add_argument("--lam", type=float, help="Arrival rate lambda (required unless --scenario).")
    parser.add_argument("--mu", type=float, help="Service rate mu (required unless --scenario).")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "C"],
        help="Scenario shortcut. If provided, lambda is adjusted to keep rho constant per server.",
    )
    parser.add_argument(
        "--g-list",
        type=str,
        default="1,2,3,4",
        help='Comma-separated list of server counts to evaluate (e.g. "1,2,3,4").',
    )
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--warmup", type=float, default=10_000.0, help="Warm-up time.")
    parser.add_argument("--horizon", type=float, default=200_000.0, help="Simulation horizon.")
    parser.add_argument("--replications", type=int, default=20, help="Replications per g.")
    parser.add_argument(
        "--results-out",
        type=Path,
        default=Path("outputs/compare_results.csv"),
        help="CSV where per-replication results will be stored.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/compare_summary.csv"),
        help="CSV with aggregated statistics per g.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where comparison figures will be written.",
    )
    parser.add_argument(
        "--decision-mode",
        type=str,
        choices=["weights", "costs"],
        default="weights",
        help="Strategy for recommending g: weighted metric or explicit costs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for Wq_mean when decision-mode=weights.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Weight for Pwait_sim_mean when decision-mode=weights.",
    )
    parser.add_argument(
        "--c-server",
        type=float,
        default=1.0,
        dest="c_server",
        help="Server cost coefficient when decision-mode=costs.",
    )
    parser.add_argument(
        "--c-wait",
        type=float,
        default=1.0,
        dest="c_wait",
        help="Waiting cost coefficient when decision-mode=costs.",
    )
    return parser.parse_args()


def resolve_base_rates(args: argparse.Namespace) -> tuple[float, float, float | None]:
    if args.scenario:
        params = get_params(args.scenario, seed=args.seed, warmup=args.warmup, horizon=args.horizon)
        lam = params.lam
        mu = params.mu
        target_rho = lam / mu
    else:
        if args.lam is None or args.mu is None:
            raise SystemExit("Either --scenario or both --lam and --mu must be provided.")
        lam = args.lam
        mu = args.mu
        target_rho = None
    return lam, mu, target_rho


def run_replications_for_g(
    lam: float, mu: float, g: int, args: argparse.Namespace
) -> Iterable[dict[str, float]]:
    for rep in trange(args.replications, desc=f"g={g}", unit="rep"):
        params = MM1Params(
            lam=lam,
            mu=mu,
            seed=args.seed + rep,
            warmup=args.warmup,
            horizon=args.horizon,
        )
        result = run_mmg(params, g=g)
        payload = result.as_dict()
        payload["model"] = "mmg"
        return_dict = payload
        yield return_dict


def summarize_by_g(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("L", "L"),
        ("Lq", "Lq"),
        ("W_mean", "W"),
        ("Wq_mean", "Wq"),
        ("Pwait_sim", "Pwait_sim"),
        ("utilization", "utilization"),
        ("lambda_hat", "lambda_hat"),
    ]
    rows = []
    for g, group in df.groupby("g"):
        row = {
            "g": int(g),
            "replications": len(group),
            "lam": float(group["lam"].iloc[0]),
            "mu": float(group["mu"].iloc[0]),
            "rho_theory": float(group["rho_theory"].iloc[0]),
            "Pwait_theory": float(group["Pwait_theory"].iloc[0]),
        }
        row["L_theory"] = float(group["L_theory"].iloc[0])
        row["Lq_theory"] = float(group["Lq_theory"].iloc[0])
        row["W_theory"] = float(group["W_theory"].iloc[0])
        row["Wq_theory"] = float(group["Wq_theory"].iloc[0])
        for column, alias in metrics:
            series = group[column]
            mean = float(series.mean())
            std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(series)) if len(series) > 1 else 0.0
            row[f"{alias}_mean"] = mean
            row[f"{alias}_ci95"] = half
        rows.append(row)
    return pd.DataFrame(rows).sort_values("g")


def decision_score(
    row: pd.Series,
    mode: str,
    alpha: float,
    beta: float,
    c_server: float,
    c_wait: float,
) -> float:
    if mode == "weights":
        return alpha * row["Wq_mean"] + beta * row["Pwait_sim_mean"]
    return c_server * row["g"] + c_wait * row["Wq_mean"]


def annotate_decision(summary: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, int | None]:
    if summary.empty:
        return summary, None

    summary = summary.copy()
    summary["decision_mode"] = args.decision_mode
    summary["alpha"] = args.alpha
    summary["beta"] = args.beta
    summary["c_server"] = args.c_server
    summary["c_wait"] = args.c_wait
    summary["score"] = summary.apply(
        lambda row: decision_score(
            row,
            args.decision_mode,
            args.alpha,
            args.beta,
            args.c_server,
            args.c_wait,
        ),
        axis=1,
    )
    best_idx = summary["score"].idxmin() if not summary.empty else None
    best_g = int(summary.loc[best_idx, "g"]) if best_idx is not None else None
    return summary, best_g


def plot_bar_metrics(summary: pd.DataFrame, reports_dir: Path) -> None:
    metrics = ["L", "Lq", "W", "Wq"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.flatten()
    gs = summary["g"].astype(int).tolist()
    for ax, label in zip(axes, metrics):
        means = summary[f"{label}_mean"]
        errors = summary[f"{label}_ci95"]
        ax.bar(gs, means, yerr=errors, capsize=5)
        ax.set_title(label)
        ax.set_xlabel("g")
        ax.set_ylabel(label)
    fig.suptitle("Metricas M/M/g por numero de servidores (IC95)")
    fig.tight_layout()
    fig.savefig(reports_dir / "mm1_mmg_barras.png", dpi=150)
    plt.close(fig)


def plot_metrics_vs_g(summary: pd.DataFrame, reports_dir: Path) -> None:
    g_values = summary["g"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    series = ["L", "Lq", "W", "Wq"]
    for label in series:
        ax.plot(g_values, summary[f"{label}_mean"], marker="o", label=f"{label} (sim)")
        theory_col = f"{label}_theory"
        ax.plot(
            g_values,
            summary[theory_col],
            linestyle="--",
            marker="x",
            label=f"{label} (teo)",
        )
    ax2 = ax.twinx()
    ax2.plot(g_values, summary["Pwait_sim_mean"], color="black", marker="s", label="Pwait (sim)")
    ax2.plot(
        g_values,
        summary["Pwait_theory"],
        color="gray",
        linestyle="--",
        marker="^",
        label="Pwait (teo)",
    )
    ax.set_xlabel("g")
    ax.set_ylabel("L, Lq, W, Wq")
    ax2.set_ylabel("Pwait")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    ax.set_title("Metricas vs. numero de servidores")
    fig.tight_layout()
    fig.savefig(reports_dir / "metricas_vs_g.png", dpi=150)
    plt.close(fig)


def plot_pwait(summary: pd.DataFrame, reports_dir: Path) -> None:
    g_values = summary["g"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_values, summary["Pwait_sim_mean"], marker="o", label="Simulacion")
    ax.plot(g_values, summary["Pwait_theory"], linestyle="--", marker="x", label="Teoria")
    ax.set_xlabel("g")
    ax.set_ylabel("P(wait)")
    ax.set_title("Probabilidad de espera vs. numero de servidores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "pwait_vs_g.png", dpi=150)
    plt.close(fig)


def plot_decision(summary: pd.DataFrame, best_g: int | None, reports_dir: Path) -> None:
    if summary.empty:
        return
    g_values = summary["g"].astype(int).tolist()
    scores = summary["score"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_values, scores, marker="o", label="Score")
    ax.set_xlabel("g")
    ax.set_ylabel("Score (menor es mejor)")
    ax.set_title("Funcion de decision vs. numero de servidores")
    if best_g is not None:
        best_row = summary[summary["g"] == best_g].iloc[0]
        ax.scatter([best_g], [best_row["score"]], color="black", zorder=5, label="g*")
        ax.text(
            best_g,
            best_row["score"],
            " g*",
            va="bottom",
            ha="left",
            fontsize=10,
            color="black",
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "decision_vs_g.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    g_values = parse_g_list(args.g_list)
    lam_base, mu, target_rho = resolve_base_rates(args)

    all_results = []
    for g in g_values:
        lam = lam_base
        if target_rho is not None:
            lam = target_rho * g * mu
        theory = mmg_theory(lam, mu, g)
        for rep in run_replications_for_g(lam, mu, g, args):
            rep["lam"] = lam
            rep["mu"] = mu
            rep["g"] = g
            rep["Pwait_theory"] = theory.Pwait
            rep["L_theory"] = theory.L
            rep["Lq_theory"] = theory.Lq
            rep["W_theory"] = theory.W
            rep["Wq_theory"] = theory.Wq
            rep["rho_theory"] = theory.rho
            all_results.append(rep)

    if not all_results:
        raise SystemExit("No se generaron resultados; revise los parametros.")

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.results_out, index=False)

    summary_df = summarize_by_g(results_df)
    summary_df, best_g = annotate_decision(summary_df, args)
    summary_df.to_csv(args.summary_out, index=False)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    plot_bar_metrics(summary_df, args.reports_dir)
    plot_metrics_vs_g(summary_df, args.reports_dir)
    plot_pwait(summary_df, args.reports_dir)
    plot_decision(summary_df, best_g, args.reports_dir)

    print(f"Resultados por replicacion: {args.results_out.resolve()}")
    print(f"Resumen comparativo: {args.summary_out.resolve()}")
    print(f"Graficos guardados en {args.reports_dir.resolve()}")


if __name__ == "__main__":
    main()
