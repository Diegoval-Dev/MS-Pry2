"""Utility to generate figures comparing simulation vs. theory."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    from sim import mm1_theory
except ModuleNotFoundError:  # pragma: no cover
    from .sim import mm1_theory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from simulation results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/results.csv"),
        help="CSV produced by src.run_sim.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where PNG files will be saved.",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Results file is empty. Run the simulation first.")
    return df


def compute_metrics(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    lam = float(df["lam"].iloc[0])
    mu = float(df["mu"].iloc[0])
    theory = mm1_theory(lam, mu).as_dict()
    sim_means = df[["L", "Lq", "W_mean", "Wq_mean", "utilization"]].mean()
    return sim_means, theory


def compute_error_bars(df: pd.DataFrame, metrics: list[str]) -> list[float]:
    if len(df) < 2:
        return [0.0 for _ in metrics]
    errs = []
    n = len(df)
    for metric in metrics:
        std = float(df[metric].std(ddof=1))
        errs.append(1.96 * std / math.sqrt(n))
    return errs


def plot_bar_comparison(
    df: pd.DataFrame, sim_means: pd.Series, theory: Dict[str, float], out: Path
) -> None:
    labels = ["L", "Lq", "W", "Wq"]
    sim_values = [sim_means["L"], sim_means["Lq"], sim_means["W_mean"], sim_means["Wq_mean"]]
    theory_values = [theory["L"], theory["Lq"], theory["W"], theory["Wq"]]
    errors = compute_error_bars(df, ["L", "Lq", "W_mean", "Wq_mean"])

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], theory_values, width=width, label="Teoria")
    ax.bar(
        [i + width / 2 for i in x],
        sim_values,
        width=width,
        label="Simulacion",
        yerr=errors,
        capsize=5,
        error_kw={"elinewidth": 1, "alpha": 0.8},
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor")
    ax.set_title("Comparacion teoria vs. simulacion (IC95)")
    ax.legend()
    for i, err in enumerate(errors):
        if err <= 0:
            continue
        ax.text(
            i + width / 2,
            sim_values[i] + err + 0.02 * max(sim_values),
            f"±{err:.2f}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_histogram(series: pd.Series, title: str, xlabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = min(20, max(5, len(series)))
    ax.hist(series, bins=bins, color="#4c72b0", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_replication_series(series: pd.Series, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series.index + 1, series.values, marker="o")
    ax.set_xlabel("Replicacion")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} por replicacion")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_results(args.results)
    sim_means, theory = compute_metrics(df)

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    plot_bar_comparison(df, sim_means, theory, args.reports_dir / "comp_teo_sim.png")
    plot_histogram(
        df["Wq_mean"], "Distribucion de Wq_mean", "Wq_mean", args.reports_dir / "hist_wq.png"
    )
    plot_replication_series(df["L"], "L promedio", args.reports_dir / "serie_L.png")

    print(f"Figuras guardadas en {args.reports_dir.resolve()}")


if __name__ == "__main__":
    main()
