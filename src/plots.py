"""Utility to generate figures comparing simulation vs. theory."""

from __future__ import annotations

import argparse
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


def plot_bar_comparison(sim_means: pd.Series, theory: Dict[str, float], out: Path) -> None:
    labels = ["L", "Lq", "W", "Wq"]
    sim_values = [sim_means["L"], sim_means["Lq"], sim_means["W_mean"], sim_means["Wq_mean"]]
    theory_values = [theory["L"], theory["Lq"], theory["W"], theory["Wq"]]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], theory_values, width=width, label="Teoría")
    ax.bar([i + width / 2 for i in x], sim_values, width=width, label="Simulación")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación teoría vs. simulación")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_histogram(series: pd.Series, title: str, xlabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series, bins=min(10, len(series)), color="#4c72b0", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_replication_series(series: pd.Series, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series.index + 1, series.values, marker="o")
    ax.set_xlabel("Replicación")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} por replicación")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_results(args.results)
    sim_means, theory = compute_metrics(df)

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    plot_bar_comparison(sim_means, theory, args.reports_dir / "comp_teo_sim.png")
    plot_histogram(df["Wq_mean"], "Distribución de Wq_mean", "Wq_mean", args.reports_dir / "hist_wq.png")
    plot_replication_series(df["L"], "L promedio", args.reports_dir / "serie_L.png")

    print(f"Figuras guardadas en {args.reports_dir.resolve()}")


if __name__ == "__main__":
    main()
