"""Command line interface to run M/M/1 simulations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from tqdm import trange

try:
    from sim import (
        MM1Params,
        SimulationResult,
        get_params,
        list_scenarios,
        mm1_theory,
        relative_error,
        run_mm1,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when executed as package
    from .sim import (
        MM1Params,
        SimulationResult,
        get_params,
        list_scenarios,
        mm1_theory,
        relative_error,
        run_mm1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run replicated simulations for an M/M/1 queue."
    )
    parser.add_argument("--lam", type=float, help="Arrival rate lambda (required unless --scenario).")
    parser.add_argument("--mu", type=float, help="Service rate mu (required unless --scenario).")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(list_scenarios()),
        help="Named scenario shortcut (A, B, C).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--warmup", type=float, default=1_000.0, help="Warm-up time to discard.")
    parser.add_argument("--horizon", type=float, default=50_000.0, help="Total simulation time.")
    parser.add_argument("--replications", type=int, default=5, help="Number of replications.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs/results.csv"),
        help="Path where the CSV summary will be written.",
    )
    return parser.parse_args()


def resolve_rates(args: argparse.Namespace) -> Tuple[float, float]:
    """Return lambda and mu based on scenario or explicit arguments."""
    if args.scenario:
        params = get_params(args.scenario, seed=args.seed, warmup=args.warmup, horizon=args.horizon)
        return params.lam, params.mu

    if args.lam is None or args.mu is None:
        raise SystemExit("Either --scenario or both --lam and --mu must be provided.")
    return args.lam, args.mu


def run_replications(
    lam: float,
    mu: float,
    args: argparse.Namespace,
) -> Iterable[SimulationResult]:
    """Yield SimulationResult for each replication."""
    for rep in trange(args.replications, desc="Simulating", unit="rep"):
        params = MM1Params(
            lam=lam,
            mu=mu,
            seed=args.seed + rep,
            warmup=args.warmup,
            horizon=args.horizon,
        )
        yield run_mm1(params)


def summarize(results: Iterable[SimulationResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.as_dict() for r in results])
    numeric_cols = [
        "L",
        "Lq",
        "utilization",
        "W_mean",
        "Wq_mean",
        "n_samples",
        "obs_time",
        "arrivals_obs",
        "lambda_hat",
        "rho_theory",
        "little_L_error",
        "little_Lq_error",
    ]
    if not df.empty:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def compute_summary(df: pd.DataFrame, theory: dict[str, float]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if df.empty:
        return pd.DataFrame(), {}

    n = len(df)
    lam = float(df["lam"].iloc[0])
    mu = float(df["mu"].iloc[0])
    warmup = float(df["warmup"].iloc[0])
    horizon = float(df["horizon"].iloc[0])

    rows = []
    lookup: dict[str, dict[str, float]] = {}

    def add_metric(name: str, theory_key: str | None) -> None:
        series = df[name]
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if n > 1 else 0.0
        half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        rel_half = (half / mean * 100) if mean else 0.0
        theory_value = theory[theory_key] if theory_key else (lam if name == "lambda_hat" else float("nan"))
        rel_err = relative_error(mean, theory_value) * 100 if theory_key else float("nan")

        row = {
            "metric": name,
            "mean": mean,
            "std": std,
            "ci95_halfwidth": half,
            "ci95_rel_pct": rel_half,
            "theory": theory_value,
            "relative_error_pct": rel_err,
            "replications": n,
            "lam": lam,
            "mu": mu,
            "warmup": warmup,
            "horizon": horizon,
        }
        rows.append(row)
        lookup[name] = row

    metric_mapping = [
        ("L", "L"),
        ("Lq", "Lq"),
        ("W_mean", "W"),
        ("Wq_mean", "Wq"),
        ("utilization", "rho"),
    ]
    for sim_key, th_key in metric_mapping:
        add_metric(sim_key, th_key)

    add_metric("lambda_hat", None)
    add_metric("little_L_error", None)
    add_metric("little_Lq_error", None)

    return pd.DataFrame(rows), lookup


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    lam, mu = resolve_rates(args)

    results = list(run_replications(lam, mu, args))
    df = summarize(results)

    ensure_parent(args.outputs)
    df.to_csv(args.outputs, index=False)

    theory = mm1_theory(lam, mu).as_dict()
    summary_df, summary_lookup = compute_summary(df, theory)
    summary_path = args.outputs.parent / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    sim_means = df[["L", "Lq", "W_mean", "Wq_mean", "utilization", "lambda_hat"]].mean()
    obs_time = df["obs_time"].iloc[0] if not df.empty else 0.0
    arrivals_total = int(df["arrivals_obs"].sum())
    little_L_gap = df["little_L_error"].mean()
    little_Lq_gap = df["little_Lq_error"].mean()

    print("\nTeoria M/M/1:")
    for key, value in theory.items():
        print(f"  {key:<3}: {value:>10.6f}")

    print("\nSimulacion (promedios empiricos):")
    for key in ["L", "Lq", "W_mean", "Wq_mean", "utilization", "lambda_hat"]:
        print(f"  {key:<11}: {sim_means[key]:>10.6f}")

    print("\nVentana de observacion:")
    print(f"  obs_time   : {obs_time:>10.2f}")
    print(f"  arrivals   : {arrivals_total:>10d}")
    print(f"  lambda_hat : {sim_means['lambda_hat']:>10.6f}")

    print("\nErrores relativos:")
    mapping = {"L": "L", "Lq": "Lq", "W_mean": "W", "Wq_mean": "Wq", "utilization": "rho"}
    for sim_key, th_key in mapping.items():
        err = relative_error(sim_means[sim_key], theory[th_key])
        readable = th_key if sim_key == "utilization" else sim_key
        print(f"  {readable:<11}: {err * 100:>9.3f}%")

    print("\nVerificacion Ley de Little:")
    print(f"  L vs lambda*W   : {little_L_gap * 100:>9.3f}%")
    print(f"  Lq vs lambda*Wq : {little_Lq_gap * 100:>9.3f}%")

    if summary_lookup:
        print("\nIC95 (media +/- half-width):")
        for metric in ("L", "W_mean"):
            row = summary_lookup.get(metric)
            if not row:
                continue
            print(
                f"  {metric:<11}: +/-{row['ci95_halfwidth']:>10.6f} "
                f"({row['ci95_rel_pct']:>6.3f}% del promedio)"
            )

    print(f"\nResultados guardados en {args.outputs.resolve()}")
    print(f"Resumen guardado en {summary_path.resolve()}")


if __name__ == "__main__":
    main()
