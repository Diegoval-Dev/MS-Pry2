"""Command line interface to run M/M/1 simulations."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--lam", type=float, help="Arrival rate λ (required unless --scenario).")
    parser.add_argument("--mu", type=float, help="Service rate μ (required unless --scenario).")
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
    """Return λ and μ based on scenario or explicit arguments."""
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
    numeric_cols = ["L", "Lq", "utilization", "W_mean", "Wq_mean", "n_samples"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


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
    sim_means = df[["L", "Lq", "W_mean", "Wq_mean", "utilization"]].mean()

    print("\nTeoría M/M/1:")
    for key, value in theory.items():
        print(f"  {key:<3}: {value:>10.6f}")

    print("\nSimulación (promedios empíricos):")
    for key in ["L", "Lq", "W_mean", "Wq_mean", "utilization"]:
        print(f"  {key:<11}: {sim_means[key]:>10.6f}")

    print("\nErrores relativos:")
    mapping = {"L": "L", "Lq": "Lq", "W_mean": "W", "Wq_mean": "Wq", "utilization": "rho"}
    for sim_key, th_key in mapping.items():
        err = relative_error(sim_means[sim_key], theory[th_key])
        readable = th_key if sim_key == "utilization" else sim_key
        print(f"  {readable:<11}: {err * 100:>9.3f}%")

    print(f"\nResultados guardados en {args.outputs.resolve()}")


if __name__ == "__main__":
    main()
