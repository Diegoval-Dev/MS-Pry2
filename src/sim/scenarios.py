"""Pre-defined simulation scenarios with varying traffic intensities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from .mm1_core import MM1Params


@dataclass(frozen=True)
class Scenario:
    name: str
    lam: float
    mu: float


SCENARIOS: Dict[str, Scenario] = {
    "A": Scenario(name="A", lam=0.6, mu=1.0),  # ρ ≈ 0.60
    "B": Scenario(name="B", lam=0.85, mu=1.0),  # ρ ≈ 0.85
    "C": Scenario(name="C", lam=0.95, mu=1.0),  # ρ ≈ 0.95
}


def list_scenarios() -> Iterable[str]:
    """Return available scenario identifiers."""
    return sorted(SCENARIOS.keys())


def get_params(name: str, seed: int, warmup: float, horizon: float) -> MM1Params:
    """Return `MM1Params` for a named scenario."""
    key = name.upper()
    if key not in SCENARIOS:
        raise KeyError(f"Scenario '{name}' is not defined. Available: {list_scenarios()}")
    scenario = SCENARIOS[key]
    return MM1Params(
        lam=scenario.lam,
        mu=scenario.mu,
        seed=seed,
        warmup=warmup,
        horizon=horizon,
    )
