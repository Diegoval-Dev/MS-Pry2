"""Closed-form performance metrics for an M/M/1 queue."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class MM1Theory:
    """Bundle of theoretical steady-state metrics for an M/M/1 system."""

    rho: float
    L: float
    Lq: float
    W: float
    Wq: float

    def as_dict(self) -> Mapping[str, float]:
        """Return the metrics as a plain dictionary (handy for printing)."""
        return asdict(self)


def rho(lam: float, mu: float) -> float:
    """Return the traffic intensity λ/μ validating the input domain."""
    if lam < 0:
        raise ValueError("Arrival rate lam must be non-negative.")
    if mu <= 0:
        raise ValueError("Service rate mu must be strictly positive.")
    return lam / mu


def mm1_theory(lam: float, mu: float) -> MM1Theory:
    """
    Compute steady-state M/M/1 metrics.

    Raises:
        ValueError: when ρ ≥ 1 (system unstable) or inputs are invalid.
    """
    r = rho(lam, mu)
    if r >= 1.0:
        raise ValueError("Unstable system: rho must be < 1 for M/M/1.")

    if lam == 0:
        return MM1Theory(rho=0.0, L=0.0, Lq=0.0, W=0.0, Wq=0.0)

    denom = 1.0 - r
    L = r / denom
    Lq = (r * r) / denom
    W = L / lam
    Wq = Lq / lam
    return MM1Theory(rho=r, L=L, Lq=Lq, W=W, Wq=Wq)


def relative_error(sim_value: float, reference_value: float) -> float:
    """Return |sim-ref| / ref guarding division by zero."""
    if reference_value == 0:
        return 0.0 if sim_value == 0 else float("inf")
    return abs(sim_value - reference_value) / abs(reference_value)
