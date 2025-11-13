"""Closed-form metrics for the M/M/g queue (Erlang-C)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MMGTheory:
    """Bundle of steady-state metrics for an M/M/g system."""

    rho: float
    Pwait: float
    L: float
    Lq: float
    W: float
    Wq: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def mmg_theory(lam: float, mu: float, g: int) -> MMGTheory:
    """
    Compute M/M/g steady-state metrics using the Erlang-C formulas.

    Raises:
        ValueError: if inputs are outside the stability region (rho >= 1)
                    or g < 1 / mu <= 0 / lam < 0.
    """
    if lam < 0:
        raise ValueError("Arrival rate lam must be non-negative.")
    if mu <= 0:
        raise ValueError("Service rate mu must be strictly positive.")
    if g < 1:
        raise ValueError("Number of servers g must be >= 1.")

    if lam == 0:
        return MMGTheory(rho=0.0, Pwait=0.0, L=0.0, Lq=0.0, W=0.0, Wq=0.0)

    a = lam / mu
    c = g
    rho = a / c
    if rho >= 1.0:
        raise ValueError("Unstable system: rho must be < 1 for M/M/g.")

    sum_terms = sum((a**k) / math.factorial(k) for k in range(c))
    tail = (a**c / math.factorial(c)) * (c / (c - a))
    inv_P0 = sum_terms + tail
    P0 = 1.0 / inv_P0
    Pwait = tail * P0

    Wq = Pwait * (1.0 / mu) * (1.0 / (c - a))
    Lq = lam * Wq
    W = Wq + 1.0 / mu
    L = lam * W

    return MMGTheory(rho=rho, Pwait=Pwait, L=L, Lq=Lq, W=W, Wq=Wq)
