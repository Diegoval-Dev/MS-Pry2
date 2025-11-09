"""Discrete-event simulation core for the M/M/1 queue."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
import simpy


@dataclass(frozen=True)
class MM1Params:
    """Simulation parameters bundled for convenience."""

    lam: float
    mu: float
    seed: int
    warmup: float
    horizon: float

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError("Arrival rate lam must be non-negative.")
        if self.mu <= 0:
            raise ValueError("Service rate mu must be strictly positive.")
        if self.horizon <= 0:
            raise ValueError("Simulation horizon must be positive.")
        if self.warmup < 0:
            raise ValueError("Warm-up period must be non-negative.")


@dataclass
class SimulationResult:
    """Container for the aggregated outputs of one replication."""

    lam: float
    mu: float
    seed: int
    warmup: float
    horizon: float
    L: float
    Lq: float
    utilization: float
    W_mean: float
    Wq_mean: float
    n_samples: int
    obs_time: float
    arrivals_obs: int
    lambda_hat: float
    rho_theory: float
    little_L_error: float
    little_Lq_error: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class MM1System:
    """Encapsulates the SimPy processes and on-the-fly measurements."""

    def __init__(self, env: simpy.Environment, params: MM1Params):
        self.env = env
        self.params = params
        self.rng = np.random.default_rng(seed=params.seed)
        self.server_busy = False
        self.queue: Deque[float] = deque()
        self.wait_samples: List[float] = []
        self.system_samples: List[float] = []
        self.in_system = 0
        self.in_queue = 0
        self.area_L = 0.0
        self.area_Lq = 0.0
        self.busy_time = 0.0
        self.last_event_time = 0.0
        self.busy_start: Optional[float] = None
        self.arrivals_obs = 0

    def _exp(self, rate: float) -> float:
        """Draw an exponential sample handling the λ=0 case upstream."""
        return self.rng.exponential(1.0 / rate)

    def arrival_process(self):
        """Generate arrivals until the horizon."""
        if self.params.lam == 0:
            return

        while True:
            inter_arrival = self._exp(self.params.lam)
            yield self.env.timeout(inter_arrival)
            if self.env.now > self.params.horizon:
                break
            self._handle_arrival()

    def _handle_arrival(self):
        self._update_time_integrals()
        arrival_time = self.env.now
        self.in_system += 1
        if arrival_time >= self.params.warmup:
            self.arrivals_obs += 1
        if self.server_busy:
            self.queue.append(arrival_time)
            self.in_queue += 1
        else:
            self.server_busy = True
            self.busy_start = self.env.now
            self.env.process(self._serve_customer(arrival_time))

    def _serve_customer(self, arrival_time: float):
        # Customers that reach service immediately have wait = 0.
        wait = self.env.now - arrival_time
        service_time = self._exp(self.params.mu)
        yield self.env.timeout(service_time)
        self._update_time_integrals()
        departure_time = self.env.now
        self.in_system -= 1

        if arrival_time >= self.params.warmup:
            self.wait_samples.append(wait)
            self.system_samples.append(departure_time - arrival_time)

        if self.queue:
            queued_arrival = self.queue.popleft()
            self.in_queue -= 1
            # The server stays busy; immediately start next customer.
            self.env.process(self._serve_customer(queued_arrival))
        else:
            self._accumulate_busy_time(departure_time)
            self.server_busy = False
            self.busy_start = None

    def _update_time_integrals(self, target_time: float | None = None):
        """Integrate areas for L and Lq restricted to the observation window."""
        now = self.env.now if target_time is None else target_time
        start = self.last_event_time
        self.last_event_time = now

        window_start = max(start, self.params.warmup)
        window_end = min(now, self.params.horizon)
        dt = window_end - window_start
        if dt <= 0:
            return

        self.area_L += self.in_system * dt
        self.area_Lq += self.in_queue * dt

    def _accumulate_busy_time(self, end_time: float) -> None:
        """Accumulate server busy time intersected with the observation window."""
        if self.busy_start is None:
            return
        window_start = max(self.busy_start, self.params.warmup)
        window_end = min(end_time, self.params.horizon)
        dt = window_end - window_start
        if dt > 0:
            self.busy_time += dt


def run_mm1(params: MM1Params) -> SimulationResult:
    """Run one M/M/1 replication and return aggregated statistics."""
    obs_time = max(params.horizon - params.warmup, 0.0)
    rho_theory = params.lam / params.mu
    if params.lam == 0:
        return SimulationResult(
            lam=params.lam,
            mu=params.mu,
            seed=params.seed,
            warmup=params.warmup,
            horizon=params.horizon,
            L=0.0,
            Lq=0.0,
            utilization=0.0,
            W_mean=0.0,
            Wq_mean=0.0,
            n_samples=0,
            obs_time=obs_time,
            arrivals_obs=0,
            lambda_hat=0.0,
            rho_theory=rho_theory,
            little_L_error=0.0,
            little_Lq_error=0.0,
        )

    env = simpy.Environment()
    system = MM1System(env, params)
    env.process(system.arrival_process())
    env.run(until=params.horizon)
    system._update_time_integrals(target_time=params.horizon)
    if system.server_busy:
        system._accumulate_busy_time(params.horizon)

    if obs_time == 0:
        L = Lq = utilization = 0.0
    else:
        L = system.area_L / obs_time
        Lq = system.area_Lq / obs_time
        utilization = system.busy_time / obs_time

    W_mean = float(np.mean(system.system_samples)) if system.system_samples else 0.0
    Wq_mean = float(np.mean(system.wait_samples)) if system.wait_samples else 0.0
    lambda_hat = system.arrivals_obs / obs_time if obs_time > 0 else 0.0

    little_L = lambda_hat * W_mean
    little_Lq = lambda_hat * Wq_mean
    little_L_error = 0.0 if L == 0 else abs(L - little_L) / L
    little_Lq_error = 0.0 if Lq == 0 else abs(Lq - little_Lq) / Lq

    return SimulationResult(
        lam=params.lam,
        mu=params.mu,
        seed=params.seed,
        warmup=params.warmup,
        horizon=params.horizon,
        L=L,
        Lq=Lq,
        utilization=utilization,
        W_mean=W_mean,
        Wq_mean=Wq_mean,
        n_samples=len(system.system_samples),
        obs_time=obs_time,
        arrivals_obs=system.arrivals_obs,
        lambda_hat=lambda_hat,
        rho_theory=rho_theory,
        little_L_error=little_L_error,
        little_Lq_error=little_Lq_error,
    )
