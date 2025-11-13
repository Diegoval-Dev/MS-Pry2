"""Discrete-event simulation core for an M/M/g queue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import simpy

from .metrics_mmg import mmg_theory
from .mm1_core import MM1Params, SimulationResult


@dataclass
class MMGSimulationState:
    """Mutable containers for aggregated statistics."""

    wait_samples: List[float]
    system_samples: List[float]
    waited_count: int = 0
    arrivals_obs: int = 0
    in_system: int = 0
    in_queue: int = 0
    busy_servers: int = 0
    area_L: float = 0.0
    area_Lq: float = 0.0
    busy_area: float = 0.0
    last_event_time: float = 0.0


class MMGSystem:
    """Encapsulates SimPy processes and measurements for M/M/g."""

    def __init__(self, env: simpy.Environment, params: MM1Params, g: int):
        self.env = env
        self.params = params
        self.g = g
        self.state = MMGSimulationState(wait_samples=[], system_samples=[])
        self.rng = np.random.default_rng(seed=params.seed)
        self.server = simpy.Resource(env, capacity=g)

    def _exp(self, rate: float) -> float:
        return self.rng.exponential(1.0 / rate)

    def arrival_process(self):
        if self.params.lam == 0:
            return
        while True:
            inter_arrival = self._exp(self.params.lam)
            yield self.env.timeout(inter_arrival)
            if self.env.now > self.params.horizon:
                break
            self.env.process(self._customer())

    def _customer(self):
        self._update_time_integrals()
        arrival_time = self.env.now
        self.state.in_system += 1
        if arrival_time >= self.params.warmup:
            self.state.arrivals_obs += 1
        self._refresh_queue_count()

        with self.server.request() as req:
            yield req
            self._update_time_integrals()
            wait = self.env.now - arrival_time
            self.state.busy_servers += 1
            self._refresh_queue_count()

            service_time = self._exp(self.params.mu)
            yield self.env.timeout(service_time)
            self._update_time_integrals()

            departure_time = self.env.now
            self.state.busy_servers -= 1
            self.state.in_system -= 1
            self._refresh_queue_count()

            if arrival_time >= self.params.warmup:
                self.state.wait_samples.append(wait)
                self.state.system_samples.append(departure_time - arrival_time)
                if wait > 0:
                    self.state.waited_count += 1

    def _refresh_queue_count(self) -> None:
        waiting = self.state.in_system - self.state.busy_servers
        self.state.in_queue = max(waiting, 0)

    def _update_time_integrals(self, target_time: Optional[float] = None) -> None:
        now = self.env.now if target_time is None else target_time
        start = self.state.last_event_time
        self.state.last_event_time = now

        window_start = max(start, self.params.warmup)
        window_end = min(now, self.params.horizon)
        dt = window_end - window_start
        if dt <= 0:
            return

        self.state.area_L += self.state.in_system * dt
        self.state.area_Lq += self.state.in_queue * dt
        self.state.busy_area += self.state.busy_servers * dt


def run_mmg(params: MM1Params, g: int) -> SimulationResult:
    """Run one replication of an M/M/g queue."""
    if g < 1:
        raise ValueError("Number of servers g must be >= 1.")

    obs_time = max(params.horizon - params.warmup, 0.0)

    if params.lam == 0:
        rho_theory = 0.0
        pwait_theory = 0.0
        return SimulationResult(
            model="mmg",
            g=g,
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
            Pwait_sim=0.0,
            Pwait_theory=pwait_theory,
            n_samples=0,
            obs_time=obs_time,
            arrivals_obs=0,
            lambda_hat=0.0,
            rho_theory=rho_theory,
            little_L_error=0.0,
            little_Lq_error=0.0,
        )

    theory = mmg_theory(params.lam, params.mu, g)
    rho_theory = theory.rho
    pwait_theory = theory.Pwait

    env = simpy.Environment()
    system = MMGSystem(env, params, g)
    env.process(system.arrival_process())
    env.run(until=params.horizon)
    system._update_time_integrals(target_time=params.horizon)

    if obs_time == 0:
        L = Lq = utilization = 0.0
    else:
        L = system.state.area_L / obs_time
        Lq = system.state.area_Lq / obs_time
        busy_mean = system.state.busy_area / obs_time
        utilization = busy_mean / g

    W_mean = float(np.mean(system.state.system_samples)) if system.state.system_samples else 0.0
    Wq_mean = float(np.mean(system.state.wait_samples)) if system.state.wait_samples else 0.0
    lambda_hat = system.state.arrivals_obs / obs_time if obs_time > 0 else 0.0
    n_samples = len(system.state.system_samples)
    pwait_sim = (system.state.waited_count / n_samples) if n_samples > 0 else 0.0

    little_L = lambda_hat * W_mean
    little_Lq = lambda_hat * Wq_mean
    little_L_error = 0.0 if L == 0 else abs(L - little_L) / L
    little_Lq_error = 0.0 if Lq == 0 else abs(Lq - little_Lq) / Lq

    return SimulationResult(
        model="mmg",
        g=g,
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
        Pwait_sim=pwait_sim,
        Pwait_theory=pwait_theory,
        n_samples=n_samples,
        obs_time=obs_time,
        arrivals_obs=system.state.arrivals_obs,
        lambda_hat=lambda_hat,
        rho_theory=rho_theory,
        little_L_error=little_L_error,
        little_Lq_error=little_Lq_error,
    )
