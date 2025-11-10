## M/M/1 Simulation Project

Discrete-event simulation of an M/M/1 queue that contrasts empirical metrics against theory, saves CSV/PNG artefacts, and validates Little's law.

### Setup

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Recommended Run (Scenario B, rho ~ 0.85)

```bash
python -m src.run_sim --scenario B --replications 20 --seed 123 --warmup 10000 --horizon 200000
```

Outputs:

- `outputs/results.csv` with per-replication metrics (`L`, `Lq`, `W_mean`, `Wq_mean`, `utilization`, `obs_time`, `arrivals_obs`, `lambda_hat`, `rho_theory`, `little_*_error`, ...).
- `outputs/summary.csv` gathering means, theoretical comparisons, and IC95 half-widths for the key metrics (notably `L` and `W_mean`).
- Console report with theory vs. simulation, relative errors, observation window diagnostics (`[warmup, horizon]`), Little's law gaps, and IC95 relative widths.

Adapt the same command for scenarios A/C or custom rates via `--lam` and `--mu`.

### Methodology

All time-based metrics (`L`, `Lq`, utilization) integrate only over the observation window `[warmup, horizon]`. The simulation also tracks the effective arrival rate `lambda_hat = arrivals_obs / (horizon - warmup)` to validate Little's law (`L ~ lambda_hat * W`, `Lq ~ lambda_hat * Wq`).

### Plots

```bash
python -m src.plots
```

Generates `reports/comp_teo_sim.png` (con barras de error IC95), `reports/hist_wq.png`, y `reports/serie_L.png`.

### Tests

```bash
pytest -q
```

Confirms analytical formulas plus a smoke test ensuring Little's law deviations stay within 10%.
