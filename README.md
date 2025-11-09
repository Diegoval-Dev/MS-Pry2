## M/M/1 Simulation Project

Discrete-event simulation of an M/M/1 queue that contrasts empirical metrics against theory, saves CSV/PNG artefacts, and validates Little's law.

### Setup

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Recommended Run (Scenario B, rho ≈ 0.85)

```bash
python -m src.run_sim --scenario B --replications 20 --seed 123 --warmup 10000 --horizon 200000
```

Outputs:

- `outputs/results.csv` with per-replication metrics (`L`, `Lq`, `W_mean`, `Wq_mean`, `utilization`, `obs_time`, `arrivals_obs`, `lambda_hat`, `little_*_error`, etc.).
- Console report with theory vs. simulation, relative errors, observation window diagnostics, and Little's law gaps.

Adapt the same command for scenarios A/C or custom rates via `--lam` and `--mu`.

### Plots

```bash
python -m src.plots
```

Generates `reports/comp_teo_sim.png`, `reports/hist_wq.png`, and `reports/serie_L.png`.

### Tests

```bash
pytest -q
```

Confirms analytical formulas for the M/M/1 reference metrics.
