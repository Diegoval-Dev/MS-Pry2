## M/M/1 Simulation Project

Discrete-event simulation of an M/M/1 queue with empirical vs. theoretical comparison.

### Setup

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run a Scenario

```bash
python -m src.run_sim --scenario B --replications 10 --seed 123 --warmup 1000 --horizon 50000
```

This generates `outputs/results.csv`, prints theory vs. simulation, and reports relative errors.

### Custom Parameters

Explicit λ and μ can be supplied instead of a scenario:

```bash
python -m src.run_sim --lam 0.8 --mu 1.0 --replications 5
```

### Plots

```bash
python -m src.plots
```

Creates PNGs inside `reports/` comparing theory vs. simulation and summarising replications.

### Tests

Run analytical consistency tests with:

```bash
pytest -q
```
