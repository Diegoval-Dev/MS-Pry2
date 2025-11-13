# Simulacion de colas M/M/1 y M/M/g

Proyecto de simulacion de eventos discretos para analizar sistemas de colas con llegadas Poisson y servicios exponenciales. El nucleo incluye medicion en ventana estacionaria, comparacion con teoria M/M/1 y Erlang-C (M/M/g), generacion de CSV, figuras y scripts de comparacion.

## 1. Objetivos
- Construir motores reproducibles M/M/1 (un servidor) y M/M/g (g servidores) con warm-up y horizonte configurables.
- Calcular metricas teoricas (`L`, `Lq`, `W`, `Wq`, utilizacion, `Pwait`) y validar estabilidad (`rho < 1`) y la ley de Little.
- Ofrecer CLI para replicaciones, escenarios A/B/C y exportacion a `outputs/`.
- Producir figuras listas para informes en `reports/`.
- Añadir pruebas unitarias para las formulas analiticas.

## 2. Requisitos
- Python 3.11+
- Librerias (`requirements.txt`): `simpy`, `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`, `pytest`.
- Windows/macOS/Linux (WSL valido).

## 3. Instalacion rapida
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1     # PowerShell (Windows)
# source .venv/bin/activate      # Bash (Linux/macOS)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Estructura clave
```
src/
  sim/
    mm1_core.py       # Motor M/M/1 (SimPy) con mediciones en ventana
    mmg_core.py       # Motor M/M/g (Resource(capacity=g))
    metrics.py        # Formulas M/M/1
    metrics_mmg.py    # Formulas Erlang-C (M/M/g)
    scenarios.py      # Escenarios A/B/C
  run_sim.py          # CLI principal
  compare.py          # Barrido de g y graficas
  plots.py            # Figuras base teoria vs simulacion
tests/
  test_metrics.py
  test_metrics_mmg.py
outputs/              # CSV (results, summary, compare_results, compare_summary)
reports/              # PNG (comp_teo_sim, hist_wq, serie_L, mm1_mmg_barras, metricas_vs_g, pwait_vs_g)
```

## 5. CLI principal (`src.run_sim`)

## Reproducibilidad oficial (Escenario B, ρ≈0.85)

### 1) M/M/1
python -m src.run_sim --model mm1 --scenario B --replications 20 --seed 123 --warmup 10000 --horizon 200000

### 2) M/M/g (barrido g ∈ {1,2,3,4})
python -m src.compare --scenario B --g-list "1,2,3,4" --replications 20 --seed 123 --warmup 10000 --horizon 200000

### 3) Métrica de decisión (ponderaciones por defecto α=0.6, β=0.4)
python -m src.compare --scenario B --g-list "1,2,3,4" --replications 20 --seed 123 --warmup 10000 --horizon 200000 --decision-mode weights --alpha 0.6 --beta 0.4

### Artefactos generados
- outputs/results.csv, outputs/summary.csv
- outputs/compare_results.csv, outputs/compare_summary.csv
- reports/comp_teo_sim.png, reports/hist_Wq.png, reports/serie_L.png
- reports/metricas_vs_g.png, reports/mm1_mmg_barras.png, reports/pwait_vs_g.png
- reports/decision_vs_g.png
- reports/analysis/conclusions.md

Bandera nuevas:
- `--model {mm1,mmg}` (default `mm1`)
- `--g INT` (solo para `mmg`, numero de servidores)

Si se usa `--scenario` con `mmg`, se ajusta `lambda = rho_objetivo * g * mu` para mantener la misma utilizacion por servidor que en el escenario M/M/1.

Ejemplo oficial (Addendum v1.2, escenario B, rho≈0.85):
```bash
python -m src.run_sim --model mm1 --scenario B \
    --replications 20 --seed 123 \
    --warmup 10000 --horizon 200000
```

Mismo escenario con tres servidores:
```bash
python -m src.run_sim --model mmg --g 3 --scenario B \
    --replications 20 --seed 123 \
    --warmup 10000 --horizon 200000
```

Salida:
- `outputs/results.csv`: una fila por replicacion con `model`, `g`, `L`, `Lq`, `W_mean`, `Wq_mean`, `utilization`, `Pwait_sim`, `Pwait_theory`, `obs_time`, `arrivals_obs`, `lambda_hat`, `rho_theory`, `little_*_error`.
- `outputs/summary.csv`: promedios, IC95 (half-width y % relativo) y errores relativos vs teoria (`L`, `Lq`, `W_mean`, `Wq_mean`, `utilization`, `Pwait_sim`, `lambda_hat`).
- Consola: tabla teoria vs. simulacion, ventana `[warmup, horizon]`, errores relativos, verificacion de Little y half-width para `L` y `W_mean`.

## 6. Script de comparacion M/M/1 vs. M/M/g
`src.compare` ejecuta barridos de `g` y genera tablas y figuras comparativas.

```bash
python -m src.compare --scenario B \
    --g-list "1,2,3,4" \
    --replications 20 --seed 123 \
    --warmup 10000 --horizon 200000
```

Produce:
- `outputs/compare_results.csv`: resultados por replicacion (todas las columnas de `run_sim` + teoricos por replica).
- `outputs/compare_summary.csv`: media e IC95 por `g` para `L`, `Lq`, `W`, `Wq`, `Pwait`, `utilization`, `lambda_hat`.
- Figuras en `reports/`:
  - `mm1_mmg_barras.png`: barras con IC95 para `L`, `Lq`, `W`, `Wq`.
  - `metricas_vs_g.png`: lineas teoria vs simulacion para `L`, `Lq`, `W`, `Wq` y `Pwait`.
  - `pwait_vs_g.png`: probabilidad de espera por `g`.

## 7. Visualizaciones base
```bash
python -m src.plots --results outputs/results.csv
```
Genera:
- `comp_teo_sim.png`
- `hist_wq.png`
- `serie_L.png`

`src.plots` detecta el modelo (`mm1` o `mmg`) y aplica las formulas correspondientes.

## 8. Metodologia y validacion
- Las areas de `L`, `Lq` y la utilizacion solo integran la ventana observada (`obs_time = horizon - warmup`).
- Para M/M/g se integra el numero de servidores ocupados y se calcula `utilization = busy_area / (g * obs_time)`.
- Se registra `lambda_hat = arrivals_obs / obs_time` y el numero de clientes con espera (`Pwait_sim = waited / arrivals_obs`).
- Las corridas deben respetar:
  - Error relativo medio ≤ 10 % en `L` y `W`.
  - Utilizacion dentro de ±3 % respecto a `rho`.
  - `Pwait_sim` dentro de ±10 % absoluto de la teoria (Erlang-C).
  - Ley de Little (`L` y `Lq`) con discrepancia ≤ 10 %.

## 9. Pruebas
```bash
pytest -q
```
Incluye:
- `tests/test_metrics.py`: formulas M/M/1 y guardas de dominio.
- `tests/test_metrics_mmg.py`: reduccion a M/M/1 (`g=1`), caso `g=2` con `rho=0.85`, y excepciones (`g<1`, `rho>=1`).

## Metodología (resumen)
- Ventana de observación: solo se miden L, Lq y utilización en [warmup, horizon]; warmup=10 000, horizon=200 000.
- Replicaciones: N=20; se reportan medias e IC95.
- Validación:
  - Teoría M/M/1 y M/M/g (Erlang-C) vs simulación (error objetivo ≤10%).
  - Ley de Little usando λ̂ (llegadas observadas / tiempo observado), tolerancia ≤10%.
- M/M/g: para escenario B se fija λ = ρ_objetivo · g · μ (conserva ρ por servidor).

## 10. Checklist de entrega
- [ ] `outputs/results.csv` y `outputs/summary.csv`.
- [ ] `outputs/compare_results.csv` y `outputs/compare_summary.csv`.
- [ ] Figuras `comp_teo_sim.png`, `hist_wq.png`, `serie_L.png`.
- [ ] Figuras `mm1_mmg_barras.png`, `metricas_vs_g.png`, `pwait_vs_g.png`.
- [ ] Figura `decision_vs_g.png` y reporte `reports/analysis/conclusions.md`.
- [ ] Errores relativos y verificaciones de Little dentro de tolerancias.
- [ ] README y Agents.md actualizados.
- [ ] `pytest -q` exitoso.

## 11. Comandos rapidos
```
python -m src.run_sim --model mm1 --scenario B --replications 20 --seed 123 --warmup 10000 --horizon 200000
python -m src.run_sim --model mmg --g 3 --scenario B --replications 20 --seed 123 --warmup 10000 --horizon 200000
python -m src.compare --scenario B --g-list "1,2,3,4" --replications 20 --seed 123 --warmup 10000 --horizon 200000
python -m src.plots
pytest -q
```

## Limitaciones y extensiones
No se incluyeron M/M/1/K (pérdidas), prioridades ni tiempos no exponenciales (M/G/g). Para ρ→1 se requieren horizontes mayores; considerar batch means. Extensiones naturales: M/M/1/K con tasa de rechazo, M/G/g con aproximaciones (Kingman) y políticas de scheduling.
