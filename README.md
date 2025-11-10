# Simulación de Colas M/M/1 (Eventos Discretos)

## 1. Introducción
Este proyecto implementa y valida una simulación de eventos discretos para un sistema de colas M/M/1 bajo una disciplina FIFO. Se contrasta el comportamiento empírico con las fórmulas teóricas, se reportan estadísticas por replicación, y se generan visualizaciones y tablas listas para informes académicos.

## 2. Objetivos
- Construir un motor de simulación reproducible para escenarios M/M/1 con warm‑up configurable.
- Calcular métricas teóricas y empíricas (L, Lq, W, Wq, utilización) y verificar la ley de Little.
- Exponer un CLI para configurar parámetros y exportar resultados en CSV.
- Generar figuras comparativas en formato PNG para el informe.
- Proveer documentación y pruebas automatizadas de soporte.

## 3. Requisitos
- Python 3.11 o superior.
- Librerías: `simpy`, `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`, `pytest` (listadas en `requirements.txt`).
- Sistema operativo: Windows, macOS o Linux (WSL válido).

## 4. Instalación
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1            # PowerShell (Windows)
# ó source .venv/bin/activate           # Bash (Linux/Mac)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Estructura Principal
```
src/
 ├─ sim/
 │   ├─ mm1_core.py     # Motor de eventos y métricas empíricas
 │   ├─ metrics.py      # Fórmulas teóricas M/M/1
 │   └─ scenarios.py    # Escenarios predefinidos (A, B, C)
 ├─ run_sim.py          # CLI para correr replicaciones y guardar CSV
 └─ plots.py            # Generación de figuras en reports/
tests/
 ├─ test_metrics.py     # Validación de las fórmulas teóricas
 └─ test_mm1_core.py    # Prueba de Ley de Little (tolerancia 10 %)
outputs/                # CSV de resultados (se versionan .gitkeep)
reports/                # PNG generados (se versionan .gitkeep)
```

## 6. Ejecución Principal
Escenario recomendado (B, ρ≈0.85) siguiendo el addendum v1.2:
```bash
python -m src.run_sim --scenario B \
    --replications 20 \
    --seed 123 \
    --warmup 10000 \
    --horizon 200000
```
Genera:
- `outputs/results.csv`: métricas por replicación (`L`, `Lq`, `W_mean`, `Wq_mean`, `utilization`, `obs_time`, `arrivals_obs`, `lambda_hat`, `rho_theory`, `little_*_error`, etc.).
- `outputs/summary.csv`: tabla de promedios vs. teoría, errores relativos e intervalos de confianza 95 % (half-width absoluto y relativo) para las métricas clave (`L`, `W_mean`, etc.).
- Consola con: teoría vs. promedios, diagnósticos de la ventana `[warmup, horizon]`, validación de Little y half-width de IC95 para `L` y `W_mean`.

Parámetros alternos (`--lam`, `--mu`) pueden sustituir `--scenario`. Solo ajustar warm-up / horizonte según la estabilidad deseada.

## 7. Visualizaciones
```bash
python -m src.plots
```
Produce en `reports/`:
- `comp_teo_sim.png`: barras teoría vs. simulación con barras de error IC95.
- `hist_wq.png`: histograma de `Wq_mean` (bins automáticos).
- `serie_L.png`: serie de `L` promedio por replicación.

## 8. Metodología de medición
Las métricas temporales (`L`, `Lq`, utilización) se integran únicamente sobre la ventana `[warmup, horizon]`. Se registra `lambda_hat = arrivals_obs / (horizon - warmup)` para comprobar la ley de Little: `L ≈ lambda_hat * W` y `Lq ≈ lambda_hat * Wq`. El CSV incluye los errores relativos resultantes.

## 9. Pruebas
```bash
pytest -q
```
Confirma:
- Consistencia de las fórmulas analíticas (`tests/test_metrics.py`).
- Verificación de la ley de Little con tolerancia del 10 % en una corrida corta (`tests/test_mm1_core.py`).

## 10. Checklist de entrega
- [x] `outputs/results.csv` y `outputs/summary.csv` generados para el escenario oficial.
- [x] Figuras actualizadas (`reports/comp_teo_sim.png`, `reports/hist_wq.png`, `reports/serie_L.png`).
- [x] Error relativo promedio < 10 % para `L` y `W`, utilización ≈ ρ (±3 %), Little < 10 %.
- [x] README documentando comandos, metodología y resultados.
- [x] Pruebas (`pytest -q`) exitosas.

## 11. Referencias
- Allen, A. O. “Probability, Statistics, and Queueing Theory”. Academic Press.
- Banks, J. et al. “Discrete-Event System Simulation”. Pearson.
