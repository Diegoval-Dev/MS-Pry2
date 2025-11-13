# Agents.md — Proyecto de Simulación de Colas M/M/1 (Eventos Discretos)

> Documento operativo para que **Codex** (o agente local) implemente, ejecute y valide el proyecto **end‑to‑end**. Todo está atomizado: objetivos, alcance, requisitos, estructura, tareas, criterios de aceptación, pruebas, métricas, y extensiones.

---

## 1) Objetivo General

Implementar y validar una **simulación de un sistema de colas M/M/1** (llegadas Poisson con tasa λ, tiempos de servicio exponenciales con tasa μ y un servidor, disciplina FIFO) usando **simulación de eventos discretos**. El sistema debe:

* Generar resultados reproducibles.
* Calcular métricas empíricas (L, Lq, W, Wq, utilización) y compararlas contra las fórmulas teóricas.
* Exportar resultados (CSV) y figuras (PNG) para el informe.

## 2) Objetivos Específicos (atomizados)

1. **Core de simulación**: construir el motor de eventos discretos con warm‑up y horizonte total.
2. **Métricas teóricas**: implementar fórmulas M/M/1 y validaciones de estabilidad (ρ < 1).
3. **CLI reproducible**: exponer parámetros clave (λ, μ, semillas, replicaciones, tiempos) vía línea de comandos.
4. **Persistencia**: guardar resultados crudos en `outputs/` y figuras en `reports/`.
5. **Validación**: contrastar promedios simulados vs. teoría (error relativo ≤ 10%).
6. **Documentación**: organizar código modular y legible con docstrings y README breve.

## 3) Alcance

* **Incluido**: M/M/1 con buffer infinito, política FIFO, una estación de servicio, llegada/servicio estocásticos, warm‑up. Escenarios con diferentes ρ. Gráficas básicas.
* **Excluido**: prioridades, colas en red, patrones de llegada/servicio no exponenciales.
* **Extensión opcional**: M/M/1/K (pérdidas) o M/M/2 (dos servidores), sección 12.

## 4) Requisitos Técnicos

* **Lenguaje**: Python 3.11+
* **Librerías**: `simpy`, `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`, `pytest`.
* **Sistema**: Windows/macOS/Linux (WSL válido). Sin dependencias del sistema fuera de las citadas.
* **Estilo**: líneas ≤ 100 columnas; docstrings concisos; sin emojis en código/comentarios.

### 4.1 Instalación

```bash
python -m venv .venv
source .venv/bin/activate     # Windows PowerShell: . .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5) Estructura de Repositorio (obligatoria)

```
mm1-simulation/
├─ src/
│  ├─ sim/
│  │  ├─ mm1_core.py          # lógica de eventos (SimPy) y recolección de métricas
│  │  ├─ metrics.py            # fórmulas M/M/1 (ρ, L, Lq, W, Wq) y validaciones
│  │  └─ scenarios.py          # definiciones de escenarios (A, B, C) y utilidades
│  ├─ run_sim.py               # CLI: corre replicaciones y guarda outputs/results.csv
│  └─ plots.py                 # genera figuras en reports/
├─ tests/
│  └─ test_metrics.py          # pruebas unitarias de consistencia teórica
├─ notebooks/                  # opcional, exploración
│  └─ 00_exploracion.ipynb
├─ outputs/                    # resultados crudos (CSV/NPY)
├─ reports/                    # figuras y tablas publicables
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 6) Parámetros del Sistema y Terminología

* **λ (lambda)**: tasa de llegada (eventos por unidad de tiempo). Interarribos ~ Exp(λ).
* **μ (mu)**: tasa de servicio. Duración de servicio ~ Exp(μ).
* **ρ (rho)**: intensidad de tráfico = λ/μ. Para estabilidad se requiere ρ < 1.
* **Warm‑up**: período inicial a descartar para evitar sesgo transiente.
* **Horizonte**: tiempo total de simulación (incluye warm‑up).
* **Replicación**: corrida independiente con distinta semilla; se promedian resultados.

## 7) Funcionalidades esperadas (end‑to‑end)

1. **Simulación de eventos discretos** con procesos de llegada y servicio.
2. **Cálculo on‑the‑fly** de áreas bajo la curva para L (nº en sistema) y Lq (nº en cola).
3. **Muestreo de tiempos** de espera (Wq) y de permanencia en el sistema (W).
4. **Cómputo de utilización** del servidor (tiempo ocupado / horizonte).
5. **Control de warm‑up**: excluir métricas previas a `warmup`.
6. **CLI** para configurar λ, μ, warm‑up, horizonte, replicaciones y semilla base.
7. **Persistencia** en `outputs/results.csv` con columnas estándar.
8. **Comparación teórica** impresa en consola y utilidades en `metrics.py`.
9. **Gráficas** en `reports/` (barras teoría vs. simulación, histogramas, series si aplica).

## 8) Interfaz de Línea de Comandos (CLI)

Ejemplo:

```bash
python -m src.run_sim \
  --lam 0.85 --mu 1.0 \
  --replications 10 \
  --seed 123 \
  --warmup 1000 \
  --horizon 50000
```

**Salida esperada**:

* `outputs/results.csv` con columnas: `L, Lq, utilization, W_mean, Wq_mean, n_samples`.
* Impresión en consola de la teoría (`ρ, L, Lq, W, Wq`) y promedios simulados.

## 9) Detalle de Implementación (tareas del agente)

### 9.1 `src/sim/mm1_core.py`

**Responsables**: motor de eventos, estado del servidor y la cola, mediciones continuas.

* Crear clase `MM1Params` con `lam, mu, seed, warmup, horizon`.
* Crear clase `MM1System` con:

  * Estado: `server_busy`, `queue` (FIFO), contadores `in_system`, `in_queue`.
  * Medidas continuas: `last_event_time`, `area_L`, `area_Lq`, `busy_time`.
  * Muestras: `wait_times`, `system_times`.
  * Generadores de tiempos: `exp(rate)` con `numpy.random.Generator`.
* Procesos:

  * `arrival_process()`: genera interarribos ~ Exp(λ); en llegada, si servidor libre inicia servicio; si ocupado, encola.
  * `_departure(arrival_time, service_time)`: completa servicio de quien entró directo.
  * `_serve_from_queue(arrival_time, service_time)`: atiende siguiente cliente de cola y registra `wait`.
* Acumuladores:

  * `_update_areas()`: integrar `area_L` y `area_Lq` con rectángulos (estado × Δt).
* Finalización y retorno:

  * Función `run_mm1(params)` que orquesta SimPy, corre hasta `horizon` y devuelve dict con métricas.

### 9.2 `src/sim/metrics.py`

* Función `rho(lam, mu)`.
* Función `mm1_theory(lam, mu)` que:

  * Verifica `rho < 1` (si no, lanzar excepción o mensaje claro).
  * Calcula: `L = ρ/(1−ρ)`, `Lq = ρ²/(1−ρ)`, `W = L/λ`, `Wq = Lq/λ`.
* Utilidades de error relativo (opcional) para validación.

### 9.3 `src/sim/scenarios.py`

* Definir escenarios estandarizados con distintas ρ:

  * A: ρ≈0.60 (p. ej., λ=0.6, μ=1.0)
  * B: ρ≈0.85 (p. ej., λ=0.85, μ=1.0)
  * C: ρ≈0.95 (p. ej., λ=0.95, μ=1.0)
* Funciones:

  * `list_scenarios() -> list[str]`
  * `get_params(name: str, seed: int, warmup: float, horizon: float) -> MM1Params`

### 9.4 `src/run_sim.py`

* Parseo de argumentos: `--lam`, `--mu`, `--seed`, `--warmup`, `--horizon`, `--replications`.
* Bucle de replicaciones: variar `seed+k`.
* Acumular dicts de `run_mm1` en `pandas.DataFrame` y **guardar CSV** en `outputs/`.
* Imprimir teoría (`mm1_theory`) y promedios simulados.

### 9.5 `src/plots.py`

* Cargar `outputs/results.csv`.
* Generar al menos:

  1. **Barra comparativa** teoría vs. simulación (L, Lq, W, Wq).
  2. **Histograma** de `Wq` (si se exportan muestras) o barras de `Wq_mean`.
  3. **Serie temporal** de longitud de cola/sistema (si se guarda traza de estados; opcional).
* Guardar en `reports/` con nombres claros: `comp_teo_sim.png`, `hist_Wq.png`, `serie_L.png`.

## 10) Criterios de Aceptación (definidos y medibles)

1. **Reproducibilidad**: misma semilla base ⇒ mismos promedios (variaciones menores por rounding).
2. **Validez**: para un caso estable (ρ<1), el error relativo de `L` y `W` entre simulación y teoría ≤ **10%**.
3. **Persistencia**: al ejecutar la CLI se genera `outputs/results.csv` con columnas esperadas.
4. **Gráficas**: `python -m src.plots` produce al menos dos figuras PNG en `reports/` sin errores.
5. **Pruebas**: `pytest -q` pasa con éxito los tests del módulo de métricas.

## 11) Plan de Pruebas

### 11.1 Pruebas unitarias (`tests/test_metrics.py`)

* Verificar que `rho(lam, mu)` devuelva λ/μ correctamente.
* Para un caso con ρ=0.5 (λ=0.5, μ=1.0):

  * `L = ρ/(1−ρ) = 1`, `W = L/λ = 2` y que `L ≈ λ·W` (Ley de Little).
* Comprobar que se lanza excepción si ρ≥1.

### 11.2 Pruebas de humo

* Correr `run_sim.py` con `--horizon` pequeño (p. ej. 2000) y verificar que crea `outputs/results.csv`.

### 11.3 Validación empírica vs. teoría

* Para ρ=0.85: 10 replicaciones; comparar `L`, `W` simulados vs. `mm1_theory` (≤10%).

## 12) Extensiones Opcionales (si hay tiempo)

* **M/M/1/K**: límite K de buffer; registrar tasa de bloqueo (pérdidas) y comparar con teoría.
* **M/M/2**: dos servidores; calcular métricas teóricas correspondientes y comparar.
* **Trazas**: exportar series de `L(t)` y `Lq(t)` con timestamp para análisis temporal fino.

## 13) Entregables

* **Código** fuente completo bajo `src/` y **tests** en `tests/`.
* **CSV** en `outputs/results.csv` con resumen de replicaciones.
* **Figuras** en `reports/`.
* **README.md** con instrucciones mínimas de ejecución.

## 14) Convenciones y Estilo

* Nombres descriptivos; comentarios breves y precisos.
* Docstrings en funciones públicas (qué hace, entradas, salidas, supuestos).
* No introducir dependencias no listadas; no hardcodear rutas fuera de `outputs/` y `reports/`.

## 15) Métricas y Cálculo esperado

* `L` (promedio en sistema) = `area_L / horizon`.
* `Lq` (promedio en cola) = `area_Lq / horizon`.
* `Utilization` = `busy_time / horizon`.
* `W_mean` = promedio de tiempos en sistema de clientes **post‑warm‑up**.
* `Wq_mean` = promedio de tiempos en cola **post‑warm‑up**.
* **Little**: verificar `L ≈ λ·W` y `Lq ≈ λ·Wq` en resultados medios.

## 16) Flujo de Trabajo recomendado

1. Instalar dependencias.
2. Implementar `metrics.py` y pruebas unitarias.
3. Implementar `mm1_core.py` y ejecutar un caso pequeño.
4. Implementar `run_sim.py` y guardar CSV.
5. Implementar `plots.py` y generar figuras.
6. Validar escenarios A/B/C; ajustar horizonte/warm‑up si el error > 10%.

## 17) Comandos Clave (resumen)

```bash
# Instalar
pip install -r requirements.txt

# Correr base (ρ≈0.85)
python -m src.run_sim --lam 0.85 --mu 1.0 --replications 10 --warmup 1000 --horizon 50000

# Graficar
python -m src.plots

# Probar
pytest -q
```

## 18) Checklist de Cierre

* [ ] `outputs/results.csv` presente y con columnas correctas.
* [ ] `reports/comp_teo_sim.png` y al menos otra figura generada.
* [ ] Error relativo ≤ 10% en `L` y `W` para caso estable documentado.
* [ ] Tests de métricas pasan (`pytest -q`).
* [ ] README actualizado con instrucciones para replicar.

---

## 19) Addendum v2 — Comparación M/M/1 vs M/M/g

1. **Nuevos módulos**:
   * `src/sim/mmg_core.py`: simulación M/M/g con `simpy.Resource(capacity=g)` aplicando la ventana `[warmup, horizon]`, registro de `Pwait_sim`, áreas `L/Lq` y utilización promedio por servidor.
   * `src/sim/metrics_mmg.py`: fórmulas Erlang-C (`P0`, `Pwait`, `L`, `Lq`, `W`, `Wq`, `ρ`) con validaciones (`g ≥ 1`, `ρ < 1`).
   * `tests/test_metrics_mmg.py`: pruebas unitarias (caso `g=1`, caso `g=2` con `ρ=0.85`, excepciones).
2. **CLI (`src/run_sim.py`)**:
   * Nuevas banderas `--model {mm1,mmg}` (por defecto `mm1`) y `--g INT`.
   * Si se usa `--scenario` con `--model mmg`, ajustar `λ = rho_target * g * μ`.
   * Imprimir siempre `obs_time`, `arrivals_obs`, `lambda_hat`, `rho_theory`, y para M/M/g añadir `g` y `Pwait_sim`.
   * CSV por replicación debe contener `model`, `g`, `Pwait_sim`, `Pwait_theory`, `obs_time`, `arrivals_obs`, `lambda_hat`, `rho_theory`.
3. **Script de comparación (`src/compare.py`)**:
   * Ejecuta barridos `g ∈ {1,2,3,4}` (configurable con `--g-list`) generando `outputs/compare_results.csv` y `outputs/compare_summary.csv`.
   * Ajusta `λ` según escenario (mantener `ρ` por servidor); si no hay escenario, usa `λ` constante.
   * Genera figuras en `reports/`:
     - `mm1_mmg_barras.png` (barras con IC95 para `L, Lq, W, Wq` vs `g`).
     - `metricas_vs_g.png` (líneas teoría vs simulación para `L, Lq, W, Wq, Pwait`).
     - `pwait_vs_g.png` (probabilidad de espera).
4. **Criterios de aceptación adicionales**:
   * Para `g=1`, `mmg_theory` coincide con `mm1_theory` (error ≤ 1%).
   * Para todo `g` estable: error relativo medio ≤10% en `L` y `W`, utilización dentro del 3% de `ρ`, `Pwait_sim` dentro de ±10% absoluto de `Pwait_teo`.
   * Ley de Little usando `lambda_hat`: `|L - lambda_hat·W|/L ≤ 10%` y `|Lq - lambda_hat·Wq|/Lq ≤ 10%`.
5. **Checklist extendido**:
   * [ ] `outputs/compare_results.csv` y `outputs/compare_summary.csv`.
   * [ ] Figuras `mm1_mmg_barras.png`, `metricas_vs_g.png`, `pwait_vs_g.png`.
   * [ ] README documenta `--model`, `--g`, `src.compare` y los comandos de ejemplo.

---

> **Nota**: Si se habilita la extensión M/M/1/K o M/M/2, duplicar esta estructura en nuevos módulos (`mm1k_core.py`, `mm2_core.py`) y extender `plots.py` con gráficos y comparativas adicionales.

---

## Addendum v1.1 — Correcciones tras primera corrida (ρ=0.85, warmup=1000, horizon=5000)

### 1) Diagnóstico de las discrepancias observadas

* **Errores relativos altos (≈20–27%)** en L, Lq, W, Wq.
* **Causas probables**:

  1. **Promedios de tiempo (L, Lq, utilización) calculados sobre TODO el horizonte**, incluyendo el **warm‑up**. La teoría asume **régimen estacionario**; mezclar transiente + estacionario sesga al alza.
  2. **Utilización/áreas acumuladas sin recortar** al inicio del período de observación.
  3. **Horizonte corto para ρ=0.85** (colas largas → alta varianza). 5 000 unidades es insuficiente para estabilizar promedios.
  4. **Muestreo inconsistente**: W y Wq se calculan post‑warm‑up (bien), pero L y Lq por tiempo no estaban recortados al mismo intervalo.

### 2) Cambios obligatorios en implementación (ajustes al core)

**Objetivo**: medir todas las métricas **solo en la ventana [warmup, horizon]**.

1. **Integración de áreas condicionada a la ventana de observación**

   * En `_update_areas()` calcular `dt_eff = max(0, min(now, horizon) - max(last_event_time, warmup))`.
   * Acumular `area_L += in_system * dt_eff` y `area_Lq += in_queue * dt_eff`.
   * Actualizar siempre `last_event_time = now`.

2. **Cómputo de utilización post‑warm‑up**

   * Mantener `busy_start` y, en cada cambio de estado del servidor, sumar **solo** la intersección del intervalo ocupado con `[warmup, horizon]`:
     `busy_time += max(0, min(now, horizon) - max(busy_start, warmup))`.

3. **Muestreo de clientes**

   * Conservar la regla: **solo registrar** `wait_times` y `system_times` de clientes con `arrival_time ≥ warmup`.
   * (Opcional) Contar llegadas efectivas post‑warm‑up `arrivals_obs` para estimar `λ̂ = arrivals_obs / (horizon - warmup)` y verificar **Little**: `L ≈ λ̂·W`, `Lq ≈ λ̂·Wq`.

4. **Cálculo final de promedios**

   * Usar `obs_time = horizon - warmup` para: `L = area_L / obs_time`, `Lq = area_Lq / obs_time`, `utilization = busy_time / obs_time`.

5. **Corrección menor de cola/servicio**

   * Eliminar no‑ops como `self.wait_times[-1] = self.wait_times[-1]`.
   * Asegurar que, cuando la cola pasa de vacía a no vacía, se fije correctamente `busy_start = now` (para no subestimar/duplicar ocupación).

### 3) Parámetros y políticas de corrida recomendados

* **Warm‑up sugerido**: `warmup ∈ [5 000, 20 000]` (mayor para ρ cercano a 1).
* **Horizonte mínimo** por escenario (base):

  * A (ρ≈0.60): `horizon ≥ 50 000`
  * B (ρ≈0.85): `horizon ≥ 200 000`
  * C (ρ≈0.95): `horizon ≥ 500 000`
* **Replications**: 20 (mínimo 10).
* **Semillas**: `seed + k` para k∈[0,replications).
* **Regla de parada por precisión (opcional)**: detener cuando el **IC 95%** del estimador de `W` y `L` tenga **half‑width ≤ 5%** del valor medio.

### 4) Nuevos criterios de aceptación (sustituyen/ajustan la Sección 10)

1. Cálculo de **L, Lq, utilización** usando **solo** `obs_time = horizon - warmup`.
2. Para ρ<1 y con horizontes sugeridos:

   * **Error relativo medio ≤ 10%** en `L` y `W` (respecto a teoría M/M/1).
   * **Utilización ≈ ρ** con diferencia ≤ 3%.
3. **Consistencia de Little**: `|L - λ̂·W|/L ≤ 10%` y `|Lq - λ̂·Wq|/Lq ≤ 10%`.

### 5) Cambios en CLI y trazabilidad

* Añadir flag `--scenario {A,B,C}` (mantener `--lam/--mu` para override).
* Imprimir `obs_time`, `arrivals_obs`, `λ̂` y chequeos de Little.
* Guardar en CSV columnas extra: `obs_time, arrivals_obs, lambda_hat, rho_theory`.

### 6) Plan de re‑ejecución recomendado

1. Refactor del core según 2.1–2.4.
2. Re‑correr **Escenario B** con: `--replications 20 --warmup 10000 --horizon 200000`.
3. Verificar criterios de aceptación (Sección 4) y regenerar figuras.

### 7) Impacto esperado

* Los promedios deberán bajar y alinearse con la teoría; la **utilización** convergerá hacia **ρ**; los errores relativos se mantendrán ≤ 10%.

### 8) Notas de validación adicional

* Si los errores persisten, aumentar `horizon` y/o aplicar **batch means** (particionar la serie temporal de L en 20–30 bloques para IC por bloques).
* Para ρ≥0.95 la varianza crece sustancialmente; usar horizontes más largos y warm‑up extendido.

---

## Addendum v1.2 — Validación lograda y cierre de especificación

**Resultado**: Para ρ=0.85 con `replications=20`, `warmup=10000`, `horizon=200000`, se obtuvieron errores relativos ≤ 0.65% en L, Lq, W y Wq; utilización ≈ ρ con 0.036% de diferencia; Little verificado (0.004%). **Criterios de aceptación cumplidos con holgura.**

### A) Parámetros “oficiales” del proyecto

* Escenario B (ρ≈0.85): `--replications 20 --seed 123 --warmup 10000 --horizon 200000`.
* Mantener la misma configuración para reproducibilidad en el informe y la demo.

### B) Requisitos finales añadidos

1. **Intervalos de confianza** (IC95) para L y W por replicación (media ± 1.96·s/√n). Reportar half‑width relativo (%) y confirmar ≤10%.
2. **CSV ampliado**: asegurarse de incluir `obs_time, arrivals_obs, lambda_hat, rho_theory, L, Lq, W_mean, Wq_mean, utilization` por replicación, más medias/IC en un archivo resumen.
3. **Figuras oficiales** (guardar en `reports/`):

   * `comp_teo_sim.png` (barras teoría vs sim con etiquetas de error, si aplica).
   * `hist_Wq.png` (histograma de Wq_mean por replicación con bins=10–20).
   * `serie_L.png` (L promedio por replicación; opcional añadir IC como barras).
4. **Sección de metodología**: documentar explícitamente la ventana de observación `[warmup, horizon]` y el cálculo de `λ̂`.
5. **Control de calidad**: `pytest -q` debe pasar; añadir un test de Little tolerancia 10% sobre datos de simulación corta.

### C) Opcional (para subir nota)

* **Batch means**: estimar IC usando 20–30 bloques en una corrida larga única y comparar con IC por replicaciones.
* **Escenarios A y C**: repetir protocolo y reportar cómo crece el half‑width cuando ρ→1.
* **Extensión M/M/1/K**: añadir tasa de rechazo y comparar con teoría.

### D) Checklist de entrega (actualizado)

* [ ] `outputs/results.csv` por replicación y `outputs/summary.csv` con medias e IC.
* [ ] `reports/comp_teo_sim.png`, `reports/hist_Wq.png`, `reports/serie_L.png` regenerados con parámetros oficiales.
* [ ] Informe con: fenómeno, teoría, método (ventana), resultados (tablas/figuras/IC), validación (errores ≤10%), discusión y conclusiones.
* [ ] README con comandos exactos para reproducibilidad.
