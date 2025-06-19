# TFM — World Models en índices bursátiles intradía

Repositorio de **código, notebooks y scripts** para mi Trabajo de Fin de Máster.  
Los datos Parquet **no** se versionan: se accede a ellos mediante el enlace simbólico  
`data/raw → ~/datasets/data/fin_parquet`.

---

## Estructura del proyecto
```
TFM/
├── articulos/          PDFs de referencia  (fuera de Git)
├── data/
│   ├── raw/            ← link a Parquet originales
│   └── prepared/       tensores .npy generados
├── notebooks/          exploración y experimentos
├── scripts/            pipelines (train_wm.py, eval_policies.py…)
├── models/             checkpoints de WM y políticas
├── runs/               TensorBoard / métricas csv
├── environment.yml     entorno Conda reproducible
└── README.md           este archivo
```

---

## Reproducir el entorno
```bash
conda env create -f environment.yml
conda activate tfm
jupyter notebook          # abre los cuadernos con el kernel "Python (tfm)"
```

---

## Flujo de trabajo

| Paso | Descripción | Cuaderno / script |
|------|-------------|-------------------|
| 0 | **exploración** de datos | `notebooks/00_exploracion.ipynb` |
| 1 | **preparar ventanas** 128 × F | `notebooks/01_prepara_windows.ipynb` |
| 2 | **pre-entrenar World Model** | `scripts/train_wm.py` |
| 3 | **entrenar Dreamer-Fin** (actor-critic) | `scripts/train_dreamer.py` |
| 4 | **model-free PPO** baseline | `scripts/train_ppo.py` |
| 5 | **evaluar políticas** (Sharpe, Sortino, MaxDD) | `scripts/eval_policies.py` |
| 6 | **figuras y tablas** para la memoria | `notebooks/05_report_figures.ipynb` |

Cada etapa escribe sus artefactos en `models/` o `runs/` y puede
reanudar desde el último checkpoint.

---

> **Licencia:** uso estrictamente académico.
