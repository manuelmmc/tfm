# Modelos del Mundo y su Aplicación en Entornos de Series Temporales Complejas

Este repositorio contiene la implementación oficial del Trabajo de Fin de Máster (TFM) sobre la aplicación de **World Models (Modelos del Mundo)** y arquitecturas **RSSM (Recurrent State-Space Models)** para la toma de decisiones (trading algorítmico) en series temporales financieras intradía.

📄 **Autor:** Manuel Moya Martín-Castaño  
🎓 **Máster:** Investigación en Inteligencia Artificial (UIMP / AEPIA)  
📍 **Tutores:** Sebastián Ventura y Antonio Moya

---

## 🚀 Resumen del Proyecto

El objetivo de este proyecto es adaptar el paradigma de los **World Models** (típicamente usado en robótica y videojuegos) a series temporales financieras estocásticas y ruidosas. Se investiga si aprender una **dinámica latente** del entorno permite tomar mejores decisiones de inversión que los métodos predictivos tradicionales.

### Conceptos Clave
* **RSSM (Recurrent State-Space Model):** Una arquitectura que descompone el estado en una parte determinista (memoria GRU) y una estocástica (variables latentes), permitiendo modelar la incertidumbre.
* **Imaginación Latente:** Capacidad del modelo para simular ("soñar") trayectorias futuras posibles sin interactuar con el mercado real, entrenando al agente sobre estas simulaciones.
* **Entrenamiento End-to-End:** Optimización conjunta de la representación (VAE/AE), la dinámica y la política de control.

---

## 🧠 Arquitecturas Implementadas

El repositorio incluye implementaciones en **PyTorch** de las siguientes estrategias:

1.  **Baselines (Reglas):** Buy & Hold, Momentum/Contrarian, Cruce de Medias, Volatility Targeting.
2.  **Modelos sin World Model:**
    * Controlador Directo (Transformer/MLP sobre la ventana causal).
    * Clasificador como Política (Señales discretas de trading).
3.  **World Models Deterministas:**
    * **AE + CLS + Controller:** Autoencoder secuencial + Clasificador direccional + Política continua.
    * Comparativa entre entrenamiento modular (fases) vs. conjunto (joint).
4.  **World Models Estocásticos (RSSM):**
    * Implementación completa de RSSM adaptado a series 1D.
    * Entrenamiento con y sin **Imaginación Latente** (rollouts del prior).

---

## 📂 Estructura del Repositorio

La estructura recomendada para organizar los scripts (actualmente en la raíz) es la siguiente:

```text
src/
├── data/           # Generación de series sintéticas (MSAR, GARCH, Hawkes) y preprocesamiento.
├── models/         # Scripts de entrenamiento de las distintas arquitecturas (Memoria, Controlador, RSSM).
└── evaluation/     # Scripts de evaluación de políticas y cálculo de métricas (Sharpe, P&L).
```


## 📊 Resultados Destacados

Los experimentos realizados sobre 8 conjuntos de datos (6 sintéticos y 2 reales: SPX, BTC) mostraron que:

* 🚀 **Superioridad de WM:** Las arquitecturas basadas en World Models superan consistentemente a los baselines de reglas y a los controladores directos.
* ⚖️ **Memoria Ponderada:** El uso de clasificación ponderada para la memoria direccional mejora el Sharpe Ratio frente a la regresión directa.
* 🔮 **Imaginación:** La "imaginación latente" aporta valor en entornos con dinámicas estables (como la familia de "motivos" o BTC), aunque su efectividad disminuye ante cambios de régimen bruscos.

---

## 📜 Referencias

Este trabajo se inspira en World Models (Ha & Schmidhuber, 2018) y Dream to Control (Hafner et al., 2019), adaptándolos al dominio financiero.

> **Nota:** Este código es parte de un trabajo de investigación académica para el Máster en Investigación en IA.
