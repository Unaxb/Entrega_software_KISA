# Paquete para procesamiento y métricas de variables (R / Python)

Este repositorio contiene implementaciones propias (R y Python) de funciones útiles para trabajar con datasets: discretización, métricas por atributo (varianza, entropía, AUC), normalización/estandarizado, filtrado por métricas, información mutua por pares, matriz de relaciones y funciones de visualización (ROC y heatmap). También incluye ejemplos/scripts para probar las funciones con datasets que se cargan desde librerías (sin descargar CSV).

Contenido principal del proyecto

utils.py — Implementación Python de las funciones (discretización, variance, entropy, auc, compute_metrics, normalize_series, standardize_series, normalize_dataframe, filter_variables, mutual_information_discrete, pairwise_correlation_mi, plot_roc_curve, plot_matrix_heatmap, ...).

utils.R — Implementación equivalente en R de las funciones.


test.ipynb / test.Rmd — Ejemplos de uso de las funciones.
