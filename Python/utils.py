"""
Funciones propias para discretización, métricas, normalización, filtrado, correlación/MI y plots.
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def equal_width_discretize(series: pd.Series, n_bins: int) -> pd.Series:
    """
    Discretiza una serie numérica en n_bins de igual anchura.
    Entradas:
      - series: pd.Series con valores numéricos
      - n_bins: int, número de bins (>0)
    Salida:
      - pd.Series categórica con etiquetas "bin_0",..."bin_{n-1}"
    """
    if n_bins <= 0:
        raise ValueError("n_bins debe ser > 0")
    s = series.copy()
    mask_na = s.isna()
    mini, maxi = s.min(), s.max()
    if pd.isna(mini) or pd.isna(maxi) or mini == maxi:
        # todo mismo valor -> asignar único bin
        out = pd.Series(pd.Categorical(["bin_0"] * len(s)), index=s.index)
        out[mask_na] = pd.NA
        return out
    edges = np.linspace(mini, maxi, n_bins + 1)
    inds = np.digitize(s.fillna(maxi), edges, right=False) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    labels = [f"bin_{i}" for i in range(n_bins)]
    cat = pd.Categorical.from_codes(inds.astype(int), categories=labels)
    out = pd.Series(cat, index=s.index)
    out[mask_na] = pd.NA
    return out, edges

def equal_frequency_discretize(series: pd.Series, n_bins: int, eps = 1e-12) -> pd.Series:
    """
    Discretiza por igual frecuencia en n_bins.
    Usa cuantiles empíricos.
    Entradas:
      - series: pd.Series con valores numéricos
      - n_bins: int, número de bins (>0)
      - eps: epsilon
    Salida:
      - pd.Series categórica con etiquetas "bin_0",..."bin_{n-1}"
    """
    if n_bins <= 0:
        raise ValueError("n_bins debe ser > 0")
    s = series.copy()
    mask_na = s.isna()
    if s.dropna().empty:
        out = pd.Series(pd.Categorical([pd.NA] * len(s)), index=s.index)
        return out
  
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = s.dropna().quantile(quantiles).values
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + eps
    inds = np.digitize(s.fillna(edges[-1]), edges, right=True) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    labels = [f"bin_{i}" for i in range(n_bins)]
    cat = pd.Categorical.from_codes(inds.astype(int), categories=labels)
    out = pd.Series(cat, index=s.index)
    out[mask_na] = pd.NA
    return out

def variance(series: pd.Series) -> float:
    """
    Varianza muestral (ddof=1) ignorando NaN. Si 1 o 0 observaciones -> NaN.
    Entrada:
      - series: pd.Series con valores numéricos
    Salida:
      - float: la varianza de la variable
    """
    return series.var(ddof=1)

def entropy(series: pd.Series, base: float = 2.0) -> float:
    """
    Entropía de una variable discreta (en bits si base=2). Calcula usando frecuencias empíricas p_i
    Entradas:
      - series: pd.Series categórica o discreta
      - base: float (opcional, por defecto 2.0) — base del logaritmo para expresar la MI
    Salida:
      - float: la entropia de la variable
    """
    s = series.dropna()
    if s.empty:
        return 0.0
    counts = s.value_counts()
    ps = counts.values / counts.values.sum()
    ent = -np.sum([p * (np.log(p) / np.log(base)) for p in ps if p > 0])
    return float(ent)

def auc(x: pd.Series, y_true: pd.Series) -> float:
    """
    AUC (ROC) para un atributo numérico x frente a y_true binaria (0/1 or False/True).
    Entradas:
    - x: pd.Series variable numerica
    - y_true: pd.Series variable binaria
    Salida:
      - float: AUC entre 0 y 1
    """
    mask = (~x.isna()) & (~y_true.isna())
    xv = x[mask].values
    yv = y_true[mask].astype(int).values
    if xv.size == 0 or len(np.unique(yv)) == 1:
        return float("nan")
    # ordenar por score descendente
    order = np.argsort(-xv)
    y_sorted = yv[order]
    # calcular TPR y FPR acumulados
    P = np.sum(y_sorted == 1)
    N = np.sum(y_sorted == 0)
    if P == 0 or N == 0:
        return float("nan")
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    # recorrer empatados para manejo correcto
    # construir arrays únicos de score para calcular curvas
    scores_sorted = xv[order]
    # iterate through unique scores descending
    idx = 0
    while idx < len(scores_sorted):
        # segmentos con mismo score
        sc = scores_sorted[idx]
        j = idx
        while j < len(scores_sorted) and scores_sorted[j] == sc:
            if y_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr.append(tp / P)
        fpr.append(fp / N)
        idx = j
    # asegurar que empieza en (0,0) y termina (1,1)
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    # integracion trapezoidal de TPR wrt FPR
    auc = np.trapz(tpr, fpr)
    return float(auc)

def compute_metrics(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calcula la metrica para cada columna (excepto class_col si se da).
    Entradas:
        - X: pd.DataFrame
        - y: pd.Series. Variable target (binaria)
    Salida:
        - pd.DataFrame con siguientes columnas:
            - type: "numeric" o "categorical"
            - variance: varianza (solo numeric)
            - auc: AUC frente a class_col (solo numeric y si class_col existe)
            - entropy: entropía (solo categorical)
    """
    rows = []

    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            var = variance(s)
            a = np.nan           
            a = auc(s, y)
            ent = np.nan
            typ = "numeric"
        else:
            var = np.nan
            a = np.nan
            try:
                ent = entropy(s)
            except Exception:
                ent = np.nan
            typ = "categorical"

        rows.append({
            "attribute": col,
            "type": typ,
            "variance": var,
            "auc": a,
            "entropy": ent
        })

    return pd.DataFrame(rows).set_index("attribute")[["type", "variance", "auc", "entropy"]]

def normalize_series(series: pd.Series, new_min: float = 0.0, new_max: float = 1.0) -> pd.Series:
    """
    Normaliza (min-max) una serie numérica al rango [new_min, new_max].
    Entradas:
      - series: pd.Series con valores numéricos
      - new_min: float, nuevo mínimo del rango
      - new_max: float, nuevo máximo del rango
    Salida:
      - pd.Series numérica escalada al rango [new_min, new_max]
    """
    s = series.copy()
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError("normalize_series solo para datos numéricos")
    mask_na = s.isna()
    mini, maxi = s.min(), s.max()
    if pd.isna(mini) or mini == maxi:
        out = pd.Series([ (new_min+new_max)/2.0 ] * len(s), index=s.index)
        out[mask_na] = pd.NA
        return out
    scaled = ((s - mini) / (maxi - mini)) * (new_max - new_min) + new_min
    scaled[mask_na] = pd.NA
    return scaled

def standardize_series(series: pd.Series) -> pd.Series:
    """    
    Estandariza una serie numérica usando z-score: (x - mean) / sd (ddof=1).
    Entradas:
      - series: pd.Series con valores numéricos
    Salida:
      - pd.Series con valores estandarizados (media 0, desviación tipo 1 cuando sd>0)
    """
    s = series.copy()
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError("standardize_series solo para datos numéricos")
    mask_na = s.isna()
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        out = pd.Series([0.0] * len(s), index=s.index)
        out[mask_na] = pd.NA
        return out
    z = (s - mu) / sd
    z[mask_na] = pd.NA
    return z

def normalize_dataframe(df: pd.DataFrame, method: str = "normalize", **kwargs) -> pd.DataFrame:
    """    
    Aplica normalización/estandarización a las columnas numéricas de un DataFrame.
    Entradas:
      - df: pd.DataFrame original
      - method: str, ("normalize", "standardize")
      - kwargs: parámetros adicionales:
          * Si method == "normalize": puede pasarse new_min (float) y new_max (float)
            para definir el rango objetivo.
    Salida:
      - pd.DataFrame con las mismas columnas; las columnas numéricas transformadas,
        las no numéricas se dejan sin cambios.
    """
    out = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if method == "normalize":
                out[col] = normalize_series(df[col], kwargs.get("new_min", 0.0), kwargs.get("new_max", 1.0))
            elif method == "standardize":
                out[col] = standardize_series(df[col])
            else:
                raise ValueError("method must be 'normalize' or 'standardize'")
    return out

def filter_variables(df: pd.DataFrame, metrics_df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Filtra las columnas de un DataFrame en base a métricas precomputadas contenidas en metrics_df.
    Entradas:
      - df: pd.DataFrame original con las variables (columnas) a filtrar.
      - metrics_df: pd.DataFrame con métricas por atributo. Debe tener como índice
        los nombres de las variables (mismas cadenas que las columnas de df) y
        columnas con las métricas (por ejemplo: "variance", "entropy", "auc", ...).
      - rules: dict donde las claves son nombres de métricas (columnas de metrics_df)
        y los valores son funciones (callables) que reciben un valor de métrica y 
        devuelven True si la variable debe conservarse, False en caso contrario.
        Ejemplo: {"variance": lambda v: v > 0.01, "entropy": lambda e: e > 0.5}
    
    Salida:
      - pd.DataFrame que contiene únicamente las columnas de df que cumplen
        todas las reglas indicadas en rules.
    """
    keep = []
    for attr in metrics_df.index:
        met = metrics_df.loc[attr]
        ok = True
        for key, fn in rules.items():
            val = met.get(key, None)
            try:
                if not fn(val):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            keep.append(attr)
    return df.loc[:, df.columns.isin(keep)]

def mutual_information_discrete(x: pd.Series, y: pd.Series, base: float = 2.0) -> float:
    """
    Calcula la información mutua empírica entre dos series discretas (o categóricas).
    Entradas:
      - x: pd.Series con la primera variable (discreta / categórica). 
      - y: pd.Series con la segunda variable (discreta / categórica). 
      - base: float (opcional, por defecto 2.0) — base del logaritmo para expresar la MI
    Salida:
      - float: valor de la información mutua I(X;Y) calculada a partir de las frecuencias empíricas P(x,y). 
    """
    # ambos discretos (o categóricos)
    x = x.dropna()
    y = y.dropna()
    idx = x.index.intersection(y.index)
    if idx.empty:
        return 0.0
    x = x.loc[idx]
    y = y.loc[idx]
    joint = pd.crosstab(x, y, normalize=True)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in joint.index:
        for j in joint.columns:
            pxy = joint.loc[i, j]
            if pxy <= 0:
                continue
            mi += pxy * (math.log(pxy / (px[i] * py[j])) / math.log(base))
    return float(mi)

def pairwise_correlation_mi(df: pd.DataFrame, n_bins_for_continuous: int = 10) -> pd.DataFrame:
    """
    Calcula una matriz simétrica de relaciones por pares entre columnas de un DataFrame,
    usando correlación Pearson para numéricas y información mutua para categóricas.
    Entradas:
      - df: pd.DataFrame con las variables.
      - n_bins_for_continuous: int, número de bins a usar
    Salida:
      - pd.DataFrame cuadrado (len(columns) x len(columns)) con índices y columnas 
      iguales a los nombres de las variables de df. Cada celda (i,j) contiene:
        - Si ambas variables son numéricas: correlación de Pearson entre ellas.
        - Si ambas son categóricas: information mutual I(X;Y).
        - Si son mixtas: se discretiza la numérica en n_bins_for_continuous cuantiles y se calcula la MI.
    """
    cols = df.columns
    n = len(cols)
    M = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    types = {c: ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical") for c in cols}
    for i in range(n):
        for j in range(i, n):
            a = df[cols[i]]
            b = df[cols[j]]
            if types[cols[i]] == "numeric" and types[cols[j]] == "numeric":
                # Pearson corr (ignorar NA pares)
                mask = a.notna() & b.notna()
                if mask.sum() < 2:
                    val = np.nan
                else:
                    val = a[mask].corr(b[mask])
            elif types[cols[i]] == "categorical" and types[cols[j]] == "categorical":
                val = mutual_information_discrete(a.astype(str), b.astype(str))
            else:
                # mixto -> discretizar la numeric
                if types[cols[i]] == "numeric":
                    ad = equal_frequency_discretize(a, n_bins_for_continuous)
                    bd = b.astype(str)
                else:
                    ad = a.astype(str)
                    bd = equal_frequency_discretize(b, n_bins_for_continuous)
                val = mutual_information_discrete(ad, bd)
            M.iat[i, j] = val
            M.iat[j, i] = val
    return M

def plot_roc_curve(x: pd.Series, y_true: pd.Series, ax: Optional[plt.Axes] = None, show: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dibuja la curva ROC (FPR vs TPR) para un vector de puntuaciones y una etiqueta binaria.
    Entradas:
      - x: pd.Series con el score del atributo.
      - y_true: pd.Series variable binaria.
      - ax: matplotlib.axes.Axes opcional. Si se proporciona, se dibuja en ese eje; si es None, se crea una nueva figura y eje.
      - show: bool. Si True, llama a plt.show() tras dibujar; si False,
      no muestra la figura automáticamente.
    Salida:
      - Tuple[np.ndarray, np.ndarray]: dos arrays (fpr, tpr) con los puntos que forman la curva ROC,
      en el mismo orden en que se han trazado.
    """
    mask = (~x.isna()) & (~y_true.isna())
    xv = x[mask].values
    yv = y_true[mask].astype(int).values
    order = np.argsort(-xv)
    y_sorted = yv[order]
    P = np.sum(y_sorted == 1)
    N = np.sum(y_sorted == 0)
    if P == 0 or N == 0:
        raise ValueError("Necesita ejemplos de ambas clases para plot ROC")
    tpr = [0.0]
    fpr = [0.0]
    tp = 0; fp = 0
    idx = 0
    scores_sorted = xv[order]
    while idx < len(scores_sorted):
        sc = scores_sorted[idx]
        j = idx
        while j < len(scores_sorted) and scores_sorted[j] == sc:
            if y_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr.append(tp / P)
        fpr.append(fp / N)
        idx = j
    tpr = np.array(tpr); fpr = np.array(fpr)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve")
    if show:
        plt.show()
    return fpr, tpr

def plot_matrix_heatmap(M: pd.DataFrame, ax: Optional[plt.Axes] = None, cmap: Optional[str]=None, show: bool=True):
    """
    Dibuja un heatmap simple a partir de una matriz (pd.DataFrame).
    Entradas:
      - M: pd.DataFrame cuadrado con índices y columnas a usar como etiquetas. Cada celda
      contiene el valor a representar (por ejemplo: correlación o información mutua).
      - ax: matplotlib.axes.Axes opcional. Si se proporciona, se dibuja en ese eje; si es None,
      se crea una nueva figura y eje con tamaño por defecto.
      - cmap: str. Nombre del mapa de color a usar; si es None
      se emplea el colormap por defecto de matplotlib.
      - show: bool. Si True, llama a plt.show() tras dibujar; si False,
      no muestra la figura automáticamente.
    Salida:
      - matplotlib.axes.Axes: el eje donde se ha dibujado el heatmap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(M.values, aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=90)
    ax.set_yticks(np.arange(len(M.index)))
    ax.set_yticklabels(M.index)
    plt.colorbar(im, ax=ax)
    if show:
        plt.tight_layout()
        plt.show()
    return ax
































