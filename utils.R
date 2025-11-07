
# --- DISCRETIZAR ---

equal_width_discretize <- function(vec, n_bins) {
    # Discretiza una serie numérica en n_bins de igual anchura.
    # Entradas:
    #   - vec: numeric vector con los valores
    #   - n_bins: entero > 0, número de bins
    # Salida:
    #   - factor con etiquetas "bin_0", ..."bin_{n-1}"

  if (!is.numeric(vec)) stop("vec debe ser numérico")
  if (n_bins <= 0) stop("n_bins debe ser > 0")
  # si todo NA
  if (all(is.na(vec))) return(factor(rep(NA, length(vec))))
  mini <- min(vec, na.rm = TRUE)
  maxi <- max(vec, na.rm = TRUE)
  # caso valores constantes o min/max inválidos
  if (is.na(mini) || is.na(maxi) || mini == maxi) {
    out <- factor(rep("bin_0", length(vec)))
    out[is.na(vec)] <- NA
    return(out)
  }
  # construir cortes de igual anchura
  edges <- seq(mini, maxi, length.out = n_bins + 1)
  # findInterval devuelve índices 1..n_bins (con all.inside=TRUE se recorta)
  inds <- findInterval(vec, edges, rightmost.closed = FALSE, all.inside = TRUE)
  # mapear a etiquetas "bin_0".."bin_{n-1}"
  labels <- paste0("bin_", 0:(n_bins-1))
  res <- factor(labels[inds], levels = labels)
  res[is.na(vec)] <- NA
  return(res)
}

equal_frequency_discretize <- function(vec, n_bins, eps = 1e-12) {
    # Discretiza por igual frecuencia en n_bins.
    # Entradas:
    #   - vec: numeric vector con los valores
    #   - n_bins: entero > 0, número de bins
    #   - eps: epsilon pequeño para garantizar monotonicidad en los cortes (por defecto 1e-12)
    # Salida:
    #   - factor con etiquetas "bin_0", ..."bin_{n-1}"
  if (!is.numeric(vec)) stop("vec debe ser numérico")
  if (n_bins <= 0) stop("n_bins debe ser > 0")
  if (all(is.na(vec))) return(factor(rep(NA, length(vec))))

  probs <- seq(0, 1, length.out = n_bins + 1)
  edges <- stats::quantile(vec, probs = probs, na.rm = TRUE, type = 7)

  # Asegurar monotonicidad estricta en edges (evitar cortes iguales)
  for (i in 2:length(edges)) {
    if (edges[i] <= edges[i - 1]) edges[i] <- edges[i - 1] + eps
  }

  labels <- paste0("bin_", 0:(n_bins - 1))
  # cut con intervals right-closed
  res <- cut(vec, breaks = edges, include.lowest = TRUE, right = TRUE, labels = labels)
  # preservar NA originales
  res[is.na(vec)] <- NA
  return(res)
}

# --- METRICAS ---

variance <- function(vec) {
    # Varianza muestral.
    # Entradas:
    #   - vec: numeric vector con valores numéricos
    # Salida:
    #   - numeric: varianza muestral calculada sobre las observaciones.
  if (!is.numeric(vec)) stop("vec debe ser numérico")
  s <- vec[!is.na(vec)]
  if (length(s) <= 1) return(NA_real_)
  return(var(s, na.rm = TRUE))
}

entropy <- function(vec, base = 2.0) {
    # Entropía de una variable discreta.
    # Entradas:
    #   - vec: vector categórico o discreto (factor o character)
    #   - base: numeric (opcional, por defecto 2.0) — base del logaritmo
    # Salida:
    #   - numeric: valor de la entropía.
  s <- vec[!is.na(vec)]
  if (length(s) == 0) return(0.0)
  counts <- table(s)
  ps <- as.numeric(counts) / sum(counts)
  ps <- ps[ps > 0]
  ent <- -sum(ps * (log(ps) / log(base)))
  return(ent)
}

auc <- function(x, y_true) {
    # Calcula el AUC (ROC) para un atributo numérico x frente a y_true binaria
    # Entradas:
    #   - x: vector numérico
    #   - y_true: vector binario
    # Salida:
    #   - numeric: valor del AUC entre 0 y 1
  mask <- !is.na(x) & !is.na(y_true)
  xv <- x[mask]
  yv <- as.integer(y_true[mask])
  
  if (length(xv) == 0 || length(unique(yv)) == 1) {
    return(NA_real_)
  }
  # Ordenar por score descendente
  order_idx <- order(-xv)
  y_sorted <- yv[order_idx]
  scores_sorted <- xv[order_idx]
  
  P <- sum(y_sorted == 1)
  N <- sum(y_sorted == 0)
  if (P == 0 || N == 0) {
    return(NA_real_)
  }
  
  tp <- 0
  fp <- 0
  tpr <- c(0.0)
  fpr <- c(0.0)
  
  idx <- 1
  while (idx <= length(scores_sorted)) {
    sc <- scores_sorted[idx]
    j <- idx
    while (j <= length(scores_sorted) && scores_sorted[j] == sc) {
      if (y_sorted[j] == 1) {
        tp <- tp + 1
      } else {
        fp <- fp + 1
      }
      j <- j + 1
    }
    tpr <- c(tpr, tp / P)
    fpr <- c(fpr, fp / N)
    idx <- j
  }
  
  # Integración trapezoidal
  auc_value <- sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
  return(as.numeric(auc_value))
}

compute_metrics <- function(df, class_col = NULL) {
    # Calcula métricas por atributo de un data.frame
    # Entradas:
    #   - df: data.frame con las variables.
    #   - class_col: character (opcional). Nombre de la columna target binaria.
    # Salida:
    #   - data.frame con una fila por atributo (excepto class_col) y columnas:
    #       * type: "numeric" o "categorical"
    #       * variance: varianza muestral (NA si no aplica)
    #       * auc: AUC frente a class_col (NA si no aplica o no se puede calcular)
    #       * entropy: entropía para categóricas (NA si no aplica)
  if (!is.data.frame(df)) stop("df debe ser un data.frame")
  has_class <- !is.null(class_col) && class_col %in% colnames(df)
  
  # comprobar columna clase válida (al menos dos clases no-NA)
  if (has_class) {
    y <- df[[class_col]]
    y_non_na <- y[!is.na(y)]
    if (length(unique(y_non_na)) < 2) {
      warning("class_col no tiene al menos dos clases distintas. Se ignorará AUC.")
      has_class <- FALSE
    }
  }
  
  attrs <- setdiff(colnames(df), if (has_class) class_col else character(0))
  res_list <- vector("list", length(attrs))
  
  for (i in seq_along(attrs)) {
    col_name <- attrs[i]
    s <- df[[col_name]]
    if (is.numeric(s)) {
      # numérico
      var_val <- tryCatch(variance(s), error = function(e) NA_real_)
      auc_val <- NA_real_
      if (has_class) {
        auc_val <- tryCatch(auc(s, df[[class_col]]), error = function(e) NA_real_)
      }
      ent_val <- NA_real_
      typ <- "numeric"
    } else {
      # categórico
      var_val <- NA_real_
      auc_val <- NA_real_
      ent_val <- tryCatch(entropy(s), error = function(e) NA_real_)
      typ <- "categorical"
    }
    res_list[[i]] <- data.frame(
      attribute = col_name,
      type = typ,
      variance = as.numeric(var_val),
      auc = as.numeric(auc_val),
      entropy = as.numeric(ent_val),
      stringsAsFactors = FALSE
    )
  }
  
  res_df <- do.call(rbind, res_list)
  rownames(res_df) <- res_df$attribute
  res_df$attribute <- NULL
  # asegurar orden de columnas
  res_df <- res_df[, c("type", "variance", "auc", "entropy")]
  return(res_df)
}

# --- NORMALIZAR ----

normalize_series <- function(series, new_min = 0.0, new_max = 1.0) {
    # Normaliza (min-max) una serie numérica al rango [new_min, new_max].
    # Entradas:
    #   - series: vector numérico
    #   - new_min: numérico, nuevo mínimo del rango
    #   - new_max: numérico, nuevo máximo del rango
    # Salida:
    #   - vector numérico escalado al rango [new_min, new_max]
  if (!is.numeric(series)) stop("normalize_series solo para datos numéricos")
  s <- series
  mask_na <- is.na(s)
  mini <- min(s, na.rm = TRUE)
  maxi <- max(s, na.rm = TRUE)
  if (is.na(mini) || mini == maxi) {
    out <- rep((new_min + new_max) / 2, length(s))
    out[mask_na] <- NA
    return(out)
  }
  scaled <- ((s - mini) / (maxi - mini)) * (new_max - new_min) + new_min
  scaled[mask_na] <- NA
  return(scaled)
}

standardize_series <- function(series) {
    # Estandariza una serie numérica usando z-score.
    # Entradas:
    #   - series: vector numérico
    # Salida:
    #   - vector numérico estandarizado
  if (!is.numeric(series)) stop("standardize_series solo para datos numéricos")
  s <- series
  mask_na <- is.na(s)
  mu <- mean(s, na.rm = TRUE)
  sd_val <- sd(s, na.rm = TRUE)
  if (is.na(sd_val) || sd_val == 0) {
    out <- rep(0.0, length(s))
    out[mask_na] <- NA
    return(out)
  }
  z <- (s - mu) / sd_val
  z[mask_na] <- NA
  return(z)
}

normalize_dataframe <- function(df, method = "normalize", new_min = 0, new_max = 1) {
  # Aplica normalización o estandarización a las columnas numéricas de un data.frame.
  # Entradas:
  #   - df: data.frame original
  #   - method: "normalize" o "standardize"
  #   - new_min, new_max: usados solo si method == "normalize"
  # Salida:
  #   - data.frame con las columnas numéricas transformadas; las no numéricas se dejan igual
  
  out <- df
  
  for (col in names(df)) {
    if (is.numeric(df[[col]])) {
      if (method == "normalize") {
        out[[col]] <- normalize_series(df[[col]], new_min = new_min, new_max = new_max)
      } else if (method == "standardize") {
        out[[col]] <- standardize_series(df[[col]])
      } else {
        stop("El argumento 'method' debe ser 'normalize' o 'standardize'")
      }
    }
  }
  
  return(out)
}


# --- FLITRAR ----

filter_variables <- function(df, metrics_df, rules) {
    # Filtra las columnas de un data.frame en base a métricas precomputadas contenidas en metrics_df.
    # Entradas:
    #   - df: data.frame original con las variables a filtrar.
    #   - metrics_df: data.frame con métricas por atributo. Debe tener como rownames
    #                 los nombres de las variables (mismas cadenas que las columnas de df)
    #                 y columnas con las métricas (por ejemplo: "variance", "entropy", "auc", ...).
    #   - rules: lista nombrada donde las claves son nombres de métricas (columnas de metrics_df)
    #            y los valores son funciones que reciben un valor de métrica y 
    #            devuelven TRUE si la variable debe conservarse, FALSE en caso contrario.
    #            Ejemplo: list(variance = function(v) v > 0.01, entropy = function(e) e > 0.5)
    # Salida:
    #   - data.frame que contiene únicamente las columnas de df que cumplen
    #     todas las reglas indicadas en rules.
  keep <- c()
  for (attr in rownames(metrics_df)) {
    met <- metrics_df[attr, , drop = FALSE]
    ok <- TRUE
    for (key in names(rules)) {
      val <- met[[key]]
      fn <- rules[[key]]
      res <- tryCatch(fn(val), error = function(e) FALSE)
      if (isFALSE(res)) {
        ok <- FALSE
        break
      }
    }
    if (ok) {
      keep <- c(keep, attr)
    }
  }
  return(df[, intersect(colnames(df), keep), drop = FALSE])
}

# --- CORRELACIÓN ---

mutual_information_discrete <- function(x, y, base = 2.0) {
    # Calcula la información mutua empírica entre dos series discretas (o categóricas).
    # Entradas:
    #   - x: vector (factor/character) con la primera variable
    #   - y: vector (factor/character) con la segunda variable
    #   - base: numeric (opcional, por defecto 2) — base del logaritmo para la MI
    # Salida:
    #   - numeric: valor de la información mutua I(X;Y) calculada a partir de las frecuencias empíricas.
  # Alinear y descartar pares con NA
  ok <- !is.na(x) & !is.na(y)
  x2 <- x[ok]
  y2 <- y[ok]
  if (length(x2) == 0) return(0.0)
  
  # Convertir a character para evitar problemas con factores con diferentes niveles
  x2c <- as.character(x2)
  y2c <- as.character(y2)
  
  joint_tab <- table(x2c, y2c)
  joint_prop <- prop.table(joint_tab) # p(x,y)
  if (sum(joint_prop) <= 0) return(0.0)
  
  px <- rowSums(joint_prop) # p(x)
  py <- colSums(joint_prop) # p(y)
  
  mi <- 0.0
  rn <- rownames(joint_prop)
  cn <- colnames(joint_prop)
  for (i in seq_along(rn)) {
    for (j in seq_along(cn)) {
      pxy <- joint_prop[i, j]
      if (pxy <= 0) next
      pi <- px[rn[i]]
      pj <- py[cn[j]]
      mi <- mi + pxy * (log(pxy / (pi * pj)) / log(base))
    }
  }
  return(as.numeric(mi))
}

pairwise_correlation_mi <- function(df, n_bins_for_continuous = 10) {
    # Calcula una matriz simétrica de relaciones por pares entre columnas de un data.frame,
    # usando correlación Pearson para numéricas y MI para categóricas. Para pares mixtos
    # discretiza la numérica en n_bins_for_continuous cuantiles.
    # Entradas:
    #   - df: data.frame con las variables (columnas)
    #   - n_bins_for_continuous: entero número de bins para discretizar continuas
    # Salida:
    #   - data.frame cuadrado con la matriz de correlaciones
  if (!is.data.frame(df)) stop("df debe ser un data.frame")
  cols <- colnames(df)
  n <- length(cols)
  M <- matrix(NA_real_, nrow = n, ncol = n)
  rownames(M) <- cols
  colnames(M) <- cols
  
  # detectar tipo: numeric -> "numeric", lo demás -> "categorical"
  types <- sapply(df, function(col) if (is.numeric(col)) "numeric" else "categorical")
  
  for (i in seq_len(n)) {
    for (j in i:n) {
      a <- df[[cols[i]]]
      b <- df[[cols[j]]]
      val <- NA_real_
      
      # both numeric -> Pearson (usar pares no-NA)
      if (types[cols[i]] == "numeric" && types[cols[j]] == "numeric") {
        mask <- !is.na(a) & !is.na(b)
        if (sum(mask) < 2) {
          val <- NA_real_
        } else {
          val <- tryCatch(cor(a[mask], b[mask], use = "complete.obs"), error = function(e) NA_real_)
        }
      
      # both categorical -> mutual information
      } else if (types[cols[i]] == "categorical" && types[cols[j]] == "categorical") {
        val <- tryCatch(mutual_information_discrete(as.character(a), as.character(b)), error = function(e) NA_real_)
      
      # mixed -> discretizar la numérica y calcular MI
      } else {
        # identificar cuál es la numérica
        if (types[cols[i]] == "numeric") {
          num <- a
          cat <- b
        } else {
          num <- b
          cat <- a
        }
        # discretizar num en cuantiles usando equal_frequency_discretize (si disponible)
        disc <- tryCatch({
          equal_frequency_discretize(num, n_bins_for_continuous)
        }, error = function(e) {
          # fallback: cortes de igual anchura
          if (all(is.na(num))) {
            factor(rep(NA, length(num)))
          } else {
            edges <- seq(min(num, na.rm = TRUE), max(num, na.rm = TRUE), length.out = n_bins_for_continuous + 1)
            cut(num, breaks = edges, include.lowest = TRUE, right = TRUE, labels = FALSE)
          }
        })
        # asegurar que disc sea tipo factor/character para MI
        disc_char <- as.character(disc)
        cat_char <- as.character(cat)
        val <- tryCatch(mutual_information_discrete(disc_char, cat_char), error = function(e) NA_real_)
      }
      
      M[i, j] <- val
      M[j, i] <- val
    }
  }
  
  # devolver como data.frame para facilitar su uso y etiquetado
  M_df <- as.data.frame(M, stringsAsFactors = FALSE)
  rownames(M_df) <- cols
  colnames(M_df) <- cols
  return(M_df)
}

# --- PLOT ---

plot_roc_curve <- function(x, y_true, show = TRUE) {
    # Dibuja la curva ROC (FPR vs TPR) para un vector de puntuaciones y una etiqueta binaria.
    # Entradas:
    #   - x: vector numérico con el score del atributo 
    #   - y_true: vector binario 
    #   - show: logical (por defecto TRUE). Si TRUE dibuja la curva; si FALSE solo la calcula.
    # Salida:
    #   - lista con dos vectores: $fpr y $tpr (en el mismo orden que se trazan)
  if (length(x) != length(y_true)) stop("x y y_true deben tener la misma longitud")
  # Alinear y remover NA
  ok <- !is.na(x) & !is.na(y_true)
  xv <- as.numeric(x[ok])
  yv <- as.integer(y_true[ok])
  if (length(xv) == 0 || length(unique(yv)) == 1) {
    stop("Necesita ejemplos de ambas clases y al menos una observación válida")
  }
  # ordenar por score descendente
  ord <- order(-xv)
  y_sorted <- yv[ord]
  scores_sorted <- xv[ord]
  P <- sum(y_sorted == 1)
  N <- sum(y_sorted == 0)
  if (P == 0 || N == 0) stop("Necesita ejemplos de ambas clases para plot ROC")
  tp <- 0L
  fp <- 0L
  tpr <- c(0)
  fpr <- c(0)
  idx <- 1L
  while (idx <= length(scores_sorted)) {
    sc <- scores_sorted[idx]
    j <- idx
    while (j <= length(scores_sorted) && scores_sorted[j] == sc) {
      if (y_sorted[j] == 1L) tp <- tp + 1L else fp <- fp + 1L
      j <- j + 1L
    }
    tpr <- c(tpr, tp / P)
    fpr <- c(fpr, fp / N)
    idx <- j
  }
  tpr <- as.numeric(tpr)
  fpr <- as.numeric(fpr)
  if (show) {
    # plot step-like curve
    plot(fpr, tpr, type = "s", xlab = "FPR", ylab = "TPR",
         main = "ROC curve", xlim = c(0,1), ylim = c(0,1))
    abline(0, 1, lty = 2) # diagonal de referencia
    grid()
  }
  return(list(fpr = fpr, tpr = tpr))
}

plot_matrix_heatmap <- function(M, show = TRUE, palette_fn = heat.colors) {
    # Dibuja un heatmap simple a partir de una matriz (data.frame o matrix).
    # Entradas:
    #   - M: data.frame o matrix cuadrada con índices y columnas (nombres usados como etiquetas)
    #   - show: logical (por defecto TRUE). Si TRUE dibuja el heatmap; si FALSE solo prepara y retorna info.
    #   - palette_fn: función para generar la paleta de colores. Por defecto usa heat.colors.
    # Salida:
    #   - lista con elementos: $matrix (matriz numérica usada para dibujar), $rownames, $colnames, 
    #   y (si show TRUE) invisibly devuelve NULL.
  if (is.data.frame(M)) {
    mat <- as.matrix(M)
  } else if (is.matrix(M)) {
    mat <- M
  } else {
    stop("M debe ser un data.frame o una matrix")
  }
  # asegurar matriz numérica
  mat_num <- matrix(as.numeric(mat), nrow = nrow(mat), ncol = ncol(mat))
  rown <- rownames(mat)
  coln <- colnames(mat)
  # si no hay nombres, crear índices
  if (is.null(rown)) rown <- as.character(seq_len(nrow(mat)))
  if (is.null(coln)) coln <- as.character(seq_len(ncol(mat)))
  # invertir el orden de filas para que la primera fila aparezca arriba (coincide con convención)
  mat_plot <- mat_num[nrow(mat_num):1, , drop = FALSE]
  # rango de valores para la paleta
  finite_vals <- mat_num[is.finite(mat_num)]
  if (length(finite_vals) == 0) {
    zlim <- c(0, 1)
  } else {
    zlim <- range(finite_vals, na.rm = TRUE)
  }
  if (show) {
    ncols <- 100
    pal <- palette_fn(ncols)
    # image requires x/y coordinates; usamos seq
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)
    # crear layout para heatmap + barra de color a la derecha
    layout(matrix(c(1,2), ncol = 2), widths = c(4,1))
    par(mar = c(5, 4, 4, 1)) # margen para heatmap
    image(1:ncol(mat_plot), 1:nrow(mat_plot), t(mat_plot), xaxt = "n", yaxt = "n",
          col = pal, zlim = zlim, xlab = "", ylab = "")
    axis(1, at = seq_len(ncol(mat_plot)), labels = coln, las = 2)
    axis(2, at = seq_len(nrow(mat_plot)), labels = rev(rown), las = 2)
    title(main = "Heatmap")
    # colorbar
    par(mar = c(5, 1, 4, 3))
    zseq <- seq(zlim[1], zlim[2], length.out = ncols)
    image(1, zseq, matrix(zseq, nrow = 1), col = pal, axes = FALSE, xlab = "")
    axis(4, at = pretty(zseq), labels = pretty(zseq))
    layout(1) # restaurar layout
  }
  return(list(matrix = mat_num, rownames = rown, colnames = coln))
}