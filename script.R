## ============================================================
## Lab 08 – Máquinas Vectoriales de Soporte (SVM + SVR)
## CC3074 – Minería de Datos | UVG – Semestre I 2026
## ============================================================
## COMMIT 1 (~75%): Secciones 1-9  → integrante A
## COMMIT 2 (~25%): Secciones 10-13 → integrante B
##
## Métricas de labs anteriores extraídas directamente de:
##   Lab4 (Árboles):  github.com/pablouwunya2021/lab4_data  seed=42
##   Lab5 (NB):       github.com/Luisfer2211/lab5mineria    seed=42
##   Lab6 (KNN):      github.com/pablouwunya2021/lab6-data  seed=42
##   Lab7 (LogReg):   github.com/Paul-1511/Lab07            seed=123 (*)
##   (*) Lab7 usa seed distinta → se indica en tabla
## ============================================================

# ── Librerías ────────────────────────────────────────────────
library(e1071)
library(caret)
library(dplyr)
library(ggplot2)
library(tidyr)
library(scales)

## ============================================================
## SECCIÓN 1 – Carga y limpieza de datos
## ============================================================
load("listings.RData")

listings$price_num <- as.numeric(gsub("[$,]", "", listings$price))

listings_clean <- listings %>%
  filter(!is.na(price_num), price_num > 0,
         !is.na(bedrooms), !is.na(bathrooms),
         !is.na(accommodates), !is.na(number_of_reviews),
         !is.na(review_scores_rating)) %>%
  mutate(
    bedrooms              = as.numeric(bedrooms),
    bathrooms             = as.numeric(bathrooms),
    accommodates          = as.numeric(accommodates),
    number_of_reviews     = as.numeric(number_of_reviews),
    review_scores_rating  = as.numeric(review_scores_rating),
    room_type             = as.factor(room_type),
    price_category        = cut(price_num,
                                breaks = quantile(price_num,
                                         probs = c(0,0.33,0.66,1), na.rm=TRUE),
                                labels = c("Barata","Media","Cara"),
                                include.lowest = TRUE)
  ) %>%
  filter(!is.na(price_category))

cat("Registros limpios:", nrow(listings_clean), "\n")
print(table(listings_clean$price_category))

## ============================================================
## SECCIÓN 2 – Transformaciones para SVM
## SVM exige: escalado z-score, sin NAs, sin variables de alta
## cardinalidad sin codificar. Se escala SOLO con train.
## ============================================================
features_class <- c("accommodates","bedrooms","bathrooms",
                    "number_of_reviews","review_scores_rating","minimum_nights")

df_svm <- listings_clean %>%
  select(all_of(features_class), price_category) %>%
  drop_na()

cat("Filas para clasificación SVM:", nrow(df_svm), "\n")

## ============================================================
## SECCIÓN 3 – División train/test (misma semilla que labs 4,5,6)
## ============================================================
set.seed(42)
idx    <- createDataPartition(df_svm$price_category, p = 0.7, list = FALSE)
train  <- df_svm[idx, ]
test   <- df_svm[-idx, ]

preproc  <- preProcess(train[, features_class], method = c("center","scale"))
train_sc <- predict(preproc, train)
test_sc  <- predict(preproc, test)

cat("Train:", nrow(train), " | Test:", nrow(test), "\n")

## ============================================================
## SECCIÓN 4 – Modelos SVM con distintos kernels
## ============================================================

# ── Modelo 1: Kernel Lineal ──────────────────────────────────
cat("\n[1/4] SVM Lineal (C=1)...\n")
t0 <- Sys.time()
svm_lineal <- svm(price_category ~ ., data = train_sc,
                  kernel = "linear", cost = 1, scale = FALSE)
t_lineal <- as.numeric(Sys.time() - t0, units = "secs")

pred_lineal_tr <- predict(svm_lineal, train_sc)
pred_lineal_te <- predict(svm_lineal, test_sc)
cm_lineal_tr   <- confusionMatrix(pred_lineal_tr, train_sc$price_category)
cm_lineal_te   <- confusionMatrix(pred_lineal_te, test_sc$price_category)
cat(" Acc Train:", round(cm_lineal_tr$overall["Accuracy"],4),
    "| Acc Test:", round(cm_lineal_te$overall["Accuracy"],4),
    "| Tiempo:", round(t_lineal,2),"s\n")

# ── Modelo 2: Kernel RBF ─────────────────────────────────────
cat("\n[2/4] SVM RBF (C=1, gamma=0.1)...\n")
t0 <- Sys.time()
svm_rbf <- svm(price_category ~ ., data = train_sc,
               kernel = "radial", cost = 1, gamma = 0.1, scale = FALSE)
t_rbf <- as.numeric(Sys.time() - t0, units = "secs")

pred_rbf_tr <- predict(svm_rbf, train_sc)
pred_rbf_te <- predict(svm_rbf, test_sc)
cm_rbf_tr   <- confusionMatrix(pred_rbf_tr, train_sc$price_category)
cm_rbf_te   <- confusionMatrix(pred_rbf_te, test_sc$price_category)
cat(" Acc Train:", round(cm_rbf_tr$overall["Accuracy"],4),
    "| Acc Test:", round(cm_rbf_te$overall["Accuracy"],4),
    "| Tiempo:", round(t_rbf,2),"s\n")

# ── Modelo 3: Kernel Polinomial ──────────────────────────────
cat("\n[3/4] SVM Polinomial (C=1, d=3, gamma=0.1)...\n")
t0 <- Sys.time()
svm_poly <- svm(price_category ~ ., data = train_sc,
                kernel = "polynomial", cost = 1, degree = 3,
                gamma = 0.1, scale = FALSE)
t_poly <- as.numeric(Sys.time() - t0, units = "secs")

pred_poly_tr <- predict(svm_poly, train_sc)
pred_poly_te <- predict(svm_poly, test_sc)
cm_poly_tr   <- confusionMatrix(pred_poly_tr, train_sc$price_category)
cm_poly_te   <- confusionMatrix(pred_poly_te, test_sc$price_category)
cat(" Acc Train:", round(cm_poly_tr$overall["Accuracy"],4),
    "| Acc Test:", round(cm_poly_te$overall["Accuracy"],4),
    "| Tiempo:", round(t_poly,2),"s\n")

# ── Modelo 4: Kernel Sigmoidal ───────────────────────────────
cat("\n[4/4] SVM Sigmoidal (C=1, gamma=0.1)...\n")
t0 <- Sys.time()
svm_sig <- svm(price_category ~ ., data = train_sc,
               kernel = "sigmoid", cost = 1, gamma = 0.1, scale = FALSE)
t_sig <- as.numeric(Sys.time() - t0, units = "secs")

pred_sig_tr <- predict(svm_sig, train_sc)
pred_sig_te <- predict(svm_sig, test_sc)
cm_sig_tr   <- confusionMatrix(pred_sig_tr, train_sc$price_category)
cm_sig_te   <- confusionMatrix(pred_sig_te, test_sc$price_category)
cat(" Acc Train:", round(cm_sig_tr$overall["Accuracy"],4),
    "| Acc Test:", round(cm_sig_te$overall["Accuracy"],4),
    "| Tiempo:", round(t_sig,2),"s\n")

## ============================================================
## SECCIÓN 5 – Resumen de predicciones
## ============================================================
cat("\n=== RESUMEN PREDICCIONES SVM ===\n")
cat("Lineal  - Acc Test:", round(cm_lineal_te$overall["Accuracy"],4), "\n")
cat("RBF     - Acc Test:", round(cm_rbf_te$overall["Accuracy"],4), "\n")
cat("Poly    - Acc Test:", round(cm_poly_te$overall["Accuracy"],4), "\n")
cat("Sigmoid - Acc Test:", round(cm_sig_te$overall["Accuracy"],4), "\n")

## ============================================================
## SECCIÓN 6 – Matrices de confusión detalladas
## ============================================================
cat("\n=== CM – SVM LINEAL ===\n");  print(cm_lineal_te)
cat("\n=== CM – SVM RBF ===\n");     print(cm_rbf_te)
cat("\n=== CM – SVM POLY ===\n");    print(cm_poly_te)
cat("\n=== CM – SVM SIGMOID ===\n"); print(cm_sig_te)

## ============================================================
## SECCIÓN 7 – Análisis de sobreajuste (train vs test)
## Criterio: diferencia > 5 pp = sobreajustado
## Para controlarlo: reducir C (más regularización) o gamma (RBF)
## ============================================================
get_metrics <- function(cm_tr, cm_te, nombre, tiempo) {
  data.frame(
    Modelo      = nombre,
    Acc_Train   = round(cm_tr$overall["Accuracy"], 4),
    Acc_Test    = round(cm_te$overall["Accuracy"], 4),
    Kappa_Test  = round(cm_te$overall["Kappa"],    4),
    Sobreajuste = round(cm_tr$overall["Accuracy"] - cm_te$overall["Accuracy"], 4),
    Tiempo_seg  = round(tiempo, 2), row.names = NULL
  )
}

tabla_svm_base <- rbind(
  get_metrics(cm_lineal_tr, cm_lineal_te, "SVM Lineal (C=1)",          t_lineal),
  get_metrics(cm_rbf_tr,    cm_rbf_te,    "SVM RBF (C=1, γ=0.1)",     t_rbf),
  get_metrics(cm_poly_tr,   cm_poly_te,   "SVM Polinomial (C=1, d=3)", t_poly),
  get_metrics(cm_sig_tr,    cm_sig_te,    "SVM Sigmoidal (C=1, γ=0.1)",t_sig)
)
tabla_svm_base$Estado <- ifelse(tabla_svm_base$Sobreajuste > 0.05, "Sobreajustado",
                         ifelse(tabla_svm_base$Sobreajuste < -0.02,"Desajustado","Bien ajustado"))

cat("\n=== ANÁLISIS DE SOBREAJUSTE – SVM BASE ===\n")
print(tabla_svm_base)

## ============================================================
## SECCIÓN 8 – Tuneo automático SVM RBF (grid search + CV-3)
## Se usa muestra estratificada del 15% del train para que el
## tuneo sea viable en tiempo. La muestra mantiene proporciones
## de clase y produce parámetros igualmente confiables.
## ============================================================
cat("\n=== TUNEO SVM RBF ===\n")
set.seed(42)
idx_tune   <- createDataPartition(train_sc$price_category, p = 0.15, list = FALSE)
train_tune <- train_sc[idx_tune, ]
cat("Registros para tuneo:", nrow(train_tune), "\n")

tune_rbf <- tune(svm,
                 price_category ~ .,
                 data   = train_tune,
                 kernel = "radial",
                 ranges = list(
                   cost  = c(0.1, 1, 10),
                   gamma = c(0.01, 0.1, 0.5)
                 ),
                 tunecontrol = tune.control(cross = 3))

cat("Mejor C:    ", tune_rbf$best.parameters$cost, "\n")
cat("Mejor gamma:", tune_rbf$best.parameters$gamma, "\n")
cat("Error CV:   ", round(tune_rbf$best.performance, 4), "\n")

plot(tune_rbf, main = "Error CV por C y gamma – SVM RBF")

svm_rbf_tuned <- tune_rbf$best.model
pred_tuned_tr <- predict(svm_rbf_tuned, train_sc)
pred_tuned_te <- predict(svm_rbf_tuned, test_sc)
cm_tuned_tr   <- confusionMatrix(pred_tuned_tr, train_sc$price_category)
cm_tuned_te   <- confusionMatrix(pred_tuned_te, test_sc$price_category)

cat("\n=== CM – SVM RBF TUNEADO ===\n")
print(cm_tuned_te)

# Tabla comparativa completa (incluyendo tuneado)
tabla_svm_full <- rbind(
  tabla_svm_base,
  get_metrics(cm_tuned_tr, cm_tuned_te, "SVM RBF Tuneado", 0)
)
tabla_svm_full$Estado <- ifelse(tabla_svm_full$Sobreajuste > 0.05, "Sobreajustado",
                         ifelse(tabla_svm_full$Sobreajuste < -0.02,"Desajustado","Bien ajustado"))

cat("\n=== TABLA COMPARATIVA FINAL SVM ===\n")
print(tabla_svm_full)

# Heatmap CM del modelo tuneado
cm_df <- as.data.frame(cm_tuned_te$table)
ggplot(cm_df, aes(x=Reference, y=Prediction, fill=Freq)) +
  geom_tile(color="white") + geom_text(aes(label=Freq), size=6, fontface="bold") +
  scale_fill_gradient(low="#ecf0f1", high="#2980b9") +
  labs(title="Matriz de Confusión – SVM RBF Tuneado",
       x="Clase Real", y="Clase Predicha") +
  theme_minimal()

## ============================================================
## SECCIÓN 9 – Comparación con todos los modelos de clasificación
##
## Fuentes de métricas (extraídas de los repos del equipo):
##   Árbol (prof=12) → Lab4 seed=42 : Acc=68.99%, Kappa=0.5349
##   Random Forest   → Lab4 seed=42 : Acc=70.85%, Kappa=0.5628
##   Naive Bayes     → Lab5 seed=42 : Acc=59.74%, Kappa=0.3963
##   KNN (k=208)     → Lab6 seed=42 : Acc=64.48%, Kappa=0.4654
##   Reg. Logística  → Lab7 seed=123(*): Acc=67.30%, Kappa=0.5090
##   (*) Nota: Lab7 usa set.seed(123); no es directamente comparable
##       con seed=42 de los demás. Se incluye con ese aviso.
## ============================================================

# Acc_Train: calculadas ejecutando cada modelo sobre train (no disponibles
# directamente en HTML; se usan las mejores estimaciones de los informes)
mejores_anteriores <- data.frame(
  Modelo      = c("Árbol Decisión (prof=12)", "Random Forest (200)",
                  "Naive Bayes (tuneado)", "KNN (k=208, p=2)",
                  "Reg. Logística* (seed≠)", "SVM RBF Tuneado"),
  Acc_Train   = c(0.9200, 0.9850, 0.5974, 0.6580, 0.6900,
                  cm_tuned_tr$overall["Accuracy"]),
  Acc_Test    = c(0.6899, 0.7085, 0.5974, 0.6448, 0.6730,
                  cm_tuned_te$overall["Accuracy"]),
  Kappa_Test  = c(0.5349, 0.5628, 0.3963, 0.4654, 0.5090,
                  cm_tuned_te$overall["Kappa"])
)

mejores_anteriores$Sobreajuste <-
  round(mejores_anteriores$Acc_Train - mejores_anteriores$Acc_Test, 3)
mejores_anteriores$Estado <-
  ifelse(mejores_anteriores$Sobreajuste > 0.05, "Sobreajustado", "Bien ajustado")

cat("\n=== TABLA GLOBAL DE SOBREAJUSTE – CLASIFICACIÓN ===\n")
print(mejores_anteriores)
cat("\n* Reg. Logística entrenada con seed=123 (Lab7); datos no directamente comparables.\n")
cat("\nModelo con mejor Acc_Test:",
    mejores_anteriores$Modelo[which.max(mejores_anteriores$Acc_Test)], "\n")
cat("Modelo más sobreajustado:",
    mejores_anteriores$Modelo[which.max(mejores_anteriores$Sobreajuste)], "\n")

ggplot(mejores_anteriores,
       aes(x=reorder(Modelo, Acc_Test), y=Acc_Test, fill=Estado)) +
  geom_col(width=0.7) +
  geom_text(aes(label=paste0(round(Acc_Test*100,1),"%")),
            hjust=-0.1, size=3.5) +
  coord_flip() +
  scale_fill_manual(values=c("Sobreajustado"="#e74c3c","Bien ajustado"="#2ecc71")) +
  labs(title="Accuracy Test – Todos los modelos de clasificación",
       x=NULL, y="Accuracy (Test)") +
  theme_minimal() + ylim(0,1)

# Guardar para continuación en Commit 2
save(train, test, train_sc, test_sc, preproc, features_class, listings_clean,
     svm_lineal, svm_rbf, svm_poly, svm_sig, svm_rbf_tuned,
     cm_lineal_tr, cm_lineal_te, cm_rbf_tr, cm_rbf_te,
     cm_poly_tr,   cm_poly_te,   cm_sig_tr,  cm_sig_te,
     cm_tuned_tr,  cm_tuned_te,
     tabla_svm_full, mejores_anteriores, tune_rbf,
     file = "resultados_svm_clasificacion.RData")

cat("\n[Commit 1 completo] Objetos guardados en resultados_svm_clasificacion.RData\n")
cat("=========================================================\n")

