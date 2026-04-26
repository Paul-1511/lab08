## ========================================================
## LABORATORIO 8 - SVM (Actividades 1 a 7) - VERSIÓN OPTIMIZADA
## ========================================================

library(caret)
library(e1071)
library(ggplot2)

# ====================== ACTIVIDAD 1 ======================
load("listings.RData")

listings$price <- as.numeric(gsub("[$,]", "", listings$price))

if (!"precio_cat" %in% names(listings)) {
  q33 <- quantile(listings$price, 0.33, na.rm = TRUE)
  q66 <- quantile(listings$price, 0.66, na.rm = TRUE)
  
  listings$precio_cat <- cut(listings$price,
                             breaks = c(-Inf, q33, q66, Inf),
                             labels = c("económico", "medio", "caro"),
                             include.lowest = TRUE)
}

set.seed(123)
index <- createDataPartition(listings$precio_cat, p = 0.7, list = FALSE)
trainData <- listings[index, ]
testData  <- listings[-index, ]

# ====================== ACTIVIDAD 2 ======================
predictors <- c("room_type", "accommodates", "bedrooms", "bathrooms", 
                "review_scores_rating", "number_of_reviews", 
                "host_is_superhost", "instant_bookable",
                "neighbourhood_cleansed", "property_type")

vars_modelo <- c("precio_cat", predictors)

trainData <- trainData[complete.cases(trainData[, vars_modelo]), ]
testData  <- testData[complete.cases(testData[, vars_modelo]), ]

# Convertir factores
trainData$precio_cat <- factor(trainData$precio_cat)
testData$precio_cat  <- factor(testData$precio_cat)

# Limpiar niveles raros (muy importante para SVM)
for (col in c("neighbourhood_cleansed", "property_type", "room_type")) {
  trainData[[col]] <- as.factor(trainData[[col]])
  testData[[col]]  <- as.factor(testData[[col]])
  
  levels(trainData[[col]]) <- make.names(levels(trainData[[col]]))
  levels(testData[[col]])  <- make.names(levels(testData[[col]]))
}

cat("→ Datos listos para SVM\n")

# REDUCCIÓN DE TAMAÑO (clave para que no se congele)
set.seed(123)
trainData <- trainData[sample(nrow(trainData), min(5000, nrow(trainData))), ]

cat("→ Submuestra aplicada:", nrow(trainData), "filas\n")

# ====================== ACTIVIDAD 3 ======================
# Variable respuesta: precio_cat

# ====================== ACTIVIDAD 4 ======================
ctrl <- trainControl(method = "cv", number = 5)

set.seed(123)

cat("\nEntrenando SVM LINEAL...\n")
svm_linear <- train(precio_cat ~ .,
                    data = trainData[, c("precio_cat", predictors)],
                    method = "svmLinear",
                    trControl = ctrl,
                    tuneLength = 3)

cat("\nEntrenando SVM RADIAL...\n")
svm_radial <- train(precio_cat ~ .,
                    data = trainData[, c("precio_cat", predictors)],
                    method = "svmRadial",
                    trControl = ctrl,
                    tuneLength = 3)

cat("\nEntrenando SVM POLINOMIAL...\n")
svm_poly <- train(precio_cat ~ .,
                  data = trainData[, c("precio_cat", predictors)],
                  method = "svmPoly",
                  trControl = ctrl,
                  tuneLength = 3)

cat("\n=== RESULTADOS MODELOS ===\n")
print(svm_linear)
print(svm_radial)
print(svm_poly)

# ====================== ACTIVIDAD 5 ======================
cat("\nGenerando predicciones...\n")

pred_linear <- predict(svm_linear, testData)
pred_radial <- predict(svm_radial, testData)
pred_poly   <- predict(svm_poly, testData)

# ====================== ACTIVIDAD 6 ======================
cat("\n=== MATRIZ LINEAL ===\n")
cm_linear <- confusionMatrix(pred_linear, testData$precio_cat)
print(cm_linear)

cat("\n=== MATRIZ RADIAL ===\n")
cm_radial <- confusionMatrix(pred_radial, testData$precio_cat)
print(cm_radial)

cat("\n=== MATRIZ POLINOMIAL ===\n")
cm_poly <- confusionMatrix(pred_poly, testData$precio_cat)
print(cm_poly)

# ====================== ACTIVIDAD 7 ======================
cat("\nCalculando errores...\n")

train_pred_linear <- predict(svm_linear, trainData)
train_pred_radial <- predict(svm_radial, trainData)
train_pred_poly   <- predict(svm_poly, trainData)

error_train_linear <- 1 - mean(train_pred_linear == trainData$precio_cat)
error_test_linear  <- 1 - cm_linear$overall["Accuracy"]

error_train_radial <- 1 - mean(train_pred_radial == trainData$precio_cat)
error_test_radial  <- 1 - cm_radial$overall["Accuracy"]

error_train_poly <- 1 - mean(train_pred_poly == trainData$precio_cat)
error_test_poly  <- 1 - cm_poly$overall["Accuracy"]

cat("\n=== ERRORES ===\n")
cat("Linear -> Train:", round(error_train_linear,4), " Test:", round(error_test_linear,4), "\n")
cat("Radial -> Train:", round(error_train_radial,4), " Test:", round(error_test_radial,4), "\n")
cat("Poly   -> Train:", round(error_train_poly,4), " Test:", round(error_test_poly,4), "\n")

cat("\n=== ANÁLISIS SOBREAJUSTE ===\n")
cat("Train << Test → Overfitting\n")
cat("Ambos altos → Underfitting\n")
cat("Similares → Buen ajuste\n")