# ============================================================
# NETFLIX STOCK PRICE PREDICTION AND FORECASTING USING ML IN R
# ============================================================

# ============================================================
# 1. INSTALL AND LOAD REQUIRED PACKAGES
# ============================================================

packages <- c(
  "quantmod",
  "TTR",
  "dplyr",
  "ggplot2",
  "randomForest",
  "xgboost",
  "Metrics"
)

installed <- rownames(installed.packages())

for (p in packages) {
  if (!(p %in% installed)) {
    install.packages(p)
  }
}

library(quantmod)
library(TTR)
library(dplyr)
library(ggplot2)
library(randomForest)
library(xgboost)
library(Metrics)


# ============================================================
# 2. DOWNLOAD NETFLIX STOCK DATA
# ============================================================

getSymbols(
  "NFLX",
  src = "yahoo",
  from = "2015-01-01",
  to = Sys.Date(),
  auto.assign = TRUE
)

data <- data.frame(
  Date = index(NFLX),
  Open = as.numeric(Op(NFLX)),
  High = as.numeric(Hi(NFLX)),
  Low = as.numeric(Lo(NFLX)),
  Close = as.numeric(Cl(NFLX)),
  Volume = as.numeric(Vo(NFLX)),
  Adjusted = as.numeric(Ad(NFLX))
)

data <- na.omit(data)


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

print(head(data))
print(tail(data))
print(dim(data))
print(summary(data))

cat("\nMissing Values:\n")
print(colSums(is.na(data)))


# ============================================================
# 4. PLOT NETFLIX CLOSING PRICE
# ============================================================

ggplot(data, aes(x = Date, y = Close)) +
  geom_line() +
  labs(
    title = "Netflix (NFLX) Closing Price",
    x = "Date",
    y = "Closing Price (USD)"
  ) +
  theme_minimal()


# ============================================================
# 5. CALCULATE DAILY RETURNS
# ============================================================

data$Return <- c(
  NA,
  diff(data$Close) / head(data$Close, -1)
)

ggplot(data, aes(x = Date, y = Return)) +
  geom_line() +
  labs(
    title = "Netflix Daily Returns",
    x = "Date",
    y = "Daily Return"
  ) +
  theme_minimal()


# ============================================================
# 6. MOVING AVERAGES
# ============================================================

data$MA_10 <- SMA(data$Close, n = 10)
data$MA_20 <- SMA(data$Close, n = 20)
data$MA_50 <- SMA(data$Close, n = 50)


# ============================================================
# 7. RSI INDICATOR
# ============================================================

data$RSI <- RSI(data$Close, n = 14)


# ============================================================
# 8. MACD INDICATOR
# ============================================================

macd <- MACD(
  data$Close,
  nFast = 12,
  nSlow = 26,
  nSig = 9
)

data$MACD <- as.numeric(macd[, "macd"])
data$Signal <- as.numeric(macd[, "signal"])


# ============================================================
# 9. BOLLINGER BANDS
# ============================================================

bb <- BBands(
  data$Close,
  n = 20
)

data$BB_up <- as.numeric(bb[, "up"])
data$BB_mid <- as.numeric(bb[, "mavg"])
data$BB_dn <- as.numeric(bb[, "dn"])


# ============================================================
# 10. ROLLING VOLATILITY
# ============================================================

data$Volatility_20 <- runSD(
  data$Return,
  n = 20
)


# ============================================================
# 11. LAG FEATURES
# ============================================================

data$Lag_1 <- lag.xts(data$Close, 1)
data$Lag_2 <- lag.xts(data$Close, 2)
data$Lag_3 <- lag.xts(data$Close, 3)
data$Lag_5 <- lag.xts(data$Close, 5)
data$Lag_10 <- lag.xts(data$Close, 10)

data$Return_Lag_1 <- lag.xts(data$Return, 1)
data$Return_Lag_2 <- lag.xts(data$Return, 2)
data$Return_Lag_5 <- lag.xts(data$Return, 5)


# ============================================================
# 12. CREATE TARGET VARIABLE
# ============================================================

# Target = next trading day's closing price

data$Target <- lead(data$Close, 1)


# ============================================================
# 13. CREATE UP/DOWN DIRECTION VARIABLE
# ============================================================

data$Direction <- ifelse(
  data$Target > data$Close,
  1,
  0
)


# ============================================================
# 14. REMOVE MISSING VALUES
# ============================================================

data_ml <- na.omit(data)

data_ml <- as.data.frame(data_ml)


# ============================================================
# 15. SELECT FEATURES
# ============================================================

features <- c(
  "Open",
  "High",
  "Low",
  "Close",
  "Volume",
  "MA_10",
  "MA_20",
  "MA_50",
  "RSI",
  "MACD",
  "Signal",
  "BB_up",
  "BB_mid",
  "BB_dn",
  "Volatility_20",
  "Lag_1",
  "Lag_2",
  "Lag_3",
  "Lag_5",
  "Lag_10",
  "Return_Lag_1",
  "Return_Lag_2",
  "Return_Lag_5"
)


# ============================================================
# 16. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

n <- nrow(data_ml)

train_size <- floor(0.80 * n)

train_data <- data_ml[1:train_size, ]
test_data <- data_ml[(train_size + 1):n, ]

cat("\nTraining observations:", nrow(train_data))
cat("\nTesting observations:", nrow(test_data), "\n")


# ============================================================
# 17. CREATE TRAINING AND TESTING DATA
# ============================================================

X_train <- train_data[, features]
X_test <- test_data[, features]

y_train <- train_data$Target
y_test <- test_data$Target


# ============================================================
# 18. MODEL 1 - LINEAR REGRESSION
# ============================================================

lm_model <- lm(
  Target ~ .,
  data = train_data[, c(features, "Target")]
)

print(summary(lm_model))


# ============================================================
# 19. LINEAR REGRESSION PREDICTION
# ============================================================

lm_pred <- predict(
  lm_model,
  newdata = test_data[, features]
)


# ============================================================
# 20. LINEAR REGRESSION EVALUATION
# ============================================================

lm_rmse <- rmse(
  y_test,
  lm_pred
)

lm_mae <- mae(
  y_test,
  lm_pred
)

cat("\nLinear Regression Results")
cat("\nRMSE:", lm_rmse)
cat("\nMAE:", lm_mae, "\n")


# ============================================================
# 21. MODEL 2 - RANDOM FOREST REGRESSION
# ============================================================

set.seed(123)

rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(length(features))),
  importance = TRUE
)

print(rf_model)


# ============================================================
# 22. RANDOM FOREST PREDICTION
# ============================================================

rf_pred <- predict(
  rf_model,
  X_test
)


# ============================================================
# 23. RANDOM FOREST EVALUATION
# ============================================================

rf_rmse <- rmse(
  y_test,
  rf_pred
)

rf_mae <- mae(
  y_test,
  rf_pred
)

cat("\nRandom Forest Results")
cat("\nRMSE:", rf_rmse)
cat("\nMAE:", rf_mae, "\n")


# ============================================================
# 24. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print(importance(rf_model))

varImpPlot(
  rf_model,
  main = "Random Forest Feature Importance"
)


# ============================================================
# 25. MODEL 3 - XGBOOST REGRESSION
# ============================================================

X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

set.seed(123)

xgb_model <- xgboost(
  data = X_train_matrix,
  label = y_train,
  nrounds = 500,
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)


# ============================================================
# 26. XGBOOST PREDICTION
# ============================================================

xgb_pred <- predict(
  xgb_model,
  X_test_matrix
)


# ============================================================
# 27. XGBOOST EVALUATION
# ============================================================

xgb_rmse <- rmse(
  y_test,
  xgb_pred
)

xgb_mae <- mae(
  y_test,
  xgb_pred
)

cat("\nXGBoost Results")
cat("\nRMSE:", xgb_rmse)
cat("\nMAE:", xgb_mae, "\n")


# ============================================================
# 28. MODEL COMPARISON
# ============================================================

results <- data.frame(
  Model = c(
    "Linear Regression",
    "Random Forest",
    "XGBoost"
  ),
  RMSE = c(
    lm_rmse,
    rf_rmse,
    xgb_rmse
  ),
  MAE = c(
    lm_mae,
    rf_mae,
    xgb_mae
  )
)

print(results)


# ============================================================
# 29. MODEL COMPARISON PLOT
# ============================================================

ggplot(results, aes(x = Model, y = RMSE)) +
  geom_col() +
  labs(
    title = "Model Comparison Using RMSE",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()


# ============================================================
# 30. ACTUAL VS PREDICTED VALUES
# ============================================================

comparison <- data.frame(
  Date = test_data$Date,
  Actual = y_test,
  Linear_Regression = lm_pred,
  Random_Forest = rf_pred,
  XGBoost = xgb_pred
)


# ============================================================
# 31. ACTUAL VS XGBOOST PREDICTION PLOT
# ============================================================

ggplot(
  comparison,
  aes(x = Date)
) +
  geom_line(
    aes(y = Actual, linetype = "Actual")
  ) +
  geom_line(
    aes(y = XGBoost, linetype = "XGBoost Prediction")
  ) +
  labs(
    title = "Netflix Actual vs XGBoost Predicted Price",
    x = "Date",
    y = "Closing Price (USD)",
    linetype = "Series"
  ) +
  theme_minimal()


# ============================================================
# 32. ACTUAL VS RANDOM FOREST PREDICTION PLOT
# ============================================================

ggplot(
  comparison,
  aes(x = Date)
) +
  geom_line(
    aes(y = Actual, linetype = "Actual")
  ) +
  geom_line(
    aes(y = Random_Forest, linetype = "Random Forest Prediction")
  ) +
  labs(
    title = "Netflix Actual vs Random Forest Predicted Price",
    x = "Date",
    y = "Closing Price (USD)",
    linetype = "Series"
  ) +
  theme_minimal()


# ============================================================
# 33. RANDOM FOREST CLASSIFICATION - PRICE DIRECTION
# ============================================================

y_train_direction <- as.factor(train_data$Direction)
y_test_direction <- as.factor(test_data$Direction)

set.seed(123)

rf_classifier <- randomForest(
  x = X_train,
  y = y_train_direction,
  ntree = 500,
  mtry = floor(sqrt(length(features))),
  importance = TRUE
)


# ============================================================
# 34. DIRECTION PREDICTION
# ============================================================

direction_pred <- predict(
  rf_classifier,
  X_test
)


# ============================================================
# 35. CONFUSION MATRIX
# ============================================================

confusion_matrix <- table(
  Actual = y_test_direction,
  Predicted = direction_pred
)

print(confusion_matrix)


# ============================================================
# 36. DIRECTIONAL ACCURACY
# ============================================================

direction_accuracy <- mean(
  direction_pred == y_test_direction
)

cat(
  "\nDirectional Accuracy:",
  round(direction_accuracy * 100, 2),
  "%\n"
)


# ============================================================
# 37. PRECISION, RECALL AND F1 SCORE
# ============================================================

TP <- confusion_matrix["1", "1"]
TN <- confusion_matrix["0", "0"]
FP <- confusion_matrix["0", "1"]
FN <- confusion_matrix["1", "0"]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

F1 <- 2 * (
  precision * recall
) / (
  precision + recall
)

cat("\nPrecision:", round(precision, 4))
cat("\nRecall:", round(recall, 4))
cat("\nF1 Score:", round(F1, 4), "\n")


# ============================================================
# 38. CLASSIFICATION FEATURE IMPORTANCE
# ============================================================

varImpPlot(
  rf_classifier,
  main = "Feature Importance for Price Direction"
)


# ============================================================
# 39. PREDICT NEXT TRADING DAY PRICE
# ============================================================

latest_data <- tail(
  data_ml,
  1
)

latest_features <- latest_data[, features]

next_day_prediction <- predict(
  xgb_model,
  as.matrix(latest_features)
)

current_price <- latest_data$Close

cat(
  "\nCurrent Netflix Closing Price:",
  round(current_price, 2),
  "USD\n"
)

cat(
  "Predicted Next Trading Day Price:",
  round(next_day_prediction, 2),
  "USD\n"
)


# ============================================================
# 40. PREDICTED PRICE CHANGE
# ============================================================

predicted_change <- (
  next_day_prediction - current_price
) / current_price * 100

cat(
  "Predicted Percentage Change:",
  round(predicted_change, 2),
  "%\n"
)


# ============================================================
# 41. PREDICT NEXT DAY DIRECTION
# ============================================================

next_day_direction <- predict(
  rf_classifier,
  latest_features
)

if (next_day_direction == "1") {
  direction_text <- "UP"
} else {
  direction_text <- "DOWN"
}

cat(
  "Predicted Next Trading Day Direction:",
  direction_text,
  "\n"
)


# ============================================================
# 42. FINAL PREDICTION REPORT
# ============================================================

prediction_report <- data.frame(
  Date = latest_data$Date,
  Current_Price = current_price,
  Predicted_Price = next_day_prediction,
  Predicted_Change_Percent = predicted_change,
  Predicted_Direction = direction_text
)

print(prediction_report)


# ============================================================
# 43. SAVE MODEL RESULTS
# ============================================================

write.csv(
  results,
  "model_comparison.csv",
  row.names = FALSE
)

write.csv(
  comparison,
  "actual_vs_predicted.csv",
  row.names = FALSE
)

write.csv(
  prediction_report,
  "next_day_prediction.csv",
  row.names = FALSE
)


# ============================================================
# 44. SAVE FEATURE-ENGINEERED DATA
# ============================================================

write.csv(
  data_ml,
  "Netflix_ML_Dataset.csv",
  row.names = FALSE
)


# ============================================================
# 45. FINAL OUTPUT
# ============================================================

cat("\n")
cat("============================================================\n")
cat("NETFLIX STOCK FORECASTING PROJECT COMPLETED\n")
cat("============================================================\n")

cat("\nModel Performance:\n")
print(results)

cat("\nCurrent Price:",
    round(current_price, 2),
    "USD")

cat("\nPredicted Next Trading Day Price:",
    round(next_day_prediction, 2),
    "USD")

cat("\nPredicted Change:",
    round(predicted_change, 2),
    "%")

cat("\nPredicted Direction:",
    direction_text)

cat("\nDirectional Accuracy:",
    round(direction_accuracy * 100, 2),
    "%")

cat("\n============================================================\n")