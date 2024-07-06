library(MASS) # Used to generate normal distribution data
library(caret) # Used to split training and testing dataset
library(Metrics) # Calculate mean square value
library(dplyr)
library(diversityForest)
library(randomForest)


############################## Dataset 2 ##############################
set.seed(42)
#################### Generate data ####################
# Define coefficients
beta_0_2 <- 1
betas_2 <- runif(10, min = 1, max = 5) # Random generate

# Generate features
X_2 <- mvrnorm(1000, mu = rep(1, 10), Sigma = diag(1, 10)) # multinormal distribution

# feature name
feature_names2 <- paste0('X', 1:10)

# coefficients of interaction term
beta_16 <- 2
beta_89 <- -1
beta_234 <- 2.5

# Calculate Y
Y_complex1 <- beta_0_2 + X_2 %*% betas_2 + beta_16 * X_2[,1] * X_2[,6] + beta_89 * sin(X_2[,8]) * X_2[,9] + beta_234 * X_2[,2] * X_2[,3] * X_2[,4] + rnorm(1000, mean = 0, sd = 0.1)

# Create dataframe
Synthetic_data2 <- data.frame(X_2)
names(Synthetic_data2) <- feature_names2
Synthetic_data2$Y <- Y_complex1

#################### Fit linear regression ####################

# Perform linear regression
model_linear2 <- lm(Y ~ ., data = Synthetic_data2)

# Predict Y values
Y_pred_linear2 <- predict(model_linear2, Synthetic_data2)

# Print coefficients
print(coef(model_linear2))

# Evaluate the model
mse_linear2 <- mse(Synthetic_data2$Y, Y_pred_linear2)
r2_linear2 <- summary(model_linear2)$r.squared
print(paste("Mean Squared Error:", mse_linear2))
print(paste("R-squared:", r2_linear2))

#################### Fit interaction forest ####################
interaction_forest_2 = interactionfor(
  Y ~ .,
  data = Synthetic_data2,
  num.trees = 2000,
  importance = 'both',
  simplify.large.n = TRUE,
  num.trees.eim.large.n = 20000
)
print(interaction_forest_2)
interaction_forest_2$eim.univ.sorted
interaction_forest_2$eim.qual.sorted
interaction_forest_2$eim.quant.sorted
plot(interaction_forest_2)

#################### Try interaction ####################
# Add new interaction term
Synthetic_data2$X2_X4 <- Synthetic_data2$X2 * Synthetic_data2$X4
Synthetic_data2$X3_X4 <- Synthetic_data2$X3 * Synthetic_data2$X4
Synthetic_data2$X2_X3 <- Synthetic_data2$X2 * Synthetic_data2$X3
Synthetic_data2$X1_X6 <- Synthetic_data2$X1 * Synthetic_data2$X6

interaction_forest_22 = interactionfor(
  Y ~ .,
  data = Synthetic_data2,
  num.trees = 2000,
  importance = 'both',
  simplify.large.n = TRUE,
  num.trees.eim.large.n = 20000
)
print(interaction_forest_22)

# Perform linear regression
model_linear22 <- lm(Y ~ ., data = Synthetic_data2)

# Predict Y values
Y_pred_linear22 <- predict(model_linear22, Synthetic_data2)

# Print coefficients
print(coef(model_linear22))

# Evaluate the model
mse_linear22 <- mse(Synthetic_data2$Y, Y_pred_linear22)
r2_linear22 <- summary(model_linear22)$r.squared
print(paste("Mean Squared Error:", mse_linear22))
print(paste("R-squared:", r2_linear22))

# Random forest
# Use cross-validation to optimize random forest
control2 <- trainControl(method = "cv", number = 5)
tunegrid2 <- expand.grid(.mtry = c(2, 3, 4, 5, 6))
ntree_values <- c(100, 300, 500, 700, 1000)

best_mse <- Inf
best_model <- NULL
best_mtry <- NULL
best_ntree <- NULL

for (ntree in ntree_values) {
  rf_model <- train(Y ~ ., data = Synthetic_data2, method = "rf", 
                    trControl = control2, tuneGrid = tunegrid2, ntree = ntree)
  
  mse <- min(rf_model$results$RMSE)
  
  if (mse < best_mse) {
    best_mse <- mse
    best_model <- rf_model
    best_mtry <- rf_model$bestTune$.mtry
    best_ntree <- ntree
  }
}
print(best_model)
print(best_ntree)
