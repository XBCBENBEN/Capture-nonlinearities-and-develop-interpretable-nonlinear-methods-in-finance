library(MASS) # Used to generate normal distribution data
library(caret) # Used to split training and testing dataset
library(Metrics) # Calculate mean square value
library(dplyr)
library(diversityForest)
library(randomForest)

set.seed(42) # set seeds
############################## Dataset 1 ##############################

#################### Generate data ####################

# Define the coefficients
beta_0 <- 1
beta_1 <- 2
beta_2 <- 3
beta_3 <- 4
beta_4 <- 5
beta_5 <- 6
beta_12 <- 1.5
beta_45 <- -2

# Generate data
X1 <- rnorm(1000, mean = 1, sd = 1)
X2 <- rnorm(1000, mean = 1, sd = 1)
X3 <- rnorm(1000, mean = 1, sd = 1)
X4 <- rnorm(1000, mean = 1, sd = 1)
X5 <- runif(1000, min = 0, max = 1)

# Calculate Y
Y_simple <- beta_0 + beta_1*X1 + beta_2*X2 + beta_3*X3 + beta_4*X4 + beta_5*X5 +
  beta_12*(X1*X2) + beta_45*(X4*X5) + rnorm(1000, mean = 0, sd = 0.1)

# Create dataframe
Synthetic_data1 <- data.frame(
  X1 = X1,
  X2 = X2,
  X3 = X3,
  X4 = X4,
  X5 = X5,
  Y = Y_simple
)

#################### Fit linear regression ####################

# Perform linear regression
model_linear1 <- lm(Y ~ ., data = Synthetic_data1)

# Predict Y values
Y_pred_linear1 <- predict(model_linear1, Synthetic_data1)

# Print coefficients
print(coef(model_linear1))

# Evaluate the model
mse_linear1 <- mse(Synthetic_data1$Y, Y_pred_linear1)
r2_linear1 <- summary(model_linear1)$r.squared
print(paste("Mean Squared Error:", mse_linear1))
print(paste("R-squared:", r2_linear1))

#################### Fit interaction forest ####################
interaction_forest_1 = interactionfor(
  Y ~ X1 + X2 + X3 + X4 + X5,
  data = Synthetic_data1,
  num.trees = 2000,
  importance = 'both',
  simplify.large.n = TRUE,
  num.trees.eim.large.n = 20000
)
print(interaction_forest_1)
interaction_forest_1$eim.univ.sorted
interaction_forest_1$eim.qual.sorted
interaction_forest_1$eim.quant.sorted
plot(interaction_forest_1)

#################### Try interaction ####################
# Add new interaction term
Synthetic_data1$X1_X2 <- Synthetic_data1$X1 * Synthetic_data1$X2

interaction_forest_11 = interactionfor(
  Y ~ X1 + X2 + X3 + X4 + X5 + X1_X2,
  data = Synthetic_data1,
  num.trees = 2000,
  importance = 'both',
  simplify.large.n = TRUE,
  num.trees.eim.large.n = 20000
)
print(interaction_forest_11)

# Perform linear regression
model_linear11 <- lm(Y ~ ., data = Synthetic_data1)

# Predict Y values
Y_pred_linear11 <- predict(model_linear11, Synthetic_data1)

# Print coefficients
print(coef(model_linear11))

# Evaluate the model
mse_linear11 <- mse(Synthetic_data1$Y, Y_pred_linear11)
r2_linear11 <- summary(model_linear11)$r.squared
print(paste("Mean Squared Error:", mse_linear11))
print(paste("R-squared:", r2_linear11))

# Random forest
# Use cross-validation to optimize random forest
control1 <- trainControl(method = "cv", number = 5)
tunegrid1 <- expand.grid(.mtry = c(2, 3, 4, 5, 6))
ntree_values <- c(100, 300, 500, 700, 1000)

best_mse <- Inf
best_model <- NULL
best_mtry <- NULL
best_ntree <- NULL

for (ntree in ntree_values) {
  rf_model <- train(Y ~ ., data = Synthetic_data1, method = "rf", 
                    trControl = control1, tuneGrid = tunegrid1, ntree = ntree)
  
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