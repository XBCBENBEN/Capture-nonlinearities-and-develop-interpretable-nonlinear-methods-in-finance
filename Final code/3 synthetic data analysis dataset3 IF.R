library(MASS) # Used to generate normal distribution data
library(caret) # Used to split training and testing dataset
library(Metrics) # Calculate mean square value
library(dplyr)
library(diversityForest)
library(randomForest)


############################## Dataset 3 ##############################
set.seed(42)
#################### Generate data ####################
# Define parameters
beta_0_3 <- 1
betas_3 <- runif(5, min = 1, max = 5)  # random generation

# Generate features
X_3 <- matrix(runif(5000, min = 0, max = 1), ncol = 5)

# Calculate Y
Y_nonlinear <- beta_0_3 + betas_3[1] * sin(X_3[,1]) + betas_3[2] * log(X_3[,2]) + betas_3[3] * X_3[,3] + betas_3[4] * (X_3[,4]^2) + betas_3[5] * X_3[,5]

# Create data frame
Synthetic_data3 <- data.frame(X_3)
colnames(Synthetic_data3) <- paste0("X", 1:5)
Synthetic_data3$Y <- Y_nonlinear

#################### Fit linear regression ####################

# Perform linear regression
model_linear3 <- lm(Y ~ ., data = Synthetic_data3)

# Predict Y values
Y_pred_linear3 <- predict(model_linear3, Synthetic_data3)

# Print coefficients
print(coef(model_linear3))

# Evaluate the model
mse_linear3 <- mse(Synthetic_data3$Y, Y_pred_linear3)
r2_linear3 <- summary(model_linear3)$r.squared
print(paste("Mean Squared Error:", mse_linear3))
print(paste("R-squared:", r2_linear3))

#################### Fit interaction forest ####################
interaction_forest_3 = interactionfor(
  Y ~ .,
  data = Synthetic_data3,
  num.trees = 2000,
  importance = 'both',
  simplify.large.n = TRUE,
  num.trees.eim.large.n = 20000
)
print(interaction_forest_3)
interaction_forest_3$eim.univ.sorted
interaction_forest_3$eim.qual.sorted
interaction_forest_3$eim.quant.sorted
plot(interaction_forest_3)

#################### Try interaction ####################
# Add new interaction term
Synthetic_data3$X2_X4 <- Synthetic_data3$X2 * Synthetic_data3$X4
Synthetic_data3$X2_X5 <- Synthetic_data3$X2 * Synthetic_data3$X5
Synthetic_data3$X2_X3 <- Synthetic_data3$X2 * Synthetic_data3$X3
Synthetic_data3$X1_X2 <- Synthetic_data3$X1 * Synthetic_data3$X2

interaction_forest_33 = interactionfor(
  Y ~ .,
  data = Synthetic_data3,
  num.trees = 2000,
  importance = 'both',
  simplify.large.n = TRUE,
  num.trees.eim.large.n = 20000
)
print(interaction_forest_33)

# Perform linear regression
model_linear33 <- lm(Y ~ ., data = Synthetic_data3)

# Predict Y values
Y_pred_linear33 <- predict(model_linear33, Synthetic_data3)

# Print coefficients
print(coef(model_linear33))

# Evaluate the model
mse_linear33 <- mse(Synthetic_data3$Y, Y_pred_linear33)
r2_linear33 <- summary(model_linear33)$r.squared
print(paste("Mean Squared Error:", mse_linear33))
print(paste("R-squared:", r2_linear33))

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
  rf_model <- train(Y ~ ., data = Synthetic_data3, method = "rf", 
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
