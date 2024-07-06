library(MASS) # Used to generate normal distribution data
library(caret) # Used to split training and testing dataset
library(Metrics) # Calculate mean square value
library(dplyr)
library(diversityForest)
library(randomForest)


############################## log_scope_2 ##############################
set.seed(42)
##Read data
file_path1 = "C:/Users/13976/Desktop/M4R Project/R_data_cleaned.csv"
Data = read.csv(file_path1)

#################### Fit linear regression ####################

# Perform linear regression
model_linear2 <- lm(RET. ~ log_scope_2+LOGSIZE_t.1+Winsorised.BM_t.1+Winsorised.Leverage_t.1+Winsorised.INVESTA_t.1+Winsorised.ROE_t.1+LOGPPE_t.1+MOM_duplicate_t.1+beta_t.1+VOLAT_t.1+Winsorised.SALESGR_t.1+EPSGR_t.1, data = Data)

# Predict Y values
Y_pred_linear2 <- predict(model_linear2, Data)

# Print coefficients
print(coef(model_linear2))

# Evaluate the model
mse_linear2 <- mse(Data$RET., Y_pred_linear2)
r2_linear2 <- summary(model_linear2)$r.squared
print(paste("Mean Squared Error:", mse_linear2))
print(paste("R-squared:", r2_linear2))

#################### Fit interaction forest ####################
interaction_forest_2 = interactionfor(
  RET. ~ log_scope_2+LOGSIZE_t.1+Winsorised.BM_t.1+Winsorised.Leverage_t.1+Winsorised.INVESTA_t.1+Winsorised.ROE_t.1+LOGPPE_t.1+MOM_duplicate_t.1+beta_t.1+VOLAT_t.1+Winsorised.SALESGR_t.1+EPSGR_t.1,
  data = Data,
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
Data$MOM_duplicate_t.1_VOLAT_t.1 <- Data$MOM_duplicate_t.1 * Data$VOLAT_t.1
Data$beta_t.1_VOLAT_t.1 <- Data$beta_t.1 * Data$VOLAT_t.1
Data$VOLAT_t.1_Winsorised.SALESGR_t.1 <- Data$VOLAT_t.1 * Data$Winsorised.SALESGR_t.1

interaction_forest_22 = interactionfor(
  RET. ~ log_scope_1+VOLAT_t.1_Winsorised.SALESGR_t.1+MOM_duplicate_t.1_VOLAT_t.1+beta_t.1_VOLAT_t.1+LOGSIZE_t.1+Winsorised.BM_t.1+Winsorised.Leverage_t.1+Winsorised.INVESTA_t.1+Winsorised.ROE_t.1+LOGPPE_t.1+MOM_duplicate_t.1+beta_t.1+VOLAT_t.1+Winsorised.SALESGR_t.1+EPSGR_t.1,
  data = Data,
  num.trees = 2000,
  importance = 'both',
  simplify.large.n = TRUE,
  num.trees.eim.large.n = 20000
)
print(interaction_forest_22)

# Perform linear regression
model_linear22 <- lm(RET. ~ log_scope_1+VOLAT_t.1_Winsorised.SALESGR_t.1+MOM_duplicate_t.1_VOLAT_t.1+beta_t.1_VOLAT_t.1+LOGSIZE_t.1+Winsorised.BM_t.1+Winsorised.Leverage_t.1+Winsorised.INVESTA_t.1+Winsorised.ROE_t.1+LOGPPE_t.1+MOM_duplicate_t.1+beta_t.1+VOLAT_t.1+Winsorised.SALESGR_t.1+EPSGR_t.1, data = Data)

# Predict Y values
Y_pred_linear22 <- predict(model_linear22, Data)

# Print coefficients
print(coef(model_linear22))

# Evaluate the model
mse_linear22 <- mse(Data$RET., Y_pred_linear22)
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
  rf_model <- train(RET. ~ log_scope_1+LOGSIZE_t.1+Winsorised.BM_t.1+Winsorised.Leverage_t.1+Winsorised.INVESTA_t.1+Winsorised.ROE_t.1+LOGPPE_t.1+MOM_duplicate_t.1+beta_t.1+VOLAT_t.1+Winsorised.SALESGR_t.1+EPSGR_t.1, data = Data, method = "rf", 
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



for (ntree in ntree_values) {
  rf_model <- train(RET. ~ log_scope_1+VOLAT_t.1_Winsorised.SALESGR_t.1+MOM_duplicate_t.1_VOLAT_t.1+beta_t.1_VOLAT_t.1+MOM_duplicate_t.1_VOLAT_t.1+LOGSIZE_t.1+Winsorised.BM_t.1+Winsorised.Leverage_t.1+Winsorised.INVESTA_t.1+Winsorised.ROE_t.1+LOGPPE_t.1+MOM_duplicate_t.1+beta_t.1+VOLAT_t.1+Winsorised.SALESGR_t.1+EPSGR_t.1, data = Data, method = "rf", 
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