##################################################
# ECON 418-518 Homework 3
# Sowadhana Sim
# The University of Arizona
# sowadhana@arizona.edu 
# 08 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

# Set sead
set.seed(418518)

# Set working directory
setwd("`/Downloads")

# Data Table
dt <- read_csv("ECON_418-518_HW3_Data.csv")
dt <- as.data.table(dt)

#####################
# Problem 1
#####################


#################
# Question (i)
#################

# Code

# Drop columns
dt[, fnlwgt := NULL]
dt[, occupation := NULL]
dt[, relationship := NULL]
dt[, `capital-gain` := NULL]
dt[, `capital-loss` := NULL]
dt[, `educational-num` := NULL]

#################
# Question (ii)
#################

# Code

# Convert income column
dt$income <- ifelse(dt$income == ">50K", 1, 0)

# Convert race column
dt$race <- ifelse(dt$race == "White", 1, 0)

# Convert gender column
dt$gender <- ifelse(dt$gender == "Male", 1, 0)

# Convert workclass column
dt$workclass <- ifelse(dt$workclass == "Private", 1, 0)

# Convert native_country column
dt$`native-country` <- ifelse(dt$`native-country` == "United-States", 1, 0)

# Convert marital_status column
dt$`marital-status`<- ifelse(dt$`marital-status` == "Married-civ-spouse", 1, 0)

# Convert education column
dt$education <- ifelse(dt$education %in% c("Bachelors", "Masters", "Doctorate"), 1,0)

# Create age_sq variable
dt$age_sq <- dt$age^2

# Standardize age, age squared, hours variable
dt$age <- scale(dt$age)
dt$age_sq <- scale(dt$age_sq)
dt$`hours-per-week` <- scale(dt$`hours-per-week`)


#################
# Question (iii)
#################

# Code

# Proportion of each column
mean(dt$income == 1)
mean(dt$workclass == 1)
mean(dt$`marital-status` == 1)
mean(dt$gender == 0)

# Total NAs
totalnavalues <- sum(is.na(dt))
totalnavalues

# Convert income to a factor variable
dt$income <- as.factor(dt$income)

#################
# Question (iv)
#################

# Code

# Training, validation, and test set size
train_size <- floor(0.7 * nrow(dt))
test_size <- floor(0.3 * nrow(dt))

# Shuffle the data
shuffled_indices <- sample(nrow(dt))

# Split the indices
train_indices <- shuffled_indices[1:train_size]
test_indices <- shuffled_indices[(train_size + 1):nrow(dt)]

# Create the datasets
dt_train <- dt[train_indices, ]
dt_test <- dt[test_indices, ]

#################
# Question (v)
#################

# Code

# Load Package
install.packages("caret")
library(caret)

# Set seed for reproducibility
set.seed(418518)  

# Define the grid of lambda values
lambda_grid <- 10^(seq(5, -2, length = 50))

# Train lasso regression model
cv <- train(
  income ~ ., 
  data = dt, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)  # alpha = 1 for lasso
)

# Optimal lambda
best_lambda <- cv$bestTune$lambda

# Best accuracy
best_accuracy <- max(cv$results$Accuracy)

best_lambda
best_accuracy

# Coefficients of the best model
lasso_coefficients <- coef(cv$finalModel, s = cv$bestTune$lambda)

# Identify variables with coefficients approximately zero
zero_coeff_vars <- rownames(lasso_coefficients)[lasso_coefficients == 0]
zero_coeff_vars

# Subset the dataset with non-zero coefficient variables
non_zero_vars <- rownames(lasso_coefficients)[lasso_coefficients != 0]
filtered_data <- dt[, c(non_zero_vars, "income")]

# Train lasso and ridge models on filtered data
ridge_model <- train(
  income ~ ., 
  data = dt, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid)  # alpha = 0 for ridge
)

lasso_model_filtered <- train(
  income ~ ., 
  data = dt, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)

# Compare best accuracies
ridge_best_accuracy <- max(ridge_model$results$Accuracy)
lasso_best_accuracy <- max(lasso_model_filtered$results$Accuracy)

ridge_best_accuracy
lasso_best_accuracy

#################
# Question (vi)
#################

# Code

install.packages("randomForest")
install.packages("caret")
library(randomForest)
library(caret)

set.seed(418518)

# Define a grid for the random forest hyperparameters
rf_grid <- expand.grid(mtry = c(2, 5, 9))

# Train models with different numbers of trees
rf_100 <- train(
  income ~ ., 
  data = dt,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = rf_grid,
  ntree = 100
)

rf_200 <- train(
  income ~ ., 
  data = dt,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = rf_grid,
  ntree = 200
)

rf_300 <- train(
  income ~ ., 
  data = dt,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = rf_grid,
  ntree = 300
)

# Extract the best model accuracy for each ntree setting
rf_100_best <- max(rf_100$results$Accuracy)
rf_200_best <- max(rf_200$results$Accuracy)
rf_300_best <- max(rf_300$results$Accuracy)

# Determine which setting gives the highest accuracy
best_accuracy <- max(rf_100_best, rf_200_best, rf_300_best)
best_accuracy

# Best accuracy from Part (v)
lasso_ridge_best_accuracy <- max(lasso_model_filtered$results$Accuracy, ridge_model$results$Accuracy)

# Compare
comparison <- best_accuracy > lasso_ridge_best_accuracy
comparison

best_rf_model <- rf_300
predictions <- predict(best_rf_model, dt)

confusion_matrix <- confusionMatrix(predictions, dt$income)
confusion_matrix

# Extract false positives (FP) and false negatives (FN)
FP <- confusion_matrix$table[1, 2]  # False positives
FN <- confusion_matrix$table[2, 1]  # False negatives

FP
FN

#################
# Question (vii)
#################

# Code

# Model regression
best_rf_model <- train(
  income ~ ., 
  data = dt,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(mtry = rf_grid), 
  ntree = 300
)

# Make predictions
predictions <- predict(best_rf_model, dt_test)

# Compute accuracy
confusion_matrix <- confusionMatrix(predictions, dt_test$income)
classification_accuracy <- confusion_matrix$overall["Accuracy"]

classification_accuracy
