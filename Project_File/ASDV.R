
# Concrete Compressive Strength Analysis Project
# ---------------------------------------------
# ----------------Load Libraries and Dataset
# Install necessary libraries
install.packages("readxl")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("corrplot")
install.packages("stats")
install.packages("car")
install.packages("fitdistrplus")
install.packages("tidyverse")

install.packages("xgboost")
install.packages("randomForest")

# Load libraries
library(readxl)
library(ggplot2)
library(dplyr)
library(corrplot)
library(car)
library(fitdistrplus)
library(tidyverse)
library(tidyr)

library(randomForest)
library(xgboost)
# ---------------------------------------------
# Load the dataset
Concrete_data <- read_excel("concrete compressive strength.xlsx")

# Manually rename columns
colnames(Concrete_data) <- c(
  "Cement", 
  "Blast_Furnace_Slag", 
  "Fly_Ash", 
  "Water", 
  "Superplasticizer", 
  "Coarse_Aggregate", 
  "Fine_Aggregate", 
  "Age", 
  "Concrete_Category", 
  "Contains_Fly_Ash", 
  "Compressive_Strength"
)


# Verify the new column names
colnames(Concrete_data)

# ---------------------------------------------
# Overview of the data
str(Concrete_data)
summary(Concrete_data)
head(Concrete_data)
tail(Concrete_data)

# ---------------------------------------------
# Check for duplicates
anyDuplicated(Concrete_data)


# ---------------------------------------------
# Outlier Detection Using IQR

# Function to detect outliers using IQR
detect_outliers <- function(column) {
  Q1 <- quantile(column, 0.25)
  Q3 <- quantile(column, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  which(column < lower_bound | column > upper_bound)
}

# Apply outlier detection to numeric columns
numeric_columns <- Concrete_data[, sapply(Concrete_data, is.numeric)]
outliers <- lapply(numeric_columns, detect_outliers)

# Display outliers
outliers

# Count outliers in each numeric column
outlier_counts <- sapply(outliers, length)
outlier_data <- data.frame(Column = names(outlier_counts), Count = outlier_counts)
print(outlier_data)

# Plot outlier counts
ggplot(outlier_data, aes(x = Column, y = Count, fill = Column)) +
  geom_bar(stat = "identity", color = "black") +
  labs(title = "Number of Outliers in Each Numeric Column") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set3")



# Replace outliers with the median value of the column
for (col_name in names(outliers)) {
  outlier_indices <- outliers[[col_name]]
  median_value <- median(Concrete_data[[col_name]], na.rm = TRUE)
  Concrete_data[outlier_indices, col_name] <- median_value
}

# ---------------------------------------------
# Data Cleaning and Inspection

# Verify that outliers have been replaced and no further outliers exist
sapply(Concrete_data, function(x) sum(is.na(x)))


# -------------------------------------------------------------------------------
# Check the class of the variables
class(Concrete_data$`Concrete_Category`)
class(Concrete_data$`Contains_Fly_Ash`)

# Convert categorical variables to factors
Concrete_data$`Concrete_Category` <- as.factor(Concrete_data$`Concrete_Category`)
Concrete_data$`Contains_Fly_Ash` <- as.factor(Concrete_data$`Contains_Fly_Ash`)

# Check if conversion was successful
is.factor(Concrete_data$`Concrete_Category`)
is.factor(Concrete_data$`Contains_Fly_Ash`)

# View contingency table for categorical variables
table(Concrete_data$`Concrete_Category`, Concrete_data$`Contains_Fly_Ash`)

# ---------------------------------------------
# Exploratory Data Analysis (EDA)
# Boxplot for Compressive Strength by Concrete Category
ggplot(Concrete_data, aes(x = `Concrete_Category`, y = `Compressive_Strength`)) +
  geom_boxplot(fill = "orange") +
  labs(title = "Boxplot of Compressive Strength by Concrete Category", 
       x = "Concrete Category", y = "Compressive Strength (MPa)")

# Barplot for Contains Fly Ash
ggplot(Concrete_data, aes(x = `Contains_Fly_Ash`)) +
  geom_bar(fill = "blue") +
  labs(title = "Distribution of Concrete Categories", x = "Contains Fly Ash", y = "Frequency")


# ---------------------------------------------
# Continuous Variables - Histograms with Density Overlay
library(dplyr)
# Explicitly using dplyr's select function
continuous_data <- dplyr::select(Concrete_data, -Concrete_Category, -Contains_Fly_Ash)


# Plot histograms with density overlays
long_data <- continuous_data %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

ggplot(long_data, aes(x = Value)) +
  geom_histogram(aes(y = ..density..), bins = 10, fill = "blue", color = "black", alpha = 0.7) +
  geom_density(alpha = 0.2, fill = "red") +
  labs(title = "Distribution of Variables", x = "Value", y = "Density") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal()

# ---------------------------------------------
# Correlation Analysis

# Creating and plot correlation matrix
corr_matrix <- cor(continuous_data)
corrplot(corr_matrix, method = "number", type = "upper")


# ---------------------------------------------
# Simple Linear Regression (SLR) - Cement vs Compressive Strength

model_1 <- lm(Compressive_Strength ~ Cement, data = continuous_data)
summary(model_1)

# Scatter plot for SLR with regression line
ggplot(continuous_data, aes(x = Cement, y = Compressive_Strength)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Simple Linear Regression: Cement vs Compressive Strength",
       x = "Cement (kg/m³)", y = "Concrete Compressive Strength (MPa)")

# ------Plotting assumptions for SLR----------

plot(model_1, 1)

plot(model_1, 2)

plot(model_1, 3)

# Plot SLR with regression line
ggplot(continuous_data, aes(x = Cement, y = Compressive_Strength)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "SLR: Cement vs Compressive Strength", x = "Cement (kg/m³)", y = "Compressive Strength (MPa)")

# ------------------------MLR equation-----------------------------
# ==============================================================================
model_2 <- lm(Compressive_Strength ~ 
                Cement + 
                Superplasticizer
              , data = continuous_data)

summary(model_2)

# ========================================================================

model_3 <- lm(Compressive_Strength ~ 
                Cement + 
                Superplasticizer + 
                Age
              , data = continuous_data)

summary(model_3)


# ============================================================================

model_4 <- lm(Compressive_Strength ~ 
                Cement + 
                Superplasticizer + 
                Age + Blast_Furnace_Slag
              , data = Concrete_data)

summary(model_4)


model_5 <- lm(Compressive_Strength ~ 
                Cement + 
                Superplasticizer + 
                Age +
                Blast_Furnace_Slag +
                Water
              , data = Concrete_data)

summary(model_5)


# Scatterplot matrix with linear regression lines
pairs(
  Concrete_data[, c(11,1, 5, 8, 2, 4)],  # Select relevant columns for the scatterplot
  lower.panel = NULL,  # Remove lower panel
  upper.panel = function(x, y) {
    points(x, y, pch = 19, cex = 0.2)  # Add points
    abline(lm(y ~ x), col = "red", lwd = 2)  # Add linear regression line in red
  }
)

# --------------1. Residuals’ Independence----------------------------
# Check this assumption by examining a scatterplot of “residuals versus fits”; 
# the correlation should be approximately 0

plot(model_5, 1)

# --------------2. NORMALITY OF RESIDUALS------------------------------
# The residuals must be approximately normally distributed

plot(model_5, 2)

# --------------3. Equal variances of the residuals (Homoscedasticity)----
# Verify that there is no clear pattern among the residuals

plot(model_5, 3)







# Replace zeros or negative values with a small constant for log transformation
# Apply log transformation to columns and handle zero or negative values by replacing with 0.001
Concrete_data <- Concrete_data %>%
  mutate(
    log_Cement = log(ifelse(Cement > 0, Cement, 0.001)),
    log_Superplasticizer = log(ifelse(Superplasticizer > 0, Superplasticizer, 0.001)),
    log_Age = log(ifelse(Age > 0, Age, 0.001)),
    log_Blast_Furnace_Slag = log(ifelse(`Blast_Furnace_Slag` > 0, `Blast_Furnace_Slag`, 0.001)),
    log_Water = log(ifelse(Water > 0, Water, 0.001))
  )


# Fit the MLR model with log-transformed variables
model_log <- lm(Compressive_Strength ~ 
                  log_Cement + 
                  log_Superplasticizer + 
                  log_Age + 
                  log_Blast_Furnace_Slag + 
                  log_Water,
                data = Concrete_data)

# Display the summary of the model
summary(model_log)

# Check multicollinearity with VIF
vif(model_log)

# --------------2. Residuals’ Independence----------------------------
# Check this assumption by examining a scatterplot of “residuals versus fits”; 
# the correlation should be approximately 0

plot(model_log, 1)

# --------------3. NORMALITY OF RESIDUALS------------------------------
# The residuals must be approximately normally distributed

plot(model_log, 2)

# --------------4. Equal variances of the residuals (Homoscedasticity)----
# Verify that there is no clear pattern among the residuals

plot(model_log, 3)




# Model evaluation
predictions <- predict(model_log, newdata = Concrete_data)
rmse <- sqrt(mean((predictions - Concrete_data$`Compressive_Strength`)^2))
rsq <- 1 - sum((predictions - Concrete_data$`Compressive_Strength`)^2) / 
          sum((Concrete_data$`Compressive_Strength` - mean(Concrete_data$`Compressive_Strength`))^2)

cat("RMSE: ", rmse, "\n")
cat("R-squared: ", rsq, "\n")






# ----------------------------------------------------------------
newConcrete_data <- dplyr::select(Concrete_data, 
                                  Cement, 
                                  Blast_Furnace_Slag, 
                                  Fly_Ash, 
                                  Water, 
                                  Superplasticizer, 
                                  Coarse_Aggregate, 
                                  Fine_Aggregate, 
                                  Age, 
                                  Concrete_Category, 
                                  Contains_Fly_Ash, 
                                  Compressive_Strength)

# View the first few rows of the new dataset
head(newConcrete_data)

# Fit the random forest regression model using all predictors
rf_model <- randomForest(Compressive_Strength ~ ., data = newConcrete_data, ntree = 500)

# Display the model summary
print(rf_model)

# Plot the variable importance
varImpPlot(rf_model, main = "Variable Importance Plot", col = "darkblue", pch = 19, cex = 1.2)

# Predict using the random forest model
rf_predictions <- predict(rf_model, newdata = newConcrete_data)

# Evaluate model performance
rf_rmse <- sqrt(mean((rf_predictions - newConcrete_data$Compressive_Strength)^2))
cat("Random Forest Model RMSE: ", rf_rmse, "\n")

# -----------------------------------------------------------------------


# Handle categorical variables by converting them into factors or dummy variables
newConcrete_data$Concrete_Category <- as.factor(newConcrete_data$Concrete_Category)
newConcrete_data$Contains_Fly_Ash <- as.factor(newConcrete_data$Contains_Fly_Ash)

# Select predictor variables (all columns except 'Compressive_Strength')
X <- dplyr::select(newConcrete_data, -Compressive_Strength)

# Convert all columns to numeric (including factors) using model.matrix() for one-hot encoding of categorical variables
X_matrix <- model.matrix(~ ., data = X)

# Convert target variable to numeric
y <- newConcrete_data$Compressive_Strength
y_matrix <- as.matrix(y)

# Fit the XGBoost model
xgb_model <- xgboost(data = X_matrix, label = y_matrix, objective = "reg:squarederror", nrounds = 500)

# Display the model summary
print(xgb_model)

# Predict using the XGBoost model
xgb_predictions <- predict(xgb_model, newdata = X_matrix)

# Calculate RMSE for XGBoost model
xgb_rmse <- sqrt(mean((xgb_predictions - y_matrix)^2))
cat("XGBoost Model RMSE: ", xgb_rmse, "\n")







# -------------------- Performance Comparison --------------------

# The model with the lower RMSE is the better performing model
if (rf_rmse < xgb_rmse) {
  cat("Random Forest performs better than XGBoost.\n")
} else if (xgb_rmse < rf_rmse) {
  cat("XGBoost performs better than Random Forest.\n")
} else {
  cat("Both Random Forest and XGBoost perform equally well.\n")
}





# Calculate R-squared for Random Forest model
rf_rsq <- 1 - sum((rf_predictions - y)^2) / sum((y - mean(y))^2)
cat("Random Forest R-squared: ", rf_rsq, "\n")

# Calculate R-squared for XGBoost model
xgb_rsq <- 1 - sum((xgb_predictions - y)^2) / sum((y - mean(y))^2)
cat("XGBoost R-squared: ", xgb_rsq, "\n")









# ---------------------------------------------
# Hypothesis Testing

# Hypothesis 1: Does Superplasticizer significantly affect Compressive Strength?
model_superplasticizer <- lm(Compressive_Strength ~ Superplasticizer, data = continuous_data)
summary(model_superplasticizer)

# Hypothesis 2: Does Water content reduce Compressive Strength?
cor_test_water <- cor.test(continuous_data$Water, continuous_data$Compressive_Strength)
cor_test_water

# Hypothesis 3: Does Fly Ash presence reduce Compressive Strength?
t_test_fly_ash <- t.test(Compressive_Strength ~ Contains_Fly_Ash, data = Concrete_data)
summary(t_test_fly_ash)

# Hypothesis 4: Is Age positively correlated with Compressive Strength?
age_corr <- cor.test(continuous_data$Age, continuous_data$Compressive_Strength)
age_corr

# Hypothesis 5: Does Concrete Category significantly affect Compressive Strength?
anova_category <- aov(Compressive_Strength ~ Concrete_Category, data = Concrete_data)
summary(anova_category)

# ---------------------------------------------
# Two-Way ANOVA - Interaction Effect between Concrete Category and Fly Ash Presence
two_way_anova <- aov(Compressive_Strength ~ Concrete_Category * Contains_Fly_Ash, data = Concrete_data)
summary(two_way_anova)

# ---------------------------------------------
# Conclusion

cat("Summary of Findings:\n")
cat("1. Cement and Superplasticizer are significant predictors of compressive strength.\n")
cat("2. Positive correlation between Age and compressive strength.\n")
cat("3. ANOVA shows significant differences in strength across concrete categories.\n")
cat("4. Water has a weak negative association with compressive strength.\n")

cat("\nConclusion:\n")
cat("This analysis provides insights on factors affecting concrete compressive strength, with Cement, Age, and Superplasticizer identified as key variables.\n")
cat("Further investigation with non-linear models or interaction terms could yield additional insights.")








