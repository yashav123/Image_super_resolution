import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('lifeexp.csv')

# Clean the column names by stripping any leading or trailing spaces
data.columns = data.columns.str.strip()

# Drop rows with missing values
data_clean = data.dropna()

# Exclude non-numeric columns for correlation analysis
numeric_data = data_clean.select_dtypes(include=[np.number])

# Perform correlation analysis
correlation_matrix = numeric_data.corr()
print(correlation_matrix['Life expectancy'].sort_values(ascending=False))

# Function to calculate RMSE
def calculate_rmse(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    y_pred = X_b.dot(theta_best)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return rmse

# Target variable
y = data_clean['Life expectancy'].values

# Evaluate RMSE for each feature
features = numeric_data.columns.drop('Life expectancy')
rmse_values = {}

for feature in features:
    X = data_clean[[feature]].values
    rmse = calculate_rmse(X, y)
    rmse_values[feature] = rmse
    print(f"Feature: {feature}, RMSE: {rmse}")

# Find the feature with the least RMSE
best_feature = min(rmse_values, key=rmse_values.get)
print(f"Best feature: {best_feature} with RMSE: {rmse_values[best_feature]}")

# Visualize the best feature
X_best = data_clean[[best_feature]].values
X_b_best = np.c_[np.ones((X_best.shape[0], 1)), X_best]
theta_best = np.linalg.inv(X_b_best.T.dot(X_b_best)).dot(X_b_best.T).dot(y)
y_pred_best = X_b_best.dot(theta_best)

plt.scatter(data_clean[best_feature], y, color='blue', label='Actual Life Expectancy')
plt.scatter(data_clean[best_feature], y_pred_best, color='red', label='Predicted Life Expectancy')
plt.xlabel(best_feature)
plt.ylabel('Life Expectancy')
plt.legend()
plt.title(f'Life Expectancy vs. {best_feature}')
plt.show()
