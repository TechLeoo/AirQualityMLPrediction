# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 01:11:25 2023

@author: lEO
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, RidgeCV
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from No4_Further_Data_Preparation import x_train, x_test, y_train, y_test

# (1) Model Training
regressor = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state = 0)
# regressor = LinearRegression()
# regressor = Lasso(alpha = 0.01)
# regressor = RidgeCV(alphas = 0.01)
# regressor = SVR(kernel = "linear", C = 0.05, epsilon = 0.5)
# regressor = KNeighborsRegressor()
# regressor = MLPRegressor(max_iter = 1000, activitation = "tanh", learning_rate = "adaptive", warm_start = True)
model = regressor.fit(x_train, y_train)

# (2) Model Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)
print(f"Predictions from Training Data: \n{y_pred}")
print(f"Predictions from Test Data: \n{y_pred1}")

# (3) Model Evaluation
# Training Evaluation
rmse_training = np.sqrt(mean_squared_error(y_train, y_pred))
r2_training = r2_score(y_train, y_pred)
print(f"RMSE for Training Data: \n{rmse_training}")
print(f"R-Squared for Training Data: \n{r2_training}")

# Test Evaluation
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred1))
r2_test = r2_score(y_test, y_pred1)
print(f"RMSE for Test Data: \n{rmse_test}")
print(f"R-Squared for Test Data: \n{r2_test}")

# (4) Model Cross Validation
score = cross_val_score(regressor, x_test, y_test, cv = 10)
score_mean = round((score.mean() * 100), 2)
score_std_dev = round((score.std() * 100), 2)
print(f"Cross Validation Mean: {score_mean}")
print(f"Cross Validation Standard Deviation: {score_std_dev}")

# (5) Feature Importance
imp_features = pd.DataFrame({"Features": model.feature_names_in_, "Score": model.feature_importances_})
print(f"Important Features from Training: \n{imp_features}")