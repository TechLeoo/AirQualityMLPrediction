# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:27:08 2023

@author: lEO
"""

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from Further_DataPreparation_and_Segregation import x, y, x_train, x_test, y_train, y_test

warnings.filterwarnings("ignore")

# # MODEL TRAINING AND EVALUATION
# # (1) Training
# # regressor = GradientBoostingRegressor(random_state = 0, criterion = "squared_error",)
# regressor = LinearRegression()
regressor = RandomForestRegressor(random_state = 0)
# regressor = DecisionTreeRegressor(random_state = 0,)
# # regressor = BernoulliNB()
# regressor = XGBRegressor()
model = regressor.fit(x_train, y_train)

# (2) Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)

# (3) Evaluation
mse = mean_squared_error(y_test, y_pred1)
r2 = r2_score(y_test, y_pred1)

# (4) Validation
score = cross_val_score(regressor, x, y, cv = 10)
score_mean = round((score.mean() * 100), 2)
score_std_dev = round((score.std() * 100), 2)