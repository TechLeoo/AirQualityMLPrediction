# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:13:36 2023

@author: lEO
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
import warnings
from Data_Preparation import data

warnings.filterwarnings("ignore")

# # FURTHER DATA PREPARATION AND SEGREGATION
# # (1) Dropping irrelevant columns due to Multicollinearity
# data = data.drop(["NMHC(GT)"], axis = 1)
# data = data.drop(["C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "PT08.S5(O3)", "NO2(GT)", "PT08.S4(NO2)", "AH", "RH", "T(C)", "PeakTime", "ValleyTime"], axis = 1)
data = data.drop(["PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)", "PT08.S4(NO2)", "AH", "RH", "T(C)", "PeakTime", "ValleyTime"], axis = 1)
# data = data.drop(["PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)", "PT08.S4(NO2)", "PeakTime", "ValleyTime"], axis = 1)



# (2) Fixing Missing Values
impute = SimpleImputer(strategy = "median")
# clean_data = pd.DataFrame(impute.fit_transform(data_removed_outliers,), columns = impute.feature_names_in_)
data1 = pd.DataFrame(impute.fit_transform(data.select_dtypes("number")), columns = impute.feature_names_in_)
data2 = data.select_dtypes("object")
data = data2.join(data1)


# (3) Extracting Features from Date and Time to Create New Features
data['Datetime'] = data['Date'] + " " + data['Time']
data.set_index("Datetime", inplace = True)
data.index = pd.to_datetime(data.index)
data = data.sort_index(axis = 0)

data["Year"] = data.index.year
data["Month"] = data.index.month
data["Day"] = data.index.day
data["HourTime"] = data.index.hour
data["DayOfWeek"] = data.index.day_of_week
data["Quarter"] = data.index.quarter


# (4) Dropping the Columns we won't be needing
data = data.drop(["Date", 'Time', 'DayOfWeekName'], axis = 1)

# (5) Grouping dependent and independent variables
x = data.drop(["CO(GT)"], axis = 1) 
y = data["CO(GT)"]

# (6) Splitting the dataset (80:20)
# x_train = x[x.index < "2005-01-01"]
# x_test = x[x.index >= "2005-01-01"]

# y_train = y[y.index < "2005-01-01"]
# y_test = y[y.index >= "2005-01-01"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# # (7) Feature Selection
# selector = SelectKBest(score_func = f_regression, k = 5)
# selector = RFE(RandomForestRegressor(random_state = 0), n_features_to_select = 5)
# x = pd.DataFrame(selector.fit_transform(x, y), columns = selector.get_feature_names_out())

# ---> Columns Score
# features_score = pd.DataFrame({"Features": selector.feature_names_in_, "Score": selector.scores_})
