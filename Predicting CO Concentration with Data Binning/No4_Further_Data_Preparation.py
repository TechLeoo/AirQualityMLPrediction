# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 01:03:21 2023

@author: lEO
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from No2_Data_Preparation import data

# (1) Dropping other ground truth readings
data = data.drop(["C6H6(GT)", "NOx(GT)", "NO2(GT)", "NMHC(GT)"], axis = 1)
count_null = data.isnull().sum()
print(data)
print(f"\n\n\nMissing Values in Columns: \n{count_null}")

# (2) Dropping all missing values in our label CO(GT) to improve our prediction and allow us train the model om the True Labels
data = data.dropna()
count_null = data.isnull().sum()
print(data)
print(f"\n\n\nMissing Values in Columns: \n{count_null}")

# Data Binning
data["PT08.S1(CO)"] = pd.cut(data["PT08.S1(CO)"], bins = 25, labels = False)
data["PT08.S2(NMHC)"] = pd.cut(data["PT08.S2(NMHC)"], bins = 25, labels = False)
data["PT08.S3(NOx)"] = pd.cut(data["PT08.S3(NOx)"], bins = 25, labels = False)
data["PT08.S4(NO2)"] = pd.cut(data["PT08.S4(NO2)"], bins = 25, labels = False)
data["PT08.S5(O3)"] = pd.cut(data["PT08.S5(O3)"], bins = 25, labels = False)

# # (3) Fixing Missing Values
# impute = SimpleImputer(strategy = "most_frequent")
# data.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10]] = impute.fit_transform(data.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10]])
# count_null = data.isnull().sum()
# print(data)
# print(f"\n\n\nMissing Values in Columns: \n{count_null}")

# (4) Extracting Features from Date and Time to Create New Features
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

print(data)

# (5) Dropping additional columns we won't be needing
data = data.drop(["Date", 'Time'], axis = 1)
print(data)

# (6) Grouping dependent and independent variables for prediction
x = data.drop(["CO(GT)"], axis = 1) 
y = data["CO(GT)"]
print(f"Independent Variables: \n{x}")
print(f"\n\nDependent Variables: \n{y}")

# (7) Splitting the dataset (80:20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(f"x_train: \n{x_train}")
print(f"y_train: \n{y_train}")
print(f"x_test: \n{x_test}")
print(f"y_test: \n{y_test}")