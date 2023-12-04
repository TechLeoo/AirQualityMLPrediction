# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 03:21:00 2023

@author: lEO
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from No2_Data_Preparation import data

# (1) Creating my AQI standard
def AQI(dataframe):
    data_with_aqi = {
        "CO(GT)": [],
        "C6H6(GT)": [],
        "NO2(GT)": [],
        "NOx(GT)": [],
        "NMHC(GT)": [],
    }
    
    for index, row in dataframe.iterrows():
        # CO(GT)
        if pd.isna(row["CO(GT)"]):
            data_with_aqi["CO(GT)"].append(np.nan)
        elif 0 <= row["CO(GT)"] <= 4.4:
            data_with_aqi["CO(GT)"].append(0)
        elif 4.5 <= row["CO(GT)"] <= 9.4:
            data_with_aqi["CO(GT)"].append(1)
        elif 9.5 <= row["CO(GT)"] <= 14.4:
            data_with_aqi["CO(GT)"].append(2)
        elif 14.5 <= row["CO(GT)"] <= 24.4:
            data_with_aqi["CO(GT)"].append(3)
        elif row["CO(GT)"] > 24.4:
            data_with_aqi["CO(GT)"].append(4)
        else:
            data_with_aqi["CO(GT)"].append(np.nan)

        # C6H6(GT)
        if pd.isna(row["C6H6(GT)"]):
            data_with_aqi["C6H6(GT)"].append(np.nan)
        elif 0 <= row["C6H6(GT)"] <= 0.54:
            data_with_aqi["C6H6(GT)"].append(0)
        elif 0.55 <= row["C6H6(GT)"] <= 2.4:
            data_with_aqi["C6H6(GT)"].append(1)
        elif 2.5 <= row["C6H6(GT)"] <= 4.4:
            data_with_aqi["C6H6(GT)"].append(2)
        elif 4.5 <= row["C6H6(GT)"] <= 8.4:
            data_with_aqi["C6H6(GT)"].append(3)
        elif row["C6H6(GT)"] > 8.4:
            data_with_aqi["C6H6(GT)"].append(4)
        else:
            data_with_aqi["C6H6(GT)"].append(np.nan)

        # NO2(GT)
        if pd.isna(row["NO2(GT)"]):
            data_with_aqi["NO2(GT)"].append(np.nan)
        elif 0 <= row["NO2(GT)"] <= 25:
            data_with_aqi["NO2(GT)"].append(0)
        elif 26 <= row["NO2(GT)"] <= 50:
            data_with_aqi["NO2(GT)"].append(1)
        elif 51 <= row["NO2(GT)"] <= 100:
            data_with_aqi["NO2(GT)"].append(2)
        elif 101 <= row["NO2(GT)"] <= 200:
            data_with_aqi["NO2(GT)"].append(3)
        elif row["NO2(GT)"] > 200:
            data_with_aqi["NO2(GT)"].append(4)
        else:
            data_with_aqi["NO2(GT)"].append(np.nan)

        # NOx(GT)
        if pd.isna(row["NOx(GT)"]):
            data_with_aqi["NOx(GT)"].append(np.nan)
        elif 0 <= row["NOx(GT)"] <= 30.4:
            data_with_aqi["NOx(GT)"].append(0)
        elif 30.5 <= row["NOx(GT)"] <= 60.4:
            data_with_aqi["NOx(GT)"].append(1)
        elif 60.5 <= row["NOx(GT)"] <= 90.4:
            data_with_aqi["NOx(GT)"].append(2)
        elif 90.5 <= row["NOx(GT)"] <= 120.4:
            data_with_aqi["NOx(GT)"].append(3)
        elif row["NOx(GT)"] > 120.4:
            data_with_aqi["NOx(GT)"].append(4)
        else:
            data_with_aqi["NOx(GT)"].append(np.nan)

        # NMHC(GT)
        if pd.isna(row["NMHC(GT)"]):
            data_with_aqi["NMHC(GT)"].append(np.nan)
        elif 0 <= row["NMHC(GT)"] <= 50:
            data_with_aqi["NMHC(GT)"].append(0)
        elif 51 <= row["NMHC(GT)"] <= 100:
            data_with_aqi["NMHC(GT)"].append(1)
        elif 101 <= row["NMHC(GT)"] <= 150:
            data_with_aqi["NMHC(GT)"].append(2)
        elif 151 <= row["NMHC(GT)"] <= 200:
            data_with_aqi["NMHC(GT)"].append(3)
        elif row["NMHC(GT)"] > 200:
            data_with_aqi["NMHC(GT)"].append(4)
        else:
            data_with_aqi["NMHC(GT)"].append(np.nan)
            
    return pd.DataFrame(data_with_aqi)

# Assuming 'data' is a DataFrame
dataframe = AQI(data)
data["AQI"] = np.max(dataframe, axis = 1)
print(f"Hourly AQI: \n{dataframe}")
print(f"\n\nData with AQI: \n{data}")

# (2) Dropping columns that won't be useful for AQI prediction
data = data.drop(["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"], axis = 1)
count_null = data.isnull().sum()
print(data)
print(f"\n\n\nMissing Values in Columns: \n{count_null}")

# (3) Removing all missing values across the rows to improve integrity of prediction after training
data = data.dropna()
count_null = data.isnull().sum()
print(data)
print(f"\n\n\nMissing Values in Columns: \n{count_null}")

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
x = data.drop(["AQI"], axis = 1) 
y = data["AQI"]
print(f"Independent Variables: \n{x}")
print(f"\n\nDependent Variables: \n{y}")

# (7) Splitting the dataset (80:20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(f"x_train: \n{x_train}")
print(f"y_train: \n{y_train}")
print(f"x_test: \n{x_test}")
print(f"y_test: \n{y_test}")

# (8) Dealing with an Unbalanced Dataset
unbalanced_model_fix = SMOTE()
x_train, y_train = unbalanced_model_fix.fit_resample(x_train, y_train)
y_train_class_count = y_train.value_counts()
print(f"x_train: \n{x_train}")
print(f"y_train: \n{y_train}")
print(f"\n\nClass Count: \n{y_train_class_count}")