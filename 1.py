# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:13:44 2023

@author: lEO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")



# ---> Data Ingestion
# (1) Getting the dataset
dataset = pd.read_csv("AirQuality.csv")



# TASK 1: CO Prediction



# ---> Data Preparation
# (1) Dropping the empty columns
data = dataset.drop(["Unnamed: 15", "Unnamed: 16"], axis = 1)

# (2) Replacing all values with -200 with np.nan
data[data == -200] = np.nan

# (3) Creating the day of the week column
data["DayOfWeek"] = pd.to_datetime(data["Date"], dayfirst = True).dt.day_name()

# (4) Creating a column indicating Peak Time or Not. 
# (8AM-12PM and 6-10PM on working days) and (9am-12pm on non-working days).
def peak_time(dayofweek):
    WorkingDays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    NonWorkingDays = ["Saturday", "Sunday"]

    if (dayofweek["DayOfWeek"] in WorkingDays) and (dayofweek["Time"] >= "08:00:00" and dayofweek["Time"] <= "12:00:00"):
        return 1
    elif (dayofweek["DayOfWeek"] in WorkingDays) and (dayofweek["Time"] >= "18:00:00" and dayofweek["Time"] <= "22:00:00"):
        return 1
    elif (dayofweek["DayOfWeek"] in NonWorkingDays) and (dayofweek["Time"] >= "09:00:00" and dayofweek["Time"] <= "12:00:00"):
        return 1
    else:
        return 0
 
data["PeakTime"] = data.apply(peak_time, axis = 1)

# (5) Creating a column indicating Valley Road Usage or Not
# (Valley road usage is during the central hours of the night (2-6am))
def valley_time(time):
    if time["Time"] >= "02:00:00" and time["Time"] <= "06:00:00":
        return 1
    else:
        return 0

data["ValleyTime"] = data.apply(valley_time, axis = 1)



# ---> Visualization
def plot_normal_distribution_curve():
    data_numerical_columns = data.select_dtypes("number")
    count = 0
    while count < 13:
        data_to_plot = data_numerical_columns.iloc[:, count]
        
        # Plotting the histogram
        plt.figure(figsize = (15, 10))
        plt.hist(data_to_plot, bins = 10, density = True, alpha=0.7, rwidth = 8.5)
        plt.title(f"Distribution of {data_to_plot.name}", pad = 10, size = 25)
        plt.xlabel(f"{data_to_plot.name}")
        
        # Plotting the normal distribution
        x_values = np.linspace(start = min(data_to_plot), stop = max(data_to_plot), num = 200)
        y_values = norm.pdf(x = x_values, loc = data_to_plot.mean(), scale = data_to_plot.std())
        
        plt.plot(x_values, y_values, color='blue', label='Normal Distribution', linewidth = 1)
        plt.show()
        
        count += 1
        
data_histogram = data.hist(bins = 10, figsize = (30, 15), alpha=0.7, color='brown')
data_distribution = plot_normal_distribution_curve()



# ---> Further Data Preparation and Segregation 
# (1) Extracting Features from Date
data["Year"] = pd.to_datetime(data["Date"]).dt.year
data["Month"] = pd.to_datetime(data["Date"]).dt.month
data["Day"] = pd.to_datetime(data["Date"]).dt.day
data["Hour"] = pd.to_datetime(data["Time"], format = '%H:%M:%S').dt.hour
data = pd.get_dummies(data, columns = ["DayOfWeek"], dtype = np.int64, drop_first = True, prefix = "Date")
data = data.drop(["Date", "Time"], axis = 1)

x = data.drop(["CO(GT)"], axis = 1) 
y = data["CO(GT)"]

# (2) Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_distinct_count = data.nunique()
data_correlation_matrix = x.corr() # Get the correlation matrix of the independent variables
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
            # ---> More Visuals
plt.figure(figsize = (30, 10))
data_heatmap = sns.heatmap(data_correlation_matrix, annot = True, cmap = "coolwarm")
plt.title('Correlation Matrix of Independent Variables')
plt.show()

# (3) Fixing Missing Values

# # (4) Removing outliers in the data
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# clean_data = data[data > -3 & data < 3]






























































































