# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:09:01 2023

@author: lEO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

import warnings
from Initial_EDA import dataset

warnings.filterwarnings("ignore")

# DATA PREPARATION
# (1) Dropping the empty columns
data = dataset.drop(["Unnamed: 15", "Unnamed: 16"], axis = 1)

# (2) Replacing all values with -200 with np.nan
data[data == -200] = np.nan

# (3) Creating the day of the week column
data["DayOfWeekName"] = pd.to_datetime(data["Date"]).dt.day_name()

# (4) Creating a column indicating Peak Time or Not. 
# (8AM-12PM and 6-10PM on working days) and (9am-12pm on non-working days).
def peak_time(dayofweek):
    WorkingDays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    NonWorkingDays = ["Saturday", "Sunday"]

    if (dayofweek["DayOfWeekName"] in WorkingDays) and (dayofweek["Time"] >= "08:00:00" and dayofweek["Time"] <= "12:00:00"):
        return 1
    elif (dayofweek["DayOfWeekName"] in WorkingDays) and (dayofweek["Time"] >= "18:00:00" and dayofweek["Time"] <= "22:00:00"):
        return 1
    elif (dayofweek["DayOfWeekName"] in NonWorkingDays) and (dayofweek["Time"] >= "09:00:00" and dayofweek["Time"] <= "12:00:00"):
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

# (6) Visualization
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