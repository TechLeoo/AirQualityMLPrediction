# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 01:00:02 2023

@author: lEO
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from No2_Data_Preparation import data

# (1) Exploratory Data Analysis
data.info()
data_head = data.head()
print(f"Data Head: \n\n{data_head}")
data_tail = data.tail()
print(f"Data Tail: \n\n{data_tail}")
data_descriptive_statistic = data.describe()
print(f"Descriptive Statistics: \n\n{data_descriptive_statistic}")
data_distinct_count = data.nunique()
print(f"Data Distinct Count: \n\n{data_distinct_count}")
data_correlation_matrix = data.corr() 
print(f"Correlation Matrix: \n\n{data_correlation_matrix}")
data_null_count = data.isnull().sum()
print(f"Missing Values in each Column: \n\n{data_null_count}")
data_total_null_count = data.isnull().sum().sum()
print(f"Data Total Missing Values: {data_total_null_count}")

            # ---> Visualization
data_histogram = data.hist(bins = 10, figsize = (30, 15), alpha=0.7, color='brown')
plt.figure(figsize = (30, 10))
data_heatmap = sns.heatmap(data_correlation_matrix, annot = True, cmap = "coolwarm")
plt.title('Correlation Matrix of Independent Variables')
plt.show()

# (2) Visualization
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
        
data_distribution = plot_normal_distribution_curve()