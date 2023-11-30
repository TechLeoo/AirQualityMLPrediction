# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:21:25 2023

@author: lEO
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# DATA INGESTION
# Getting the dataset
dataset = pd.read_csv("AirQuality.csv")

# Initial descriptive statistics
print("Data Schema")
dataset.info()

data_head = dataset.head()
print(f"\n\nTop 5 rows in the Data: \n{data_head}")

data_tail = dataset.tail()
print(f"\n\nBottom 5 rows in the Data: \n{data_tail}")

data_descriptive_statistic = dataset.describe()
print(f"\n\nData descriptive statistics: \n{data_descriptive_statistic}")

data_distinct_count = dataset.nunique()
print(f"\n\nUnique values in columns: \n{data_distinct_count}")

data_correlation_matrix = dataset.corr() # Get the correlation matrix of the independent variables
print(f"\n\nData correlation matrix: \n{data_correlation_matrix}")

data_null_count = dataset.isnull().sum()
print(f"\n\nCounting empty rows in each column: \n{data_null_count}")

data_total_null_count = dataset.isnull().sum().sum()
print(f"\n\nCounting total empty rows in the data: \n{data_total_null_count}")

data_column_mode = dataset.mode()
print(f"\n\nData Mode: \n{data_column_mode}")

            # ---> More Visuals
data_histogram = dataset.hist(bins = 10, figsize = (30, 15), alpha=0.7, color='brown')
plt.figure(figsize = (30, 10))
data_heatmap = sns.heatmap(data_correlation_matrix, annot = True, cmap = "coolwarm")
plt.title('Correlation Matrix of Independent Variables')
plt.show()
