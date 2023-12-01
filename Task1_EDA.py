# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:23:04 2023

@author: lEO
"""

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Data_Preparation import data

warnings.filterwarnings("ignore")

# Exploratory Data Analysis
print("Data Schema")
data.info()

data_head = data.head()
print(f"\n\nTop 5 rows in the Data: \n{data_head}")

data_tail = data.tail()
print(f"\n\nBottom 5 rows in the Data: \n{data_tail}")

data_descriptive_statistic = data.describe()
print(f"\n\nData descriptive statistics: \n{data_descriptive_statistic}")

data_distinct_count = data.nunique()
print(f"\n\nUnique values in columns: \n{data_distinct_count}")

data_correlation_matrix = data.corr() # Get the correlation matrix of the independent variables
print(f"\n\nData correlation matrix: \n{data_correlation_matrix}")

data_null_count = data.isnull().sum()
print(f"\n\nCounting empty rows in each column: \n{data_null_count}")

data_total_null_count = data.isnull().sum().sum()
print(f"\n\nCounting total empty rows in the data: \n{data_total_null_count}")

data_column_mode = data.mode()
print(f"\n\nData Mode: \n{data_column_mode}")

            # ---> More Visuals
plt.figure(figsize = (30, 10))
data_heatmap = sns.heatmap(data_correlation_matrix, annot = True, cmap = "coolwarm")
plt.title('Correlation Matrix of Independent Variables')
plt.show()