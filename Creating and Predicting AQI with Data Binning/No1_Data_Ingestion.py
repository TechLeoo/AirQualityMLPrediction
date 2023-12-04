# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 00:44:29 2023

@author: lEO
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Getting the dataset
dataset = pd.read_csv("AirQuality.csv")
print(dataset)

# Initial Exploratory dataset Analysis
dataset.info()
dataset_head = dataset.head()
print(f"dataset Head: \n\n{dataset_head}")
dataset_tail = dataset.tail()
print(f"dataset Tail: \n\n{dataset_tail}")
dataset_descriptive_statistic = dataset.describe()
print(f"Descriptive Statistics: \n\n{dataset_descriptive_statistic}")
dataset_distinct_count = dataset.nunique()
print(f"dataset Distinct Count: \n\n{dataset_distinct_count}")
dataset_correlation_matrix = dataset.corr() 
print(f"Correlation Matrix: \n\n{dataset_correlation_matrix}")
dataset_null_count = dataset.isnull().sum()
print(f"Missing Values in each Column: \n\n{dataset_null_count}")
dataset_total_null_count = dataset.isnull().sum().sum()
print(f"dataset Total Missing Values: {dataset_total_null_count}")

            # ---> Visualization
dataset_histogram = dataset.hist(bins = 10, figsize = (30, 15), alpha=0.7, color='brown')
plt.figure(figsize = (30, 10))
dataset_heatmap = sns.heatmap(dataset_correlation_matrix, annot = True, cmap = "coolwarm")
plt.title('Correlation Matrix of Independent Variables')
plt.show()