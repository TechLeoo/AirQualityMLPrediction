# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:21:25 2023

@author: lEO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# DATA INGESTION
# Getting the dataset
dataset = pd.read_csv("AirQuality.csv")

# Initial descriptive statistics
print("Data Schema")
dataset.info()

dataset_descriptive_statistic = dataset.describe()
print(f"\n\nData descriptive statistics: \n{dataset_descriptive_statistic}")



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

# # (6) Visualization
# def plot_normal_distribution_curve():
#     data_numerical_columns = data.select_dtypes("number")
#     count = 0
#     while count < 13:
#         data_to_plot = data_numerical_columns.iloc[:, count]
        
#         # Plotting the histogram
#         plt.figure(figsize = (15, 10))
#         plt.hist(data_to_plot, bins = 10, density = True, alpha=0.7, rwidth = 8.5)
#         plt.title(f"Distribution of {data_to_plot.name}", pad = 10, size = 25)
#         plt.xlabel(f"{data_to_plot.name}")
        
#         # Plotting the normal distribution
#         x_values = np.linspace(start = min(data_to_plot), stop = max(data_to_plot), num = 200)
#         y_values = norm.pdf(x = x_values, loc = data_to_plot.mean(), scale = data_to_plot.std())
        
#         plt.plot(x_values, y_values, color='blue', label='Normal Distribution', linewidth = 1)
#         plt.show()
        
#         count += 1
        
# data_distribution = plot_normal_distribution_curve()

# (7) Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr() # Get the correlation matrix of the independent variables
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_column_mode = data.mode()

            # ---> More Visuals
data_histogram = data.hist(bins = 10, figsize = (30, 15), alpha=0.7, color='brown')
plt.figure(figsize = (30, 10))
data_heatmap = sns.heatmap(data_correlation_matrix, annot = True, cmap = "coolwarm")
plt.title('Correlation Matrix of Independent Variables')
plt.show()

# (8) Dropping the NMHC columns readings due to excessive missing values in the columns
data = data.drop(["NMHC(GT)"], axis = 1)



# FURTHER DATA PREPARATION AND SEGREGATION
# (1) Dropping other ground truths and other irrelevant columns
data = data.drop(["C6H6(GT)", "NOx(GT)", "NO2(GT)",], axis = 1)

# (2) Dropping all missing values in our label CO(GT) to improve our prediction and allow us train the model om the True Labels
data = data.dropna(subset = "CO(GT)")

# (3) Fixing Missing Values
impute = SimpleImputer(strategy = "median")
data.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10]] = impute.fit_transform(data.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10]])


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

# (5) Dropping additional columns we won't be needing
data = data.drop(["Date", 'Time', 'DayOfWeekName'], axis = 1)

# (6) Grouping dependent and independent variables for prediction
x = data.drop(["CO(GT)"], axis = 1) 
y = data["CO(GT)"]

# (7) Splitting the dataset (80:10:10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# # MODEL TRAINING AND EVALUATION
# # (1) Training
# regressor = XGBRegressor()
regressor = XGBRegressor(n_estimators=1000, learning_rate=0.1)
model = regressor.fit(x_train, y_train)

# (2) Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)

# (3) Evaluation
rmse_training = np.sqrt(mean_squared_error(y_train, y_pred))
r2_training = r2_score(y_train, y_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred1))
r2_test = r2_score(y_test, y_pred1)

# (3) Cross Validation
score = cross_val_score(regressor, x_test, y_test, cv = 10)
score_mean = round((score.mean() * 100), 2)
score_std_dev = round((score.std() * 100), 2)

# (4) Feature Importance
imp_features = pd.DataFrame({"Features": model.feature_names_in_, "Score": model.feature_importances_})

# # (5) Model Tuning
# def finding_the_best_k_KNN_method1(cv_num):
#     # Define the parameter grid
#     param_grid = {'n_neighbors': range(1, 21)}
    
#     # Perform grid search using cross-validation
#     grid_search = GridSearchCV(KNeighborsClassifier(metric = 'euclidean'), param_grid, cv=cv_num)
#     grid_search.fit(x_train, y_train)
    
#     # Print the best parameter and best score
#     print("Best k value: ", grid_search.best_params_['n_neighbors'])
#     print("Best score: ", grid_search.best_score_)















