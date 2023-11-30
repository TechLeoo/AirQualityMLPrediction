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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeRegressor
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
        
# data_histogram = data.hist(bins = 10, figsize = (30, 15), alpha=0.7, color='brown')
# data_distribution = plot_normal_distribution_curve()


# (8) Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr() # Get the correlation matrix of the independent variables
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_column_mode = data.mode()

# #             # ---> More Visuals
# # plt.figure(figsize = (30, 10))
# # data_heatmap = sns.heatmap(data_correlation_matrix, annot = True, cmap = "coolwarm")
# # plt.title('Correlation Matrix of Independent Variables')
# # plt.show()

# (9) Dropping the Columns we won't be needing
data = data.drop(["NMHC(GT)"], axis = 1)






# # FURTHER DATA PREPARATION AND SEGREGATION
# # (1) Dropping irrelevant columns due to Multicollinearity
# data = data.drop(["C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "PT08.S5(O3)", "NO2(GT)", "PT08.S4(NO2)", "AH", "RH", "T(C)", "PeakTime", "ValleyTime"], axis = 1)
# data = data.drop(["C6H6(GT)", "NOx(GT)", "NO2(GT)", "PeakTime", "ValleyTime"], axis = 1)
# data = data.drop(["PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)", "PT08.S4(NO2)", "AH", "RH", "T(C)", "PeakTime", "ValleyTime"], axis = 1)
# data = data.drop(["PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)", "PT08.S4(NO2)", "PeakTime", "ValleyTime"], axis = 1) # BEST SO FAR --> Random Forest
# data = data.drop(["PeakTime", "ValleyTime",], axis = 1) # BEST --> XGBoost
# data = data.drop(["PT08.S3(NOx)", "PT08.S5(O3)", "NO2(GT)", "PT08.S4(NO2)", "AH", "RH", "T(C)", "PeakTime", "ValleyTime"], axis = 1)
# data = data.drop(["PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)", "PT08.S4(NO2)", "PeakTime", "ValleyTime"], axis = 1)
# data = data.drop(["AH", "RH", "T(C)", "PeakTime", "ValleyTime"], axis = 1)
data = data.drop(["C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "PT08.S5(O3)", "NO2(GT)", "PT08.S4(NO2)", "PeakTime", "ValleyTime"], axis = 1)


data = data.dropna(subset = "CO(GT)")

# (3) Fixing Missing Values
# impute = SimpleImputer(strategy = "most_frequent")
impute = SimpleImputer(strategy = "median")
data.iloc[:, [3, 4, 5, 6]] = impute.fit_transform(data.iloc[:, [3, 4, 5, 6]])

# (7) Extracting Features from Date and Time to Create New Features
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

# (4) Grouping dependent and independent variables
x = data.drop(["CO(GT)"], axis = 1) 
y = data["CO(GT)"]

# # (5) Feature Selection
# selector = SelectKBest(score_func = f_regression, k = 10)
# # selector = SelectFromModel(max_features = 5)
# # selector = RFE(RandomForestRegressor(random_state = 0), n_features_to_select = 5)
# x = pd.DataFrame(selector.fit_transform(x, y), columns = selector.get_feature_names_out())

# # ---> Columns Score
# features_score = pd.DataFrame({"Features": selector.feature_names_in_, "Score": selector.scores_})

# (6) Splitting the dataset (80:20)
# x_train = x[x.index < "2005-01-01"]
# x_test = x[x.index >= "2005-01-01"]

# y_train = y[y.index < "2005-01-01"]
# y_test = y[y.index >= "2005-01-01"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)




# # MODEL TRAINING AND EVALUATION
# # (1) Training
# # regressor = GradientBoostingRegressor(random_state = 0, criterion = "squared_error",)
# regressor = LinearRegression()
regressor = RandomForestRegressor(random_state = 0)
# regressor = DecisionTreeRegressor(random_state = 0,)
# # regressor = BernoulliNB()
# regressor = XGBRegressor()
model = regressor.fit(x_train, y_train)

# (2) Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)


mse = mean_squared_error(y_test, y_pred1)
r2 = r2_score(y_test, y_pred1)


# Cross Validation
score = cross_val_score(regressor, x, y, cv = 10)
score_mean = round((score.mean() * 100), 2)
score_std_dev = round((score.std() * 100), 2)


# Feature Importance
imp_features = pd.DataFrame({"Features": model.feature_names_in_, "Score": model.feature_importances_})