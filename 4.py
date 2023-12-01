# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:40:17 2023

@author: lEO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Fixing missing values
from sklearn.impute import SimpleImputer
# Fixing unbalanced dataset
from imblearn.over_sampling import SMOTE
# Splitting the data
from sklearn.model_selection import train_test_split
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
# Selecting the best K for KNN
from sklearn.model_selection import GridSearchCV
# Cross validation
from sklearn.model_selection import cross_val_score
# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# DATA INGESTION
# Getting the dataset
dataset = pd.read_csv("AirQuality.csv")

# Initial descriptive statistics
dataset.info()
dataset_descriptive_statistic = dataset.describe()



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



# FURTHER DATA PREPARATION AND SEGREGATION
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
        elif 0 <= row["CO(GT)"] <= 2.4:
            data_with_aqi["CO(GT)"].append(0)
        elif 2.5 <= row["CO(GT)"] <= 4.4:
            data_with_aqi["CO(GT)"].append(1)
        elif 4.5 <= row["CO(GT)"] <= 8.4:
            data_with_aqi["CO(GT)"].append(2)
        elif 8.5 <= row["CO(GT)"] <= 30.4:
            data_with_aqi["CO(GT)"].append(3)
        elif 30.5 <= row["CO(GT)"] <= 100.4:
            data_with_aqi["CO(GT)"].append(4)
        elif row["CO(GT)"] > 100:
            data_with_aqi["CO(GT)"].append(5)
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
        elif 8.5 <= row["C6H6(GT)"] <= 30.4:
            data_with_aqi["C6H6(GT)"].append(4)
        elif row["C6H6(GT)"] > 30.5:
            data_with_aqi["C6H6(GT)"].append(5)
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
        elif 201 <= row["NO2(GT)"] <= 400:
            data_with_aqi["NO2(GT)"].append(4)
        elif row["NO2(GT)"] > 401:
            data_with_aqi["NO2(GT)"].append(5)
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
        elif 120.5 <= row["NOx(GT)"] <= 180.4:
            data_with_aqi["NOx(GT)"].append(4)
        elif row["NOx(GT)"] > 180.5:
            data_with_aqi["NOx(GT)"].append(5)
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
        elif 201 <= row["NMHC(GT)"] <= 300:
            data_with_aqi["NMHC(GT)"].append(4)
        elif row["NMHC(GT)"] > 300:
            data_with_aqi["NMHC(GT)"].append(5)
        else:
            data_with_aqi["NMHC(GT)"].append(np.nan)
            
    return pd.DataFrame(data_with_aqi)

# Assuming 'data' is a DataFrame
dataframe = AQI(data)
data["AQI"] = np.max(dataframe, axis = 1)

# (2) Dropping columns that won't be useful for AQI prediction
data = data.drop(["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"], axis = 1)

# (3) Dropping all missing values in our label CO(GT) to improve our prediction
data = data.dropna(subset = "AQI")

# (4) Fixing missing columns
impute = SimpleImputer(strategy = "most_frequent")
# impute = SimpleImputer(strategy = "median")
data.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9]] = impute.fit_transform(data.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9]])

# (5) Extracting Features from Date and Time to Create New Features
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

# (6) Dropping additional columns we won't be needing
data = data.drop(["Date", 'Time', 'DayOfWeekName'], axis = 1)

# (7) Grouping dependent and independent variables
x = data.drop(["AQI"], axis = 1) 
y = data["AQI"]

# (8) Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr() # Get the correlation matrix of the independent variables
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_class_count = y.value_counts()

            # ---> More Visuals
data_histogram = data.hist(bins = 10, figsize = (30, 15), alpha=0.7, color='brown')
plt.figure(figsize = (30, 10))
data_heatmap = sns.heatmap(data_correlation_matrix, annot = True, cmap = "coolwarm")
plt.title('Correlation Matrix of Independent Variables')
plt.show()

# (9) Splitting the dataset (80:20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# (9) Dealing with an Unbalanced Dataset
unbalanced_model_fix = SMOTE()
x_train, y_train = unbalanced_model_fix.fit_resample(x_train, y_train)
y_train_class_count = y_train.value_counts()





# Model Training and Evaluation
# (1) Base Model Training
# classifier = LogisticRegression(random_state = 0)
classifier = XGBClassifier()
# classifier = GradientBoostingClassifier(random_state = 0)
# classifier = RandomForestClassifier(random_state= 0,)
# classifier = DecisionTreeClassifier(random_state= 0)
model = classifier.fit(x_train, y_train)

# (2) Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)

# (3) Evaluation
training_analysis = confusion_matrix(y_train, y_pred)
training_class_report = classification_report(y_train, y_pred)
training_accuracy = accuracy_score(y_train, y_pred)
training_precision = precision_score(y_train, y_pred, average='weighted')
training_recall = recall_score(y_train, y_pred, average='weighted')
training_f1_score = f1_score(y_train, y_pred, average='weighted')
# training_roc_auc = roc_auc_score(y_train, y_pred, multi_class = 'ovo', average='weighted')

test_analysis = confusion_matrix(y_test, y_pred1)
test_class_report = classification_report(y_test, y_pred1)
test_accuracy = accuracy_score(y_test, y_pred1)
test_precision = precision_score(y_test, y_pred1, average='weighted')
test_recall = recall_score(y_test, y_pred1, average='weighted')
test_f1_score = f1_score(y_test, y_pred1, average='weighted')
# test_roc_auc = roc_auc_score(y_test, y_pred1, multi_class = 'ovo', average='weighted')

# (4) Cross Validation
score = cross_val_score(classifier, x_train, y_train, cv = 10)    
score_mean = round((score.mean() * 100), 2)
score_std_dev = round((score.std() * 100), 2)

# (5) Feature Importance
imp_features = pd.DataFrame({"Features": model.feature_names_in_, "Score": model.feature_importances_})

# # (6) Model Tuning
# def hyper_parameter_tuning(cv_num):
#     # Define the parameter grid
#     param_grid = {'n_neighbors': range(1, 21)}
    
#     # Perform grid search using cross-validation
#     grid_search = GridSearchCV(KNeighborsClassifier(metric = 'euclidean'), param_grid, cv=cv_num)
#     grid_search.fit(x_train, y_train)
    
#     # Print the best parameter and best score
#     print("Best k value: ", grid_search.best_params_['n_neighbors'])
#     print("Best score: ", grid_search.best_score_)

