# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 03:32:13 2023

@author: lEO
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from No4_Further_Data_Preparation import x_train, x_test, y_train, y_test

# (1) Base Model Training
classifier = RandomForestClassifier(random_state= 0,)
model = classifier.fit(x_train, y_train)

# (2) Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)
print(f"Predictions from Training Data: \n{y_pred}")
print(f"Predictions from Test Data: \n{y_pred1}")

# (3) Training Evaluation
training_analysis = confusion_matrix(y_train, y_pred)
training_class_report = classification_report(y_train, y_pred)
training_accuracy = accuracy_score(y_train, y_pred)
training_precision = precision_score(y_train, y_pred, average='weighted')
training_recall = recall_score(y_train, y_pred, average='weighted')
training_f1_score = f1_score(y_train, y_pred, average='weighted')

# (4) Test Evaluation
test_analysis = confusion_matrix(y_test, y_pred1)
test_class_report = classification_report(y_test, y_pred1)
test_accuracy = accuracy_score(y_test, y_pred1)
test_precision = precision_score(y_test, y_pred1, average='weighted')
test_recall = recall_score(y_test, y_pred1, average='weighted')
test_f1_score = f1_score(y_test, y_pred1, average='weighted')

# (5) Cross Validation
score = cross_val_score(classifier, x_test, y_test, cv = 10)    
score_mean = round((score.mean() * 100), 2)
score_std_dev = round((score.std() * 100), 2)
print(f"Cross Validation Mean: {score_mean}")
print(f"Cross Validation Standard Deviation: {score_std_dev}")

# (6) Feature Importance
imp_features = pd.DataFrame({"Features": model.feature_names_in_, "Score": model.feature_importances_})
print(f"Important Features from Training: \n{imp_features}")