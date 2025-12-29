# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 10:15:50 2025

@author: LAB
"""

import numpy as np
#from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
#data=fetch_california_housing()
data=load_diabetes()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
rf_clf=RandomForestRegressor(n_estimators=100,random_state=42)
rf_regressor=RandomForestRegressor(n_estimators=100,random_state=42)
rf_regressor.fit(x_train,y_train)
y_pred=rf_regressor.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error(MSE) :", mse)
print("Mean Absolute Error(MAE) :", mse)
print("R-Squared(R2) Score: ",r2)
feature_importances=rf_regressor.feature_importances_
print("\nFeature Importance: ")
for name,importance in zip(data.feature_names,feature_importances):
    print(f"{name}:{importance: .4f}")