# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 09:33:35 2025

@author: LAB
"""

import numpy as np
from sklearn.datasets import load_diabetes
#from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
data=load_diabetes()
#data=fetch_california_housing()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
regressor=DecisionTreeRegressor(random_state=42)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Aquared Error(MSE) :", mse)
print("R-Squared(R2) Score: ",r2)
print("\n Sample predictions:")
for true ,pred in zip(y_test[:5],y_pred[:5]):
    print(f"Actual : {true:.2f} , predicted: {pred:.2f} ")