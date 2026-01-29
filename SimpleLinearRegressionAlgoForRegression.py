import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
np.random.seed(42)
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
linear_regressor=LinearRegression()
linear_regressor.fit(x_train, y_train)
y_pred=linear_regressor.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error(MSE) :",mse)
print("Mean Absolute Error(MAE) :", mse)
print("R-Squared(R2) Score: ",r2)
print("\n Model Coefficients: ")
print(f"Intercept: {linear_regressor.intercept_[0]:.2f}")
print(f"Slope: {linear_regressor.coef_[0][0]:.2f}")
