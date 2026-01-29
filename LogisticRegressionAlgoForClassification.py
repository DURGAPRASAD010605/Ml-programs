import numpy as np
import pandas as pdf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.datasets import load_iris
data=load_iris()
x=data.data
y=data.target
data_binary=(y<2)
x=x[data_binary]
y=y[data_binary]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
logistic_regressor=LogisticRegression(random_state=42)
logistic_regressor.fit(x_train,y_train)
y_pred=logistic_regressor.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)
print("\n Classification Report: \n",classification_report(y_test,y_pred))
