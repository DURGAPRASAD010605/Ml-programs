# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 09:59:09 2025

@author: LAB
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
rf_clf=RandomForestClassifier(n_estimators=100,random_state=42)
rf_clf.fit(x_train,y_train)
y_pred=rf_clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("\n Classification Report: \n",classification_report(y_test,y_pred))
feature_importances=rf_clf.feature_importances_
print("\nFeature Importance: ")
for name,importance in zip(data.feature_names,feature_importances):
    print(f"{name}:{importance: .4f}")