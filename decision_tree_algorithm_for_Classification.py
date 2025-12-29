import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report
data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.3,random_state=42)
clf=DecisionTreeClassifier(random_state=42)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Initial Model Accuracy: ", accuracy_score(y_test, y_pred))
print("\n Classification Report : \n", classification_report(y_test,y_pred))
param_grid={
    'criterion':['gini','entropy'],
    'max_depth':[None,5,10,15],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
    }
grid_search=GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
    )
grid_search.fit(x_train,y_train)
print("\n Best Parameters :", grid_search.best_params_)
print("Best Cross-validation Score:",grid_search.best_score_)
best_model=grid_search.best_estimator_
y_pred_best=best_model.predict(x_test)
print("\n Turned Model Accuracy : ", accuracy_score(y_test, y_pred_best))
print("\n Classification Report (Turned Model): \n",classification_report(y_test, y_pred_best))