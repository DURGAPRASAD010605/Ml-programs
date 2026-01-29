import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.3,random_state=42)
svc_classifier=SVC(kernel='linear',random_state=42)
svc_classifier.fit(x_train,y_train)
y_pred=svc_classifier.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy: ",accuracy)
print("\n classification report :\n ",classification_report(y_test, y_pred))
