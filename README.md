# Codesoft1
this is second Internship project,whcih is made by me 
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()


x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)



y_pred = model.predict(x_test)


from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


iris.target_names

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


