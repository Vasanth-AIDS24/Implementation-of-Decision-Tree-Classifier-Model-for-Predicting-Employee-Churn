# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vasanth P
RegisterNumber:212224230295
/*

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]]) 

*/
```

## Output:
![decision tree classifier model](sam.png)\
DATA HEAD
<img width="1391" height="257" alt="image" src="https://github.com/user-attachments/assets/ec2661fd-fdc7-462b-b43c-a6b2504ad2d7" />

DATASET INFO:
<img width="498" height="362" alt="image" src="https://github.com/user-attachments/assets/2531c5fe-54e4-46e1-8d65-eb2a09ce6420" />

NULL DATASET:
<img width="312" height="255" alt="image" src="https://github.com/user-attachments/assets/b51eb219-dd41-439d-9a12-4a0085781e0d" />

Values count in left column:
<img width="312" height="97" alt="image" src="https://github.com/user-attachments/assets/2036cc58-dd46-4531-81d1-e146d6d70e9d" />

Dataset transformed head:
<img width="1371" height="242" alt="image" src="https://github.com/user-attachments/assets/6e117b90-757e-4709-b243-9869bec7b7a5" />

X.head:
<img width="1228" height="232" alt="image" src="https://github.com/user-attachments/assets/fb08b852-287b-4b60-a0cf-4a66dd66d7f3" />

Accuracy:
<img width="257" height="52" alt="image" src="https://github.com/user-attachments/assets/9e515205-fd67-41a8-acda-d862c41fb6c1" />

Data prediction:
<img width="1372" height="97" alt="image" src="https://github.com/user-attachments/assets/07f8f873-dc46-495a-8099-36fde5b62f92" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
