# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values. 


## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JAYATHRAA V
RegisterNumber: 212219220018 

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:
Data Head:
![image](https://user-images.githubusercontent.com/107881970/174664200-7dea3fac-6b83-4833-80df-07cd09202d6a.png)


Data Info:

![image](https://user-images.githubusercontent.com/107881970/174664218-51f0ce5b-ca5b-468e-9679-6ed5fcfbb0c3.png)


Data Isnull:

![image](https://user-images.githubusercontent.com/107881970/174664247-a1fa7370-86e6-42e4-8e41-c422b1dad58d.png)


Data Head:

![image](https://user-images.githubusercontent.com/107881970/174664269-04b61501-7907-48db-823f-505e6c87a631.png)

MSE:

![image](https://user-images.githubusercontent.com/107881970/174664296-751d41b2-4cda-472d-b1f6-fe2ce1336f81.png)


R2:

![image](https://user-images.githubusercontent.com/107881970/174664361-286086a1-2786-42c2-a0bb-97ef3ad2393e.png)


Predicted Value:

![image](https://user-images.githubusercontent.com/107881970/174664389-0c739f08-760d-4e66-9b41-37c721d7b32f.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
