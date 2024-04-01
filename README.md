# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ABINAYA S
RegisterNumber:  212222230002
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_

```

## Output:
## 1)HEAD:
![image](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/1af631da-608b-42f6-8fee-be4ddf3f6efa)
## GRAPH OF PLOTTED DATA:
![image](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/b64d7aea-ec3d-452f-8ffe-2ac5e0f82f4a)
## TRAINED DATA:
![image](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/5d83b9ed-e8e7-4aa9-90c6-8b5a9a40752b)
## LINE OF REGRESSION:
![image](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/b1bb8601-57d8-4a6e-9ae0-f60fef3b3100)
## COEFFICIENT AND INTERCEPT VALUES:
![image](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/56041fcb-9309-49da-8f89-022e74c65c45)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
