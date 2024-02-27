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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('/content/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
df.head()
![ml1](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/a6a6342d-9a5c-4fe6-b230-e9a64ff48e89)
df.tail()
![ml2](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/e024bf5a-8e78-458e-a6ff-cff38dabbe60)
Array value of X
![ml3](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/ec9f4b6c-f932-4672-99b1-ccbb63eb75cf)
Array value of Y
![ml4](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/b045c04d-1325-477a-9150-7c384c821a5e)

Values of Y prediction
![ml5](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/50df8a52-21f6-4790-b09b-c9c4ab130b0c)

Array values of Y test
![ml6](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/eb5b4707-d801-452f-b296-b4be6c02a194)

Training and Testing set
![ml7](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/351d34f7-5975-422d-9fc1-b07cdecbe62d)
![ml8](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/c2661a45-9dd5-4b39-8ecb-a27a6cbe78e9)


Values of MSE,MAE and RMSE
![ml9](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/c076007b-3ea8-4b0f-867a-6361ce8c4049)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
