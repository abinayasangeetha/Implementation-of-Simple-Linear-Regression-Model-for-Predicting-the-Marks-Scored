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
![ML21](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/c9b4e4b8-d079-416a-b87d-c348a17453b6)
![ML22](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/2ec628e0-d2b2-4833-ab0d-8ecc910807ce)
![ML23](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/829bdef5-fbd0-4242-a07a-60ee1cfdcecc)
![ML24](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/fec02645-27dd-4f0c-a2a2-8c3880bd69cb)
![ML25](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/d0d81c1b-30f5-4cfc-a6f7-379b36883417)
#### Training and Testing set

![ML26](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/d38af052-5fe6-4f49-969c-08fc8ab0c54a)
![ML27](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/d04941d1-c6bc-42d3-80c5-e6f78fdc59bd)
#### Values of MSE,MAE and RMSE
![ML28](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/1d5b020d-0016-4880-a788-77072d15ad1f)
![ML29](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/4029b3d2-67ad-4acb-9551-31ccc82507c9)
![ML30](https://github.com/abinayasangeetha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393675/9f5f78af-bec5-49a7-86ab-99bcafea889c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
