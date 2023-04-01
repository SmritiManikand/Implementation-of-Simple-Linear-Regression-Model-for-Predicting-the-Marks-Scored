# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Aim:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.

2. Assign hours to X and scores to Y.

3. Implement training set and test set of the dataframe

4. Plot the required graph both for test data and training data.

5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Smriti M
RegisterNumber:  212221040157
*/
import pandas as pd
import numpy as np
dataset=pd.read_csv('/student_scores.csv')
print(dataset)


X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

![s1](https://user-images.githubusercontent.com/113674204/229070636-2cba6cbc-74a2-4e94-b764-409b48f77e3c.png)

![s2](https://user-images.githubusercontent.com/113674204/229070663-b6654a9b-3397-4159-a69f-9874610b9ef6.png)

## Training Set Graph

![s3](https://user-images.githubusercontent.com/113674204/229070707-bd59cd92-0cd4-4ce7-8fb9-1c71bca0fb88.png)

## Test Set Graph

![s4](https://user-images.githubusercontent.com/113674204/229070746-5c7854cb-9a98-41d4-b700-8e714ca3ac5a.png)

## Values of MSE, MAE and RMSE

![s5](https://user-images.githubusercontent.com/113674204/229070778-70e53c59-41eb-400f-b963-40c1524388df.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
