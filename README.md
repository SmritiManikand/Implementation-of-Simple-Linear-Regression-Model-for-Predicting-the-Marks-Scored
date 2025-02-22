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

## df.head()

![s4](https://user-images.githubusercontent.com/113674204/229329319-a45805bd-f2f8-4bf3-b296-ddbe4998fe7e.png)

## df.tail()

![s5](https://user-images.githubusercontent.com/113674204/229329322-d46dd646-9693-486b-9b88-81b28b8013f4.png)

## Array value of X

![s1](https://user-images.githubusercontent.com/113674204/229329470-05a23176-edc1-46de-9d8f-042795b3ed6c.png)

## Array value of Y

![s2](https://user-images.githubusercontent.com/113674204/229329483-840b3da3-b82f-4a1b-a7ee-90dd232fdc92.png)

## Values of Y prediction

![s4](https://user-images.githubusercontent.com/113674204/229329690-de2fbdc6-7636-4e13-a5af-aecadb20d9bc.png)

## Array values of Y test

![s3](https://user-images.githubusercontent.com/113674204/229329631-a31d7b98-cc88-4b30-8191-6b5f09c5f110.png)

## Training Set Graph

![s3](https://user-images.githubusercontent.com/113674204/229070707-bd59cd92-0cd4-4ce7-8fb9-1c71bca0fb88.png)

## Test Set Graph

![s4](https://user-images.githubusercontent.com/113674204/229070746-5c7854cb-9a98-41d4-b700-8e714ca3ac5a.png)

## Values of MSE, MAE and RMSE

![s5](https://user-images.githubusercontent.com/113674204/229070778-70e53c59-41eb-400f-b963-40c1524388df.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
