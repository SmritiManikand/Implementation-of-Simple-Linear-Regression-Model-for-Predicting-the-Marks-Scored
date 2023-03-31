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
plt.show

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

![s1](https://user-images.githubusercontent.com/113674204/229026653-570ac064-5952-4996-9bf6-c4991c425076.png)

![s2](https://user-images.githubusercontent.com/113674204/229026712-39efab45-7b9c-4c40-9e77-bac989463fff.png)

![s3](https://user-images.githubusercontent.com/113674204/229026753-5e085da4-e14c-445d-b543-fc228c62d970.png)

![s4](https://user-images.githubusercontent.com/113674204/229026788-e3e2e72d-c619-41c3-9a30-52fb355e3c89.png)

![s5](https://user-images.githubusercontent.com/113674204/229026816-fa2f4712-6abe-4d50-b337-b93b1f9c41ad.png)

![s6](https://user-images.githubusercontent.com/113674204/229026925-b4f1382b-4ba4-4adf-b2e1-bc017b359d1c.png)

![s7](https://user-images.githubusercontent.com/113674204/229026986-4bfe9345-96d9-4b0f-8825-1cf1c78e18ee.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
