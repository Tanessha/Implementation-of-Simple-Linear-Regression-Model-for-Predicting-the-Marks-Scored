# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: TANESSHA KANNAN 
RegisterNumber: 212223040225

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### DataSet:
![image](https://github.com/user-attachments/assets/e0ab006e-1f89-433d-958f-754ef15972ac)


### Hard Values:
![image](https://github.com/user-attachments/assets/d9de9064-6ed0-4db1-afee-abd29c2e26b3)


### Tail Values:
![image](https://github.com/user-attachments/assets/252bad47-81d7-4cf0-a816-d44685c26a46)


### X and Y Values:
![image](https://github.com/user-attachments/assets/8d996a47-162b-4435-8c83-f090631b65c5)


### Prediction of X and Y:
![image](https://github.com/user-attachments/assets/2db25cb3-5cdd-4f39-b3b2-87348479ea91)


### MSE, MAE and RMSE:
![image](https://github.com/user-attachments/assets/e74fcf05-18dd-4bd2-b33b-6a962bc63af9)


### Training Set:
![image](https://github.com/user-attachments/assets/2eb2cc04-fb5c-4a2c-93ae-7799a7edbe6f)
![image](https://github.com/user-attachments/assets/e30b193c-4fce-4438-87ec-e790f0e9cc91)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
