## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored


```
Program to implement the simple linear regression
model for predicting the marks scored
Developed by: Hariharan A
RegisterNumber: 212223110013
```


## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1 : Start

Step 2 : Import the standard Libraries.

Step 3 : Set variables for assigning dataset values.

Step 4 : Import linear regression from sklearn.

Step 5 : Assign the points for representing in the graph.

Step 6 : Predict the regression for marks by using the representation of the graph.

Step 7 : Compare the graphs and hence we obtained the linear regression for the given datas. 

Step 8 : Stop

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

![Screenshot 2024-08-24 135442](https://github.com/user-attachments/assets/9ee37ac3-13ec-4ad6-8ca7-c6a64a4bc359)

### Hours vs Scores (Training Set):
![Screenshot 2024-08-24 135403](https://github.com/user-attachments/assets/9095779b-3fa1-4674-a649-ac7a4041f72e)

![Screenshot 2024-08-24 135352](https://github.com/user-attachments/assets/c69b81ed-e356-4cff-8e97-292eab36bda2)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
