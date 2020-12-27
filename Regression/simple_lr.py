# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values   #X ia a 1D array of features
y = dataset.iloc[:, 1].values     #y is a vector of dependent variable X

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression    #LinearRegression is a class
regressor = LinearRegression()                       #regressor is an object of class LinearRegression
regressor.fit(X_train,y_train)                       #fit is a method to fit the correlations of X_train & y_train

#Predicting the Test set results
y_pred = regressor.predict(X_test)      #vector of predictions of predicted values

#Visualizing the Training set results
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary vs Experince (Training Set)")
plt.xlabel("Years of Expeience")
plt.ylabel("Salary")
plt.show()

"""#Visualising the Test set results
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
#X_train itself should be used as this the simple linear regression model
#If we use X_test, a new regression model is formed with the X_test data set
plt.title("Salary vs Experince (Test Set)")
plt.xlabel("Years of Expeience")
plt.ylabel("Salary")
plt.show()

#Visualising the Predicted set results
plt.scatter(X_train,y_train,color="red")
plt.scatter(X_test,y_test,color="black")
plt.scatter(X_test,y_pred,color="yellow")
plt.plot(X_train,regressor.predict(X_train),color="blue")
#X_train itself should be used as this the simple linear regression model
#If we use X_test, a new regression model is formed with the X_test data set
plt.title("Salary vs Experince (Training,Test,Predicted Set)")
plt.xlabel("Years of Expeience")
plt.ylabel("Salary")
plt.show()"""





