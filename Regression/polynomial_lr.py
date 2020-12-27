# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Visualising the dataset
"""plt.scatter(X,y,color="red")
plt.title("Position vs Salary")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()"""

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
#Since we have very few data and our final model needs to be precise
#Hence there is no test splitting here

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear REgression to the dataset
from sklearn.linear_model import LinearRegression as LR
lin_reg = LR()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures as PF
poly_reg = PF(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LR()
lin_reg_2.fit(X_poly,y)

#Visualising the Linear Regression model
plt.scatter(X,y,color="red")
plt.plot(X, lin_reg.predict(X) , color = "blue",)
plt.title("Position vs Salary (Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualising the Polynmial Regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Position vs Salary (Polyynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5,]])))