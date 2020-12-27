# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap, sklearn already does it, but you can do it mannualy as well
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Function to predict values using MulpipleLinearRegression
def MultipleLinearRegression(X_train,X_test,y_train,y_test):

    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # The score and coefficients
    print('Coefficients: \n', regressor.coef_)
    print('Score:',regressor.score(X_test,y_pred))
    return y_pred

# Predicting the Test set results
y_pred = MultipleLinearRegression(X_train,X_test,y_train,y_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

#Mannual Method
# To calculate ordinary least squares and optimize the model
"""
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]                                          #OLS - ordinary linear class
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()                      #endog - dependent variable x, exog - regressor variable y
X_opt = X[:, [0, 1, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
X_opt = X[:, [0, 3, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
X_opt = X[:, [0, 3]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressorOLS.summary())
"""

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm


"""# Using p_values for back_elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
"""
# Using both p_values and Adjusted R squared values for backward elimination
def backwardElimination_improved(x, sl):
    numVars = len(x[0])
    flag_elimination=0
    for i in range(0, numVars):
        if(i and flag_elimination):
            regressor_OLS = sm.OLS(y, x[:,[a for a in range(np.shape(x)[1] -1)]]).fit()
            r_sq_adj_new = regressor_OLS.rsquared_adj
            if(r_sq_adj_new > r_sq_adj_prev):
                x=np.delete(x,-1,1)                                     #Deleting the last column if adjusted r-squared value has improved
        regressor_OLS = sm.OLS(y, x).fit()
        r_sq_adj_prev = regressor_OLS.rsquared_adj
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            flag_elimination=1
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    for k in range(np.shape(x)[0]):                     #Exchanging the column to delete and the last column
                                x[k][j],x[k][-1] = x[k][-1],x[k][j]
        else:
            flag_elimination=0
    print(regressor_OLS.summary())
    return x

SL = 0.03
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)   #To have a column of ones for constant term b0 (here it is 0)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = np.delete(backwardElimination_improved(X_opt, SL),0,1)        #Getting the new X data set with only relevant features
#X_Modeled = np.delete(backwardElimination(X_opt, SL),0,1)
               #Deletion of initially added ones column for term b0

# Splitting the new modeled X dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

# Predicting the Test set results with new modeled X after backward elimination
y_pred_new = MultipleLinearRegression(X_train,X_test,y_train,y_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_new))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred_new))

#To see the error of predection
change=0
for i in range(len(y_test)):
    print(y_test[i],'\t',y_pred[i]-y_test[i],'\t',y_pred_new[i]-y_test[i],end="\t")
    if(abs(y_pred[i]-y_test[i]) > abs(y_pred_new[i]-y_test[i])):
        print('y')     #if it prints y, it means the error for each value has reduced, i.e. a better prediction
        change+=1
    else:
        print('n')
print("Number of better predictions:",change,'/',len(y_test))







