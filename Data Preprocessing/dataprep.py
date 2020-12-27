#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Data.csv")
#spliting arrays for x and y
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
#used for making all empty values to nan and making those nan values to the mean of the column
imputer = Imputer(missing_values = "NaN",strategy ="mean", axis=0)
#used to fit array x
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x= LabelEncoder()
#used to make string values to numerical values
x[:,0] = label_encoder_x.fit_transform(x[:,0])
#used to make numerical values in one column as separate columns with either 0 or 1(present)
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
#used to fit array y
label_encoder_y= LabelEncoder()
y = label_encoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
#spliting arrays to corresponding variables
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling for x_train & y_train
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_train)






