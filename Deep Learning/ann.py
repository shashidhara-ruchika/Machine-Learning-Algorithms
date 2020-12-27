# Artificial Neural Networks

# Part 1: Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13 ].values
y = dataset.iloc[:, 13 ].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2= LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:]   # Avoiding Dummy variable trap for Country column i.e. removing one column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2: Creating the ANN:

# Importing the keras libraries and packages
import keras
from keras.models import Sequential       # initializing the ANN
from keras.layers import Dense            # connecting the fully connected layers to the ANN

# Rectifier activation function for the hidden layers
# Sigmoid activation function for the output layer.
# Choose the number of nodes in the hidden layer as the average of the number of nodes
# in the input layer and the number of nodes in the output layer

# Initializing the ANN by sequence of layers
classifier = Sequential()

# Adding the input layer and first hidden layer
# output_dim ((11+1)/2 = 6) - number of nodes in (next)first hidden layer
# init - initializing weights as random uniform numbers
# activation function - relu (rectifier function)
# input_dim - number of nodes in the (previous) input layer used only for this layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
# activation would be softmax i.e. sigmoid function applied to dependent variable that has more than 2 categories
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# optimizer - algorithm to find optimal set of weights
# loss - loss function of stochastic gradient method
# if dependent variable has binary outcome then use binary_crossentropy
# if dependent variable has more than two outcomes use categorical_crossentropy
# metrics - criterion to use improve the model
classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# batch_size - number of observations after which the weights will be updated
# nb_epoch - number of epochs
classifier.fit( X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3: Making the predictions and evaluating the model:

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred_int = []
for i in y_pred:
    if i==True:
        y_pred_int.append(1)
    else:
        y_pred_int.append(0)

# Making the Confusion Matrix
# It contains the number of correct and incorrect predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the performance of the model
performance = []
accuracy = np.diag(cm).sum()/ cm.sum()
precision = cm[1][1]/ cm.sum(axis=0)[1]
recall = cm[1][1]/ cm.sum(axis=1)[1]
f1_score = 2*precision*recall / (precision + recall)
performance.extend([accuracy,precision,recall,f1_score])










