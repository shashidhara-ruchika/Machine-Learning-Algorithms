# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# quoting = 3 will ignore the quotes
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Cleaning the texts
# getting rid of capital letters(into small caps),numbers, common words like the an of etc
# stemming for regrouping the same versions of the word. Eg: considering love for words such as love, loved, loving
# Tokenization preprocessing: spits all the different reviews into different words, these words are then given columns,
#    then for each review each column will contain the number of times the associated word appears in the review
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# corpus is a collection of text of the same type
corpus = []
# iterating through each review of the dataset
for i in range(len(dataset)):
    # sub keeps only letters, it removes punctuations, numbers but does not remove letters in [] and a space i.e. ' '
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    # replacing all the upper to lower cases
    review = review.lower()
    # making a list of the words in each review
    review = review.split()
    # creating a class to apply stemming
    ps = PorterStemmer()
    # removing common words such as the an of this, etc then applying stemming to the selected words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # joining the string of words into a single word
    review = ' '.join(review)
    # appending the cleaned review to the list corpus
    corpus.append(review)

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Naive Bayes Classification

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
"""
# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression as logreg
classifier = logreg(random_state = 0)
classifier.fit(X_train, y_train)
"""
"""
# Fitting K-Nearest-Neighbors to the training set
from sklearn.neighbors import KNeighborsClassifier as knn
classifier = knn(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train, y_train)
"""
"""
# Fitting SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
"""
"""
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC( kernel = 'precomputed', random_state = 0)
classifier.fit(X_train, y_train)
"""
"""
# Fitting Decission Tree classifier to the Training set
from sklearn.tree import DecisionTreeClassifier as DTC
classifier = DTC(criterion = 'entropy', random_state = 0)
#entroy is used as we want each class/ groups to be as homogeneous as the previous one
classifier.fit(X_train, y_train)
"""
"""
# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier as RFC
classifier = RFC( n_estimators = 10, criterion = 'entropy', random_state = 0 )
classifier.fit(X_train, y_train)
"""
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the performance of the model
performance = []
accuracy = np.diag(cm).sum()/ cm.sum()
precision = cm[1][1]/ cm.sum(axis=0)[1]
recall = cm[1][1]/ cm.sum(axis=1)[1]
f1_score = 2*precision*recall / (precision + recall)
performance.extend([accuracy,precision,recall,f1_score])







