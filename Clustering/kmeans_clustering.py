# K-means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
# Calculating the value of WCSS
k = 10
wcss= []
for i in range(1,k+1):
    kmeans = KMeans( n_clusters = i, init="k-means++", max_iter = 300, n_init = k, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# Plotting WCSS vs the Number of Clusters
plt.plot(range(1,k+1),wcss)
plt.title("Elbow method")
plt.xlabel("Number of clusters: k")
plt.ylabel("WCSS")
plt.show()
# Finding the elbow point
import math
scale = math.pow(10,int(math.log10(sum(wcss)/ k)))
# angle at each point in radians
angle = [ (math.atan((wcss[i+1] - wcss[i]) / scale )) - (math.atan((wcss[i+2] - wcss[i+1]) / scale )) for i in range(k-2) ]
# number of clusters
n = angle.index(min(angle)) + 2

# Applying kmeans to the dataset
kmeans = KMeans( n_clusters = n, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
# fit_predict method returns values for each observation which cluster it belongs to
# i.e. it will return a single vector which carries which cluster a data point belongs to
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters in 2D
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()




