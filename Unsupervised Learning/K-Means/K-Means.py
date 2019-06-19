import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Color Array
colors = ['r', 'g', 'b', 'y', 'c', 'm']

X = pd.read_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\K-Means\k-means_data.csv")

#Converting the dataframe to array
X = X.values 

#K-means algorithm
kmeans = KMeans(n_clusters=5)
kmeans = kmeans.fit(X)

#Finding the centroids
centroids = kmeans.cluster_centers_

#Finding the label of each point
labels = kmeans.labels_

#Plotting the 1st column on x-axis and 2nd column on y-axix and chaning colour for each label
plt.scatter(X[:,0], X[:,1], c = labels, s=1)  
#Plotting the centroids
plt.scatter(centroids[:,0],centroids[:,1],color='m', s=100 ,marker="*")
            

#Predicting the label of any given point
data = X[0,:]
label = kmeans.predict([data])
plt.scatter(data[0],data[1], c = 'k', s=50, marker = 'o') 

plt.show()