from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from sklearn.decomposition import PCA
import pandas as pd
from keras.models import Sequential
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

#Selecting an image
img = train_data[0]

#Preprocess the data:
#Reshaping the data from 3D to 2D (num_channel , num_features)
#Normalising the data
train_data = train_data.reshape((60000, 28 * 28))
train_data = train_data.astype('float32') / 255

test_data = test_data.reshape((10000, 28 * 28))
test_data = test_data.astype('float32') / 255

#print(train_data.shape)
#print(train_labels.shape)
#print(test_data.shape)
#print(test_labels.shape)


#Creating an instance of PCA
#scikit-learn will choose minimum number of principal features such that 85% of the variance is retained.
variance_retained = 0.85
pca = PCA(variance_retained)

#Fit pca on the training set 
pca.fit(train_data)

#Show the number of principal features
num_features = pca.n_components_
#print(num_features)

#Apply the mapping(transform) to both training and test set
train_components = pca.transform(train_data)
test_components = pca.transform(test_data)

#Apply inverse transform to recover original image data
train_reconstructed = pca.inverse_transform(train_components)
test_reconstructed = pca.inverse_transform(test_components)

#Reshaping and scaling the reconstructed image
reconstructed_img = (train_reconstructed[0].reshape(28,28))*255

#################################################################################################################
# Plot the original and reconstructed image

plt.figure(figsize=(8,4));

# Original Image
plt.subplot(1, 2, 1);
plt.imshow(img, cmap = 'gray', interpolation='nearest',clim=(0, 255));
plt.xlabel('784 components', fontsize = 14)
plt.title('Original Image', fontsize = 20);

# 154 principal components
plt.subplot(1, 2, 2);
plt.imshow(reconstructed_img, cmap = 'gray', interpolation='nearest',clim=(0, 255));
plt.xlabel((str(num_features)+" Features"), fontsize = 14)
plt.title(str(variance_retained*100)+"% of Explained Variance", fontsize = 20);
plt.show()
################################################################################################################