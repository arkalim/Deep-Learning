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

#Check the dimensions:
#print(train_data.shape)
#print(train_labels.shape)
#print(test_data.shape)
#print(test_labels.shape)

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
pca = PCA(0.85)

#Fit pca on the training set 
pca.fit(train_data)

#Show the number of principal features
num_features = pca.n_components_
#print(num_features)

#Apply the mapping(transform) to both training and test set
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)

########################################################################################################
#Classification using the transformed features

#Defining the model
model = Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(num_features,)))
model.add(layers.Dense(10, activation='softmax'))

#Compiling the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training the model on the principle features
history = model.fit(train_data, train_labels, validation_data=(test_data,test_labels),epochs=1000, batch_size=60000)

#Evaluating the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test Accuracy:",test_acc)
print("Test Loss:",test_loss)

# Measure accuracy
pred = model.predict(test_data)
pred = np.argmax(pred, axis =1)

#########################################################################################################
plt.figure(figsize=(12,6));
#Plotting the confusion matrix

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot Normalised Confusion Matrix
plt.subplot(1,2,1)    
cm = confusion_matrix(test_labels, pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print('Normalized confusion matrix')
#print(cm_normalized)
plot_confusion_matrix(cm_normalized, np.unique(test_labels) , title='Normalized confusion matrix')

########################################################################################################
#Printing the accuracy and loss curves
plt.subplot(1,2,2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss,color = 'b', label='Training loss')
plt.plot(epochs, val_loss, color = 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#######################################################################################################

