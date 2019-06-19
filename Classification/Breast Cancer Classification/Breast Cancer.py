import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from Preprocess import *
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Set the desired TensorFlow output level for this example 
df = pd.read_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Jeff Heaton Course Material\data\wcbreast_wdbc.csv",na_values=['NA','?'])

# Encode feature vector
df.drop('id',axis=1,inplace=True)
diagnosis = encode_text_index(df,'diagnosis')
num_classes = len(diagnosis)

df.to_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Breast Cancer Classification\wcbreast_wdbc_edit.csv")

# Create x & y for training

# Create the x-side (feature vectors) of the training
x, y = to_xy(df,'diagnosis')
    
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 

# Build network
model = Sequential()
model.add(Dense(400, input_dim=x.shape[1],kernel_initializer='normal', activation='relu'))
model.add(Dense(300,kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(100,kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(10,kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(y.shape[1],activation='sigmoid'))

Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=250, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=0,epochs=50)
#model.load_weights(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Breast Cancer Classification\model_weights.h5") # load weights from best model

# Measure accuracy
pred = model.predict(x_test)
print(pred.shape)
pred = np.argmax(pred,axis=1)

y_compare = np.argmax(y_test,axis=1)
score = metrics.accuracy_score(y_compare, pred)
print("Final accuracy: {}".format(score))

################################################################################
#Printing the accuracy and loss curves

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss,color = 'b', label='Training loss')
plt.plot(epochs, val_loss, color = 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

################################################################################
#Plot Normalised Confusion Matrix

cm = confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized,diagnosis, title='Normalized confusion matrix')
plt.show()

#################################################################################