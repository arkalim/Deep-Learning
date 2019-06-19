import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from Preprocess import *

preprocess = True

df = pd.read_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Miles Per Gallon\auto-mpg.csv",na_values=['NA','?'])

# create feature vector
missing_median(df, 'horsepower')
#encode_text_dummy(df, 'origin')

df.drop('name',1,inplace=True)
if preprocess:
    encode_numeric_zscore(df, 'horsepower')
    encode_numeric_zscore(df, 'weight')
    encode_numeric_zscore(df, 'cylinders')
    encode_numeric_zscore(df, 'displacement')
    encode_numeric_zscore(df, 'acceleration')
    
df.to_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Miles Per Gallon\auto-mpg-edit.csv")    

# Encode to a 2D matrix for training
x,y = to_xy(df,'mpg')
print(y.shape)

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)

model = Sequential()
model.add(Dense(500, input_dim=x.shape[1], activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=1, mode='auto')
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1000)

# Predict and measure RMSE
pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Score (RMSE): {}".format(score))

# Plot the chart
chart_regression(pred.flatten(),y_test)

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