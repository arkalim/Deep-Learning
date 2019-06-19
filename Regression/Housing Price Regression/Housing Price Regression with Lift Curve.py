import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load dataset
dataframe = pd.read_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Housing Price Regression\housing.csv", delim_whitespace=True, header=None)
dataframe = dataframe.sort_values(13)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

# define base model

model = Sequential()
model.add(Dense(500, input_shape=(13,), kernel_initializer='normal', activation='relu'))
model.add(Dense(300, kernel_initializer='normal', activation='relu'))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')      


scaled_X = MinMaxScaler()
scaled_X.fit(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100, batch_size=5)

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
#Lift Curve
sort = True
pred = model.predict(x_test)
lift_data = list(zip(pred.flatten(),y_test))
lift_data = pd.DataFrame(lift_data)

if sort == True:
    lift_data = lift_data.sort_values(1)
    
print(lift_data)    

plt.plot(range(len(pred)), lift_data.iloc[:,0],color = 'b', label='Prediction')
plt.plot(range(len(pred)), lift_data.iloc[:,1], color = 'g', label='Output')
plt.title('Lift Curve')
plt.legend()
plt.show()

#################################################################################