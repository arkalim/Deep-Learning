from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense , Dropout , Conv2D , MaxPooling2D , Flatten
import numpy as np
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import TensorBoard
from time import time

num_classes = 10
NAME = "CIFAR.{}".format(int(time()))

# TensorBoard data will be logged in log_dir
tensorboard = TensorBoard(log_dir='log_dir/{}'.format(NAME))

# loading the data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train.shape[1:])  #reading the shape from position 1(excluding the num_channels)

#Normalising the training and test data
x_train = normalize(x_train , axis = 1)
x_test = normalize(x_test , axis = 1)

#one hot encoding the train and test labels (used when the loss function is categorical cross entropy) 
#y_train = np_utils.to_categorical(y_train, num_classes)
#y_test = np_utils.to_categorical(y_test, num_classes)

#underfitting model
model = Sequential()
model.add(Conv2D( 8 , (3,3) , activation='relu' , input_shape = x_train.shape[1:] , padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D( 16 , (3,3) , activation='relu' , padding='same'))
model.add(Conv2D( 16 , (3,3) , activation='relu' , padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(32 , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10 , activation='softmax'))

# Compiling the model
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

# Training the model
# Note that tensorboard works only for model.fit() and not for model.fit_generator()
model.fit(x_train , y_train , epochs=50  , validation_split=0.2 , batch_size=200 , callbacks=[tensorboard])

# Evaluating the model
test_loss , test_accuracy = model.evaluate(x_test , y_test)
print(test_loss , (test_accuracy)*100)

# Predict any sample of the test data
img = x_test[0]  #considering the first test image
img = np.expand_dims(img,axis = 0)
prediction = np.argmax(model.predict(img))
print(prediction)