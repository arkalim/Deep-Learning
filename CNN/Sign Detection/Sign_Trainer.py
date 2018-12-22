from keras.models import Sequential
from keras.layers.core import Dense , Flatten , Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from keras.layers import Input
from keras.models import Model
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import time

image_dim = 64

train_path = r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\DataSet\Train"
valid_path = r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\DataSet\Valid"
test_path =  r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\DataSet\Test"
train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(train_path, target_size=(image_dim,image_dim), classes=['Ball', 'Left','Return','Right','Stop'], batch_size = 40)
valid_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(valid_path, target_size=(image_dim,image_dim), classes=['Ball', 'Left','Return','Right','Stop'], batch_size = 12)
test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test_path, target_size=(image_dim,image_dim), classes=['Ball', 'Left','Return','Right','Stop'], batch_size = 1)

image_input = Input(shape=(image_dim, image_dim, 3))
model = Sequential()

model.add(Conv2D(32, (3, 3),input_shape=(image_dim,image_dim,3),activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.summary();

print(image.shape)
model.compile(Adam(lr = 0.0001), loss = 'categorical_crossentropy',metrics = ['accuracy'])  
history = model.fit_generator(train_batches, steps_per_epoch=10, validation_data=valid_batches, validation_steps=5, epochs =40, verbose = 2)
#
image = cv2.imread(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\DataSet\Test\Return\Return.83 (2018_12_08 11_07_44 UTC).jpg")


print('saving the model')
#model.save(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\Sign_detection_model1.h5")
print('model saved')
#
image = cv2.resize(image , (image_dim,image_dim))
#cv2.imshow("img",image)
image = image.astype('float32')
image = image/255
image = np.expand_dims(image, axis=0)
#print(image.shape)
#
prediction = model.predict(image)
print(np.argmax(prediction))

################################################################################
#Printing the accuracy and loss curves

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, color = 'b',label='Training acc')
plt.plot(epochs, val_acc, color = 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss,color = 'b', label='Training loss')
plt.plot(epochs, val_loss, color = 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

################################################################################    
