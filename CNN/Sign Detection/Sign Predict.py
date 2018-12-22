
import cv2
import numpy as np
from keras.models import load_model

image_dim = 24

print("Loading the model....")
model = load_model(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\Sign_detection_model.h5")
print('Model Loaded...!')
    

image = cv2.imread(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\DataSet\Test\Return\Return.87 (2018_12_08 11_07_44 UTC).jpg")

image = cv2.resize(image , (image_dim,image_dim))
cv2.imshow("img",image)
image = np.expand_dims(image, axis=0)
print(image.shape)

prediction = model.predict(image)
print(np.argmax(prediction))
