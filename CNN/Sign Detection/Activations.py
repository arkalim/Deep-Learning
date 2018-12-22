from keras import models
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

image_dim = 64
layer_count = 8

model = load_model(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\Sign_detection_model1.h5")

model.summary()

layer_outputs = [layer.output for layer in model.layers[:layer_count]]       #show outputs upto layer_count layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

image = cv2.imread(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\DataSet\Test\Ball\Ball.87 (2018_12_08 11_07_44 UTC).jpg")
#
image = cv2.resize(image , (image_dim,image_dim))
image = image.astype('float32')
image = image/255.
image = np.expand_dims(image, axis=0)

activations = activation_model.predict(image)

########################################################################################################
#To output only one activation
#first_layer_activation = activations[0]
#plt.matshow(first_layer_activation[0, :, :, 0], cmap='gray')

#########################################################################################################
#To output all the activations 

layer_names = []

for layer in model.layers[:layer_count]:
    layer_names.append(layer.name)
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='gray')
    
#########################################################################################################    