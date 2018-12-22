from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\VGGFace model with weights.h5")

#model.summary()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
  
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
  
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

epsilon = 0.45

def verifyFace(img1, img2):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data\%s" % (img1)))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data\%s" % (img2)))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    print("Cosine similarity: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)
    
    if(cosine_similarity < epsilon):
        print("verified... they are same person")
    else:
        print("unverified! they are not same person!")
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data\%s" % (img1)))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data\%s" % (img2)))
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)
    
    print("-----------------------------------------")
    
verifyFace("User.1.2.jpg","User.1.69.jpg")
