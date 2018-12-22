import numpy as np
import cv2
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as img

print("Loading Model")
model = load_model(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\facenet.h5")
print("Model Loaded")

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (200, 200, 0)

def preprocess_image(image_path):
    image = img.imread(image_path)
    #print(img.shape)
    image = cv2.resize(image , (160,160))
    image = np.expand_dims(image, axis=0)
    #image = preprocess_input(image)
    print(image.shape)
    return image
  
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
  
#vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

epsilon = 0.30

def verifyFace(img2):
    img1_representation = model.predict(cropped_face)[0,:]
    img2_representation = model.predict(preprocess_image(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data\%s" % (img2)))[0,:]
    
    print(img1_representation.shape)
    print(img2_representation.shape)
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    print("Cosine similarity: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)
    
    if(cosine_similarity < epsilon):
        Id="ARK"
    else:
        Id="Unknown"
    cv2.putText(frame,str(Id), (x,y+h+50),font, 1,fontcolor)

face_cascade = cv2.CascadeClassifier(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cropped_face = []

while(True):
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(155,200,0),2)
        cropped_face = frame[y:y+h,x:x+w]
        cropped_face = cv2.resize(cropped_face ,(160,160))
        cropped_face = np.expand_dims(cropped_face, axis=0)
        #print(cropped_face.shape)
        verifyFace("ARK.jpg")
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
