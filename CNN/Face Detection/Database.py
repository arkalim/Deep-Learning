import numpy as np
import cv2
import os
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

database = {}

identity = []
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (200, 100, 0) 

def prepare_database():
    os.chdir(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data")
    for file in os.listdir(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data"):
        identity = os.path.splitext(os.path.basename(file))[0]
        
        image = img.imread(file)
        image = cv2.resize(image ,(160,160))
        image = np.expand_dims(image, axis=0)
        database[identity] = model.predict(image)[0,:]
    return database

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

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

##################################################################################################################################################
            

def who_is_it(cropped_face):
    
    encoding = model.predict(cropped_face)[0,:]
    encoding = l2_normalize(encoding)
    
    min_dist = 100
    identity = None

    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        db_enc = l2_normalize(db_enc)
        dist = findEuclideanDistance(encoding, db_enc)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.30:
        return None
    else:
        return identity
   
########################################################################################################################################################
face_cascade = cv2.CascadeClassifier(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cropped_face = []
prepare_database()

while(True):
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cropped_face = frame[y-50:y+h+50,x-50:x+w+50]
        cv2.imshow('face' ,cropped_face)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(155,200,0),2)
        
        cropped_face = cv2.resize(cropped_face ,(160,160))
        cropped_face = np.expand_dims(cropped_face, axis=0)
        identity = who_is_it(cropped_face)
        cv2.putText(frame,str(identity), (x,y+h+50),font, 2,fontcolor,3)
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
##########################################################################################################################################################    
