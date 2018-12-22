import cv2
import numpy as np
from keras.models import load_model

cap=cv2.VideoCapture(0)                               #Start video capture from webcam
area= []
maxarea=0

print("Loading the model....")
model = load_model(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\Sign_detection_model1.h5")
print('Model Loaded...!')

pts1 = np.float32([[0,0],[0,300],[450,300],[450,0]])
pts2 = np.float32([[450,0],[0,0],[0,300],[450,300]])

while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([63,54,71])                 
    upper_blue = np.array([97,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)   #mask the selected colour
    
    kernel = np.ones((3,3),np.uint8)                            
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    _, contours,_ = cv2.findContours(opening,1,2)                #finding contours        

    for contour in contours:
        area.append(cv2.contourArea(contour))
    if len(area)!=0:
        maxarea = np.argmax(area)

        epsilon = 0.1*cv2.arcLength(contours[maxarea],True)
        approx = cv2.approxPolyDP(contours[maxarea],epsilon,True)
        if(area[maxarea] > 300):
            cv2.drawContours(frame,[approx],0,(0,0,255),2)
        #print(approx)
        if(approx.shape[0]==4):
            points = approx.reshape(4,2)
            points = np.float32(points)
            if(points[0][0]<=300):
                M = cv2.getPerspectiveTransform(points,pts1)
            if(points[0][0]>=300):
                M = cv2.getPerspectiveTransform(points,pts2)    
            result = cv2.warpPerspective(frame,M,(450,300))
            if(area[maxarea] > 300):
                cv2.imshow('result',result)
                sign = cv2.resize(result,(64,64))
                sign = np.expand_dims(sign,axis=0)
                
                prediction = model.predict(sign)
                print(np.argmax(prediction))
            #print(points)

                           #colour frame with no change
        area = []
        
    
    cv2.imshow('frame',frame)                         #colour frame with no change
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
