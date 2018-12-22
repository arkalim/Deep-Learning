import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
ID = str(input('Enter the ID: '))
sampleNum=0

while(True):
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
               
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite(r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Face Detection\Personal Data/User."+ID +'.'+ str(sampleNum) + ".jpg", frame[y-50:y+h+50,x-50:x+w+50])       
        cv2.rectangle(frame,(x,y),(x+w,y+h),(155,200,0),2)
        cv2.imshow('frame',frame)
        time.sleep(0.1)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum > 10:
        break       
cap.release()
cv2.destroyAllWindows()

