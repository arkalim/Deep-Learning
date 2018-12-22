import cv2
import numpy as np

cap=cv2.VideoCapture(0)                               #Start video capture from webcam
area= []
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
        print(maxarea)
        cv2.drawContours(frame, contours,maxarea,(250,120,0), 2)
        area = []
    
    cv2.imshow('frame',frame)                         #colour frame with no change
    if cv2.waitKey(1) == 27:                          #wait for esc key to be pressed
        break
cv2.destroyAllWindows()
