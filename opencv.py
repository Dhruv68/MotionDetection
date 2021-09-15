# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:35:06 2019

@author: Dhruv
"""

# Import the Liberaries
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import time
import cv2 

# Code to detect the face from the image

# Create a CascadeClassifier object
"""face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# Reading the image as it is
img = cv2.imread("D:\\Photos\\Friend\\IMG_0474.jpg",1)

face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

# convert the image in gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# TO detect the faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)

for x,y,w,h in faces:
   img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 3)
   
resized = cv2.resize(img, (int(img.shape[1]/3),int(img.shape[0]/3)))
   
cv2.imshow("Mine",resized)

cv2.waitKey(0)

cv2.destroyAllWindows()"""


# This code is to capture the face from single frame video

"""video = cv2.VideoCapture(0)

check, frame = video.read()

print(check)
print(frame)

time.sleep(5)

cv2.imshow("capture",frame)

cv2.waitKey(0)

video.release()

cv2.destroyAllWindows()"""

# This code is to capture the real time face from live video

"""video = cv2.VideoCapture(0)

a = 1
 
while True:
  a = a + 1
  check, frame = video.read()
  print(frame)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow("capture",frame)
  key = cv2.waitKey(1)
  if key == ord('q'):
   break

print(a) #This will print the number of frames
video.release()
cv2.destroyAllWindows()"""

# This code will detect your face from your webcamp

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

a = 1

face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
 
while True:
  a = a + 1
  _, img = video.read()

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors=5)
  
  for x,y,w,h in faces:
      cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 3)
      eye_gray = gray[y:y+h, x:x+w]
      eye_color = img[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(eye_gray)
      
      
      
      for ex,ey,ew,eh in eyes:
        cv2.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (255,0,0), 3)
   
  cv2.imshow("capture",img)
  key = cv2.waitKey(1)
  if key & 0xFF == ord('q'):
        break
  if key & 0xFF == ord('c'):
        crop_img = img[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
        cv2.imwrite("my_image.jpg", crop_img)

print(a) #This will print the number of frames
video.release()
cv2.destroyAllWindows()

# This code is for Motion Detector

"""from datetime import datetime

first_frame = None
status_list = [None,None]
times = []
df = pd.DataFrame(columns = ["Start","End"])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    
    if first_frame is None:
      first_frame = gray
      continue
    
  delta_frame = cv2.absdiff(first_frame,gray)
  
  thresh_delta = cv2.threshold(delta_frame, 30,225)"""