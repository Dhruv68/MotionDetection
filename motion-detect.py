# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:55:59 2019

@author: Dhruv
"""

import time
import numpy as np
import cv2

# init camera
camera = cv2.VideoCapture('tes1.mp4') # Add the video path or the webcamp or the external camera.
camera.set(3,800)
camera.set(4,500)
time.sleep(0.5)

# master frame
master = None

while 1:

    # grab a frame
    (grabbed,frame0) = camera.read()
    
    # end of feed
    if not grabbed:
        break

    # gray frame
    frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

    # blur frame
    frame2 = cv2.GaussianBlur(frame1,(21,21),0)

    # initialize master
    if master is None:
        master = frame2
        continue

    # delta frame
    frame3 = cv2.absdiff(master,frame2)

    # threshold frame
    frame4 = cv2.threshold(frame3,15,255,cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes
    kernel = np.ones((5,5),np.uint8)
    frame5 = cv2.dilate(frame4,kernel,iterations=4)

    # find Contours on thresholded image
    contours , nada = cv2.findContours(frame5.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # make Contours frame
    frame6 = frame0.copy()
    
   
    # target contours
    targets = []

    # loop over the contours
    for c in contours:
        
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500: # make sure this has a less than sign, not an html escape
                continue

        # contour data
        M = cv2.moments(c)#;print( M )
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(c)
        rx = x+int(w/2)
        ry = y+int(h/2)
        ca = cv2.contourArea(c)

        # plot the box
        #cv2.drawContours(frame6,[c],0,(0,0,255),2)
        cv2.rectangle(frame6,(x,y),(x+w,y+h),(0,255,255),2)
        #cv2.circle(frame6,(cx,cy),2,(0,0,255),2)
        #cv2.circle(frame6,(rx,ry),2,(0,255,0),2)
        
        # save target contours
        targets.append((rx,ry,ca))

    # make target
    area = sum([x[2] for x in targets])
    mx = 0
    my = 0
    if targets:
        for x,y,a in targets:
            mx += x
            my += y
        mx = int(round(mx/len(targets),0))
        my = int(round(my/len(targets),0))

    # plot target
    tr = 50
    
      
    # update master
    master = frame2

    # display
    cv2.imshow("Frame6: Contours",frame6)

        
    # key delay and action
    key = cv2.waitKey(200) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:',[chr(key)])

# release camera
camera.release()

# close all windows
cv2.destroyAllWindows()