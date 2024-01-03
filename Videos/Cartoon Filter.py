#!/usr/bin/env python
# coding: utf-8

# #  Cartoon Face Mask
# ## 

# In[7]:


import cv2
import os
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        faceROI = frame[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

        # Display the resulting frame
        cv2.imshow('Face Video', frame)
        cv2.imshow("Face ROI", faceROI)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[1]:


import cv2
import os
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

# mask_image = cv2.imread('popeye.png')
# mask_image = cv2.imread('pig.png')
# mask_image = cv2.imread('mickey.png')
# mask_image = cv2.imread('donald.png')
# mask_image = cv2.imread('trump1.png')
# mask_image = cv2.imread('trump2.png')
# mask_image = cv2.imread('pumpkin.png')
mask_image = cv2.imread('popeye.png')
mask_image = cv2.imread('crazy1.png')
mask_image = cv2.imread('crazy2.png')

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        
        faceROI = frame[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        
        
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

        mask_image_res = cv2.resize(mask_image, (w, h))
        mask_image_gray = cv2.cvtColor(mask_image_res, cv2.COLOR_BGR2GRAY)
        _, face_mask = cv2.threshold(mask_image_gray, 25, 255, cv2.THRESH_BINARY_INV)
        face_mask_no_face = cv2.bitwise_and(faceROI, faceROI, mask=face_mask)
        final_face_mask = cv2.add(face_mask_no_face, mask_image_res)
        
        frame[y:y+h,x:x+w] = final_face_mask
        
        # Display the resulting frame
        cv2.imshow("Face mask", face_mask_no_face)
        cv2.imshow("Final face mask", final_face_mask)
        cv2.imshow("Google Face filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

