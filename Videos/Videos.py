#!/usr/bin/env python
# coding: utf-8

# # Video Processing
# ## This notebook outlines the concepts behind the Video processing using OpenCV library

# In[2]:


import numpy as np
import cv2


# In[3]:


capture = cv2.VideoCapture(0)


# In[4]:


ret, frame = capture.read()


# In[5]:


cv2.imshow('Original Webcam', frame)
cv2.waitKey(0)


# In[6]:


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# In[7]:


cv2.imshow('GrayScale Webcam', gray)
cv2.waitKey(0)


# In[9]:


while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
 
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Display the resulting frame
    cv2.imshow('Original Webcam', frame)
    cv2.imshow('GrayScale Webcam', gray)

    if cv2.waitKey(0) &0XFF == ord('q'):
        break
        
# Release the capture
capture.release()
cv2.destroyAllWindows()


# In[12]:


import numpy as np
import cv2
 
cap = cv2.VideoCapture('testvideo1.mp4')
 
while(cap.isOpened()):
    ret, frame = cap.read()
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('Gray Video',gray)
    cv2.imshow('Color Video', frame)
    if cv2.waitKey(1) &0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[17]:


import cv2

def save_webcam(outPath,fps,mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)

    currentFrame = 0

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outPath, fourcc, fps, (int(width), int(height)))

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            if mirror == True:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)
            # Saves for video
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Video view', frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            break

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    save_webcam('videos/output.avi', 30.0,mirror=True)

if __name__ == '__main__':
    main()

