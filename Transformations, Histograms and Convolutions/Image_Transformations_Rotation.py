#!/usr/bin/env python
# coding: utf-8

# # Image Transformations - Rotation
# ## This notebook outlines the different techniques used on images using OpenCV library
# 
# 

# In[1]:


from __future__ import print_function
import argparse
import cv2
import numpy as np


# In[2]:


image = cv2.imread("image.jpg")


# In[3]:


image


# In[4]:


image = cv2.imread("image.jpg")
cv2.imshow("Original", image)
cv2.waitKey(0)


# In[17]:


(h, w) = image.shape[:2]
center = (w // 2, h // 2)
center


# In[18]:


R = cv2.getRotationMatrix2D(center, 45, 1.0)


# In[19]:


rotated = cv2.warpAffine(image, R, (w, h))


# In[20]:


cv2.imshow("Rotated by 45 Degrees", rotated)
cv2.waitKey(0)


# In[21]:


R = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, R, (w, h))
cv2.imshow("Rotated by -90 Degrees", rotated)
cv2.waitKey(0)


# In[22]:


R = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated = cv2.warpAffine(image, R, (w, h))
cv2.imshow("Rotated by 180 Degrees", rotated)
cv2.waitKey(0)


# In[23]:


def rotate(image, angle, center = None, scale = 1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]
    
    # Calculate the center
    if center is None:
        center = (w // 2, h // 2)
        
    # Perform the rotation
    R = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, R, (w, h))
    
    return rotated


# In[27]:


rotated = rotate(image, 10)
cv2.imshow("Rotated 10 degrees around center", rotated)
cv2.waitKey(0)


# In[29]:


rotated = rotate(image, 10, center=(200,200))
cv2.imshow("Rotated 10 degrees around 200,200", rotated)
cv2.waitKey(0)

