#!/usr/bin/env python
# coding: utf-8

# # Image Transformations - Resizing
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


# In[30]:


r = 150.0 / image.shape[1]


# In[31]:


dim = (150, int(image.shape[0] * r))


# In[32]:


resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


# In[33]:


cv2.imshow("Resized (Width)", resized)
cv2.waitKey(0)


# In[34]:


r = 50.0 / image.shape[0]
dim = (int(image.shape[1] * r), 50)

# Perform the resizing
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resized (Height)", resized)
cv2.waitKey(0)


# In[35]:


def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


# In[37]:


resized = resize(image, width = 100)
cv2.imshow("Resized via Function", resized)
cv2.waitKey(0)

