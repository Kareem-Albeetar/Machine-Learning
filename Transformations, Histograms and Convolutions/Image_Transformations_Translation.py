#!/usr/bin/env python
# coding: utf-8

# # Image Transformations - Translation
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


# In[14]:


M = np.float32([[1, 0, 25], [0, 1, 50]])


# In[15]:


shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


# In[ ]:


M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shifted Down and Right", shifted)
cv2.waitKey(0)


# In[10]:


M = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shifted Up and Left", shifted)
cv2.waitKey(0)


# In[12]:


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


# In[13]:


shifted = translate(image, 0, 100)
cv2.imshow("Shifted Down", shifted)
cv2.waitKey(0)

