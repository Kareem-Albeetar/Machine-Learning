#!/usr/bin/env python
# coding: utf-8

# # Image Manipulations
# ## This notebook outlines the different techniques used on images using OpenCV library
# 
# 

# In[2]:


from __future__ import print_function
import argparse
import cv2
import numpy as np


# In[4]:


image = cv2.imread("image.jpg")


# In[5]:


image


# In[4]:


image = cv2.imread("image.jpg")
cv2.imshow("Original", image)
cv2.waitKey(0)


# In[5]:


cropped = image[30:120 , 240:335]
cv2.imshow("T-Rex Face", cropped)
cv2.waitKey(0)


# In[6]:


flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)
cv2.waitKey(0)


# In[7]:


flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)
cv2.waitKey(0)


# In[8]:


flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally & Vertically", flipped)
cv2.waitKey(0)


# In[33]:


mask = np.zeros(image.shape[:2], dtype = "uint8")
mask.shape


# In[34]:


(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
(cX, cY)


# In[35]:


cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75 , cY + 75), 255, -1)


# In[36]:


cv2.imshow("NewMask", mask)
cv2.waitKey(0)


# In[37]:


masked = cv2.bitwise_and(image, image, mask = mask)


# In[38]:


cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)


# In[39]:


mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask, (225, 25), (325 , 125), 255, -1)
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)


# In[41]:


mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

