#!/usr/bin/env python
# coding: utf-8

# # Images
# ## This notebook outlines the different Color Spaces used in OpenCV library
# 
# 

# In[2]:


from __future__ import print_function
import argparse
import cv2
import numpy as np


# In[3]:


image = cv2.imread("image.jpg")


# In[4]:


image


# In[5]:


image = cv2.imread("image.jpg")
cv2.imshow("Original", image)
cv2.waitKey(0)


# In[6]:


image = cv2.imread("image.jpg")
(B, G, R) = cv2.split(image)


# In[7]:


cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)


# In[8]:


merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)


# In[10]:


zeros = np.zeros(image.shape[:2], dtype = "uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)


# In[11]:


cv2.destroyAllWindows()


# In[33]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)


# In[34]:


gray


# In[36]:


gray.shape


# In[35]:


image


# In[37]:


image.shape


# In[38]:


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)


# In[39]:


lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0)

