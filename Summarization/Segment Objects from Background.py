#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from matplotlib import pyplot as plt


# In[3]:


# Read image & convert to gray scale
im = cv2.imread('Pepsi.jpeg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# convert to grayscale image
im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
plt.imshow(im_gray, cmap='gray')
plt.show()


# In[9]:


# threshold value
b = 72
_, im_binary = cv2.threshold(im_gray, b, 255, cv2.THRESH_BINARY)
plt.imshow(im_binary, cmap='gray')
plt.show()


# In[10]:


im_edge = cv2.adaptiveThreshold(im_gray, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, 9, 2)

plt.imshow(im_edge, cmap='gray')
plt.show()


# In[13]:


_, im_otsu = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(im_otsu, cmap='gray')
plt.show()

