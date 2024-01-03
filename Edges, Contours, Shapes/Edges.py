#!/usr/bin/env python
# coding: utf-8

# # EDGES IN IMAGES
# ## This notebook outlines the concepts used in Edge detection in Image Processing

# In[1]:


import cv2
import numpy as np


# In[2]:


image = cv2.imread("coins.jpg")
cv2.imshow("Coins", image)
cv2.waitKey(0)


# In[3]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[4]:


lap = cv2.Laplacian(image, cv2.CV_64F)


# In[5]:


lap = np.uint8(np.absolute(lap))


# In[12]:


cv2.imshow("Laplacian RGB", lap)
cv2.waitKey(0)


# In[7]:


lap_gray = cv2.Laplacian(gray, cv2.CV_64F)


# In[8]:


lap_gray = np.uint8(np.absolute(lap_gray))


# In[13]:


cv2.imshow("Laplacian Gray", np.hstack([lap_gray]))
cv2.waitKey(0)


# In[14]:


sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)


# In[15]:


sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))


# In[17]:


sobelCombined = cv2.bitwise_or(sobelX, sobelY)


# In[18]:


cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)


# In[ ]:


image = cv2.imread("coins.jpg")
cv2.imshow("Coins", image)
cv2.waitKey(0)


# In[19]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[20]:


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# In[21]:


canny = cv2.Canny(blurred, 30, 150)


# In[23]:


cv2.imshow("Canny", canny)
cv2.waitKey(0)

