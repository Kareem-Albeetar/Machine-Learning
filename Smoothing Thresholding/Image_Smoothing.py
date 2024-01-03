#!/usr/bin/env python
# coding: utf-8

# # Image Smoothing
# ## This notebook outlines the techniques used to smooth an image

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[4]:


image = cv2.imread("image.jpg")


# In[3]:


blur = cv2.blur(image, (3, 3))


# In[4]:


cv2.imshow("Smoothing Image", np.hstack([image, blur]))
cv2.waitKey(0)


# In[6]:


blur3 = cv2.blur(image, (3, 3))
blur5 = cv2.blur(image, (5, 5))
blur7 = cv2.blur(image, (7, 7))
blur9 = cv2.blur(image, (9, 9))


# In[7]:


cv2.imshow("Smoothed Images using Averaging Blur", np.hstack([image, blur3, blur5, blur7, blur9]))
cv2.waitKey(0)


# In[8]:


image = cv2.imread("image.jpg")


# In[10]:


gaussianblur = cv2.GaussianBlur(image, (3, 3), 0)


# In[12]:


cv2.imshow("Smoothing Images using Gaussian", np.hstack([image, gaussianblur]))
cv2.waitKey(0)


# In[14]:


gaussianblur3 = cv2.GaussianBlur(image, (3, 3), 0)
gaussianblur5 = cv2.GaussianBlur(image, (5, 5), 0)
gaussianblur7 = cv2.GaussianBlur(image, (7, 7), 0)
gaussianblur9 = cv2.GaussianBlur(image, (9, 9), 0)


# In[15]:


cv2.imshow("Smoothing Images using Gaussian Blur", np.hstack([image, gaussianblur3, gaussianblur5, gaussianblur7, gaussianblur9]))
cv2.waitKey(0)


# In[16]:


cv2.imshow("Smoothed Images using Averaging Blur Method", np.hstack([image, blur3, blur5, blur7, blur9]))
cv2.waitKey(0)


# In[17]:


image = cv2.imread("image.jpg")


# In[19]:


medianblur = cv2.medianBlur(image, 3)


# In[20]:


cv2.imshow("Smoothing Images using Median", np.hstack([image, medianblur]))
cv2.waitKey(0)


# In[21]:


medianblur3 = cv2.medianBlur(image, 3)
medianblur5 = cv2.medianBlur(image, 5)
medianblur7 = cv2.medianBlur(image, 7)
medianblur9 = cv2.medianBlur(image, 9)


# In[22]:


cv2.imshow("Smoothing Images using Median", np.hstack([image, medianblur3, medianblur5, medianblur7, medianblur9]))
cv2.waitKey(0)


# In[24]:


cv2.imshow("Smoothed Images using Averaging Blur Method", np.hstack([image, blur3, blur5, blur7, blur9]))
cv2.imshow("Smoothing Images using Gaussian Blur", np.hstack([image, gaussianblur3, gaussianblur5, gaussianblur7, gaussianblur9]))
cv2.waitKey(0)


# In[25]:


image = cv2.imread("image.jpg")


# In[26]:


bilblur = cv2.bilateralFilter(image, 3, 21, 21)


# In[27]:


cv2.imshow("Smoothing Images using Bilateral Blur", np.hstack([image, bilblur]))
cv2.waitKey(0)


# In[28]:


bilblur3 = cv2.bilateralFilter(image, 3, 21, 21)
bilblur5 = cv2.bilateralFilter(image, 5, 21, 21)
bilblur7 = cv2.bilateralFilter(image, 7, 21, 21)
bilblur9 = cv2.bilateralFilter(image, 9, 21, 21)


# In[29]:


cv2.imshow("Smoothing Images using Bilateral Blur with varying diameters", np.hstack([image, bilblur3, bilblur5, bilblur7, bilblur9]))
cv2.waitKey(0)


# In[30]:


bilblur0 = cv2.bilateralFilter(image, 5, 21, 21)
bilblur1 = cv2.bilateralFilter(image, 7, 31, 31)
bilblur2 = cv2.bilateralFilter(image, 9, 41, 41)


# In[31]:


cv2.imshow("Smoothing Images using Bilateral Blur with varying hyperparameters", np.hstack([image, bilblur0, bilblur1, bilblur2]))
cv2.waitKey(0)


# In[42]:


bilblurlargesigma = cv2.bilateralFilter(image, 17, 11, 11)
cv2.imshow("Test", bilblurlargesigma)
cv2.waitKey(0)

