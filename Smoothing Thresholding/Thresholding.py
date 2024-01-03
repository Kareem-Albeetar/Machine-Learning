#!/usr/bin/env python
# coding: utf-8

# # Thresholding
# ## This notebook outlines the concepts behind Thresholding used in Image processing

# In[1]:


import cv2
import numpy as np


# In[2]:


image = cv2.imread("image.jpg")


# In[3]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[4]:


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# In[5]:


(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)


# In[7]:


cv2.imshow("Thresholding of Image", np.hstack([gray, blurred, thresh]))
cv2.waitKey(0)


# In[8]:


(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)


# In[9]:


cv2.imshow("Thresholding of Images", np.hstack([gray, blurred, thresh, threshInv]))
cv2.waitKey(0)


# In[12]:


foreground = cv2.bitwise_and(image, image, mask = threshInv)


# In[17]:


cv2.imshow("Thresholding of Images", np.hstack([gray, blurred, thresh, threshInv]))
cv2.imshow("Foreground extraction", np.hstack([image, foreground]))
cv2.waitKey(0)


# In[18]:


import cv2
import numpy as np


# In[19]:


image = cv2.imread("image.jpg")


# In[20]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[21]:


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# In[22]:


thresh = cv2.adaptiveThreshold(blurred, 
                               255, 
                               cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY_INV, 
                               11, 
                               4
)


# In[23]:


cv2.imshow("Adaptive Thresholding of Image", np.hstack([gray, blurred, thresh]))
cv2.waitKey(0)


# In[24]:


gaussianthresh = cv2.adaptiveThreshold(blurred, 
                               255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 
                               15, 
                               3
)


# In[25]:


cv2.imshow("Adaptive Thresholding of Images", np.hstack([gray, blurred, thresh, gaussianthresh]))
cv2.waitKey(0)


# In[26]:


foregroundgaussian = cv2.bitwise_and(image, image, mask = gaussianthresh)


# In[27]:


cv2.imshow("Gaussian Adaptive Thresholding of Images", np.hstack([gray, blurred, thresh, gaussianthresh]))
cv2.imshow("Foreground extraction using Gaussian", np.hstack([image, foregroundgaussian]))
cv2.waitKey(0)


# In[31]:


import cv2
import numpy as np
import mahotas


# In[28]:


image = cv2.imread("image.jpg")


# In[29]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[30]:


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# In[32]:


T = mahotas.thresholding.otsu(blurred)


# In[33]:


T


# In[34]:


thresh = image.copy()


# In[35]:


thresh[thresh > T] = 255


# In[36]:


thresh[thresh < 255] = 0


# In[37]:


thresh = cv2.bitwise_not(thresh)


# In[39]:


cv2.imshow("OTSU Thresholding of Images", np.hstack([image, thresh]))
cv2.waitKey(0)


# In[40]:


T_RC = mahotas.thresholding.rc(blurred)


# In[41]:


thresh_RC = image.copy()


# In[42]:


thresh_RC[thresh_RC > T_RC] = 255


# In[43]:


thresh_RC[thresh_RC < 255] = 0


# In[44]:


thresh_RC = cv2.bitwise_not(thresh_RC)


# In[45]:


cv2.imshow("RC Thresholding of Images", np.hstack([image, thresh_RC]))
cv2.waitKey(0)

