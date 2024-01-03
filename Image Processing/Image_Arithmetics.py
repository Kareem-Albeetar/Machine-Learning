#!/usr/bin/env python
# coding: utf-8

# # Images
# ## This notebook outlines the math of the Image manipulation used in OpenCV library
# 
# 
# 
# 

# In[5]:


from __future__ import print_function
import argparse
import cv2
import numpy as np


# In[6]:


image = cv2.imread("image.jpg")


# In[7]:


image


# 

# In[13]:


print(f"wrap around: {np.uint8([200]) + np.uint8([100])}")
print(f"wrap around: {np.uint8([50]) - np.uint8([100])}")


# In[14]:


print(f"Max of 255: {cv2.add(np.uint8([200]), np.uint8([100]))}")
print(f"Min of 0: {cv2.subtract(np.uint8([50]), np.uint8([100]))}")


# In[15]:


M = np.ones(image.shape, dtype = "uint8") * 100
M


# In[16]:


image


# In[17]:


added_image = cv2.add(image, M)
added_image


# In[19]:


cv2.imshow("Original", image)
cv2.imshow("Added", added_image)
cv2.waitKey(0)


# In[20]:


M = np.ones(image.shape, dtype = "uint8") * 100
M


# In[22]:


subtracted_image = cv2.subtract(image, M)
subtracted_image


# In[23]:


cv2.imshow("Original", image)
cv2.imshow("Subtracted", subtracted_image)
cv2.waitKey(0)


# In[25]:


rectangle = np.zeros((300, 300), dtype = "uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)


# In[26]:


circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
cv2.waitKey(0)


# In[27]:


bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)


# In[28]:


bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)


# In[29]:


bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)


# In[31]:


bitwiseNotCircle = cv2.bitwise_not(circle)
cv2.imshow("NOT CIRCLE", bitwiseNotCircle)

bitwiseNotRect = cv2.bitwise_not(rectangle)
cv2.imshow("NOT RECT", bitwiseNotRect)
cv2.waitKey(0)

