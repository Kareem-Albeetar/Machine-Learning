#!/usr/bin/env python
# coding: utf-8

# # Images
# ## This notebook outlines the techniques to be used in working with images using OpenCV library

# In[1]:


from __future__ import print_function
import argparse
import cv2


# In[2]:


image = cv2.imread("image.jpg")


# In[3]:


image.shape


# In[4]:


def get_details(image_file):
    image = cv2.imread(image_file)
    width = image.shape[1]
    height = image.shape[0]
    channels = image.shape[2]
    return width, height, channels


# In[5]:


width, height, channels = get_details("image.jpg")
print(f"Width = {width} \nHeight = {height} \nChannels = {channels}")


# In[6]:


cv2.imshow("Image", image)
cv2.waitKey(0)


# In[7]:


cv2.imwrite("newimage.jpg", image)


# In[8]:


get_ipython().system(' ls')


# In[9]:


new_image = cv2.imread("newimage.jpg")
new_image.shape


# In[10]:


cv2.imshow("NewImage", new_image)
cv2.waitKey(0)


# In[11]:


(b, g, r) = image[10, 10]
print("Pixel at (10, 10) - Red: {}, Green: {}, Blue: {}".format(r, g, b))


# In[13]:


image[10, 10] = (0, 0, 255)
(b, g, r) = image[10, 10]
print("Pixel at (10, 10) - Red: {}, Green: {}, Blue: {}".format(r, g, b))


# In[14]:


cv2.imshow("Changed Image", image)
cv2.waitKey(0)


# In[15]:


corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)
cv2.waitKey(0)


# In[16]:


image[0:100, 0:100] = (0, 255, 0)


# In[17]:


cv2.imshow("Updated", image)
cv2.waitKey(0)


# In[3]:


import numpy as np
import cv2
canvas = np.zeros((300, 300, 3), dtype = "uint8")
canvas.shape


# In[4]:


cv2.imshow("Canvas", canvas)
cv2.waitKey(0)


# In[5]:


white_canvas = cv2.bitwise_not(canvas)
cv2.imshow("White Canvas", white_canvas)
cv2.waitKey(0)


# In[6]:


red = (0, 0, 255)
cv2.line(white_canvas, (0, 0), (300, 300), red)
cv2.imshow("Red Line on White Canvas", white_canvas)
cv2.waitKey(0)


# In[7]:


red = (0, 0, 255)
cv2.line(white_canvas, (0, 0), (300, 300), red, 5)
cv2.imshow("Red Thick Line on White Canvas", white_canvas)
cv2.waitKey(0)


# In[9]:


white_canvas = cv2.bitwise_not(canvas)
cv2.rectangle(white_canvas, (10, 10), (60, 60), red)
cv2.imshow("Red Square", white_canvas)
cv2.waitKey(0)


# In[10]:


white_canvas = cv2.bitwise_not(canvas)
cv2.rectangle(white_canvas, (50, 200), (200, 225), red, 5)
cv2.imshow("Red Thick Square Canvas", white_canvas)
cv2.waitKey(0)


# In[11]:


white_canvas = cv2.bitwise_not(canvas)
blue = (255, 0, 0)
cv2.rectangle(white_canvas, (200, 50), (225, 125), blue, -1)
cv2.imshow("Blue Rectangle Canvas", white_canvas)
cv2.waitKey(0)


# In[12]:


canvas = np.zeros((300, 300, 3), dtype = "uint8")
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

for r in range(0, 175, 25):
    cv2.circle(canvas, (centerX, centerY), r, white)

cv2.imshow("Circles Canvas", canvas)
cv2.waitKey(0)


# In[15]:


for i in range(0, 25):
    radius = np.random.randint(5, high = 200)
    color = np.random.randint(0, high = 256, size = (3,)).tolist()
    pt = np.random.randint(0, high = 300, size = (2,))
    cv2.circle(canvas, tuple(pt), radius, color, -1)


cv2.imshow("Random Circles Canvas", canvas)
cv2.waitKey(0)

