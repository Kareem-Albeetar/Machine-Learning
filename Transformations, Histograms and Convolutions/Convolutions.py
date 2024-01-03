#!/usr/bin/env python
# coding: utf-8

# # Understanding Convolutions
# ## This notebook outlines the concepts of Convolutions

# 
#    
# ### **Convolution**
# 
# Image: multi-dimensional matrix
#    - width (Number of columns)
#    - height (Number of rows)
#    
# Kernel or Convolutional matrix
#    - Tiny matrix
#    - Usually a square matrix
# 
# Tiny kernel **sits** on top of the big image and **slides** from left-to-right and top-to-bottom, applying a mathematical operation (i.e., a convolution) at each (x, y)-coordinate of the original image.
# 
# It’s normal to **hand-define kernels** to obtain various image processing functions

# ![Convolution](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Conv-1.png)

# We are sliding the kernel from left-to-right and top-to-bottom along the original image.
# 
# At each (x, y)-coordinate of the original image, we stop and examine the neighborhood of pixels located at the center of the image kernel. We then take this neighborhood of pixels, convolve them with the kernel, and obtain a single output value. This output value is then stored in the output image at the same (x, y)-coordinates as the center of the kernel.

# #### Sqaure 3 x 3 kernel
# ![Convolution](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Conv-2.png)

# Use an **odd kernel** size to ensure there is a valid integer (x, y)-coordinate at the center of the image

# ![Convolution](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Conv-3.png)

# With a 3 x 3 matrix, the center of the matrix is obviously located at **x=1, y=1** where the top-left corner of the matrix is used as the origin and our coordinates are zero-indexed.
# 
# With a 2 x 2 matrix, the center of this matrix would be located at **x=0.5, y=0.5** 
# 
# There is no such thing as pixel location (0.5, 0.5) — our pixel coordinates must be **integers**! 
# 
# This reasoning is exactly why we use **odd kernel sizes** — to always ensure there is a valid (x, y)-coordinate at the center of the kernel

# A convolution requires three components:
# - An input image
# - A kernel matrix that we are going to apply and slide on the input image
# - An output image to store the output of the input image convolved with the kernel

# **Convolution Steps**
# 
# - Select an (x, y)-coordinate from the original image
# - Place the **center** of the kernel at this (x, y)-coordinate
# - Take the element-wise multiplication of the input image region and the kernel, then sum up the values of these multiplication operations into a single value. The sum of these multiplications is called the **kernel output**
# - Use the same (x, y)-coordinates from Step #1, but this time, store the kernel output in the same (x, y)-location as the output image

# Convolution Output = Kernel Matrix (3 x 3) * Image (3 x 3)

# ![Convolution](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Conv-4.png)

# ![Convolution](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Conv-5.png)

# In[2]:


import numpy as np
import cv2


# In[8]:


get_ipython().system(' pip install scikit-image')


# In[10]:


from skimage.exposure import rescale_intensity


# In[11]:


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    
    # allocate memory for the output image, taking care to "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    
    # loop over the input image, "sliding" the kernel across each (x, y)-coordinate from left-to-right and top to bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the *center* region of the current (x, y)-coordinates dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            
            # perform the actual convolution by taking the element-wise multiplicate between the ROI and the kernel, then summing the matrix
            k = (roi * kernel).sum()
            
            # store the convolved value in the output (x,y)-coordinate of the output image
            output[y - pad, x - pad] = k
    
    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    
    return output


# In[12]:


smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))


# In[13]:


sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")


# In[14]:


image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[15]:


convolveOutput = convolve(gray, smallBlur)


# In[ ]:


convolveOutput = convolve(gray, largeBlur)


# In[16]:


OpenCVOutput = cv2.filter2D(gray, -1, smallBlur)


# In[18]:


cv2.imshow("Convolutions", np.hstack([convolveOutput, OpenCVOutput]))
cv2.waitKey(0)

