#!/usr/bin/env python
# coding: utf-8

# # WATERSHED SEGMENTATION ALGORITHM
# 
# 

# In[1]:


import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed


# In[3]:


image = cv2.imread("coins_overlap.jpg")
cv2.imshow("Coins", image)
cv2.waitKey(0)


# In[6]:


shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)


# In[7]:


gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)


# In[8]:


thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# In[9]:


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


# In[10]:


def grab_contours(cnts): 
    if len(cnts) == 2: 
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else:
        raise Exception(("Contours tuple must have length 2 or "
                        "3, otherwise OpenCV changed their cv2.findContours " 
                        "return signature yet again. "
                        "Refer to OpenCV’s documentation in that case."))
    return cnts


# In[11]:


cnts = grab_contours(cnts)


# In[12]:


len(cnts)


# In[14]:


coins = image.copy()
for (i, c) in enumerate(cnts):
    # draw the contour
    ((x, y), _) = cv2.minEnclosingCircle(c)
    cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.drawContours(coins, [c], -1, (0, 255, 0), 2)


# In[15]:


cv2.imshow("Extraction of Coins", np.hstack([image, coins]))
cv2.waitKey(0)


# In[16]:


# Uncomment the following to install scipy, scikit-image, imutils
# ! pip install scipy scikit-image imutils


# In[18]:


from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


# #### Euclidean Distance Transform (EDT)
# Computes the Euclidean distance to the closest zero (background) pixel for each of the white (foreground) pixels and builds a distance map
# ![Distance Map](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Watershed-EDT-DistanceMap.png)

# In[19]:


D = ndimage.distance_transform_edt(thresh)


# In[20]:


localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)


# #### Connected Component Analysis
# [Wiki]https://en.wikipedia.org/wiki/Connected-component_labeling
# 
# Use **8-connectivity**

# In[21]:


markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]


# In[23]:


labels = watershed(-D, markers, mask = thresh)


# In[25]:


len(np.unique(labels)) - 1


# In[27]:


for label in np.unique(labels):
    if label == 0:
        continue
        
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    
    cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# In[28]:


cv2.imshow("Watershed Extraction of coins", np.hstack([coins, image]))
cv2.waitKey(0)

