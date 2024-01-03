#!/usr/bin/env python
# coding: utf-8

# # CONTOURS IN IMAGES
# ## This notebook outlines the concepts used in Contours in the field of Image Processing

# In[3]:


import cv2
import numpy as np


# In[26]:


image = cv2.imread("coins.png")


# In[27]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[28]:


blurred = cv2.GaussianBlur(gray, (9, 9), 0)


# In[29]:


edged = cv2.Canny(blurred, 30, 150)


# In[30]:


cv2.imshow("Contour_Interim", np.hstack([gray, blurred, edged]))
cv2.waitKey(0)


# In[31]:


(cnts, _) = cv2.findContours(edged.copy(), 
                             cv2.RETR_EXTERNAL, 
                             cv2 .CHAIN_APPROX_SIMPLE
)


# In[32]:


coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)


# In[33]:


cv2.imshow("Coins", np.hstack([image, coins]))
cv2.waitKey(0)


# In[34]:


len(cnts)


# In[35]:


for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print("Coin #{}".format(i + 1))
    coin = image[y:y + h, x:x + w]
    cv2.imshow("Coin", coin)
    
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask))
    cv2.waitKey(0)


# In[2]:


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

