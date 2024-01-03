#!/usr/bin/env python
# coding: utf-8

# # SHAPES OF OBJECTS
# ## This notebook outlines the concepts used in Shape detection of objects in an image

# In[1]:


import cv2
import numpy as np


# ![Shapes](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Shapes.png)

# In[15]:


image = cv2.imread("Shapes.png")


# In[16]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[17]:


blurred = cv2.GaussianBlur(gray, (7, 7), 0)


# In[18]:


edged = cv2.Canny(blurred, 30, 150)


# In[19]:


cv2.imshow("Shapes_Interim", np.hstack([gray, blurred, edged]))
cv2.waitKey(0)


# In[20]:


(cnts, _) = cv2.findContours(edged.copy(), 
                             cv2.RETR_EXTERNAL, 
                             cv2 .CHAIN_APPROX_SIMPLE
)


# In[21]:


M = cv2.moments(cnts[0])
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])


# In[9]:


shapes = image.copy()
cv2.drawContours(shapes, [cnts[0]], -1, (0, 255, 0), 2)
cv2.circle(shapes, (cX, cY), 7, (255, 255, 255), -1)
cv2.putText(shapes, 
            "center", 
            (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2
)


# In[10]:


cv2.imshow("Shapes", np.hstack([image, shapes]))
cv2.waitKey(0)


# In[11]:


for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    cv2.drawContours(shapes, [c], -1, (0, 255, 0), 2)
    cv2.circle(shapes, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(shapes, 
                "center", 
                (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
    )
    
    cv2.imshow("Shapes", np.hstack([image, shapes]))
    cv2.waitKey(0)


# In[22]:


class ShapeDetector:
    def __init__(self):
        pass
    
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            # compute the bounding box of the contour & use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        
        return shape


# In[23]:


sd = ShapeDetector()


# In[24]:


for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    shape = sd.detect(c)
    
    cv2.drawContours(shapes, [c], -1, (0, 255, 0), 2)
    cv2.putText(shapes, 
                shape, 
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
    )
    
    cv2.imshow("Shapes", np.hstack([image, shapes]))
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

