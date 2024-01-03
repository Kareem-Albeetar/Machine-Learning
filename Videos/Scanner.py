#!/usr/bin/env python
# coding: utf-8

# # BUILD A OCR SCANNER
# ## This notebook outlines the concepts behind building an OCR Scanner for processing documents

# In[1]:


import cv2
import numpy as np


# #### Load the image
# https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Receipt.png

# In[2]:


image = cv2.imread("Receipt.png")


# In[4]:


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# In[5]:


ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(image, height = 500)


# In[6]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[8]:


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# In[9]:


edged = cv2.Canny(blurred, 75, 200)


# In[10]:


cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# In[13]:


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


# In[15]:


cnts = grab_contours(cnts)


# In[16]:


cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


# In[17]:


peri = cv2.arcLength(cnts[0], True)
approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)


# In[18]:


if len(approx) == 4:
    screenCnt = approx


# In[20]:


for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        screenCnt = approx
        break


# In[21]:


cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)


# In[22]:


cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[23]:


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# In[26]:


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


# In[27]:


warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


# In[28]:


warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


# In[30]:


# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 11)


# In[31]:


cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", resize(warped, height = 650))
cv2.waitKey(0)

