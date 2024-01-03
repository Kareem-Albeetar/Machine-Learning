#!/usr/bin/env python
# coding: utf-8

# # Face Detection
# ## This notebook outlines the concepts behind the face detection in images and videos

# ### Haar-like feaatures
# **Alfred Haar** gave the concepts of Haar wavelets, which are a sequence of **rescaled “square-shaped” functions** which together form a wavelet family or basis
# 
# #### Haar-like features 
# - features used in object recognition
# - All human faces share some universal properties of the human face
#     - eyes region is darker than its neighbour pixels
#     - nose region is brighter than the eye region
# 
# #### How to find out which region is ligther or darker?
# - Sum up the pixel values of both regions
# - Compare them
# 
# - The sum of pixel values in the darker region will be smaller than the sum of pixels in the lighter region
# - If one side is lighter than the other, it may be an **edge of an eyebrow** or sometimes the middle portion may be shinier than the surrounding boxes, which can be interpreted as a nose 
# 

# ![Haar-features](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/haar-features.png)

# ### Integral Images

# To calculate a value for each feature, we need to perform computations on all the pixels inside that particular feature
# 
# #### Issue:
# - These calculations can be very **intensive** since the number of pixels would be much greater when we are dealing with a large feature
# 
# Solution: **Integral Image**
# 
# 
# An integral image (also known as a **summed-area table**) is the name of both a data structure and an algorithm used to obtain this data structure
# - It is used as a quick and efficient way to calculate the **sum of pixel values in an image** or rectangular part of an image.
# 
# In an integral image, the value of each point is the **sum of all pixels above and to the left, including the target pixel**

# ![Integral Image](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/integral_image_1.png)

# Using these integral images, we save a lot of time calculating the summation of all the pixels in a rectangle as we only have to perform calculations on four edges of the rectangle. See the example below to understand.

# ![Integral image computation](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/integral_image_2.png)

# When we add the pixels in the blue box, we get 8 as the sum of all pixels and here we had six elements involved in your calculation. Now to calculate the sum of these same pixels using the integral image, you just need to find the corners of the rectangle and then add the vertices which are green and subtract the vertices in the red boxes.
# 
# We get the same answer and only four numbers are involved in calculations. No matter how many pixels are in the rectangle box, we will just need to compute on these 4 vertices.
# 
# Now to calculate the value of any haar-like feature, you have a simple way to calculate the difference between the sums of pixel values of two rectangles.

# ### AdaBoost

# The number of features that are present in the 24×24 detector window is nearly 160,000, but only a few of these features are important to identify a face. So we use the AdaBoost algorithm to identify the best features in the 160,000 features. 
# 
# In the Viola-Jones algorithm, each Haar-like feature represents a weak learner. To decide the type and size of a feature that goes into the final classifier, AdaBoost checks the performance of all classifiers that you supply to it.
# 
# To calculate the performance of a classifier, you evaluate it on all subregions of all the images used for training. Some subregions will produce a strong response in the classifier. Those will be classified as positives, meaning the classifier thinks it contains a human face. Subregions that don’t provide a strong response don’t contain a human face, in the classifiers opinion. They will be classified as negatives.
# 
# The classifiers that performed well are given higher importance or weight. The final result is a strong classifier, also called a boosted classifier, that contains the best performing weak classifiers.
# 
# So when we’re training the AdaBoost to identify important features, we’re feeding it information in the form of training data and subsequently training it to learn from the information to predict. So ultimately, the algorithm is setting a minimum threshold to determine whether something can be classified as a useful feature or not.

# ### Cascading Classifiers

# Maybe the AdaBoost will finally select the best features around say 2500, but it is still a time-consuming process to calculate these features for each region. We have a 24×24 window which we slide over the input image, and we need to find if any of those regions contain the face. The job of the cascade is to quickly discard non-faces, and avoid wasting precious time and computations. Thus, achieving the speed necessary for real-time face detection.
# 
# We set up a cascaded system in which we divide the process of identifying a face into multiple stages. In the first stage, we have a classifier which is made up of our best features, in other words, in the first stage, the subregion passes through the best features such as the feature which identifies the nose bridge or the one that identifies the eyes. In the next stages, we have all the remaining features.
# 
# When an image subregion enters the cascade, it is evaluated by the first stage. If that stage evaluates the subregion as positive, meaning that it thinks it’s a face, the output of the stage is maybe.
# 
# When a subregion gets a maybe, it is sent to the next stage of the cascade and the process continues as such till we reach the last stage.
# 
# If all classifiers approve the image, it is finally classified as a human face and is presented to the user as a detection.
# 
# Now how does it help us to increase our speed? Basically, If the first stage gives a negative evaluation, then the image is immediately discarded as not containing a human face. If it passes the first stage but fails the second stage, it is discarded as well. Basically, the image can get discarded at any stage of the classifier

# In[28]:


import cv2
import os


# In[29]:


cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"


# In[30]:


faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)


# In[31]:


video_capture = cv2.VideoCapture(0)


# In[33]:


while True:
    ret, frame = video_capture.read()
    cv2.imshow('Face Video', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break


# In[34]:


while True:
    ret, frame = video_capture.read()
    cv2.imshow('Face Video', frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray video', gray)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break


# In[35]:


faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE
)


# In[36]:


faces


# In[ ]:


for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 255, 0), 2)
    faceROI = frame[y:y+h,x:x+w]


# In[ ]:


eyes = eyeCascade.detectMultiScale(faceROI)


# In[ ]:


for (x2, y2, w2, h2) in eyes:
    eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
    radius = int(round((w2 + h2) * 0.25))
    frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)


# In[ ]:


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 255, 0), 2)
        faceROI = frame[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)

        # Display the resulting frame
    cv2.imshow('Face Video', frame)
    cv2.imshow("Face ROI", faceROI)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[27]:


import cv2
import os
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 255, 0), 2)
        faceROI = frame[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)

        # Display the resulting frame
    cv2.imshow('Face Video', frame)
    cv2.imshow("Face ROI", faceROI)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

