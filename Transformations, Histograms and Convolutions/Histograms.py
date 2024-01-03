#!/usr/bin/env python
# coding: utf-8

# # Histograms
# ## This notebook outlines the techniques used in Histograms of images

# In[1]:


# import the necessary packages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


# In[34]:


image = cv2.imread("image.jpg")
cv2.imshow("Original", image)
cv2.waitKey(0)


# In[35]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayScale", image)
cv2.waitKey(0)


# In[36]:


hist = cv2.calcHist([image], [0], None, [256], [0, 256])


# In[37]:


plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()


# In[41]:


image = cv2.imread("image.jpg")
cv2.imshow("Original", image)
cv2.waitKey(0)


# In[42]:


chans = cv2.split(image)
colors = ("b", "g", "r")


# In[43]:


plt.figure()
plt.title("’Flattened’ Color Histogram") 
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# Loop over the image channels
for (chan, color) in zip(chans, colors):
    # cv2.calcHist(images,channels,mask,histSize,ranges)
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])


# In[44]:


fig = plt.figure()


# In[47]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(131)
# cv2.calcHist(images,channels,mask,histSize,ranges)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)
plt.show()


# In[49]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)


# In[50]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)


# In[51]:


# Finally, let's examine the dimensionality of one of the 2D histograms
print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))


# In[52]:


hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))


# In[26]:


plt.show()


# In[27]:


# Grab the image channels, initialize the tuple of colors
# and the figure
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# Loop over the image channels
for (chan, color) in zip(chans, colors):
	# Create a histogram for the current channel and plot it
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color = color)
	plt.xlim([0, 256])

# Let's move on to 2D histograms -- I am reducing the
# number of bins in the histogram from 256 to 32 so we
# can better visualize the results
fig = plt.figure()

# Plot a 2D color histogram for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
	[32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)

# Plot a 2D color histogram for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
	[32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)

# Plot a 2D color histogram for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
	[32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)

# Finally, let's examine the dimensionality of one of
# the 2D histograms
print("2D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))

# Our 2D histogram could only take into account 2 out
# of the 3 channels in the image so now let's build a
# 3D color histogram (utilizing all channels) with 8 bins
# in each direction -- we can't plot the 3D histogram, but
# the theory is exactly like that of a 2D histogram, so
# we'll just show the shape of the histogram
hist = cv2.calcHist([image], [0, 1, 2],
	None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))

# Show our plots
plt.show()


# In[55]:


# store our largest bin size and number of bins in convenience
# variables for ease of use
image = cv2.imread("image.jpg")
size = float(5000)
bins = int(4)

# compute the color histogram for the input image
hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

# show the shape of the hostgram
print("3D histogram shape: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))

# initialize our figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection="3d")

# find the largest value in our histogram and then compute the ratio of our largest size bin to the largest in the histogram
ratio = size / np.max(hist)

# loop over the histogram planes
for (x, plane) in enumerate(hist):
    for (y, row) in enumerate(plane):
        for (z, col) in enumerate(row):
            # ensure that there is a value in the current bin
            if hist[x][y][z] > 0.0:
                # plot the bin
                siz = ratio * hist[x][y][z]
                rgb = (z / (bins - 1), y / (bins - 1), x / (bins - 1))
                ax.scatter(x, y, z, s = siz, facecolors = rgb)

# show the figures 
plt.show()


# In[7]:


# store our largest bin size and number of bins in convenience variables for ease of use
image = cv2.imread("image.jpg")
size = float(5000)
bins = int(4)


# In[6]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[8]:


# compute the color histogram for the input image
hist = cv2.calcHist([image], [0, 1, 2],None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])


# In[9]:


# show the shape of the hostgram
print("3D histogram shape: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))


# In[10]:


# initialize our figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# In[11]:


# find the largest value in our histogram and then compute the ratio
# of our largest size bin to the largest in the histogram
ratio = size / np.max(hist)


# In[12]:


# loop over the histogram planes
for (x, plane) in enumerate(hist):
	for (y, row) in enumerate(plane):
		for (z, col) in enumerate(row):
			# ensure that there is a value in the current bin
			if hist[x][y][z] > 0.0:
				# plot the bin
				siz = ratio * hist[x][y][z]
				rgb = (z / (bins - 1), y / (bins - 1), x / (bins - 1))
				ax.scatter(x, y, z, s = siz, facecolors = rgb)


# In[13]:


# show the figures 
plt.show()


# In[56]:


image = cv2.imread("image.jpg")
# cv2.imshow("Original", image)
# cv2.waitKey(0)


# In[57]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", image)
# cv2.waitKey(0)


# In[58]:


eq = cv2.equalizeHist(image)


# In[59]:


cv2.imshow("Histogram Equalization", np.hstack([image, eq]))
cv2.waitKey(0)


# In[60]:


def plot_histogram(image, title, mask = None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])


# In[64]:


plot_histogram(image, "Histogram for Original Image")


# In[61]:


mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask, (15, 15), (130, 100), 255, -1)
cv2.imshow("Mask", mask)
cv2.waitKey(0)


# In[62]:


masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Applying the Mask", masked)
cv2.waitKey(0)


# In[63]:


plot_histogram(image, "Histogram for Masked Image", mask = mask)


# In[9]:


import cv2
import numpy as np


# In[10]:


image = cv2.imread("potrait.jpg")


# In[11]:


image.shape


# In[12]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[13]:


eq = cv2.equalizeHist(image)


# In[14]:


cv2.imshow("Histogram Equalization", np.hstack([image, eq]))
cv2.waitKey(0)

