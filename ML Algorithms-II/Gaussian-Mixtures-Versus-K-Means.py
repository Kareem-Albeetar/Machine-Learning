#!/usr/bin/env python
# coding: utf-8

# # Gaussian Mixture Models Vs K-Means
# ## This notebook outlines the application of Gaussian Mixture Models versus K-Means

# # Dataset :
# 
# https://raw.githubusercontent.com/subashgandyer/datasets/main/gmm.csv
# 
# 

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[7]:


url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/gmm.csv"


# In[9]:


data = pd.read_csv(url)
data


# In[10]:


plt.figure(figsize=(10,10))
plt.scatter(data["Weight"],data["Height"])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Data Distribution')
plt.show()


# In[11]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)


pred = kmeans.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = pred
frame.columns = ['Weight', 'Height', 'cluster']


plt.figure(figsize=(10,10))
color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()


# In[13]:


data = pd.read_csv(url)
data


# In[14]:


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(data)

#predictions from gmm
labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

plt.figure(figsize=(10,10))
color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()

