#!/usr/bin/env python
# coding: utf-8

# #  Principal Component Analysis
# ## This notebook outlines the applications of the Principal Components Analysis Machine Learning Algorithm

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[8]:


from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


# In[9]:


from sklearn.decomposition import PCA as RandomizedPCA


# In[10]:


pca = RandomizedPCA(150)


# In[11]:


pca.fit(faces.data)


# In[12]:


fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')


# In[13]:


plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[14]:


pca = RandomizedPCA(150).fit(faces.data)


# In[15]:


components = pca.transform(faces.data)


# In[16]:


projected = pca.inverse_transform(components)


# In[17]:


fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
    
ax[0, 0].set_ylabel('3000-dim\noriginal')
ax[1, 0].set_ylabel('150-dim\nreconstruction');


# In[19]:


def reconstruct_images(n_components):
    pca = RandomizedPCA(n_components).fit(faces.data)
    components = pca.transform(faces.data)
    projections = pca.inverse_transform(components)
    return projections


# In[20]:


def plot_projections(projections, n):
    fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i in range(10):
        ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
        ax[1, i].imshow(projections[i].reshape(62, 47), cmap='binary_r')

    ax[0, 0].set_ylabel('3000-dim\noriginal')
    ax[1, 0].set_ylabel(f'{n}-dim\nreconstruction');


# In[21]:


components = [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200, 300, 500]
for n in components:
    projections = reconstruct_images(n)
    plot_projections(projections, n)

