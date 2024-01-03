#!/usr/bin/env python
# coding: utf-8

# # PCA Exploration with Iris dataset
# ## This notebook outlines the PCA usage in Scikit-learn

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()


# In[3]:


iris


# In[4]:


iris.keys()


# In[5]:


type(iris)


# In[7]:


X = iris.data
y = iris.target


# In[8]:


X.shape, y.shape


# In[9]:


from sklearn.decomposition import PCA


# In[21]:


model = PCA(n_components=2)


# In[22]:


model.fit(X)


# In[23]:


X_2D = model.transform(X)


# In[24]:


X.shape, X_2D.shape


# In[ ]:


X_2D.


# In[25]:


iris['PCA1'] = X_2D[:, 0]
iris['PCA1']


# In[26]:


iris['PCA2'] = X_2D[:, 1]


# In[27]:


import seaborn as sns


# In[28]:


sns.lmplot("PCA1", "PCA2", hue="species", data=iris)


# In[29]:


type(iris.data)


# In[31]:


import pandas as pd


# In[32]:


df = pd.DataFrame(iris.data)
df.head()


# In[34]:


import seaborn as sns


# In[39]:


iris = sns.load_dataset("iris")


# In[40]:


type(iris)


# In[41]:


iris


# In[42]:


X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
X.head()


# In[44]:


y = iris[['species']]
y.head()


# In[45]:


from sklearn.decomposition import PCA


# In[46]:


model = PCA(n_components = 2)


# In[47]:


model.fit(X)


# In[48]:


X_2D = model.transform(X)


# In[49]:


X_2D.shape


# In[50]:


X.shape


# In[51]:


import seaborn as sns


# In[52]:


iris


# In[54]:


iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]


# In[55]:


iris


# ### Plot the data points with the two most principal components (PCA1 & PCA2)

# In[58]:


sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)

