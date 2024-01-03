#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[9]:


df = pd.read_csv("data/raw_data.csv")
df.head()


# In[10]:


df.info()


# In[11]:


df.Country.value_counts()


# In[12]:


country = {"USA": 1, "Canada": 2, "France": 3, "India": 4, "Germany": 5, "Denmark": 6}


# In[13]:


df["Country"].replace(country)


# In[14]:


df


# In[15]:


df["Country"].replace(country, inplace=True)


# In[16]:


df


# In[17]:


df.Married.value_counts()


# In[18]:


married = {"Yes": 1, "No": 0}


# In[19]:


df["Married"].replace(married, inplace=True)


# In[20]:


df


# In[4]:


df.isnull()


# In[21]:


df["Age"].replace(np.NaN, df["Age"].mean(), inplace=True)


# In[22]:


df


# In[23]:


df["Salary"].replace(np.NaN, df["Salary"].mean(), inplace=True)


# In[24]:


df


# In[ ]:




