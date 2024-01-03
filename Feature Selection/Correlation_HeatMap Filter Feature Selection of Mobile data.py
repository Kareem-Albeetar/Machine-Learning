#!/usr/bin/env python
# coding: utf-8

# # Feature Selection from Mobile data using Correlation Heatmap filter method
# Dataset: [https://raw.githubusercontent.com/subashgandyer/datasets/main/mobile_price_train.csv]

# In[2]:


import pandas as pd


# In[3]:


url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/mobile_price_train.csv"


# In[4]:


df = pd.read_csv(url)
df.head()


# In[5]:


df.columns


# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# In[13]:


data = pd.read_csv("data/mobile_price_train.csv")


# In[14]:


X = data.iloc[:,0:20]
y = data.iloc[:,-1] 


# In[15]:


X.shape, y.shape


# In[16]:


X.shape[1]


# In[32]:


corrmat = data.corr()
top_corr_features = corrmat.index
plt.rcParams['figure.figsize'] = [20, 20]


# In[31]:


sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:





# In[ ]:




