#!/usr/bin/env python
# coding: utf-8

# # Feature Selection from Mobile data using Chi-Squared filter method
# Dataset: [https://raw.githubusercontent.com/subashgandyer/datasets/main/mobile_price_train.csv]

# In[2]:


import pandas as pd


# In[1]:


url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/mobile_price_train.csv"


# In[3]:


df = pd.read_csv(url)
df.head()


# In[4]:


df.columns


# In[5]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[6]:


data = pd.read_csv("data/mobile_price_train.csv")


# In[8]:


X = data.iloc[:,0:20]
y = data.iloc[:,-1] 


# In[9]:


X.shape, y.shape


# In[10]:


X.shape[1]


# In[11]:


bestfeatures = SelectKBest(score_func=chi2, k=10)
bestfeatures


# In[13]:


topfeatures = bestfeatures.fit(X,y)


# In[14]:


topfeatures.scores_


# In[15]:


dfscores = pd.DataFrame(topfeatures.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[20]:


featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores


# In[21]:


featureScores.columns = ['Specs', 'Score']


# In[22]:


featureScores


# In[23]:


print(featureScores.nlargest(10,'Score')) 


# In[ ]:




