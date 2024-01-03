#!/usr/bin/env python
# coding: utf-8

# # Feature Selection using Fisher Score filter method
# Dataset: [https://raw.githubusercontent.com/subashgandyer/datasets/main/mobile_price_train.csv]

# In[1]:


import pandas as pd


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import load_diabetes
db = load_diabetes()


# In[4]:


y_train = db.target
X_train = db.data


# In[5]:


X_train.shape, y_train.shape


# In[6]:


X_train.shape[1]


# In[7]:


db.keys()


# In[8]:


db.target.shape


# In[9]:


db.feature_names


# In[10]:


db.data.shape


# In[11]:


from skfeature.function.similarity_based import fisher_score


# In[12]:


ranks = fisher_score.fisher_score(X_train, y_train)
ranks


# In[13]:


dfscores = pd.DataFrame(ranks)
dfcolumns = pd.DataFrame(db.feature_names)


# In[14]:


featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores


# In[15]:


featureScores.columns = ['Specs', 'Score']


# In[16]:


featureScores


# In[17]:


print(featureScores.nlargest(10,'Score')) 


# In[ ]:




