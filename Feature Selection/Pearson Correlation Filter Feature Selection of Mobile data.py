#!/usr/bin/env python
# coding: utf-8

# # Feature Selection from Mobile data using Pearson Correlation filter method
# Dataset: [https://raw.githubusercontent.com/subashgandyer/datasets/main/mobile_price_train.csv]

# In[1]:


import pandas as pd


# In[2]:


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


# In[7]:


X = data.iloc[:,0:20]
y = data.iloc[:,-1] 


# In[8]:


X.shape, y.shape


# In[9]:


X.shape[1]


# In[10]:


feature_name = X.columns.tolist()


# In[11]:


cor = np.corrcoef(X['battery_power'], y)[0, 1]
cor


# In[12]:


cor = np.corrcoef(X['ram'], y)[0, 1]
cor


# In[13]:


cor_list = []
for i in X.columns.tolist():
    cor = np.corrcoef(X[i], y)[0, 1]
    cor_list.append(cor)


# In[14]:


cor_list


# In[15]:


cor_list = [0 if np.isnan(i) else i for i in cor_list]
cor_list


# In[16]:


cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-10:]].columns.tolist()
cor_feature


# In[17]:


# feature selection? 0 for not select, 1 for select
cor_support = [True if i in cor_feature else False for i in feature_name]
cor_support


# In[18]:


def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


# In[19]:


cor_support, cor_feature = cor_selector(X, y,num_feats=10)
cor_support, cor_feature


# In[ ]:





# In[20]:


dfscores = pd.DataFrame(cor_list)
dfcolumns = pd.DataFrame(X.columns)


# In[21]:


featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores


# In[22]:


featureScores.columns = ['Specs', 'Score']


# In[23]:


featureScores


# In[24]:


print(featureScores.nlargest(10,'Score')) 

