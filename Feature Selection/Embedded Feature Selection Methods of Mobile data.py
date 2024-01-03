#!/usr/bin/env python
# coding: utf-8

# # Embedded Feature Selection using Mobile data
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
import seaborn as sns
from matplotlib import pyplot as plt


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


y


# In[11]:


from sklearn.feature_selection import SelectFromModel


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


logreg = LogisticRegression(penalty='l1', solver='liblinear')


# In[14]:


embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=50000), max_features=5)


# In[15]:


embedded_lr_selector = embedded_lr_selector.fit(X, y)


# In[16]:


embedded_lr_support = embedded_lr_selector.get_support()
embedded_lr_support


# In[17]:


embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
embedded_lr_feature


# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


rf = RandomForestClassifier(n_estimators=100)


# In[20]:


embedded_rf_selector = SelectFromModel(rf, 
                           max_features=5
)


# In[21]:


embedded_rf_selector = embedded_rf_selector.fit(X, y)


# In[22]:


embedded_rf_support = embedded_rf_selector.get_support()
embedded_rf_support


# In[23]:


embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
embedded_rf_feature


# In[24]:


from lightgbm import LGBMClassifier


# In[25]:


lgbmc = LGBMClassifier(n_estimators=500,
                      learning_rate=0.05,
                      num_leaves=32,
                      colsample_bytree=0.2,
                      reg_alpha=3,
                      reg_lambda=1,
                      min_split_gain=0.01,
                      min_child_weight=40
)


# In[26]:


embedded_lgbm_selector = SelectFromModel(lgbmc,
                                         max_features=5
)


# In[27]:


embedded_lgbm_selector = embedded_lgbm_selector.fit(X, y)


# In[28]:


embedded_lgbm_support = embedded_lgbm_selector.get_support()
embedded_lgbm_support


# In[29]:


embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
embedded_lgbm_feature

