#!/usr/bin/env python
# coding: utf-8

# # Recursive Feature Elimination using Mobile data
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


# In[33]:


from sklearn.feature_selection import RFE


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[40]:


lr = LogisticRegression(solver='lbfgs')


# In[42]:


rfe_lr = RFE(estimator=lr, 
             n_features_to_select=5,
             step=1,
             verbose=5
)


# In[43]:


rfe_lr = rfe_lr.fit(X, y)


# In[44]:


rfe_lr_support = rfe_lr.get_support()
rfe_lr_support, type(rfe_lr_support)


# In[45]:


rfe_lr_feature = X.loc[:, rfe_lr_support].columns.tolist()
rfe_lr_feature


# In[46]:


rfe_lr.ranking_


# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


rf = RandomForestClassifier()


# In[20]:


rfe_rf = RFE(estimator=rf, 
             n_features_to_select=5,
             step=1,
             verbose=5
)


# In[21]:


rfe_rf = rfe_rf.fit(X, y)


# In[22]:


rfe_rf_support = rfe_rf.get_support()
rfe_rf_support


# In[ ]:





# In[23]:


rfe_rf_feature = X.loc[:, rfe_rf_support].columns.tolist()
rfe_rf_feature


# In[24]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.neighbors import KNeighborsClassifier


# In[25]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[29]:


efs_knn = EFS(knn, 
           min_features=2,
           max_features=4,
           scoring='accuracy',
           print_progress=True,
           cv=5
)


# In[30]:


efs_knn.fit(X, y)


# In[31]:


efs_knn.best_feature_names_


# In[32]:


efs_knn.best_score_

