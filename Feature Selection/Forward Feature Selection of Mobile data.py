#!/usr/bin/env python
# coding: utf-8

# # Feature Selection from Mobile data using SequentialFeatureSelector Forward method
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


df.price_range.value_counts()


# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# In[7]:


data = pd.read_csv("data/mobile_price_train.csv")


# In[8]:


X = data.iloc[:,0:20]
y = data.iloc[:,-1] 


# In[9]:


X.shape, y.shape


# In[10]:


X.shape[1]


# In[ ]:





# In[11]:


from mlxtend.feature_selection import SequentialFeatureSelector


# In[12]:


from sklearn.linear_model import LogisticRegression


# Solver theory 
# [https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451]

# In[13]:


lr = LogisticRegression(class_weight='balanced',
                        solver='lbfgs', 
                        random_state=42, 
                        n_jobs=-1, 
                        max_iter=500
)


# In[14]:


ffs = SequentialFeatureSelector(lr, k_features='best', forward=True, n_jobs=-1)


# In[15]:


ffs.fit(X, y)


# In[16]:


ffs.k_feature_names_


# In[17]:


features = list(ffs.k_feature_names_)
features


# In[ ]:





# In[18]:


full_feature_model = lr.fit(X, y)


# In[19]:


y_pred = full_feature_model.predict(X)
y_pred


# In[20]:


Compares = pd.DataFrame()
Compares['Predictions'] = pd.Series(y_pred)
Compares['Actuals'] = pd.Series(y)


# In[21]:


Compares['classmatch?'] = np.where(Compares['Predictions'] == Compares['Actuals'], 'True', 'False')
Compares


# In[22]:


Compares[Compares['classmatch?'] == 'False'].count()


# In[23]:


Compares[Compares['classmatch?'] == 'False'].groupby('Predictions').count()


# In[24]:


best_feature_model = lr.fit(X[features], y)


# In[25]:


y_pred = best_feature_model.predict(X[features])
y_pred


# In[26]:


y


# In[27]:


Compares = pd.DataFrame()


# In[28]:


Compares['Predictions'] = pd.Series(y_pred)
Compares['Actuals'] = pd.Series(y)
Compares.shape


# In[29]:


Compares.head()


# In[30]:


# accur = (preds == actuals)
Compares['classmatch?'] = np.where(Compares['Predictions'] == Compares['Actuals'], 'True', 'False')
Compares


# In[31]:


Compares[Compares['classmatch?'] == 'False'].groupby('Predictions').count()


# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


ffs_knn = SequentialFeatureSelector(knn, k_features='best', forward=True, n_jobs=-1)


# In[ ]:


ffs_knn.fit(X, y)


# In[ ]:


ffs_knn.k_feature_names_


# In[ ]:


features = list(ffs_knn.k_feature_names_)
features

