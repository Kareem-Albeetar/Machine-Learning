#!/usr/bin/env python
# coding: utf-8

# #  Predicting Real Estate House Prices

# ### Dataset: Real_estate.csv

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns


# In[2]:


df = pd.read_csv("data/Real_estate.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


X = df.iloc[:,:-1]
X


# In[8]:


y = df.iloc[:,-1]
y


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.05,random_state = 0)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


reg = LinearRegression()


# In[12]:


reg.fit(X_train, y_train)


# In[13]:


y_pred = reg.predict(X_test)
y_pred


# In[14]:


reg.coef_


# In[15]:


from sklearn.metrics import r2_score
print('r2 Score : ', r2_score(y_test, y_pred))


# In[16]:


sns.regplot(x="X2 house age", y="Y house price of unit area", data=df);


# In[17]:


sns.regplot(y="X3 distance to the nearest MRT station", x="Y house price of unit area", data=df);


# In[18]:


sns.regplot(y="X4 number of convenience stores", x="Y house price of unit area", data=df);

