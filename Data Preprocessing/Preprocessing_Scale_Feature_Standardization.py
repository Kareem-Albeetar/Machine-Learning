#!/usr/bin/env python
# coding: utf-8

# ## Feature Standardization
# ### This notebook outlines the Feature Standardization feature of Scikit-learn.
# - Scale
# - StandardScaler
# 
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Read training features into X_train
X_train=pd.read_csv('X_train.csv')
# Read training target into Y_train
Y_train=pd.read_csv('Y_train.csv')

# Read testing features into X_test
X_test=pd.read_csv('X_test.csv')
# Read testing target into Y_test
Y_test=pd.read_csv('Y_test.csv')

# Display the top 5 rows
print(X_train.head())


# In[3]:


X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
             .index.values].hist(figsize=[11,11])


# In[4]:


from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


X_train_scale=scale(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])


# In[6]:


log=LogisticRegression(penalty='l2',C=.01)


# In[7]:


log.fit(X_train_scale,Y_train)


# In[8]:


accuracy_score(Y_test,log.predict(X_test_scale))


# In[9]:


from sklearn import preprocessing
import numpy as np


# In[10]:


X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
scaler


# In[11]:


scaler.mean_


# In[12]:


scaler.scale_


# In[13]:


X_scaled = scaler.transform(X_train)
X_scaled


# In[14]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[15]:


X, y = make_classification(random_state=42)
print(X.shape, y.shape)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[17]:


pipe = make_pipeline(StandardScaler(), LogisticRegression())


# In[18]:


pipe.fit(X_train, y_train)  # apply scaling on training data


# In[19]:


pipe.score(X_test, y_test)


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[21]:


# Read training features into X_train
X_train=pd.read_csv('X_train.csv')
# Read training target into Y_train
Y_train=pd.read_csv('Y_train.csv')

# Read testing features into X_test
X_test=pd.read_csv('X_test.csv')
# Read testing target into Y_test
Y_test=pd.read_csv('Y_test.csv')

# Display the top 5 rows
print(X_train.head())


# In[22]:


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[23]:


X_train_numeric=scale(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_numeric=scale(X_test[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])


# In[24]:


pipe = make_pipeline(StandardScaler(), LogisticRegression())


# In[25]:


pipe.fit(X_train_numeric, Y_train)


# In[26]:


pipe.score(X_test_numeric, Y_test)


# In[27]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[28]:


# Read training features into X_train
X_train=pd.read_csv('X_train.csv')
# Read training target into Y_train
Y_train=pd.read_csv('Y_train.csv')

# Read testing features into X_test
X_test=pd.read_csv('X_test.csv')
# Read testing target into Y_test
Y_test=pd.read_csv('Y_test.csv')


# In[29]:


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[30]:


X_train_numeric=scale(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_numeric=scale(X_test[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])


# In[31]:


pipe = make_pipeline(StandardScaler(), svm.SVC())


# In[32]:


pipe.fit(X_train_numeric, Y_train)

