#!/usr/bin/env python
# coding: utf-8

# ## Feature Scaler - MinMaxScaler

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


Y_train.head()


# In[4]:


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[5]:


X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
             .index.values].hist(figsize=[11,11])


# In[6]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[7]:


min_max=MinMaxScaler()


# In[8]:


X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])


# In[9]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[10]:


knn.fit(X_train_minmax,Y_train)


# In[11]:


accuracy_score(Y_test,knn.predict(X_test_minmax))


# In[12]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[13]:


min_max=MinMaxScaler()


# In[14]:


X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])


# In[15]:


log=LogisticRegression(penalty='l2', C=0.01)


# In[16]:


log.fit(X_train_minmax,Y_train)


# In[17]:


accuracy_score(Y_test,log.predict(X_test_minmax))

