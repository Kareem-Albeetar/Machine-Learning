#!/usr/bin/env python
# coding: utf-8

# ## Label Encoding
# ### This notebook outlines the usage of Label Encoding

# In[37]:


import pandas as pd
import numpy as np


# In[38]:


data=pd.read_csv("X_train.csv")
data


# In[39]:


data.info()


# In[40]:


from sklearn import preprocessing


# In[41]:


label_encoder = preprocessing.LabelEncoder()


# In[42]:


data['Property_Area_Clean']= label_encoder.fit_transform(data['Property_Area']) 
data


# In[43]:


data[['Property_Area', 'Property_Area_Clean']]


# In[44]:


data['Gender_Clean']= label_encoder.fit_transform(data['Gender'])
data['Education_Clean']= label_encoder.fit_transform(data['Education']) 
data['Self_Employed_Clean']= label_encoder.fit_transform(data['Self_Employed'])
print(data.head())


# In[45]:


data.columns


# In[46]:


data = data[['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area_Clean', 'Gender_Clean', 'Education_Clean',
       'Self_Employed_Clean']]
data


# In[47]:


data.Dependents.value_counts()


# In[48]:


def clean_dep(x):
    return x[0]


# In[49]:


data['Dependents_Clean'] = data['Dependents'].apply(clean_dep)
data


# In[50]:


data = data[['Dependents_Clean', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area_Clean', 'Gender_Clean', 'Education_Clean',
       'Self_Employed_Clean']]
data


# In[51]:


print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[52]:


log=LogisticRegression(penalty='l2', C=0.1)
log.fit(data,Y_train)


# In[53]:


accuracy_score(Y_test,log.predict(X_test))


# In[54]:


test_data=pd.read_csv("X_test.csv")
test_data


# In[55]:


test_data.info()


# In[56]:


test_data['Property_Area_Clean']= label_encoder.fit_transform(test_data['Property_Area'])
test_data['Gender_Clean']= label_encoder.fit_transform(test_data['Gender'])
test_data['Education_Clean']= label_encoder.fit_transform(test_data['Education']) 
test_data['Self_Employed_Clean']= label_encoder.fit_transform(test_data['Self_Employed'])
test_data['Dependents_Clean'] = test_data['Dependents'].apply(clean_dep)
test_data


# In[57]:


test_data = test_data[['Dependents_Clean', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area_Clean', 'Gender_Clean', 'Education_Clean',
       'Self_Employed_Clean']]
test_data


# In[58]:


accuracy_score(Y_test,log.predict(test_data))

