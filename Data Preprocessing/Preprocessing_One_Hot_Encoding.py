#!/usr/bin/env python
# coding: utf-8

# ## One Hot Encoding
# ### This notebook outlines the usage of One Hot Encoding
# ### One Hot Encoding creates additional features based on the number of unique values in a categorical feature.
# ### In other words, it creates dummy variables

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("X_train.csv")
df


# In[3]:


df.info()


# In[4]:


from sklearn.preprocessing import OneHotEncoder


# In[5]:


onehotencoder = OneHotEncoder()


# In[6]:


df['Property_Area_Clean']= onehotencoder.fit_transform(df['Property_Area']) 
df


# In[7]:


values = df.Property_Area.values
values


# In[8]:


values = df.Property_Area.values.reshape(-1,1)
values


# In[9]:


X = onehotencoder.fit_transform(df[['Property_Area']]).toarray()


# In[10]:


df2 = pd.DataFrame(X)
df2_new = pd.concat([df,df2], axis=1)
df2_new


# In[11]:


df3=pd.get_dummies(df[["Property_Area"]])


# In[12]:


df3_new=pd.concat([df,df3],axis=1)


# In[13]:


del df3_new['Property_Area']
df3_new


# In[14]:


df4=pd.get_dummies(df[["Gender"]])
df4_new=pd.concat([df3_new,df4],axis=1)
del df4_new['Gender']
df4_new


# In[15]:


df5=pd.get_dummies(df[["Married"]])
df5_new=pd.concat([df4_new,df5],axis=1)
del df5_new['Married']
df5_new


# In[16]:


df6=pd.get_dummies(df[["Education"]])
df6_new=pd.concat([df5_new,df6],axis=1)
del df6_new['Education']
df6_new


# In[17]:


df7=pd.get_dummies(df[["Self_Employed"]])
df7_new=pd.concat([df6_new,df7],axis=1)
del df7_new['Self_Employed']
df7_new


# In[18]:


def clean_dep(x):
    return x[0]


# In[19]:


df7_new['Dependents_Clean'] = df7_new['Dependents'].apply(clean_dep)
df7_new


# In[20]:


df8=pd.get_dummies(df[["Dependents"]])
df8_new=pd.concat([df7_new,df8],axis=1)
del df8_new['Dependents']
del df8_new['Dependents_Clean']
df8_new


# In[21]:


df8_new.info()


# In[22]:


del df8_new['Loan_ID']
df8_new


# In[23]:


df8_new.info()


# In[24]:


Y_train = pd.read_csv("Y_train.csv")
Y_test = pd.read_csv("Y_test.csv")
X_test = pd.read_csv("X_test.csv")


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[26]:


log=LogisticRegression(penalty='l2', C=0.1)
log.fit(df8_new,Y_train)


# In[27]:


accuracy_score(Y_test,log.predict(X_test))


# In[28]:


test_data=pd.read_csv("X_test.csv")
test_data


# In[29]:


test_data.info()


# In[30]:


test_data


# In[31]:


test_data.info()


# In[32]:


df1=pd.get_dummies(test_data[["Property_Area"]])
df2=pd.get_dummies(test_data[["Gender"]])
df3=pd.get_dummies(test_data[["Married"]])
df4=pd.get_dummies(test_data[["Education"]])
df5=pd.get_dummies(test_data[["Self_Employed"]])
df6_new=pd.concat([test_data,df1,df2,df3,df4,df5],axis=1)
del df6_new['Property_Area']
del df6_new['Gender']
del df6_new['Married']
del df6_new['Education']
del df6_new['Self_Employed']
df6_new


# In[33]:


def clean_dep(x):
    return x[0]


# In[34]:


df6_new['Dependents_Clean'] = df6_new['Dependents'].apply(clean_dep)
df6_new


# In[35]:


df7=pd.get_dummies(df6_new[["Dependents_Clean"]])
df7_new=pd.concat([df6_new,df7],axis=1)
del df7_new['Dependents_Clean']
del df7_new['Dependents']
del df7_new['Loan_ID']
df7_new


# In[36]:


df7_new.info()


# In[37]:


X_test = df7_new


# In[38]:


accuracy_score(Y_test,log.predict(X_test))

