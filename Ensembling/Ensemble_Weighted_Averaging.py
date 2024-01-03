#!/usr/bin/env python
# coding: utf-8

# # Weighted Averaging
# ## This notebook outlines the Weighted Averaging technique in Ensembling of Machine Learning models

# ### Dataset
# Predict Loan Eligibility for Dream Housing Finance company
# 
# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
# 
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.
# 
# Train: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv
# 
# Test: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_test.csv

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[343]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[344]:


df=pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv")


# In[345]:


df


# In[346]:


df.info()


# In[347]:


df.columns


# In[348]:


cat_df = df[['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Property_Area', 'Loan_Status']]


# In[349]:


num_df = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History']]


# In[350]:


y_df = df[['Loan_Status']]


# In[351]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[352]:


cat_df.Gender.value_counts()


# In[353]:


cat_df['Gender'].fillna('Male', inplace=True)
cat_df.Gender.value_counts()


# In[354]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[355]:


cat_df.Married.value_counts()


# In[356]:


cat_df['Married'].fillna("Yes", inplace=True)
cat_df.Married.value_counts()


# In[357]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[358]:


cat_df.Dependents.value_counts()


# In[359]:


cat_df['Dependents'].fillna("0", inplace=True)
cat_df.Dependents.value_counts()


# In[360]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[361]:


cat_df.Self_Employed.value_counts()


# In[362]:


cat_df['Self_Employed'].fillna("No", inplace=True)


# In[363]:


plt.figure(figsize=(10,6))
sns.heatmap(cat_df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[364]:


plt.figure(figsize=(10,6))
sns.heatmap(num_df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[365]:


from sklearn.impute import SimpleImputer


# In[366]:


imputer = SimpleImputer(strategy='mean')


# In[367]:


imputer.fit(num_df)


# In[368]:


num_df_transform = imputer.transform(num_df)


# In[369]:


num_df = pd.DataFrame(data=num_df_transform)
num_df.columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History']
num_df


# In[370]:


plt.figure(figsize=(10,6))
sns.heatmap(num_df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[ ]:





# In[ ]:





# In[371]:


from sklearn.preprocessing import LabelEncoder


# In[372]:


label_encoder = LabelEncoder()


# In[373]:


def clean_dep(x):
    return x[0]


# In[374]:


cat_df['Property_Area']= label_encoder.fit_transform(cat_df['Property_Area'])
cat_df['Gender']= label_encoder.fit_transform(cat_df['Gender'])
cat_df['Education']= label_encoder.fit_transform(cat_df['Education']) 
cat_df['Self_Employed']= label_encoder.fit_transform(cat_df['Self_Employed'])
cat_df['Married']= label_encoder.fit_transform(cat_df['Married'])
cat_df['Dependents'] = cat_df['Dependents'].apply(clean_dep)
cat_df['Dependents']= label_encoder.fit_transform(cat_df['Dependents'])
cat_df['Loan_Status']= label_encoder.fit_transform(cat_df['Loan_Status'])
cat_df


# In[375]:


cat_df.columns


# In[376]:


cat_df = cat_df.drop('Loan_ID', axis=1)
cat_df


# In[ ]:





# In[377]:


df_transform = pd.concat([cat_df, num_df], axis=1)
df_transform


# In[378]:


plt.figure(figsize=(10,6))
sns.heatmap(df_transform.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[379]:


df_transform.info()


# In[ ]:





# In[ ]:





# In[412]:


df_transform


# In[381]:


X = df_transform.drop('Loan_Status', axis=1)
X


# In[383]:


y = df_transform['Loan_Status']
y


# In[384]:


X.shape, y.shape


# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd


# ### Load the dataset
# https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_prediction.csv

# In[4]:


df = pd.read_csv("loan_prediction.csv")
df


# In[5]:


X = df.drop('Loan_Status', axis=1)
X


# In[6]:


y = df['Loan_Status']
y


# In[7]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


# In[9]:


model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)
model3 = KNeighborsClassifier()


# In[187]:


model1.fit(xtrain, ytrain)
model2.fit(xtrain, ytrain)
model3.fit(xtrain, ytrain)


# In[188]:


pred1=model1.predict_proba(xtest)
pred2=model2.predict_proba(xtest)
pred3=model3.predict_proba(xtest)


# In[190]:


finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
finalpred

