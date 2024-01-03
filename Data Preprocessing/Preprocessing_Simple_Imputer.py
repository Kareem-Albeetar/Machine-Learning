#!/usr/bin/env python
# coding: utf-8

# ## SimpleImputer
# ### This notebook outlines the usage of Simple Imputer (Univariate Imputation).
# ### Simple Imputer substitutes missing values statistics (mean, median, ...)
# #### Dataset: [https://github.com/subashgandyer/datasets/blob/main/heart_disease.csv]

# **Demographic**
# - Sex: male or female(Nominal)
# - Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
# 
# **Behavioral**
# - Current Smoker: whether or not the patient is a current smoker (Nominal)
# - Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)
# 
# **Medical(history)**
# - BP Meds: whether or not the patient was on blood pressure medication (Nominal)
# - Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
# - Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
# - Diabetes: whether or not the patient had diabetes (Nominal)
# 
# **Medical(current)**
# - Tot Chol: total cholesterol level (Continuous)
# - Sys BP: systolic blood pressure (Continuous)
# - Dia BP: diastolic blood pressure (Continuous)
# - BMI: Body Mass Index (Continuous)
# - Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
# - Glucose: glucose level (Continuous)
# 
# **Predict variable (desired target)**
# - 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("data/heart_disease.csv")
df


# In[3]:


df.info()


# In[4]:


for i in range(len(df.columns)):
    missing_data = df[df.columns[i]].isna().sum()
    perc = missing_data / len(df) * 100
    print(f'Feature {i+1} >> Missing entries: {missing_data}  |  Percentage: {round(perc, 2)}')


# In[5]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[6]:


from sklearn.impute import SimpleImputer


# In[7]:


imputer = SimpleImputer(strategy='mean')


# In[8]:


data = df.values


# In[9]:


X = data[:, :-1]
y = data[:, -1]


# In[10]:


imputer.fit(X)


# In[11]:


X_transform = imputer.transform(X)


# In[12]:


print(f"Missing cells: {sum(np.isnan(X).flatten())}")


# In[13]:


print(f"Missing cells: {sum(np.isnan(X_transform).flatten())}")


# In[14]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[15]:


plt.figure(figsize=(10,6))
sns.heatmap(X_transform.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[16]:


df_transform = pd.DataFrame(data=X_transform)
df_transform


# In[17]:


plt.figure(figsize=(10,6))
sns.heatmap(df_transform.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[18]:


X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")
Y_test = pd.read_csv("Y_test.csv")
X_test = pd.read_csv("X_test.csv")


# In[19]:


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[20]:


plt.figure(figsize=(10,6))
sns.heatmap(X_train.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[21]:


df=pd.read_csv("data/heart_disease.csv")
X = df[df.columns[:-1]]
y = df[df.columns[-1]]


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[23]:


model = LogisticRegression()


# In[24]:


model.fit(X,y)


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# In[26]:


df=pd.read_csv("data/heart_disease.csv")
df


# In[27]:


df.shape


# In[28]:


df = df.dropna()
df.shape


# In[29]:


X = df[df.columns[:-1]]
X.shape


# In[30]:


y = df[df.columns[-1]]
y.shape


# In[31]:


pipeline = Pipeline([('model', model)])


# In[32]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[33]:


scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[34]:


scores


# In[35]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# In[37]:


df=pd.read_csv("data/heart_disease.csv")
df


# In[38]:


df.shape


# In[39]:


X = df[df.columns[:-1]]
X.shape


# In[40]:


y = df[df.columns[-1]]
y


# In[41]:


imputer = SimpleImputer(strategy='mean')


# In[42]:


model = LogisticRegression()


# In[43]:


pipeline = Pipeline([('impute', imputer), ('model', model)])


# In[44]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[45]:


scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[46]:


scores


# In[47]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# In[49]:


imputer = SimpleImputer(strategy='mean')


# In[50]:


model = RandomForestClassifier()


# In[51]:


pipeline = Pipeline([('impute', imputer), ('model', model)])


# In[52]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[53]:


scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[54]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# In[55]:


results =[]

strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model', model)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    results.append(scores)
    
for method, accuracy in zip(strategies, results):
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}   |   Max accuracy: {round(np.max(accuracy), 3)}")
          
          

