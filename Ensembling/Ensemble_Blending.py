#!/usr/bin/env python
# coding: utf-8

# # Blending
# ## This notebook outlines the Blending technique used in Ensembling of Machine Learning models.

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


# In[2]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv")


# In[4]:


df


# In[5]:


df.info()


# In[347]:


df.columns


# In[6]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ### Split them into categorical and numerical features

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





# In[25]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# ### Load the dataset
# Dataset: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_prediction.csv

# In[26]:


df = pd.read_csv("loan_prediction.csv")
df


# In[27]:


X = df.drop('Loan_Status', axis=1)
X


# In[28]:


y = df['Loan_Status']
y


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.5, 
                                                    random_state=1
)


# In[31]:


x_tn, x_val, y_tn, y_val = train_test_split(X_train, 
                                            y_train, 
                                            test_size=0.33, 
                                            random_state=1
)


# In[32]:


lr = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
svm = SVC()
bayes = GaussianNB()
rf = RandomForestClassifier(100)

models = {'lr': lr,
          'knn': knn,
          'dt': dt,
          'svm': svm,
          'bayes': bayes,
          'rf': rf
         }


# In[33]:


base_model_train_predictions = []
for algo, model in models.items():
    # Fit on the training dataset
    model.fit(x_tn, y_tn)
    # Predict on the hold-out dataset
    yhat = model.predict(x_val)
    # Store predictions for meta-model's use
    yhat = yhat.reshape(len(yhat), 1)
    base_model_train_predictions.append(yhat)


# In[34]:


base_model_train_predictions = np.hstack(base_model_train_predictions)


# In[35]:


blender = LogisticRegression()


# In[36]:


blender.fit(base_model_train_predictions, y_val)


# In[37]:


base_model_infer_predictions = []
for algo, model in models.items():
    # Predict on the test dataset
    yhat = model.predict(X_test)
    # Store predictions for meta-model's use
    yhat = yhat.reshape(len(yhat), 1)
    base_model_infer_predictions.append(yhat)


# In[39]:


base_model_infer_predictions = np.hstack(base_model_infer_predictions)


# In[40]:


blender_predictions = blender.predict(base_model_infer_predictions)


# In[41]:


score = accuracy_score(y_test, blender_predictions)
score


# In[43]:


for name, model in models.items():
    # fit the model on the training dataset
    model.fit(X_train, y_train)
    # make a prediction on the test dataset
    yhat = model.predict(X_test)
    # evaluate the predictions
    score = accuracy_score(y_test, yhat)
    # report the score
    print('>%s Accuracy: %.3f' % (name, score*100))


# In[ ]:




