#!/usr/bin/env python
# coding: utf-8

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





# ### Split them into X and y

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


# In[15]:


import pandas as pd


# In[16]:


df = pd.read_csv("loan_prediction.csv")
df


# In[17]:


X = df.drop('Loan_Status', axis=1)
X


# In[18]:


y = df['Loan_Status']
y


# In[19]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[21]:


from sklearn.ensemble import StackingClassifier


# In[22]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# In[40]:


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


# In[41]:


stacking_model = StackingClassifier(estimators = [('lr', lr),
                                         ('knn', knn),
                                         ('dt', dt),
                                         ('svm', svm),
                                         ('bayes', bayes),
                                         ('rf', rf)],
                                        final_estimator = lr,
                                        cv=5
                          )


# In[42]:


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[43]:


results = algorithms = []
for algo, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    algorithms.append(algo)
    print(f"Algorithm {algo}'s Accuracy >>> {np.mean(scores)} & Standard Deviation >>> {np.std(scores)}")


# In[34]:


algo = 'Stacking'
scores = evaluate_model(stacking_model, X, y)
print(f"Algorithm {algo}'s Accuracy >>> {np.mean(scores)} & Standard Deviation >>> {np.std(scores)}")


# In[44]:


algo = 'Stacking_with_rf'
scores = evaluate_model(stacking_model, X, y)
print(f"Algorithm {algo}'s Accuracy >>> {np.mean(scores)} & Standard Deviation >>> {np.std(scores)}")


# In[45]:


stacking_model.fit(X, y)


# In[48]:


sample1 = X.iloc[613, :].values
sample1


# In[49]:


y


# In[50]:


y1 = y.iloc[613]
y1


# In[52]:


y.value_counts()


# In[58]:


yhat = stacking_model.predict(sample1.reshape(1,-1))
print('Predicted Class: %d' % (yhat))


# In[54]:


sample2 = X.iloc[1, :].values
sample2


# In[55]:


y2 = y.iloc[1]
y2


# In[59]:


yhat = stacking_model.predict(sample2.reshape(1,-1))
print('Predicted Class: %d' % (yhat))


# In[ ]:




