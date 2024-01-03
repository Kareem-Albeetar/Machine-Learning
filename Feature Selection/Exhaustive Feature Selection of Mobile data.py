#!/usr/bin/env python
# coding: utf-8

# # Feature Selection from Mobile data using ExhaustiveFeatureSelector Exhaustive Method
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


# In[11]:


y


# In[12]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


lr = LogisticRegression(class_weight='balanced',
                        solver='lbfgs', 
                        random_state=42, 
                        n_jobs=-1, 
                        max_iter=500
)


# In[15]:


efs_lr = ExhaustiveFeatureSelector(lr, 
                                min_features=1, 
                                max_features=2,
                                scoring='accuracy',
                                print_progress=True,
                                cv=2
)


# In[16]:


efs_lr = efs_lr.fit(X, y)


# In[17]:


efs_lr.best_feature_names_


# In[18]:


efs_lr.best_score_


# In[19]:


efs_lr.subsets_


# In[ ]:





# In[20]:


metric_dict = efs_lr.get_metric_dict()
metric_dict


# ### Plotting metric_dict

# In[21]:


import matplotlib.pyplot as plt


fig = plt.figure(figsize=(50,50))

k_feat = sorted(metric_dict.keys())
avg = [metric_dict[k]['avg_score'] for k in k_feat]

upper, lower = [], []
for k in k_feat:
    upper.append(metric_dict[k]['avg_score'] +
                 metric_dict[k]['std_dev'])
    lower.append(metric_dict[k]['avg_score'] -
                 metric_dict[k]['std_dev'])

plt.fill_between(k_feat,
                 upper,
                 lower,
                 alpha=0.2,
                 color='blue',
                 lw=1)

plt.plot(k_feat, avg, color='blue', marker='o')
plt.ylabel('Accuracy +/- Standard Deviation')
plt.xlabel('Number of Features')
feature_min = len(metric_dict[k_feat[0]]['feature_idx'])
feature_max = len(metric_dict[k_feat[-1]]['feature_idx'])
plt.xticks(k_feat, 
           [str(metric_dict[k]['feature_names']) for k in k_feat], 
           rotation=90)
plt.show()


# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


rf = RandomForestClassifier()


# In[24]:


efs_rf = ExhaustiveFeatureSelector(rf, 
                                min_features=1, 
                                max_features=2, 
                                scoring='accuracy', 
                                print_progress=True, 
                                cv=2
)


# In[25]:


efs_rf = efs_rf.fit(X, y)


# In[26]:


efs_rf.best_feature_names_


# In[27]:


efs_rf.best_score_


# In[28]:


metric_dict = efs_lr.get_metric_dict()
metric_dict


# In[29]:


import matplotlib.pyplot as plt


fig = plt.figure(figsize=(50,50))

k_feat = sorted(metric_dict.keys())
avg = [metric_dict[k]['avg_score'] for k in k_feat]

upper, lower = [], []
for k in k_feat:
    upper.append(metric_dict[k]['avg_score'] +
                 metric_dict[k]['std_dev'])
    lower.append(metric_dict[k]['avg_score'] -
                 metric_dict[k]['std_dev'])

plt.fill_between(k_feat,
                 upper,
                 lower,
                 alpha=0.2,
                 color='blue',
                 lw=1)

plt.plot(k_feat, avg, color='blue', marker='o')
plt.ylabel('Accuracy +/- Standard Deviation')
plt.xlabel('Number of Features')
feature_min = len(metric_dict[k_feat[0]]['feature_idx'])
feature_max = len(metric_dict[k_feat[-1]]['feature_idx'])
plt.xticks(k_feat, 
           [str(metric_dict[k]['feature_names']) for k in k_feat], 
           rotation=90)
plt.show()


# In[30]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.neighbors import KNeighborsClassifier


# In[31]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[32]:


efs_knn = EFS(knn, 
           min_features=1,
           max_features=2,
           scoring='accuracy',
           print_progress=True,
           cv=5
)


# In[33]:


efs_knn.fit(X, y)


# In[34]:


efs_knn.best_feature_names_


# In[36]:


efs_knn.best_score_

