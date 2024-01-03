#!/usr/bin/env python
# coding: utf-8

# # Boosting
# ## This notebook outlines the main concepts behind Boosting technique used in Ensembling of Machine Learning models

# ### Algorithm
# 
# - 1. A subset is created from the original dataset.
# - 2. Initially, all data points are given equal weights.
# - 3. A base model is created on this subset.
# - 4. Use this model to make predictions on the whole dataset.
# - 5. Errors are calculated using the actual values and predicted values.
# - 6. The observations which are incorrectly predicted, are given higher weights.
# - 7. Another model is created and predictions are made on the dataset. (This model tries to correct the errors from the previous model)
# - 8. Similarly, multiple models are created, each correcting the errors of the previous model.
# - 9. The final model (strong learner) is the weighted mean of all the models (weak learners).
# 
# Thus, the boosting algorithm combines a number of weak learners to form a strong learner. The individual models would not perform well on the entire dataset, but they work well for some part of the dataset. Thus, each model actually **boosts** the performance of the ensemble, hence the name **Boosting**.
# 
# 
# ![Boosting](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/Boosting.png)

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_prediction.csv")
data        
        


# In[3]:


X = data.drop('Loan_Status', axis=1)
X


# In[4]:


y = data['Loan_Status']
y


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=1
)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[10]:


from sklearn.ensemble import AdaBoostClassifier


# In[11]:


model = AdaBoostClassifier(random_state=1)


# In[12]:


model.fit(X_train, y_train)


# In[13]:


model.score(X_test,y_test)


# In[16]:


from xgboost import XGBClassifier


# In[17]:


model = XGBClassifier(random_state=1,learning_rate=0.01)


# In[18]:


model.fit(X_train, y_train)


# In[19]:


model.score(X_test,y_test)


# ### Light GBM
# 
# Light GBM beats all the other algorithms when the dataset is **extremely large**. Compared to the other algorithms, Light GBM takes **lesser time** to run on a huge dataset.
# 
# LightGBM is a gradient boosting framework that uses tree-based algorithms and follows **leaf-wise** approach while other algorithms work in a level-wise approach pattern.
# 
# ![LightGBM](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/LightGBM_leafs.png)

# In[27]:


import lightgbm as lgb


# In[28]:


train_data=lgb.Dataset(X_train,label=y_train)


# In[29]:


params = {'learning_rate':0.001}


# In[30]:


model= lgb.train(params, train_data, 100)


# In[31]:


predictions = model.predict(X_test)


# In[32]:


predictions


# In[34]:


for i in range(0,len(predictions)):
    if predictions[i]>=0.5: 
        predictions[i]=1
    else:
        predictions[i]=0
        
predictions


# In[35]:


y_test.value_counts()


# In[37]:


predictions_df = pd.DataFrame(predictions)
predictions_df.value_counts()


# In[38]:


from sklearn.ensemble import GradientBoostingClassifier


# In[39]:


model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)


# In[40]:


model.fit(X_train, y_train)


# In[41]:


model.score(X_test,y_test)


# In[ ]:


from catboost import CatBoostClassifier


# In[ ]:


model=CatBoostClassifier()


# In[ ]:


categorical_features_indices = np.where(df.dtypes != np.float)[0]


# In[ ]:


model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))


# In[ ]:


model.score(x_test,y_test)

