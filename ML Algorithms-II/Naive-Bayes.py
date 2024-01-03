#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes
# ## This  outlines the usage  of Naive Bayes Classification Machine learning algorithm





# 
# # Tennis_dataset :
# https://raw.githubusercontent.com/subashgandyer/datasets/main/PlayTennis.csv

# In[52]:


import pandas as pd
import numpy as np


# In[14]:


url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/PlayTennis.csv"


# In[16]:


play_tennis = pd.read_csv(url)
play_tennis


# In[23]:


len(play_tennis.columns) - 1


# In[28]:


play_tennis.shape[1]


# In[24]:


play_tennis.info()


# In[25]:


play_tennis.info()


# In[26]:


play_tennis.shape[0]


# In[29]:


df = play_tennis[['Temperature', 'Play Tennis']]
df


# ![(Play Tennis Template)](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/PlayTennis_template.png)

# In[30]:


from sklearn.preprocessing import LabelEncoder


# In[31]:


number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])


# In[32]:


features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"


# In[33]:


from sklearn.model_selection import train_test_split


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(play_tennis[features], 
                                                                            play_tennis[target],
                                                                            test_size = 0.33,
                                                                            random_state = 54
)


# In[37]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[38]:


from sklearn.naive_bayes import GaussianNB


# In[39]:


model = GaussianNB()


# In[40]:


model.fit(x_train, y_train)


# In[41]:


y_pred = model.predict(x_test)


# In[42]:


from sklearn.metrics import accuracy_score


# In[44]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[53]:


sample = np.array([1, 2, 0, 1])


# In[57]:


model.predict(sample.reshape(1,-1))


# ### Newsgroup Built-in dataset

# In[1]:


from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names


# In[2]:


train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')


# In[3]:


print(train.data[5])


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.pipeline import make_pipeline


# In[ ]:


model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[5]:


model.fit(train.data, train.target)


# In[ ]:


labels = model.predict(test.data)


# In[8]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(10,10))
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[9]:


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# In[10]:


predict_category('sending a payload to the ISS')


# In[11]:


predict_category('discussing islam vs atheism')


# In[12]:


predict_category('determining the screen resolution')

