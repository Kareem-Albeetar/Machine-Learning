#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


corpus = [
          'Text of first document.',
          'Text of the second document made longer.',
          'Number three.',
          'This is number four.',
]


# In[3]:


# create CountVectorizer object
vectorizer = CountVectorizer()
vectorizer


# In[4]:


# learn the vocabulary and store CountVectorizer sparse matrix in X
X = vectorizer.fit_transform(corpus)
X


# In[5]:


# columns of X correspond to the result of this method
vectorizer.get_feature_names()


# In[6]:


# retrieving the matrix in the numpy form
X.toarray()


# In[7]:


# transforming a new document according to learn vocabulary
vectorizer.transform(['A new document.']).toarray()

