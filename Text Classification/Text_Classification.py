#!/usr/bin/env python
# coding: utf-8

# # Text Classification
# ## This notebook outlines the usage of NLP Feature extraction (CountVectorizer, TfidfVectorizer) in classification of text documents

# In[1]:


from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# In[2]:


# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
]


# In[3]:


print("Loading 20 newsgroups dataset for categories:")
print(categories)


# In[4]:


data = fetch_20newsgroups(subset='train', categories=categories)
print(f"{len(data.filenames)} documents")
print(f"{len(data.target_names)} categories")
print()


# In[5]:


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(tol=1e-3)),
])


# In[6]:


parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (20,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}


# In[7]:


grid_search = GridSearchCV(pipeline, parameters, cv=5,
                           n_jobs=-1, verbose=1)


# In[8]:


grid_search.fit(data.data, data.target)


# In[9]:


print("Best score: %0.3f" % grid_search.best_score_)


# In[10]:


print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

