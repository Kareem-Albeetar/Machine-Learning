#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling
# ## This notebook outlines the concepts involved in Topic Modeling
# 
# 
# 

# Dataset: 
# https://raw.githubusercontent.com/subashgandyer/datasets/main/kaggledatasets.csv

# In[1]:


# ! pip install gensim


# In[2]:


import nltk
get_ipython().system(" nltk.download('stopwords')")


# In[3]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from pprint import pprint
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import gensim


# In[4]:


get_ipython().system(' wget https://raw.githubusercontent.com/subashgandyer/datasets/main/kaggledatasets.csv')


# In[5]:


df = pd.read_csv("kaggledatasets.csv")
df.head()


# In[6]:


for i in df['Description'].iteritems():
    raw = str(i[1]).lower()
    print(raw)


# In[7]:


pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
en_stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[8]:


texts = []


for i in df['Description'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in en_stop]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens]
    
    # remove word containing only single char
    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]
    
    # add tokens to list
    texts.append(new_lemma_tokens)


print(texts[0])


# In[9]:


dictionary = Dictionary(texts)


# In[10]:


dictionary.filter_extremes(no_below=10, no_above=0.5)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


# In[11]:


temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token


# In[12]:


ldamodel = LdaModel(corpus, num_topics=15, id2word = id2word, passes=20)


# In[13]:


pprint(ldamodel.top_topics(corpus,topn=5))


# In[14]:


for idx in range(15):
    print("Topic #%s:" % idx, ldamodel.print_topic(idx, 10))


# In[15]:


from gensim.models import LsiModel
lsamodel = LsiModel(corpus, num_topics=10, id2word = id2word)
pprint(lsamodel.print_topics(num_topics=10, num_words=10))


# In[16]:


for idx in range(10):
    print("Topic #%s:" % idx, lsamodel.print_topic(idx, 10))
print("=" * 20)


# In[17]:


import pyLDAvis.gensim


# In[18]:


pyLDAvis.enable_notebook()


# In[19]:


pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

