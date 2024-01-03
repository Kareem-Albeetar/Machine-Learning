#!/usr/bin/env python
# coding: utf-8

# # News Modeling
# 
# Topic modeling involves **extracting features from document terms** and using
# mathematical structures and frameworks like matrix factorization and SVD to generate **clusters or groups of terms** that are distinguishable from each other and these clusters of words form topics or concepts
# 
# Topic modeling is a method for **unsupervised classification** of documents, similar to clustering on numeric data
# 
# These concepts can be used to interpret the main **themes** of a corpus and also make **semantic connections among words that co-occur together** frequently in various documents
# 
# Topic modeling can help in the following areas:
# - discovering the **hidden themes** in the collection
# - **classifying** the documents into the discovered themes
# - using the classification to **organize/summarize/search** the documents
# 
# Frameworks and algorithms to build topic models:
# - Latent semantic indexing
# - Latent Dirichlet allocation
# - Non-negative matrix factorization

# ### LDA Algorithm
# 
# - 1. For each document, **randomly initialize each word to one of the K topics** (k is chosen beforehand)
# - 2. For each document D, go through each word w and compute:
#     - **P(T |D)** , which is a proportion of words in D assigned to topic T
#     - **P(W |T )** , which is a proportion of assignments to topic T over all documents having the word W
# - **Reassign word W with topic T** with probability P(T |D)´ P(W |T ) considering all other words and their topic assignments
# 
# ![LDA](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/LDA.png)

# In[1]:


#! pip install pyLDAvis gensim spacy


# In[2]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Download the dataset
# Dataset: https://raw.githubusercontent.com/subashgandyer/datasets/main/newsgroups.json
# 
# #### 20-Newsgroups dataset
# - 11K newsgroups posts
# - 20 news topics

# In[3]:


get_ipython().system(' wget https://raw.githubusercontent.com/subashgandyer/datasets/main/newsgroups.json')


# In[4]:


df = pd.read_json("newsgroups.json")
print(df.target_names.unique())
df.head()


# In[5]:


data = df.content.values.tolist()
data


# In[6]:


data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]


# In[7]:


data


# In[8]:


data = [re.sub('\s+', ' ', sent) for sent in data]
data


# In[9]:


data = [re.sub("\'", "", sent) for sent in data]
data


# In[10]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# In[11]:


data_words = list(sent_to_words(data))

print(data_words[:1])


# In[12]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[13]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# In[14]:


data_words_nostops = remove_stopwords(data_words)


# In[15]:


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)


# In[16]:


bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[17]:


print(trigram_mod[bigram_mod[data_words[0]]])


# In[18]:


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


# In[19]:


data_words_bigrams = make_bigrams(data_words_nostops)


# In[20]:


#! python -m spacy download en


# In[21]:


nlp = spacy.load('en', disable=['parser', 'ner'])


# In[22]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[23]:


data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[24]:


print(data_lemmatized[:1])


# In[25]:


id2word = corpora.Dictionary(data_lemmatized)


# In[26]:


texts = data_lemmatized


# In[27]:


id2word.filter_extremes(no_below=10, no_above=0.5)


# In[28]:


corpus = [id2word.doc2bow(text) for text in texts]


# In[29]:


print(corpus[:1])


# In[30]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[31]:


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[32]:


print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 


# In[33]:


coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[34]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

