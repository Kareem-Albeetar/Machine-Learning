#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')


# In[2]:


import numpy as np
import random
import string


# In[3]:


import bs4 as bs
import urllib.request
import re


# In[4]:


raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')
raw_html = raw_html.read()


# In[5]:


article_html = bs.BeautifulSoup(raw_html, 'lxml')


# In[6]:


article_paragraphs = article_html.find_all('p')


# In[7]:


article_text = ''

for para in article_paragraphs:
    article_text += para.text
    
article_text


# In[8]:


corpus = nltk.sent_tokenize(article_text)
corpus


# In[9]:


for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])


# In[10]:


wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1


# In[11]:


import heapq
most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)


# In[12]:


word_idf_values = {}
for token in most_freq:
    doc_containing_word = 0
    for document in corpus:
        if token in nltk.word_tokenize(document):
            doc_containing_word += 1
    word_idf_values[token] = np.log(len(corpus)/(1 + doc_containing_word))


# In[13]:


word_tf_values = {}
for token in most_freq:
    sent_tf_vector = []
    for document in corpus:
        doc_freq = 0
        for word in nltk.word_tokenize(document):
            if token == word:
                  doc_freq += 1
        word_tf = doc_freq/len(nltk.word_tokenize(document))
        sent_tf_vector.append(word_tf)
    word_tf_values[token] = sent_tf_vector


# In[14]:


tfidf_values = []
for token in word_tf_values.keys():
    tfidf_sentences = []
    for tf_sentence in word_tf_values[token]:
        tf_idf_score = tf_sentence * word_idf_values[token]
        tfidf_sentences.append(tf_idf_score)
    tfidf_values.append(tfidf_sentences)


# In[15]:


tf_idf_model = np.asarray(tfidf_values)


# In[16]:


tf_idf_model = np.transpose(tf_idf_model)


# In[17]:


tf_idf_model


# In[18]:


import pandas as pd
pd.DataFrame(tf_idf_model, columns=most_freq)

