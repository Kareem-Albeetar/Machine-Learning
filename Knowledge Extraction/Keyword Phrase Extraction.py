#!/usr/bin/env python
# coding: utf-8

# # Keyword Phrase Extraction
# ## This notebook outlines the concepts involved in extracting keyword phrases in text

# In[1]:


# !pip install textacy==0.9.1
# !python -m spacy download en_core_web_sm


# In[2]:


import spacy
import textacy.ke
from textacy import *


# In[3]:


en = textacy.load_spacy_lang("en_core_web_sm")


# In[6]:


mytext = open('kpe_sample_text.txt').read()


# In[7]:


doc = textacy.make_spacy_doc(mytext, lang=en)


# In[13]:


print([chunk for chunk in textacy.extract.noun_chunks(doc)])


# In[8]:


textacy.ke.textrank(doc, topn=5)


# In[9]:


textacy.ke.textrank(doc, topn=20)


# In[10]:


[kps for kps, weights in textacy.ke.textrank(doc, normalize="lemma", topn=10)]


# In[11]:


[kps for kps, weights in textacy.ke.sgrank(doc, topn=10)]


# In[12]:


terms = set([term for term,weight in textacy.ke.sgrank(doc)])
print(textacy.ke.utils.aggregate_term_variants(terms))

