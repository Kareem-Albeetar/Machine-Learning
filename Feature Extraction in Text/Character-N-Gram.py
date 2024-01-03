#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk  
import numpy as np  
import random  
import string

import bs4 as bs  
import urllib.request  
import re  


# In[2]:


raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')  
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:  
    article_text += para.text


# In[3]:


raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')
raw_html


# In[4]:


raw_html = raw_html.read()
raw_html


# In[5]:


article_html = bs.BeautifulSoup(raw_html, 'lxml')


# In[6]:


article_html


# In[8]:


links = article_html.findAll('a', {'href': True})
links


# In[9]:


for link in links:
    print(link['href'])
    print(link.string)


# In[10]:


article_text


# ### Step 4: N-Grams = 3

# In[11]:


ngrams = {}
chars = 3

for i in range(len(article_text)-chars):
    seq = article_text[i:i+chars]
#     print(seq)
    if seq not in ngrams.keys():
        ngrams[seq] = []
    ngrams[seq].append(article_text[i+chars])
    
ngrams


# In[12]:


curr_sequence = article_text[0:chars]
output = curr_sequence
for i in range(200):
    if curr_sequence not in ngrams.keys():
        break
    possible_chars = ngrams[curr_sequence]
    next_char = possible_chars[random.randrange(len(possible_chars))]
    output += next_char
    curr_sequence = output[len(output)-chars:len(output)]

print(output)


# In[13]:


ngrams = {}
chars = 5

for i in range(len(article_text)-chars):
    seq = article_text[i:i+chars]
    print(seq)
    if seq not in ngrams.keys():
        ngrams[seq] = []
    ngrams[seq].append(article_text[i+chars])
    
ngrams


# In[14]:


curr_sequence = article_text[0:chars]
# print(curr_sequence)
output = curr_sequence
# print(output)
for i in range(20000):
    if curr_sequence not in ngrams.keys():
        break
    possible_chars = ngrams[curr_sequence]
#     print(possible_chars)
    next_char = possible_chars[random.randrange(len(possible_chars))]
#     print(next_char)
    output += next_char
#     print(output)
    curr_sequence = output[len(output)-chars:len(output)]
#     print(curr_sequence)

print(output)


# In[15]:


article_text

