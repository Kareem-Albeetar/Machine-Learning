#!/usr/bin/env python
# coding: utf-8

# # Summarization
# ## This notebook outlines the concepts behind Text Summarization

# In[1]:


#! pip install sumy


# In[2]:


from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


# In[3]:


url = "https://en.wikipedia.org/wiki/Automatic_summarization"
parser = HtmlParser.from_url(url, Tokenizer("english"))


# In[4]:


doc = parser.document
doc


# In[5]:


summarizer = TextRankSummarizer()


# In[6]:


summary_text = summarizer(doc, 5)
summary_text


# In[7]:


from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer


# In[8]:


lexSummarizer =  LexRankSummarizer()
luhnSummarizer = LuhnSummarizer()
lsaSummarizer = LsaSummarizer()


# In[9]:


lex_summary_text = lexSummarizer(doc, 5)
lex_summary_text


# In[10]:


luhn_summary_text = luhnSummarizer(doc, 5)
luhn_summary_text


# In[11]:


lsa_summary_text = lsaSummarizer(doc, 5)
lsa_summary_text


# In[12]:


#!pip install gensim


# In[13]:


from gensim.summarization import summarize


# In[14]:


import requests
import re
from bs4 import BeautifulSoup


# In[15]:


def get_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup


# In[16]:


def collect_text(soup):
    text = f'url: {url}\n\n'
    para_text = soup.find_all('p')
    print(f"paragraphs text = \n {para_text}")
    for para in para_text:
        text += f"{para.text}\n\n"
    return text


# In[17]:


url = "https://en.wikipedia.org/wiki/Automatic_summarization"


# In[18]:


text = collect_text(get_page(url))
text


# In[19]:


gensim_summary_text = summarize(text, word_count=200, ratio = 0.1)
gensim_summary_text


# In[20]:


# !pip install summa


# In[21]:


from summa import summarizer
from summa import keywords


# In[22]:


summa_summary_text = summarizer.summarize(text, ratio=0.1)
summa_summary_text

