#!/usr/bin/env python
# coding: utf-8

# # Text To Speech
# ## This notebook outlines the concepts invovled in Converting a piece of text into speech

# In[ ]:


# ! pip install gTTS


# In[2]:


import os
from gtts import gTTS


# In[3]:


text='Hi ,Welcome to GBC AI Program'


# In[4]:


language = 'en'


# In[5]:


speech = gTTS(text = text, lang = language, slow = False)


# In[6]:


speech.save('speech_1.wav')


# In[8]:


import IPython
IPython.display.display(IPython.display.Audio('speech_1.wav'))


# In[13]:


language = 'hi'
text='Namaste , Kaise hai aap ? Aapko hey lecture pasand aayegi'


# In[14]:


speech = gTTS(text = text, lang = language, slow = False)


# In[15]:


speech.save('speech_hindi.wav')


# In[16]:


IPython.display.display(IPython.display.Audio('speech_hindi.wav'))


# In[17]:


file = open("sample_text.txt", "r").read().replace("\n", " ")
language = 'en'
speech = gTTS(text = file, lang = language, slow = False)
speech.save('sample_text_speech.wav')
IPython.display.display(IPython.display.Audio('sample_text_speech.wav'))

