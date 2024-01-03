#!/usr/bin/env python
# coding: utf-8

# # Speech Recognition  - Speech-To-Text
# ## This notebook outlines the techniques involved in Speech to text conversion

# In[ ]:


# ! pip install SpeechRecognition


# In[2]:


import speech_recognition as sr


# In[3]:


recognizer = sr.Recognizer()


# In[4]:


recognizer


# In[32]:


with sr.Microphone() as source:
    print("Please speak up now")
    listening = recognizer.listen(source, timeout=30000)
    print(recognizer.recognize_google(listening))


# In[9]:


import IPython


# In[10]:


IPython.display.display(IPython.display.Audio('Speaker_Diarization_Example.wav'))


# In[25]:


import speech_recognition as sr
recognition = sr.Recognizer()
with sr.AudioFile('Speaker_Diarization_Example.wav') as inputs:
    file_audio = recognition.listen(inputs)
    try:
        convert_text = recognition.recognize_google(file_audio)
        print('Analysing...')
        print(convert_text)
    except sr.UnknownValueError:
         print('Sorry, Could not understand this audio...')


# In[17]:


IPython.display.display(IPython.display.Audio('Speaker_Diarization_Example2.wav'))


# In[26]:


r = sr.Recognizer()
with sr.AudioFile('Speaker_Diarization_Example2.wav') as source:
    #reads the audio file. Here we use record instead of
    #listen
    audio = r.record(source, duration=41) 
    try:
        print("The audio file contains: " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


# In[23]:


IPython.display.display(IPython.display.Audio('political_diarization_cluster_3.wav'))


# In[24]:


r = sr.Recognizer()
with sr.AudioFile('political_diarization_cluster_3.wav') as source:
    #reads the audio file. Here we use record instead of
    #listen
    audio = r.record(source) 
    try:
        print("The audio file contains: " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand this audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

