#!/usr/bin/env python
# coding: utf-8

# # Feature Extraction in Audio
# ## This notebook outlines the concepts behind extracting features from audio

# In[1]:


from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np 
import plotly.graph_objects as go 
import plotly
import IPython
from matplotlib import pyplot as plt


# In[2]:


fs, s = aIO.read_audio_file("count.wav")


# In[3]:


IPython.display.display(IPython.display.Audio("count.wav"))


# In[4]:


duration = len(s) / float(fs)
print(f'duration = {duration} seconds')


# In[5]:


win, step = 0.050, 0.050


# In[6]:


[f, fn] = aF.feature_extraction(s, 
                                fs, 
                                int(fs * win), 
                                int(fs * step)
)


# In[7]:


print(f'{f.shape[1]} frames')


# In[8]:


print(f"{f.shape[0]} short-term features")


# In[9]:


print('Feature names:')
for i, nam in enumerate(fn):
    print(f'{i}:{nam}')


# In[10]:


time = np.arange(0, duration - step, win) 


# In[11]:


time


# In[12]:


energy = f[fn.index('energy'), :]


# In[13]:


energy


# In[14]:


plt.figure(figsize=(10,10))
plt.xlabel("Time (in seconds)")
plt.ylabel("Energy")
plt.plot(time, energy)


# In[15]:


spectral_centroid = f[fn.index('spectral_centroid'), :]


# In[16]:


plt.figure(figsize=(10,10))
plt.xlabel("Time (in seconds)")
plt.ylabel("Spectral Centroid")
plt.plot(time, spectral_centroid)


# In[17]:


import sklearn
import librosa
import librosa.display


# In[18]:


audio_path = 'count.wav'
x , sr = librosa.load(audio_path)


# In[19]:


spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print(spectral_centroids.shape)


# In[20]:


frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)


# In[21]:


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# In[23]:


plt.figure(figsize=(10, 10))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')


# In[24]:


from pyAudioAnalysis import MidTermFeatures as aF


# In[25]:


mt, st, mt_n = aF.mid_feature_extraction(s, 
                                         fs, 
                                         1 * fs, 
                                         1 * fs, 
                                         0.05 * fs, 
                                         0.05 * fs
)


# In[26]:


print(f'signal duration {len(s)/fs} seconds')
print(f'{st.shape[1]} {st.shape[0]}-D short-term feature vectors extracted')
print(f'{mt.shape[1]} {mt.shape[0]}-D segment feature statistic vectors extracted')


# In[27]:


print('mid-term feature names')
for i, mi in enumerate(mt_n):
    print(f'{i}:{mi}')


# In[28]:


import librosa
import librosa.display
from matplotlib import pyplot as plt


# In[29]:


audio_path = 'count.wav'
x , sr = librosa.load(audio_path)


# In[30]:


print(sr)


# In[31]:


X = librosa.stft(x)
X.shape


# In[32]:


Xdb = librosa.amplitude_to_db(abs(X))
Xdb.shape


# In[33]:


plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


# In[34]:


import librosa
import librosa.display


# In[35]:


audio_path = 'count.wav'
x , sr = librosa.load(audio_path)


# In[36]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[37]:


zero_crossings = librosa.zero_crossings(x, pad=False)


# In[38]:


zero_crossings


# In[39]:


sum(zero_crossings)


# In[40]:


spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]


# In[41]:


plt.figure(figsize=(10, 10))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')


# In[42]:


plt.figure(figsize=(10, 10))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.plot(t, normalize(spectral_centroids), color='g')


# In[43]:


mfccs = librosa.feature.mfcc(x, sr=sr)


# In[44]:


print(mfccs.shape)


# In[45]:


plt.figure(figsize=(10, 10))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


# In[46]:


chroma = librosa.feature.chroma_stft(x, sr=sr)


# In[49]:


plt.figure(figsize=(14, 5))
librosa.display.specshow(chroma, sr=sr, x_axis='time')
plt.colorbar()

