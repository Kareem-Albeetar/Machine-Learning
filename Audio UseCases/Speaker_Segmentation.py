#!/usr/bin/env python
# coding: utf-8

# # Speaker Segmentation / Diarization
# ## This notebook outlines the concepts behind segmenting different speakers in the audio file

# In[1]:


import os, sklearn.cluster
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction as mT
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
from pyAudioAnalysis.audioSegmentation import labels_to_segments
from pyAudioAnalysis.audioTrainTest import normalize_features
import numpy as np
import scipy.io.wavfile as wavfile
import IPython


# In[2]:


input_file = "Speaker_Diarization_Example2.wav"
fs, x = read_audio_file(input_file)


# In[3]:


fs


# In[4]:


x.shape


# In[5]:


mt_size, mt_step, st_win = 1, 0.1, 0.05


# In[6]:


[mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
                            round(fs * st_win), round(fs * st_win * 0.5))


# In[7]:


(mt_feats_norm, MEAN, STD) = normalize_features([mt_feats.T])
mt_feats_norm = mt_feats_norm[0].T


# In[8]:


n_clusters = 5
x_clusters = [np.zeros((fs, )) for i in range(n_clusters)]
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(mt_feats_norm.T)
cls = k_means.labels_


# In[9]:


segs, c = labels_to_segments(cls, mt_step)  # convert flags to segment limits
for sp in range(n_clusters):                
    count_cl = 0
    for i in range(len(c)):     # for each segment in each cluster (>2 secs long)
        if c[i] == sp and segs[i, 1]-segs[i, 0] > 2:
            count_cl += 1
            # get the signal and append it to the cluster's signal (followed by some silence)
            cur_x = x[int(segs[i, 0] * fs): int(segs[i, 1] * fs)]
            x_clusters[sp] = np.append(x_clusters[sp], cur_x)
            x_clusters[sp] = np.append(x_clusters[sp], np.zeros((fs,)))
    # write cluster's signal into a WAV file
    print(f'speaker {sp}: {count_cl} segments {len(x_clusters[sp])/float(fs)} sec total dur')        
    wavfile.write(f'diarization_cluster_{sp}.wav', fs, np.int16(x_clusters[sp]))
    IPython.display.display(IPython.display.Audio(f'diarization_cluster_{sp}.wav'))


# In[10]:


from scipy.io import wavfile

fs, data = wavfile.read('Speaker_Diarization_Example.wav')            

wavfile.write('SD_1.wav', fs, data[:, 0])   
wavfile.write('SD_2.wav', fs, data[:, 1])   


# In[11]:


input_file = "SD_1.wav"
fs, x = read_audio_file(input_file)

mt_size, mt_step, st_win = 2, 0.1, 0.05

[mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
                            round(fs * st_win), round(fs * st_win * 0.5))
(mt_feats_norm, MEAN, STD) = normalize_features([mt_feats.T])
mt_feats_norm = mt_feats_norm[0].T

# perform clustering

n_clusters = 4
x_clusters = [np.zeros((fs, )) for i in range(n_clusters)]
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(mt_feats_norm.T)
cls = k_means.labels_

# save clusters to concatenated wav files
segs, c = labels_to_segments(cls, mt_step)  # convert flags to segment limits
for sp in range(n_clusters):                
    count_cl = 0
    for i in range(len(c)):     # for each segment in each cluster (>2 secs long)
        if c[i] == sp and segs[i, 1]-segs[i, 0] > 2:
            count_cl += 1
            # get the signal and append it to the cluster's signal (followed by some silence)
            cur_x = x[int(segs[i, 0] * fs): int(segs[i, 1] * fs)]
            x_clusters[sp] = np.append(x_clusters[sp], cur_x)
            x_clusters[sp] = np.append(x_clusters[sp], np.zeros((fs,)))
    # write cluster's signal into a WAV file
    print(f'political speaker {sp}: {count_cl} segments {len(x_clusters[sp])/float(fs)} sec total dur')        
    wavfile.write(f'political_diarization_cluster_{sp}.wav', fs, np.int16(x_clusters[sp]))
    IPython.display.display(IPython.display.Audio(f'political_diarization_cluster_{sp}.wav'))

