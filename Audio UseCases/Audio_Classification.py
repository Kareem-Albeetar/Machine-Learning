#!/usr/bin/env python
# coding: utf-8

# # Audio Classification
# ## This notebook outlines the concepts behind the Audio Classification

# Classify audio clips into different **musical genres**
# 
# WAV files are organized in ten folders representing **ten genres**
# - Use just two genres for our classification
# 
# GTZAN Dataset
# http://opihi.cs.uvic.ca/sound/genres.tar.gz
# 
# Source: http://marsyas.info/downloads/datasets.html
# 
# 

# In[1]:


from pyAudioAnalysis import MidTermFeatures as aF
import os
import numpy as np
from sklearn.svm import SVC
import plotly.graph_objs as go 
import plotly


# In[2]:


dirs = ["genres/classical", "genres/metal"] 


# In[3]:


class_names = [os.path.basename(d) for d in dirs] 
class_names


# In[4]:


m_win, m_step, s_win, s_step = 1, 1, 0.1, 0.05 


# In[5]:


features = [] 
for d in dirs: # get feature matrix for each directory (class) 
    f, files, fn = aF.directory_feature_extraction(d, m_win, m_step, s_win, s_step) 
    features.append(f)


# In[6]:


f1 = np.array([features[0][:, fn.index('spectral_centroid_mean')],
               features[0][:, fn.index('energy_entropy_mean')]])
f2 = np.array([features[1][:, fn.index('spectral_centroid_mean')],
               features[1][:, fn.index('energy_entropy_mean')]])


# In[7]:


p1 = go.Scatter(x=f1[0, :],  y=f1[1, :], name=class_names[0],
                marker=dict(size=10,color='rgba(255, 182, 193, .9)'),
                mode='markers')
p2 = go.Scatter(x=f2[0, :], y=f2[1, :],  name=class_names[1], 
                marker=dict(size=10,color='rgba(100, 100, 220, .9)'),
                mode='markers')
mylayout = go.Layout(xaxis=dict(title="spectral_centroid_mean"),
                     yaxis=dict(title="energy_entropy_mean"))
plotly.offline.iplot(go.Figure(data=[p1, p2], layout=mylayout))


# In[8]:


y = np.concatenate((np.zeros(f1.shape[1]), np.ones(f2.shape[1]))) 
f = np.concatenate((f1.T, f2.T), axis = 0)


# In[9]:


cl = SVC(kernel='rbf', C=20) 
cl.fit(f, y)


# In[10]:


x_ = np.arange(f[:, 0].min(), f[:, 0].max(), 0.002) 
y_ = np.arange(f[:, 1].min(), f[:, 1].max(), 0.002) 
xx, yy = np.meshgrid(x_, y_) 
Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) / 2 


# In[11]:


cs = go.Heatmap(x=x_, y=y_, z=Z, showscale=False, 
               colorscale= [[0, 'rgba(255, 182, 193, .3)'], 
                           [1, 'rgba(100, 100, 220, .3)']]) 
mylayout = go.Layout(xaxis=dict(title="spectral_centroid_mean"),
                     yaxis=dict(title="energy_entropy_mean"))
plotly.offline.iplot(go.Figure(data=[p1, p2, cs], layout=mylayout))


# In[12]:


dirs = ["genres/test"]
features = [] 
for d in dirs:
    f, files, fn = aF.directory_feature_extraction(d, m_win, m_step, s_win, s_step) 
    features.append(f)


# In[13]:


f = np.array([features[0][:, fn.index('spectral_centroid_mean')],
               features[0][:, fn.index('energy_entropy_mean')]])


# In[14]:


f.shape


# In[15]:


f = f.T
f.shape


# In[16]:


cl.predict(f)


# In[17]:


from pyAudioAnalysis.audioTrainTest import extract_features_and_train
mt, st = 1.0, 0.05
dirs = ["genres/classical", "genres/metal"] 
extract_features_and_train(dirs, mt, mt, st, st, "svm_rbf", "svm_classical_metal")


# In[20]:


import os
files_to_test = []
for dirname, _, filenames in os.walk('genres/test'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        files_to_test.append(os.path.join(dirname, filename))
        
print(len(files_to_test))


# In[21]:


from pyAudioAnalysis import audioTrainTest as aT
for f in files_to_test:
    print(f'{f}:')
    c, p, p_nam = aT.file_classification(f, "svm_classical_metal","svm_rbf")
    print(f'P({p_nam[0]}={p[0]})')
    print(f'P({p_nam[1]}={p[1]})')
    print()

