#!/usr/bin/env python
# coding: utf-8

# # Recommendation System - Content Based Filtering
# ## This notebook outlines the concepts involved in Content Based Filtering Recommendation System

# Dataset: 
# - https://raw.githubusercontent.com/subashgandyer/datasets/main/restaurant_data/restaurants.csv

# In[1]:


import numpy as np 
import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('restaurants.csv', encoding ='latin1')


# In[3]:


data['City'].value_counts(dropna = False)


# In[4]:


data_city = data.loc[data['City'] == 'New Delhi']


# In[5]:


data_new_delphi=data_city[['Restaurant Name','Cuisines','Locality','Aggregate rating']]


# In[6]:


data_new_delphi['Locality'].value_counts(dropna = False).head(5)


# In[7]:


data_new_delphi.loc[data['Locality'] == 'Connaught Place']


# In[8]:


data_sample = []


# In[9]:


location = "Connaught Place"
data_sample = data_new_delphi.loc[data_new_delphi['Locality'] == location]


# In[10]:


data_sample.reset_index(level=0, inplace=True) 


# In[11]:


data_sample['Split']="X"
for i in range(0,data_sample.index[-1]):
    split_data=re.split(r'[,]', data_sample['Cuisines'][i])
    for k,l in enumerate(split_data):
        split_data[k]=(split_data[k].replace(" ", ""))
    split_data=' '.join(split_data[:])
    data_sample['Split'].iloc[i]=split_data


# In[19]:


#Extracting Stopword
tfidf = TfidfVectorizer(stop_words='english')
#Replace NaN for empty string
data_sample['Split'] = data_sample['Split'].fillna('')
#Applying TF-IDF Vectorizer
tfidf_matrix = tfidf.fit_transform(data_sample['Split'])


# In[20]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) 


# In[21]:


# Column names are using for index
corpus_index=[n for n in data_sample['Split']]

#Construct a reverse map of indices    
indices = pd.Series(data_sample.index, index=data_sample['Restaurant Name']).drop_duplicates() 

# Collect index of the restaurant matches the cuisines of the title (restaurant)
title="Pizza Hut"
idx = indices[title]
#Aggregate rating added with cosine score in sim_score list.
sim_scores=[]
for i,j in enumerate(cosine_sim[idx]):
    k=data_sample['Aggregate rating'].iloc[i]
    if j != 0 :
        sim_scores.append((i,j,k))


# In[22]:


sim_scores = sorted(sim_scores, key=lambda x: (x[1],x[2]) , reverse=True)
# 10 similar cuisines
sim_scores = sim_scores[0:10]
rest_indices = [i[0] for i in sim_scores] 


# In[23]:


data_x =data_sample[['Restaurant Name','Aggregate rating']].iloc[rest_indices]
    
data_x['Cosine Similarity']=0
for i,j in enumerate(sim_scores):
    data_x['Cosine Similarity'].iloc[i]=round(sim_scores[i][1],2)
    
data_x


# In[24]:


data_sample=[]
def restaurant_recommender(location,title):   
    global data_sample       
    global cosine_sim
    global sim_scores
    global tfidf_matrix
    global corpus_index
    global feature
    global rest_indices
    global idx
    
    # When location comes from function ,our new data consist only location dataset
    data_sample = data_new_delphi.loc[data_new_delphi['Locality'] == location]  
    
    # index will be reset for cosine similarty index because Cosine similarty index has to be same value with result of tf-idf vectorize
    data_sample.reset_index(level=0, inplace=True) 
    
    #Feature Extraction
    data_sample['Split']="X"
    for i in range(0,data_sample.index[-1]):
        split_data=re.split(r'[,]', data_sample['Cuisines'][i])
        for k,l in enumerate(split_data):
            split_data[k]=(split_data[k].replace(" ", ""))
        split_data=' '.join(split_data[:])
        data_sample['Split'].iloc[i]=split_data
        
    #TF-IDF vectorizer
    #Extracting Stopword
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN for empty string
    data_sample['Split'] = data_sample['Split'].fillna('')
    #Applying TF-IDF Vectorizer
    tfidf_matrix = tfidf.fit_transform(data_sample['Split'])
    
    feature= tfidf.get_feature_names()
    
    #Cosine Similarity
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) 
    
    # Column names are using for index
    corpus_index=[n for n in data_sample['Split']]
       
    #Construct a reverse map of indices    
    indices = pd.Series(data_sample.index, index=data_sample['Restaurant Name']).drop_duplicates() 
    
    #index of the restaurant matchs the cuisines
    idx = indices[title]
    #Aggregate rating added with cosine score in sim_score list.
    sim_scores=[]
    for i,j in enumerate(cosine_sim[idx]):
        k=data_sample['Aggregate rating'].iloc[i]
        if j != 0 :
            sim_scores.append((i,j,k))
            
    #Sort the restaurant names based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: (x[1],x[2]) , reverse=True)
    # 10 similar cuisines
    sim_scores = sim_scores[0:10]
    rest_indices = [i[0] for i in sim_scores] 
  
    data_x =data_sample[['Restaurant Name','Aggregate rating']].iloc[rest_indices]
    
    data_x['Cosine Similarity']=0
    for i,j in enumerate(sim_scores):
        data_x['Cosine Similarity'].iloc[i]=round(sim_scores[i][1],2)
   
    return data_x


# In[25]:


restaurant_recommender('Connaught Place','Pizza Hut')


# In[26]:


restaurant_recommender('Connaught Place','Barbeque Nation')

