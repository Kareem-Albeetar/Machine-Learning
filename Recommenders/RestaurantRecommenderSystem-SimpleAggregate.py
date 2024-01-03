#!/usr/bin/env python
# coding: utf-8

# # Recommendation System
# ## This notebook outlines the concepts involved in building a simple aggregation based recommendation system

# Dataset: 
# - https://raw.githubusercontent.com/subashgandyer/datasets/main/restaurant_data/restaurants.csv
# - https://raw.githubusercontent.com/subashgandyer/datasets/main/restaurant_data/Country-Code.csv

# In[1]:


get_ipython().system(' pip install plotly')


# In[2]:


import numpy as np 
import pandas as pd 
import re
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# ! wget https://raw.githubusercontent.com/subashgandyer/datasets/main/restaurant_data/restaurants.csv


# In[4]:


#! wget https://raw.githubusercontent.com/subashgandyer/datasets/main/restaurant_data/Country-Code.csv


# In[6]:


data = pd.read_csv('restaurants.csv', encoding ='latin1')
country = pd.read_csv("Country-Code.csv")


# In[7]:


data1 = pd.merge(data, country, on='Country Code')


# In[8]:


labels = list(data1.Country.value_counts().index)
values = list(data1.Country.value_counts().values)
fig = {
    "data":[
        {
            "labels" : labels,
            "values" : values,
            "hoverinfo" : 'label+percent',
            "domain": {"x": [0, .9]},
            "hole" : 0.6,
            "type" : "pie",
            "rotation":120,
        },
    ],
    "layout": {
        "title" : "Zomato's Presence around the World",
        "annotations": [
            {
                "font": {"size":20},
                "showarrow": True,
                "text": "Countries",
                "x":0.2,
                "y":0.9,
            },
        ]
    }
}
iplot(fig)


# In[9]:


res_India = data1[data1.Country == 'India']
labels1 = list(res_India.City.value_counts().index)
values1 = list(res_India.City.value_counts().values)
labels1 = labels1[:10]
values1 = values1[:10]
fig = {
    "data":[
        {
            "labels" : labels1,
            "values" : values1,
            "hoverinfo" : 'label+percent',
            "domain": {"x": [0, .8]},
            "hole" : 0.6,
            "type" : "pie",
            "rotation":120,
        },
    ],
    "layout": {
        "title" : "",
        "annotations": [
            {
                "font": {"size":20},
                "showarrow": True,
                "text": "Cities",
                "x":0.2,
                "y":0.9,
            },
        ]
    }
}
iplot(fig)


# In[10]:


NCR = ['New Delhi','Gurgaon','Noida','Faridabad']
res_NCR = res_India[(res_India.City == NCR[0])|(res_India.City == NCR[1])|(res_India.City == NCR[2])|
                    (res_India.City == NCR[3])]
agg_rat = res_NCR[res_NCR['Aggregate rating'] > 0]
f, ax = plt.subplots(1,1, figsize = (14, 4))
ax = sns.countplot(agg_rat['Aggregate rating'])
plt.show()


# In[11]:


res_India['Cuisines'].value_counts().sort_values(ascending=False).head(10)
res_India['Cuisines'].value_counts().sort_values(ascending=False).head(10).plot(kind='pie',figsize=(10,6), 
title="Most Popular Cuisines", autopct='%1.2f%%')
plt.axis('equal')


# In[12]:


res_India = data1[data1.Country == 'India']
NCR = ['New Delhi','Gurgaon','Noida','Faridabad']
res_NCR = res_India[(res_India.City == NCR[0])|(res_India.City == NCR[1])|(res_India.City == NCR[2])|
                    (res_India.City == NCR[3])]


# In[13]:


data_new_delphi=res_NCR[['Restaurant Name','Cuisines','Locality','Aggregate rating', 'Votes']]
C = data_new_delphi['Aggregate rating'].mean()
print(C)


# In[14]:


m = data_new_delphi['Votes'].quantile(0.90)
print(m)


# In[15]:


q_restaurant = data_new_delphi.copy().loc[data_new_delphi['Votes'] >= m]
q_restaurant.shape


# In[16]:


def weighted_rating(x, m=m, C=C):
    v = x['Votes']
    R = x['Aggregate rating']
    # Calculating the score
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[17]:


q_restaurant['score'] = q_restaurant.apply(weighted_rating, axis=1)


# In[18]:


q_restaurant = q_restaurant.sort_values('score', ascending=False)


# In[19]:


q_restaurant[['Restaurant Name','Cuisines', 'Locality','Votes', 'Aggregate rating', 'score']].head(10)

