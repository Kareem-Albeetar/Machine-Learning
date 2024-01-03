#!/usr/bin/env python
# coding: utf-8

# #  Covid-19 Data Analysis
# 

# ### Data Source: 
# https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports
# 
# 
# 
# 

# 

# In[1]:


import pandas as pd
covid_data= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-10-2021.csv')
covid_data


# In[2]:


covid_data.head()


# In[3]:


covid_data.info()


# In[4]:


covid_data.isna().sum()


# In[5]:


result = covid_data.groupby('Country_Region')['Confirmed'].sum().reset_index()
result


# In[6]:


result = covid_data.groupby('Country_Region')['Deaths'].sum().reset_index()
result


# In[7]:


result = covid_data.groupby('Country_Region')['Recovered'].sum().reset_index()
result


# In[8]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
result = covid_data.groupby('Country_Region')['Active'].sum().reset_index()
result


# In[9]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
result = covid_data.groupby('Country_Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
result 


# In[10]:


data = covid_data.groupby('Country_Region')['Recovered'].sum().reset_index()
result = data[data['Recovered']==0][['Country_Region', 'Recovered']]
result


# In[11]:


data = covid_data.groupby('Country_Region')['Confirmed'].sum().reset_index()
result = data[data['Confirmed']==0][['Country_Region', 'Confirmed']]
result


# In[12]:


data = covid_data.groupby('Country_Region')['Deaths'].sum().reset_index()
result = data[data['Deaths']==0][['Country_Region', 'Deaths']]
result


# In[13]:


data = covid_data.groupby('Country_Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
result = data[data['Deaths']==0][['Country_Region', 'Confirmed', 'Deaths', 'Recovered']]
result


# In[14]:


covid_data= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-09-2021.csv', usecols = ['Last_Update', 'Country_Region', 'Confirmed', 'Deaths', 'Recovered'])
result = covid_data.groupby('Country_Region').max().sort_values(by='Confirmed', ascending=False)[:10]
pd.set_option('display.max_column', None)
result


# In[15]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
result = covid_data.groupby('Country_Region').max().sort_values(by='Active', ascending=False)[:10]
pd.set_option('display.max_column', None)
result 


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
 
r_data = covid_data.groupby(["Country_Region"])["Deaths", "Confirmed", "Recovered", "Active"].sum().reset_index()
r_data = r_data.sort_values(by='Deaths', ascending=False)
r_data = r_data[r_data['Deaths']>50000]
plt.figure(figsize=(15, 5))
plt.plot(r_data['Country_Region'], r_data['Deaths'],color='red')
plt.plot(r_data['Country_Region'], r_data['Confirmed'],color='green')
plt.plot(r_data['Country_Region'], r_data['Recovered'], color='blue')
plt.plot(r_data['Country_Region'], r_data['Active'], color='black')
 
plt.title('Total Deaths(>50,000), Confirmed, Recovered and Active Cases by Country')
plt.show()


# In[18]:


import plotly.express as px


# In[19]:


covid_data= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-09-2021.csv')


# In[20]:


covid_data.columns


# In[21]:


us_data = covid_data[covid_data['Country_Region']=='US'].drop(['Country_Region'], axis=1)
us_data = us_data[us_data.sum(axis = 1) > 0]
us_data = us_data.groupby(['Province_State'])['Deaths'].sum().reset_index()
us_data_death = us_data[us_data['Deaths'] > 0]
state_fig = px.bar(us_data_death, x='Province_State', y='Deaths', title='State wise deaths reported of COVID-19 in USA', text='Deaths')
state_fig.show()


# In[22]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
us_data = covid_data[covid_data['Country_Region']=='US'].drop(['Country_Region'], axis=1)
us_data = us_data[us_data.sum(axis = 1) > 0]
 
us_data = us_data.groupby(['Province_State'])['Active'].sum().reset_index()
us_data_death = us_data[us_data['Active'] > 0]
state_fig = px.bar(us_data_death, x='Province_State', y='Active', title='State wise recovery cases of COVID-19 in USA', text='Active')
state_fig.show()


# In[23]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
combine_us_data = covid_data[covid_data['Country_Region']=='US'].drop(['Country_Region'], axis=1)
combine_us_data = combine_us_data[combine_us_data.sum(axis = 1) > 0]
combine_us_data = combine_us_data.groupby(['Province_State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
combine_us_data = pd.melt(combine_us_data, id_vars='Province_State', value_vars=['Confirmed', 'Deaths', 'Recovered', 'Active'], value_name='Count', var_name='Case')
fig = px.bar(combine_us_data, x='Province_State', y='Count', text='Count', barmode='group', color='Case', title='USA State wise combine number of confirmed, deaths, recovered, active COVID-19 cases')
fig.show()


# In[24]:


import plotly.express as px
import plotly.io as pio


# In[25]:


grouped = covid_data.groupby('Last_Update')['Last_Update', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped, x="Last_Update", y="Confirmed",
             title="Worldwide Confirmed Novel Coronavirus(COVID-19) Cases Over Time")
fig.show()


# In[ ]:




