#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')
movies = pd.read_csv('http://bit.ly/imdbratings')
orders = pd.read_csv('http://bit.ly/chiporders', sep='\t')
orders['item_price'] = orders.item_price.str.replace('$', '').astype('float')
stocks = pd.read_csv('http://bit.ly/smallstocks', parse_dates=['Date'])
titanic = pd.read_csv('http://bit.ly/kaggletrain')
ufo = pd.read_csv('http://bit.ly/uforeports', parse_dates=['Time'])


# In[3]:


pd.__version__


# In[4]:


pd.show_versions()


# In[5]:


df = pd.DataFrame({'col one':[100, 200], 'col two':[300, 400]})
df


# In[6]:


pd.DataFrame(np.random.rand(4, 8))


# In[7]:


pd.DataFrame(np.random.rand(4, 8), columns=list('abcdefgh'))


# In[8]:


df


# In[9]:


df = df.rename({'col one':'col_one', 'col two':'col_two'}, axis='columns')


# In[10]:


df.columns = ['col_one', 'col_two']


# In[11]:


df.columns = df.columns.str.replace(' ', '_')


# In[12]:


df


# In[13]:


df.add_prefix('X_')


# In[14]:


df.add_suffix('_Y')


# In[15]:


drinks.head()


# In[16]:


drinks.loc[::-1].head()


# In[17]:


drinks.loc[::-1].reset_index(drop=True).head()


# In[18]:


drinks.loc[:, ::-1].head()


# In[19]:


drinks.dtypes


# In[20]:


drinks.select_dtypes(include='number').head()


# In[21]:


drinks.select_dtypes(include='object').head()


# In[22]:


drinks.select_dtypes(include=['number', 'object', 'category', 'datetime']).head()


# In[23]:


drinks.select_dtypes(exclude='number').head()


# In[24]:


df = pd.DataFrame({'col_one':['1.1', '2.2', '3.3'],
                   'col_two':['4.4', '5.5', '6.6'],
                   'col_three':['7.7', '8.8', '-']})
df


# In[25]:


df.dtypes


# In[26]:


df.astype({'col_one':'float', 'col_two':'float'}).dtypes


# In[27]:


pd.to_numeric(df.col_three, errors='coerce')


# In[28]:


pd.to_numeric(df.col_three, errors='coerce').fillna(0)


# In[29]:


df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
df


# In[30]:


df.dtypes


# In[31]:


drinks.info(memory_usage='deep')


# In[32]:


cols = ['beer_servings', 'continent']
small_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols)
small_drinks.info(memory_usage='deep')


# In[33]:


dtypes = {'continent':'category'}
smaller_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols, dtype=dtypes)
smaller_drinks.info(memory_usage='deep')


# In[34]:


pd.read_csv('data/stocks1.csv')


# In[35]:


pd.read_csv('data/stocks2.csv')


# In[36]:


pd.read_csv('data/stocks3.csv')


# In[37]:


from glob import glob


# In[38]:


stock_files = sorted(glob('data/stocks*.csv'))
stock_files


# In[39]:


pd.concat((pd.read_csv(file) for file in stock_files))


# In[40]:


pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)


# In[41]:


pd.read_csv('data/drinks1.csv').head()


# In[42]:


pd.read_csv('data/drinks2.csv').head()


# In[43]:


drink_files = sorted(glob('data/drinks*.csv'))


# In[44]:


pd.concat((pd.read_csv(file) for file in drink_files), axis='columns').head()


# In[45]:


df = pd.read_clipboard()
df


# In[46]:


df.dtypes


# In[47]:


df = pd.read_clipboard()
df


# In[48]:


df.index


# In[49]:


len(movies)


# In[50]:


movies_1 = movies.sample(frac=0.75, random_state=1234)


# In[51]:


movies_2 = movies.drop(movies_1.index)


# In[52]:


len(movies_1) + len(movies_2)


# In[53]:


movies_1.index.sort_values()


# In[54]:


movies_2.index.sort_values()


# In[55]:


movies.head()


# In[56]:


movies.genre.unique()


# In[57]:


movies[(movies.genre == 'Action') |
       (movies.genre == 'Drama') |
       (movies.genre == 'Western')].head()


# In[58]:


movies[movies.genre.isin(['Action', 'Drama', 'Western'])].head()


# In[59]:


movies[~movies.genre.isin(['Action', 'Drama', 'Western'])].head()


# In[60]:


counts = movies.genre.value_counts()
counts


# In[61]:


counts.nlargest(3)


# In[62]:


counts.nlargest(3).index


# In[63]:


movies[movies.genre.isin(counts.nlargest(3).index)].head()


# In[64]:


ufo.head()


# In[65]:


ufo.isna().sum()


# In[66]:


ufo.isna().mean()


# In[67]:


ufo.dropna(axis='columns').head()


# In[68]:


ufo.dropna(thresh=len(ufo)*0.9, axis='columns').head()


# In[69]:


df = pd.DataFrame({'name':['John Arthur Doe', 'Jane Ann Smith'],
                   'location':['Los Angeles, CA', 'Washington, DC']})
df


# In[70]:


df.name.str.split(' ', expand=True)


# In[71]:


df[['first', 'middle', 'last']] = df.name.str.split(' ', expand=True)
df


# In[72]:


df.location.str.split(', ', expand=True)


# In[73]:


df['city'] = df.location.str.split(', ', expand=True)[0]
df


# In[74]:


df = pd.DataFrame({'col_one':['a', 'b', 'c'], 'col_two':[[10, 40], [20, 50], [30, 60]]})
df


# In[75]:


df_new = df.col_two.apply(pd.Series)
df_new


# In[76]:


pd.concat([df, df_new], axis='columns')


# In[77]:


orders.head(10)


# In[78]:


orders[orders.order_id == 1].item_price.sum()


# In[79]:


orders.groupby('order_id').item_price.sum().head()


# In[80]:


orders.groupby('order_id').item_price.agg(['sum', 'count']).head()


# In[81]:


orders.head(10)


# In[83]:


len(orders.groupby('order_id').item_price.sum())


# In[84]:


len(orders.item_price)


# In[85]:


total_price = orders.groupby('order_id').item_price.transform('sum')
len(total_price)


# In[86]:


orders['total_price'] = total_price
orders.head(10)


# In[87]:


orders['percent_of_total'] = orders.item_price / orders.total_price
orders.head(10)


# In[88]:


titanic.head()


# In[89]:


titanic.describe()


# In[90]:


titanic.describe().loc['min':'max']


# In[91]:


titanic.describe().loc['min':'max', 'Pclass':'Parch']


# In[92]:


titanic.Survived.mean()


# In[93]:


titanic.groupby('Sex').Survived.mean()


# In[94]:


titanic.groupby(['Sex', 'Pclass']).Survived.mean()


# In[95]:


titanic.groupby(['Sex', 'Pclass']).Survived.mean().unstack()


# In[96]:


titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')


# In[97]:


titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean',
                    margins=True)


# In[98]:


titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='count',
                    margins=True)


# In[99]:


titanic.Age.head(10)


# In[100]:


pd.cut(titanic.Age, bins=[0, 18, 25, 99], labels=['child', 'young adult', 'adult']).head(10)


# In[101]:


titanic.head()


# In[102]:


pd.set_option('display.float_format', '{:.2f}'.format)


# In[103]:


titanic.head()


# In[104]:


pd.reset_option('display.float_format')


# In[105]:


stocks


# In[106]:


format_dict = {'Date':'{:%m/%d/%y}', 'Close':'${:.2f}', 'Volume':'{:,}'}


# In[107]:


stocks.style.format(format_dict)


# In[108]:


(stocks.style.format(format_dict)
 .hide_index()
 .highlight_min('Close', color='red')
 .highlight_max('Close', color='lightgreen')
)


# In[109]:


(stocks.style.format(format_dict)
 .hide_index()
 .background_gradient(subset='Volume', cmap='Blues')
)


# In[110]:


(stocks.style.format(format_dict)
 .hide_index()
 .bar('Volume', color='lightblue', align='zero')
 .set_caption('Stock Prices from October 2016')
)


# In[111]:


import pandas_profiling


# In[112]:


pandas_profiling.ProfileReport(titanic)

