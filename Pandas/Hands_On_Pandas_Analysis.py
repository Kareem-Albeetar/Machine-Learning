#!/usr/bin/env python
# coding: utf-8

# # Pandas Data Analysis 
# 
# Download [Chipotle Dataset](https://github.com/subashgandyer/datasets/blob/main/chipotle.tsv) 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# set this so the 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')


# In[3]:


chipo.head(10)


# In[4]:


chipo.shape[0]


# In[5]:


chipo.info()


# In[6]:


chipo.shape[1]


# In[7]:


chipo.columns


# In[8]:


chipo.index


# In[9]:


c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)


# In[11]:


chipo.groupby('item_name').agg({'quantity': 'sum'}).sort_values('quantity', ascending=False)[:1]


# In[13]:


c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)['quantity']


# In[14]:


c = chipo.groupby('choice_description').sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)


# In[15]:


total_items_orders = chipo.quantity.sum()
total_items_orders


# In[16]:


chipo.item_price.dtype


# In[31]:


def float_converter(x):
    return float(x[1:-1])


# In[32]:


chipo.item_price = chipo.item_price.apply(float_converter)


# In[33]:


chipo.item_price.dtype


# In[ ]:


chipo = pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/chipotle.tsv", sep="\t")
chipo


# In[20]:


float_converter = lambda x: float(x[1:-1])
chipo.item_price_float_lam = chipo.item_price.apply(float_converter)


# In[21]:


chipo.item_price_float_lam.dtype


# In[22]:


chipo.item_price.str.slice(1).astype(float)


# In[34]:


revenue = (chipo['quantity']* chipo['item_price']).sum()
revenue


# In[35]:


orders = chipo.order_id.value_counts().count()
orders


# In[36]:


chipo.order_id.nunique()


# In[37]:


chipo['revenue'] = chipo['quantity'] * chipo['item_price']
order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()['revenue']


# In[ ]:


chipo.groupby(by=['order_id']).sum().mean()['revenue']


# In[ ]:


chipo.head()


# In[ ]:


chipo['item_price'] = chipo['item_price'].str.slice(1).astype(float)


# In[ ]:


chipo['revenue'] = chipo['item_price'] * chipo['quantity']


# In[ ]:


chipo.head()


# In[ ]:


order_group = chipo.groupby('order_id').sum()


# In[ ]:


order_group


# In[ ]:


order_group.mean()


# In[ ]:


order_group.mean()['revenue']


# In[38]:


chipo.item_name.value_counts().count()


# In[ ]:


url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')


# In[ ]:


# clean the item_price column and transform it in a float
# prices = [float(value[1 : -1]) for value in chipo.item_price]
prices = chipo.item_price.str.slice(1).astype(float)

# reassign the column with the cleaned prices
chipo.item_price = prices

# delete the duplicates in item_name and quantity
chipo_filtered = chipo.drop_duplicates(['item_name','quantity','choice_description'])

# chipo_filtered

# select only the products with quantity equals to 1
chipo_one_prod = chipo_filtered[chipo_filtered.quantity == 1]
chipo_one_prod

# chipo_one_prod[chipo_one_prod['item_price']>10].item_name.nunique()
# chipo_one_prod[chipo_one_prod['item_price']>10]



# chipo.query('price_per_item > 10').item_name.nunique()


# In[ ]:


chipo['item_price'] = chipo.item_price.str.slice(1).astype(float)


# In[39]:


(chipo['item_price'] > 10).sum()


# In[40]:


products = chipo.groupby('item_name').agg({'item_price': 'max'})
products.head()


# In[41]:


(products.item_price > 10).sum()


# In[42]:


chipo.item_name.sort_values()

# OR

chipo.sort_values(by = "item_name")


# In[43]:


chipo.sort_values(by = "item_price", ascending = False).head(1)


# In[44]:


chipo.item_price.idxmax()


# In[45]:


chipo.loc[3598, :]


# In[46]:


chipo_salad = chipo[chipo.item_name == "Veggie Salad Bowl"]

len(chipo_salad)


# In[47]:


chipo[chipo.item_name == "Veggie Salad Bowl"].quantity.sum()


# In[48]:


chipo_drink_steak_bowl = chipo[(chipo.item_name == "Canned Soda") & (chipo.quantity > 1)]
len(chipo_drink_steak_bowl)


# In[49]:


chipo[chipo.item_name == "Canned Soda"]


# In[50]:


chipo[(chipo.item_name == "Canned Soda") & (chipo.quantity > 1)]


# In[51]:


chipo[(chipo.item_name == "Canned Soda") & (chipo.quantity > 1)].count()


# In[52]:


(chipo[chipo.item_name == "Canned Soda"].quantity > 1).sum()


# ### Create a Histogram of top items bought

# In[53]:


top5 = chipo.groupby('item_name').agg({"quantity": 'sum'}).sort_values("quantity", ascending=False)[:5]
top5


# In[54]:


top5.plot(kind='bar')
plt.xlabel('Items')
plt.ylabel('Number of Times Ordered')
plt.title('Most ordered Chipotle\'s Items')
plt.show()


# In[55]:


import seaborn as sns


# In[56]:


sns.barplot(
    x="item_name",
    y="quantity",
    data=top5.reset_index()
)
plt.xticks(rotation='vertical')


# In[ ]:





# In[59]:


url = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')


# In[61]:


# create a list of prices
# chipo.item_price_corrected = [float(value[1:-1]) for value in chipo.item_price] # strip the dollar sign and trailing space
chipo['item_price_corrected'] = chipo.item_price.str.slice(1).astype(float) # strip the dollar sign and trailing space

# then groupby the orders and sum
orders = chipo.groupby('order_id').sum()

# creates the scatterplot
# plt.scatter(orders.quantity, orders.item_price, s = 50, c = 'green')
plt.scatter(x = orders.item_price_corrected, y = orders.quantity, s = 50, c = 'green')

# Set the title and labels
plt.xlabel('Order Price')
plt.ylabel('Items ordered')
plt.title('Number of items ordered per order price')
plt.ylim(0)


# In[62]:


sns.scatterplot(x=orders.item_price_corrected, y=orders.quantity, data=chipo)


# In[ ]:




