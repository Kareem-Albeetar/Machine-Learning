#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
tags=pd.read_csv('T3/ml-20m/tags.csv')


# In[9]:


movies=pd.read_csv('T3/ml-20m/movies.csv')


# In[4]:


ratings=pd.read_csv('T3/ml-20m/ratings.csv')


# In[175]:


ratings


# In[176]:


tags.sort_values('movieId',inplace=True)


# In[177]:


tags


# In[178]:


# collecting tags in movies dataframe for content
for j in tags.movieId.unique():
   movies.loc[movies.loc[movies.movieId==j].index,'tags']=' '.join([i if type(i)==str else str(i) for i in tags.loc[ ( tags.movieId == j ),'tag'].unique().tolist() ])


# In[179]:


movies


# In[180]:


#splitting data for usability, larger datasets crashed kaggle and pc
from sklearn.model_selection import train_test_split
use,dontuse=train_test_split(ratings,test_size=0.995)


# In[181]:


use


# In[311]:


import gc

gc.collect()


# In[183]:


user_movies_data=pd.pivot_table(use,index='movieId',columns='userId',values='rating',fill_value=0)


# In[184]:


user_movies_data


# In[185]:


movies['tags']=movies['tags'].fillna('None')


# In[186]:


movies


# In[187]:


#Content Filtering
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(movies['tags'])
tfidf_df=pd.DataFrame(tfidf_matrix.toarray(),index=movies.index.tolist())


# In[188]:


tfidf_df.shape


# In[189]:


from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=19)
latent_matrix=svd.fit_transform(tfidf_df)


# In[190]:


n=19
latent_matrix_1_df=pd.DataFrame(latent_matrix[:,0:n],index=movies['title'].tolist())


# In[191]:


#Collaborative 
svd=TruncatedSVD(n_components=20)
latent_matrix_2=svd.fit_transform(user_movies_data)


# In[192]:


latent_matrix_2_df=pd.DataFrame(latent_matrix_2,index=[movies.loc[(movies.movieId==i),'title'].values[0] for i in (use['movieId'].unique())])


# In[193]:


latent_matrix_2_df


# In[194]:


#Finding issues with different sized matrixes
latent_matrix_1_df=latent_matrix_1_df.drop_duplicates()
latent_matrix_2_df=latent_matrix_2_df.drop_duplicates()


# In[195]:


a=latent_matrix_2_df.copy()
for i in latent_matrix_1_df.index:
    if i not in latent_matrix_2_df.index:
        a=a.append(pd.Series(np.zeros(20), index=np.arange(0,20),name=i))


# In[208]:


b=latent_matrix_1_df.copy()
for i in a.index:
    if i not in b.index:
        b=b.append(pd.Series(np.zeros(19), index=np.arange(0,19),name=i))


# In[282]:


#Same index was repeated multiple times, decided to remove duplicates
a = a[~a.index.duplicated(keep='first')]


# In[284]:


b = b[~b.index.duplicated(keep='first')]


# In[287]:


from sklearn.metrics.pairwise import cosine_similarity


# In[288]:


def recommend_similar_movies(title):
    if title in b.index:
        a_1=np.array(b.loc[title]).reshape(1,-1)
        score_content=cosine_similarity(b,a_1).reshape(-1)
    else:
        score_content=0
    if title in a.index:
        a_2=np.array(a.loc[title]).reshape(1,-1)
        score_collab=cosine_similarity(a,a_2).reshape(-1)
    else:
        score_collab=0
    
    hybrid_score=(score_content+score_collab)/2

    dictDF={'content':score_content,'collab':score_collab,'hybrid':hybrid_score}
    dictDF
    similar_movies=pd.DataFrame(dictDF,index=a.index)


    similar_movies.sort_values('hybrid',ascending=False,inplace=True)
    return similar_movies


# In[289]:


recommend_similar_movies('Toy Story (1995)')


# In[294]:


recommend_similar_movies('Mission: Impossible II (2000)')


# In[305]:


from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
Mapping_file = dict(zip(movies['title'].tolist(), movies['movieId'].tolist()))


# In[312]:


reader=Reader(rating_scale=(1,5))
data=Dataset.load_from_df(use[['userId','movieId','rating']],reader)


# In[313]:


#using split data 'use' from earlier
trainset,testset=train_test_split(data,test_size=0.25)


# In[314]:


svd=SVD()
svd.fit(trainset)
preds=svd.test(testset)


# In[315]:


accuracy.rmse(preds)


# In[318]:


def pred_user_rating(ui):
    if ui in ratings.userId.unique():
        ui_list = ratings[ratings.userId == ui].movieId.tolist()
        d = {k: v for k,v in Mapping_file.items() if not v in ui_list}        
        predictedL = []
        for i, j in d.items():     
            predicted = svd.predict(ui, j)
            predictedL.append((i, predicted[3])) 
        pdf = pd.DataFrame(predictedL, columns = ['movies', 'ratings'])
        pdf.sort_values('ratings', ascending=False, inplace=True)  
        pdf.set_index('movies', inplace=True)    
        return pdf.head(10)        
    else:
        print("User Id does not exist in the list!")
        return None


# In[320]:


pred_user_rating(100)


# In[321]:


pred_user_rating(1100)


# In[336]:


#Checking movie popularity by looking at number of user ratings
ratings['movieId'].value_counts()


# In[10]:


counts=ratings['movieId'].value_counts().values.tolist()


# In[23]:


bestmovies=[movies.loc[(movies.movieId==i),'title'].tolist()[0] for i in ratings['movieId'].value_counts().index.tolist()]


# In[27]:


pd.DataFrame(counts,index=bestmovies).iloc[:20]

