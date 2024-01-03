#!/usr/bin/env python
# coding: utf-8

# # Recommendation System - Movie Recommendation
# ## This notebook outlines the concepts involved in building a Complete Recommendation System for recommending Movies to users
# 

# In[50]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


#! wget https://raw.githubusercontent.com/subashgandyer/datasets/main/ml-100k/ml-100k.zip


# In[3]:


#! unzip ml-100k.zip


# In[4]:


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')


# In[5]:


print("\nUser Data :")
print("shape : ", users.shape)
print(users.head())


# In[6]:


users


# In[7]:


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')


# In[8]:


print("\nRatings Data :")
print("shape : ", ratings.shape)
print(ratings.head())


# In[9]:


i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')


# In[10]:


print("\nItem Data :")
print("shape : ", items.shape)
print(items.head())


# In[11]:


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')


# In[12]:


ratings_train.shape, ratings_test.shape


# In[13]:


n_users = ratings.user_id.unique().shape[0]
n_users


# In[14]:


n_items = ratings.movie_id.unique().shape[0]
n_items


# In[15]:


from scipy.sparse import csr_matrix


# In[16]:


user_movies_data = ratings.pivot(index = 'movie_id', columns = 'user_id', values = 'rating').fillna(0)


# In[17]:


user_movies_data


# In[18]:


items


# In[19]:


items['metadata'] = ""


# In[20]:


items


# In[21]:


items.Action, type(items.Action)


# In[22]:


def metadata_Action(x):
    if x == 1:
        return "Action"
    else:
        return " "


# In[23]:


items['metadata_Action'] = items.Action.apply(metadata_Action)


# In[24]:


items


# In[25]:


def metadata_Adventure(x):
    if x == 1:
        return " Adventure "
    else:
        return " "
    
items['metadata_Adventure'] = items.Adventure.apply(metadata_Adventure)


# In[26]:


items


# In[27]:


genres = ['Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


# In[28]:


def metadata_Animation(x):
    if x == 1:
        return " Animation "
    else:
        return " "
items['metadata_Animation'] = items.Animation.apply(metadata_Animation)


# In[29]:


def metadata_Childrens(x):
    if x == 1:
        return " Children's "
    else:
        return " "
    
items["metadata_Children's"] = items["Children's"].apply(metadata_Childrens)


# In[30]:


def metadata_Comedy(x):
    if x == 1:
        return " Comedy "
    else:
        return " "

items['metadata_Comedy'] = items.Comedy.apply(metadata_Comedy)


# In[31]:


def metadata_Crime(x):
    if x == 1:
        return " Crime "
    else:
        return " "
    
items['metadata_Crime'] = items.Crime.apply(metadata_Crime)


# In[32]:


def metadata_Documentary(x):
    if x == 1:
        return " Documentary "
    else:
        return " "
    
items['metadata_Documentary'] = items.Documentary.apply(metadata_Documentary)


# In[33]:


def metadata_Drama(x):
    if x == 1:
        return " Drama "
    else:
        return " "
    
items['metadata_Drama'] = items.Drama.apply(metadata_Drama)


# In[34]:


def metadata_Fantasy(x):
    if x == 1:
        return " Fantasy "
    else:
        return " "
    
items['metadata_Fantasy'] = items.Fantasy.apply(metadata_Fantasy)


# In[35]:


def metadata_FilmNoir(x):
    if x == 1:
        return " Film-Noir "
    else:
        return " "
    
items['metadata_Film-Noir'] = items["Film-Noir"].apply(metadata_FilmNoir)


# In[36]:


def metadata_Horror(x):
    if x == 1:
        return "Horror "
    else:
        return " "

items['metadata_Horror'] = items.Horror.apply(metadata_Horror)


# In[37]:


def metadata_Musical(x):
    if x == 1:
        return " Musical "
    else:
        return " "
    
items['metadata_Musical'] = items.Musical.apply(metadata_Musical)


# In[38]:


def metadata_Mystery(x):
    if x == 1:
        return " Mystery "
    else:
        return " "
    
items['metadata_Mystery'] = items.Mystery.apply(metadata_Mystery)


# In[39]:


def metadata_Romance(x):
    if x == 1:
        return " Romance "
    else:
        return " "
    
items['metadata_Romance'] = items.Romance.apply(metadata_Romance)


# In[40]:


def metadata_SciFi(x):
    if x == 1:
        return " Sci-Fi "
    else:
        return " "
    
items['metadata_Sci-Fi'] = items["Sci-Fi"].apply(metadata_SciFi)


# In[41]:


def metadata_Thriller(x):
    if x == 1:
        return " Thriller "
    else:
        return " "
    
items['metadata_Thriller'] = items.Thriller.apply(metadata_Thriller)


# In[42]:


def metadata_War(x):
    if x == 1:
        return " War "
    else:
        return " "
    
items['metadata_War'] = items.War.apply(metadata_War)


# In[43]:


def metadata_Western(x):
    if x == 1:
        return " Western "
    else:
        return " "
    
items['metadata_Western'] = items.Western.apply(metadata_Western)


# In[44]:


items


# In[45]:


items['full_metadata'] = items[['metadata_Action', 'metadata_Adventure',
'metadata_Animation', 'metadata_Children\'s', 'metadata_Comedy', 'metadata_Crime', 'metadata_Documentary', 'metadata_Drama', 'metadata_Fantasy',
'metadata_Film-Noir', 'metadata_Horror', 'metadata_Musical', 'metadata_Mystery', 'metadata_Romance', 'metadata_Sci-Fi', 'metadata_Thriller', 'metadata_War', 'metadata_Western']].apply(
                                          lambda x: ' '.join(x), axis = 1)
                                
                                


# In[46]:


items


# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(items['full_metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=items.index.tolist())
print(tfidf_df.shape)


# In[51]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=19)
latent_matrix = svd.fit_transform(tfidf_df)
# plot var expalined to see what latent dimensions to use
explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color='red')
plt.xlabel('Singular value components', fontsize= 12)
plt.ylabel('Cumulative percent of variance', fontsize=12)        
plt.show()


# In[52]:


n = 20
latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index=items.movie_title.tolist())


# In[53]:


latent_matrix_1_df.shape


# In[54]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
latent_matrix_2 = svd.fit_transform(user_movies_data)
latent_matrix_2_df = pd.DataFrame(
                             latent_matrix_2, index=items.movie_title.tolist())


# In[55]:


latent_matrix_2_df


# In[56]:


# plot variance expalined to see what latent dimensions to use
explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color='red')
plt.xlabel('Singular value components', fontsize= 12)
plt.ylabel('Cumulative percent of variance', fontsize=12)        
plt.show()


# In[57]:


from sklearn.metrics.pairwise import cosine_similarity
# take the latent vectors for a selected movie from both content 
# and collaborative matrixes
a_1 = np.array(latent_matrix_1_df.loc['Toy Story (1995)']).reshape(1, -1)
a_2 = np.array(latent_matrix_2_df.loc["Toy Story (1995)"]).reshape(1, -1)

# calculate the similartity of this movie with the others in the list
score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

# an average measure of both content and collaborative 
hybrid = ((score_1 + score_2)/2.0)

# form a data frame of similar movies 
dictDf = {'content': score_1 , 'collaborative': score_2, 'hybrid': hybrid} 
# dictDf = {'collaborative': score_2} 
similar = pd.DataFrame(dictDf, index = latent_matrix_2_df.index )

#sort it on the basis of either: content, collaborative or hybrid, 
# here : content
similar.sort_values('content', ascending=False, inplace=True)

similar[1:].head(11)


# In[58]:


similar.sort_values('collaborative', ascending=False, inplace=True)

similar[1:].head(11)


# In[59]:


similar.sort_values('hybrid', ascending=False, inplace=True)

similar[1:].head(11)


# In[60]:


def recommend_similar_movies(title):
    # take the latent vectors for a selected movie from both content 
    # and collaborative matrixes
    a_1 = np.array(latent_matrix_1_df.loc[title]).reshape(1, -1)
    a_2 = np.array(latent_matrix_2_df.loc[title]).reshape(1, -1)

    # calculate the similartity of this movie with the others in the list
    score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
    score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

    # an average measure of both content and collaborative 
    hybrid = ((score_1 + score_2)/2.0)

    # form a data frame of similar movies 
    dictDf = {'content': score_1 , 'collaborative': score_2, 'hybrid': hybrid} 
    similar = pd.DataFrame(dictDf, index = latent_matrix_2_df.index )

    #sort it on the basis of either: content, collaborative or hybrid
    similar.sort_values('hybrid', ascending=False, inplace=True)

    print(similar[1:].head(11))


# In[61]:


recommend_similar_movies("Toy Story (1995)")


# In[62]:


recommend_similar_movies("GoldenEye (1995)")


# In[63]:


recommend_similar_movies("Mission: Impossible (1996)")


# In[64]:


#! pip install turicreate


# In[65]:


import turicreate


# In[66]:


train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)


# In[67]:


popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')


# In[68]:


popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_recomm.print_rows(num_rows=25)


# In[69]:


item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')


# In[70]:


item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=5)
item_sim_recomm.print_rows(num_rows=25)


# In[71]:


class MF():

    # Initializing the user-movie rating matrix, no. of latent features, alpha and beta.
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    # Initializing user-feature and movie-feature matrix 
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # List of training samples
        self.samples = [
        (i, j, self.R[i, j])
        for i in range(self.num_users)
        for j in range(self.num_items)
        if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 20 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    # Computing total mean squared error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and moive j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_matrix(self):
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)


# In[72]:


R= np.array(ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))


# In[73]:


mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=100)
training_process = mf.train()
print()
print("P x Q:")
print(mf.full_matrix())
print()


# In[74]:


from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


# In[75]:


ratings


# In[76]:


items


# In[77]:


Mapping_file = dict(zip(items.movie_title.tolist(), items.movie_id.tolist()))


# In[78]:


# instantiate a reader and read in our rating data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader)

# train SVD on 75% of known rates
trainset, testset = train_test_split(data, test_size=.25)
algorithm = SVD()
algorithm.fit(trainset)
predictions = algorithm.test(testset)

# check the accuracy using Root Mean Square Error
accuracy.rmse(predictions)


# In[79]:


def pred_user_rating(ui):
    if ui in ratings.user_id.unique():
        ui_list = ratings[ratings.user_id == ui].movie_id.tolist()
        d = {k: v for k,v in Mapping_file.items() if not v in ui_list}        
        predictedL = []
        for i, j in d.items():     
            predicted = algorithm.predict(ui, j)
            predictedL.append((i, predicted[3])) 
        pdf = pd.DataFrame(predictedL, columns = ['movies', 'ratings'])
        pdf.sort_values('ratings', ascending=False, inplace=True)  
        pdf.set_index('movies', inplace=True)    
        return pdf.head(10)        
    else:
        print("User Id does not exist in the list!")
        return None


# In[80]:


user_id = 1
pred_user_rating(user_id)


# In[81]:


pred_user_rating(50)


# In[82]:


pred_user_rating(49)


# In[83]:


pred_user_rating(1)


# In[84]:


pred_user_rating(915)

