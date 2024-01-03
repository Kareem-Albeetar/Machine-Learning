#!/usr/bin/env python
# coding: utf-8

# # Creating a Chatbot from Scratch using Python and Scikit-learn

# In[1]:


from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB


# In[2]:


training_phrases = {
    "help-me": ' '.join(["I have a problem",
                         "Hey i need some answers",
                         "Can you help me with this?",
                         "I need help",
                         "Please help me"
                        ]),
    "alcohol-addiction": ' '.join(["I am addicted to alcohol",
                                   "I love alcohol daily",
                                   "I am an alcoholic"
                                  ]),
    "depression-problem": ' '.join(["I am depressed",
                                   "I am lonely",
                                   "I dont have friends",
                                   "I am alone",
                                   "I am always sad",
                                   "Why am I sad all the time?"
                                   ]),
    "greeting": ' '.join(["Hi",
                         "Hey there",
                          "Hola",
                          "Hi How are you doing?"
                         ])
}


# In[3]:


training_phrases


# In[4]:


training_documents = list(training_phrases.values())
labels = list(training_phrases.keys())


# In[5]:


training_documents


# In[6]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = tuple(set(stopwords.words('english'))) 

word_tokens = []
for sent in training_documents:
    word_tokens.append(word_tokenize(sent))

print(word_tokens)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

print(word_tokens)
print(filtered_sentence)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_documents)
print(X)
vectorizer.get_feature_names()


# In[8]:


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X, labels)


# In[9]:


raw_queries = ["I love everything related to C2H50H"]
queries = vectorizer.transform(raw_queries)
predictions = classifier.predict(queries)
predictions


# In[10]:


def predict(raw_queries):
    queries = vectorizer.transform(raw_queries)
    return classifier.predict(queries)

predicted = predict(["I am very much sad", "can we talk?", "Can you help me?", "i take wine everyday and i cant live without it"])
expected = ["depression-problem", "help-me", "help-me", "alcohol-addiction"]


# In[11]:


predicted


# In[12]:


evaluation = precision_recall_fscore_support(expected, predicted)
evaluation


# In[13]:


metrics = {}
(metrics['p'], metrics['r'], metrics['f1'], _) = evaluation
metrics


# In[14]:


responses = {
    "depression-response": "Hi, please do not worry. I am here to help you.",
    "help-response": "Hi there, yes ofcourse. I am at your service. How can I help you today?",
    "alcohol-response": "Good to know that. Now I can work with you in making you better.",
    "greeting-response": "Hi there my friend. How are you today?"
}


# In[15]:


responses['alcohol-response']


# In[16]:


predicted = predict(["i take wine everyday and i cant live without it"])
# expected = ["alcohol-addiction"]
predicted


# In[17]:


def send_response(raw_queries):
    predicted = predict(raw_queries)
    print(predicted[0])
    if predicted[0] == "alcohol-addiction":
        return(responses["alcohol-response"])
    else:
        return "You are not an alcoholic!"


# In[18]:


bot_response = send_response(["i take wine everyday and i cant live without it"])
print(bot_response)


# In[19]:


from nltk.corpus import stopwords


# In[22]:


stop_words = tuple(set(stopwords.words('english'))) 

for sent in training_documents:
    word_tokens = word_tokenize(sent)
    
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
filtered_sentence


# In[23]:


tokens = ['problem', 'Hey', 'need', 'some', 'answers', 'help', 'me', 'with', 'this', 'Please', 'addicted', 'alcohol', 'love', 'daily', 'alcoholic', 'depressed', 'lonely', 'dont', 'friends', 'alone', 'always', 'sad', 'Why', 'all', 'time', 'Hi', 'Hey', 'there', 'Hola', 'Hi', 'How', 'are', 'you', 'doing', '?']


# In[24]:


from difflib import get_close_matches
def spell_checker(token):
    spelling_error_flag = False
    corrected_word = ''
    if len(get_close_matches(token, tokens, n=1, cutoff=0.80)) > 0:
        corrected_word = get_close_matches(token, tokens, n=1, cutoff=0.80)[0]
        spelling_error_flag = True
        return corrected_word, spelling_error_flag
    else:
        return corrected_word, spelling_error_flag


# In[25]:


corrected_word, flag = spell_checker('lone')
corrected_word


# In[26]:


from nltk.corpus import wordnet
syns = wordnet.synsets("alone")
print(syns)

