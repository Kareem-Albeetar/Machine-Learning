#!/usr/bin/env python
# coding: utf-8

#  #### Create a Prescription Parser using CRF
# 
# 
# 

# In[1]:


get_ipython().system(' pip install sklearn-crfsuite')


# In[1]:


from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import sklearn_crfsuite
import pycrfsuite
from sklearn.model_selection import train_test_split
print(sklearn.__version__)


# In[2]:


sigs = ["for 5 to 6 days", "inject 2 units", "x 2 weeks", "x 3 days", "every day", "every 2 weeks", "every 3 days", "every 1 to 2 months", "every 2 to 6 weeks", "every 4 to 6 days", "take two to four tabs", "take 2 to 4 tabs", "take 3 tabs orally bid for 10 days at bedtime", "swallow three capsules tid orally", "take 2 capsules po every 6 hours", "take 2 tabs po for 10 days", "take 100 caps by mouth tid for 10 weeks", "take 2 tabs after an hour", "2 tabs every 4-6 hours", "every 4 to 6 hours", "q46h", "q4-6h", "2 hours before breakfast", "before 30 mins at bedtime", "30 mins before bed", "and 100 tabs twice a month", "100 tabs twice a month", "100 tabs once a month", "100 tabs thrice a month", "3 tabs daily for 3 days then 1 tab per day at bed", "30 tabs 10 days tid", "take 30 tabs for 10 days three times a day", "qid q6h", "bid", "qid", "30 tabs before dinner and bedtime", "30 tabs before dinner & bedtime", "take 3 tabs at bedtime", "30 tabs thrice daily for 10 days ", "30 tabs for 10 days three times a day", "Take 2 tablets a day", "qid for 10 days", "every day", "take 2 caps at bedtime", "apply 3 drops before bedtime", "take three capsules daily", "swallow 3 pills once a day", "swallow three pills thrice a day", "apply daily", "apply three drops before bedtime", "every 6 hours", "before food", "after food", "for 20 days", "for twenty days", "with meals"]
input_sigs = [['for', '5', 'to', '6', 'days'], ['inject', '2', 'units'], ['x', '2', 'weeks'], ['x', '3', 'days'], ['every', 'day'], ['every', '2', 'weeks'], ['every', '3', 'days'], ['every', '1', 'to', '2', 'months'], ['every', '2', 'to', '6', 'weeks'], ['every', '4', 'to', '6', 'days'], ['take', 'two', 'to', 'four', 'tabs'], ['take', '2', 'to', '4', 'tabs'], ['take', '3', 'tabs', 'orally', 'bid', 'for', '10', 'days', 'at', 'bedtime'], ['swallow', 'three', 'capsules', 'tid', 'orally'], ['take', '2', 'capsules', 'po', 'every', '6', 'hours'], ['take', '2', 'tabs', 'po', 'for', '10', 'days'], ['take', '100', 'caps', 'by', 'mouth', 'tid', 'for', '10', 'weeks'], ['take', '2', 'tabs', 'after', 'an', 'hour'], ['2', 'tabs', 'every', '4-6', 'hours'], ['every', '4', 'to', '6', 'hours'], ['q46h'], ['q4-6h'], ['2', 'hours', 'before', 'breakfast'], ['before', '30', 'mins', 'at', 'bedtime'], ['30', 'mins', 'before', 'bed'], ['and', '100', 'tabs', 'twice', 'a', 'month'], ['100', 'tabs', 'twice', 'a', 'month'], ['100', 'tabs', 'once', 'a', 'month'], ['100', 'tabs', 'thrice', 'a', 'month'], ['3', 'tabs', 'daily', 'for', '3', 'days', 'then', '1', 'tab', 'per', 'day', 'at', 'bed'], ['30', 'tabs', '10', 'days', 'tid'], ['take', '30', 'tabs', 'for', '10', 'days', 'three', 'times', 'a', 'day'], ['qid', 'q6h'], ['bid'], ['qid'], ['30', 'tabs', 'before', 'dinner', 'and', 'bedtime'], ['30', 'tabs', 'before', 'dinner', '&', 'bedtime'], ['take', '3', 'tabs', 'at', 'bedtime'], ['30', 'tabs', 'thrice', 'daily', 'for', '10', 'days'], ['30', 'tabs', 'for', '10', 'days', 'three', 'times', 'a', 'day'], ['take', '2', 'tablets', 'a', 'day'], ['qid', 'for', '10', 'days'], ['every', 'day'], ['take', '2', 'caps', 'at', 'bedtime'], ['apply', '3', 'drops', 'before', 'bedtime'], ['take', 'three', 'capsules', 'daily'], ['swallow', '3', 'pills', 'once', 'a', 'day'], ['swallow', 'three', 'pills', 'thrice', 'a', 'day'], ['apply', 'daily'], ['apply', 'three', 'drops', 'before', 'bedtime'], ['every', '6', 'hours'], ['before', 'food'], ['after', 'food'], ['for', '20', 'days'], ['for', 'twenty', 'days'], ['with', 'meals']]
output_labels = [['FOR', 'Duration', 'TO', 'DurationMax', 'DurationUnit'], ['Method', 'Qty', 'Form'], ['FOR', 'Duration', 'DurationUnit'], ['FOR', 'Duration', 'DurationUnit'], ['EVERY', 'Period'], ['EVERY', 'Period', 'PeriodUnit'], ['EVERY', 'Period', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['Method', 'Qty', 'TO', 'Qty', 'Form'], ['Method', 'Qty', 'TO', 'Qty', 'Form'], ['Method', 'Qty', 'Form', 'PO', 'BID', 'FOR', 'Duration', 'DurationUnit', 'AT', 'WHEN'], ['Method', 'Qty', 'Form', 'TID', 'PO'], ['Method', 'Qty', 'Form', 'PO', 'EVERY', 'Period', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'PO', 'FOR', 'Duration', 'DurationUnit'], ['Method', 'Qty', 'Form', 'BY', 'PO', 'TID', 'FOR', 'Duration', 'DurationUnit'], ['Method', 'Qty', 'Form', 'AFTER', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'EVERY', 'Period', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['Q46H'], ['Q4-6H'], ['Qty', 'PeriodUnit', 'BEFORE', 'WHEN'], ['BEFORE', 'Qty', 'M', 'AT', 'WHEN'], ['Qty', 'M', 'BEFORE', 'WHEN'], ['AND', 'Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'FOR', 'Duration', 'DurationUnit', 'THEN', 'Qty', 'Form', 'Frequency', 'PeriodUnit', 'AT', 'WHEN'], ['Qty', 'Form', 'Duration', 'DurationUnit', 'TID'], ['Method', 'Qty', 'Form', 'FOR', 'Duration', 'DurationUnit', 'Qty', 'TIMES', 'Period', 'PeriodUnit'], ['QID', 'Q6H'], ['BID'], ['QID'],['Qty', 'Form', 'BEFORE', 'WHEN', 'AND', 'WHEN'], ['Qty', 'Form', 'BEFORE', 'WHEN', 'AND', 'WHEN'], ['Method', 'Qty', 'Form', 'AT', 'WHEN'], ['Qty', 'Form', 'Frequency', 'DAILY', 'FOR', 'Duration', 'DurationUnit'], ['Qty', 'Form', 'FOR', 'Duration', 'DurationUnit', 'Frequency', 'TIMES', 'Period', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'Period', 'PeriodUnit'], ['QID', 'FOR', 'Duration', 'DurationUnit'], ['EVERY', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'AT', 'WHEN'], ['Method', 'Qty', 'Form', 'BEFORE', 'WHEN'], ['Method', 'Qty', 'Form', 'DAILY'], ['Method', 'Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Method', 'DAILY'], ['Method', 'Qty', 'Form', 'BEFORE', 'WHEN'], ['EVERY', 'Period', 'PeriodUnit'], ['BEFORE', 'FOOD'], ['AFTER', 'FOOD'], ['FOR', 'Duration', 'DurationUnit'], ['FOR', 'Duration', 'DurationUnit'], ['WITH', 'FOOD']]


# In[3]:


len(sigs), len(input_sigs) , len(output_labels)


# In[4]:


def tuples_maker(inp, out):
    sample_data = []
    for (inp,out) in zip(inp,out):
        #print(inp,out)
        sample_data.append((inp,out))
    return sample_data


# In[5]:


whole_data = []
for i in range(len(sigs)):
    data = tuples_maker(input_sigs[i], output_labels[i])
    whole_data.append(data)
whole_data


# In[6]:


def triples_maker(whole_data):
    sample_data = []
    for i, doc in enumerate(whole_data):

        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        # Perform POS tagging
        tagged = nltk.pos_tag(tokens)

        # Take the word, POS tag, and its label
        sample_data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    return sample_data

sample_data = triples_maker(whole_data)
sample_data


# In[7]:


def token_to_features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features


# In[8]:


from sklearn.model_selection import train_test_split

# A function for extracting features in documents
def get_features(doc):
    return [token_to_features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

#[print(doc) for doc in sample_data]
X = [get_features(doc) for doc in sample_data]

y = [get_labels(doc) for doc in sample_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[9]:


trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X, y):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 1000,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Providing a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('prescription_parser.model')


# In[10]:


test_tagger = pycrfsuite.Tagger()
test_tagger.open('prescription_parser.model')
#print(X_test[0][0])
y_pred = [test_tagger.tag(xseq) for xseq in X_test]

for k in range(len(X_test)):
    for i in range(len(X_test[k])):
         print(X_test[k][i][1])
#print(test_sig)
print(y_pred)


# In[11]:


test_sig = "1 tab x 4 days"


# In[12]:


test_sigs = []
tokens = nltk.word_tokenize(test_sig)
words = [w.lower() for w in tokens]
tags = nltk.pos_tag(words)
test_sigs.append(tags)

test_sigs


# In[13]:


test_data = []
for i, doc in enumerate(test_sigs):
    # Obtain the list of tokens in the document
    tokens = [t for t, label in doc] # Perform POS tagging
    tagged = nltk.pos_tag(tokens)
    # Take the word, POS tag, and its label
    test_data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc,tagged)])
test_data


# In[14]:


X_wild = [get_features(doc) for doc in test_data]
#X_wild


# In[15]:


test_tagger = pycrfsuite.Tagger()
test_tagger.open('prescription_parser.model')
y_wild = [test_tagger.tag(xseq) for xseq in X_wild]

# for i in range(len(X_wild[0])):
#     print(X_wild[0][i][1])
print(test_sig)
print(y_wild)


# In[16]:


def predict(sig):
    """
    predict(sig)
    Purpose: Labels the given sig into corresponding labels
    @param sig. A Sentence  # A medical prescription sig written by a doctor
    @return     A list      # A list with predicted labels (first level of labeling)
    >>> predict('2 tabs every 4 hours')
    [['Qty', 'Form', 'EVERY', 'Period', 'PeriodUnit']]
    >>> predict('2 tabs with food')
    [['Qty', 'Form', 'WITH', 'FOOD']]
    >>> predict('2 tabs qid x 30 days')
    [['Qty', 'Form', 'QID', 'FOR', 'Duration', 'DurationUnit']]
    """
    
    test_sigs = []
    tokens = nltk.word_tokenize(sig)
    words = [w.lower() for w in tokens]
    tags = nltk.pos_tag(words)
    test_sigs.append(tags)
    
    test_data = []
    for i, doc in enumerate(test_sigs):
        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc] # Perform POS tagging
        tagged = nltk.pos_tag(tokens)
        # Take the word, POS tag, and its label
        test_data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc,tagged)])
        
    X_wild = [get_features(doc) for doc in test_data]
    
    
    test_tagger = pycrfsuite.Tagger()
    test_tagger.open('prescription_parser.model')
    predictions = [test_tagger.tag(xseq) for xseq in X_wild]

    print(sig)
    print(predictions)
    
    return predictions


# In[17]:


predictions = predict("take 2 tabs every 6 hours x 10 days")


# In[18]:


predictions = predict("2 capsu for 10 day at bed")


# In[19]:


predictions = predict("2 capsu for 10 days at bed")


# In[20]:


predictions = predict("5 days 2 tabs at bed")


# In[21]:


predictions = predict("3 tabs qid x 10 weeks")


# In[22]:


predictions = predict("x 30 days")


# In[23]:


predictions = predict("x 20 months")


# In[24]:


predictions = predict("take 2 tabs po tid for 10 days")


# In[25]:


predictions = predict("take 2 capsules po every 6 hours")


# In[26]:


predictions = predict("inject 2 units pu tid")


# In[27]:


predictions = predict("swallow 3 caps tid by mouth")


# In[28]:


predictions = predict("inject 3 units orally")


# In[29]:


predictions = predict("orally take 3 tabs tid")


# In[30]:


predictions = predict("by mouth take three caps")


# In[31]:


predictions = predict("take 3 tabs orally three times a day for 10 days at bedtime")


# In[32]:


predictions = predict("take 3 tabs orally bid for 10 days at bedtime")


# In[33]:


predictions = predict("take 3 tabs bid orally at bed")


# In[34]:


predictions = predict("take 10 capsules by mouth qid")


# In[35]:


predictions = predict("inject 10 units orally qid x 3 months")


# In[36]:


prediction = predict("please take 2 tablets per day for a month in the morning and evening each day")


# In[37]:


prediction = predict("Amoxcicillin QID 30 tablets")


# In[38]:


prediction = predict("take 3 tabs TID for 90 days with food")


# In[39]:


prediction = predict("with food take 3 tablets per day for 90 days")


# In[40]:


prediction = predict("with food take 3 tablets per week for 90 weeks")


# In[41]:


prediction = predict("take 2-4 tabs")


# In[42]:


prediction = predict("take 2 to 4 tabs")


# In[43]:


prediction = predict("take two to four tabs")


# In[44]:


prediction = predict("take 2-4 tabs for 8 to 9 days")


# In[45]:


prediction = predict("take 20 tabs every 6 to 8 days")


# In[46]:


prediction = predict("take 2 tabs every 4 to 6 days")


# In[47]:


prediction = predict("take 2 tabs every 2 to 10 weeks")


# In[48]:


prediction = predict("take 2 tabs every 4 to 6 days")


# In[49]:


prediction = predict("take 2 tabs every 2 to 10 months")


# In[50]:


prediction = predict("every 60 mins")


# In[51]:


prediction = predict("every 10 mins")


# In[52]:


prediction = predict("every two to four months")


# In[53]:


prediction = predict("take 2 tabs every 3 to 4 days")


# In[54]:


prediction = predict("every 3 to 4 days take 20 tabs")


# In[55]:


prediction = predict("once in every 3 days take 3 tabs")


# In[56]:


prediction = predict("take 3 tabs once in every 3 days")


# In[57]:


prediction = predict("orally take 20 tabs every 4-6 weeks")


# In[58]:


prediction = predict("10 tabs x 2 days")


# In[59]:


prediction = predict("3 capsule x 15 days")


# In[60]:


prediction = predict("10 tabs")

