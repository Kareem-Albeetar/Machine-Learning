#!/usr/bin/env python
# coding: utf-8

# # Named Entity Recognition
# ## This notebook outlines the concepts involved in building a NER model on CoNLL dataset

# In[1]:


from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import make_scorer,confusion_matrix
from pprint import pprint
from sklearn.metrics import f1_score,classification_report
from sklearn.pipeline import Pipeline
import string


# ### Download the data
# - train.txt --> https://raw.githubusercontent.com/subashgandyer/datasets/main/conll/train.txt
# - test.txt --> https://raw.githubusercontent.com/subashgandyer/datasets/main/conll/test.txt
# - valid.txt --> https://raw.githubusercontent.com/subashgandyer/datasets/main/conll/valid.txt

# In[ ]:


#! wget FILE_URL_TO_DOWNLOAD


# In[2]:


def load__data_conll(file_path):
    myoutput,words,tags = [],[],[]
    fh = open(file_path)
    for line in fh:
        line = line.strip()
        if "\t" not in line:
            #Sentence ended.
            myoutput.append([words,tags])
            words,tags = [],[]
        else:
            word, tag = line.split("\t")
            words.append(word)
            tags.append(tag)
    fh.close()
    return myoutput


# In[3]:


train_path = 'conll/train.txt'
conll_train = load__data_conll(train_path)
print(conll_train)


# In[4]:


test_path = 'conll/test.txt'
conll_test = load__data_conll(test_path)
print(conll_test)


# In[5]:


def sent2feats(sentence):
    feats = []
    sen_tags = pos_tag(sentence) #This format is specific to this POS tagger!
    for i in range(0,len(sentence)):
        word = sentence[i]
        wordfeats = {}
       #word features: word, prev 2 words, next 2 words in the sentence.
        wordfeats['word'] = word
        if i == 0:
            wordfeats["prevWord"] = wordfeats["prevSecondWord"] = "<S>"
        elif i==1:
            wordfeats["prevWord"] = sentence[0]
            wordfeats["prevSecondWord"] = "</S>"
        else:
            wordfeats["prevWord"] = sentence[i-1]
            wordfeats["prevSecondWord"] = sentence[i-2]
        #next two words as features
        if i == len(sentence)-2:
            wordfeats["nextWord"] = sentence[i+1]
            wordfeats["nextNextWord"] = "</S>"
        elif i==len(sentence)-1:
            wordfeats["nextWord"] = "</S>"
            wordfeats["nextNextWord"] = "</S>"
        else:
            wordfeats["nextWord"] = sentence[i+1]
            wordfeats["nextNextWord"] = sentence[i+2]
        
        #POS tag features: current tag, previous and next 2 tags.
        wordfeats['tag'] = sen_tags[i][1]
        if i == 0:
            wordfeats["prevTag"] = wordfeats["prevSecondTag"] = "<S>"
        elif i == 1:
            wordfeats["prevTag"] = sen_tags[0][1]
            wordfeats["prevSecondTag"] = "</S>"
        else:
            wordfeats["prevTag"] = sen_tags[i - 1][1]

            wordfeats["prevSecondTag"] = sen_tags[i - 2][1]
            # next two words as features
        if i == len(sentence) - 2:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
            wordfeats["nextNextTag"] = "</S>"
        elif i == len(sentence) - 1:
            wordfeats["nextTag"] = "</S>"
            wordfeats["nextNextTag"] = "</S>"
        else:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
            wordfeats["nextNextTag"] = sen_tags[i + 2][1]
        #That is it! You can add whatever you want!
        feats.append(wordfeats)
    return feats


# In[6]:


def get_feats_conll(conll_data):
    feats = []
    labels = []
    for sentence in conll_data:
        feats.append(sent2feats(sentence[0]))
        labels.append(sentence[1])
    return feats, labels


# In[7]:


X_train, y_train = get_feats_conll(conll_train)


# In[8]:


X_test, y_test = get_feats_conll(conll_test)


# In[9]:


crf = CRF(algorithm='lbfgs', c1=0.1, c2=10, max_iterations=50)


# In[10]:


crf.fit(X_train, y_train)


# In[11]:


labels = list(crf.classes_)
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))


# In[12]:


y_pred = crf.predict(X_test)


# In[13]:


print(f"F-Score = {metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)}")


# In[14]:


print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))


# In[15]:


def get_confusion_matrix(y_true,y_pred,labels):
    trues,preds = [], []
    for yseq_true, yseq_pred in zip(y_true, y_pred):
        trues.extend(yseq_true)
        preds.extend(yseq_pred)
    print_confusion_matrix(confusion_matrix(trues,preds,labels),labels)


# In[16]:


def print_confusion_matrix(cm, labels):
    print("\n")
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        sum = 0
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            sum =  sum + int(cell)
            print(cell, end=" ")
        print(sum) #Prints the total number of instances per cat at the end.


# In[17]:


get_confusion_matrix(y_test, y_pred,labels=sorted_labels)


# In[18]:


def train_seq(X_train, y_train, X_test, y_test):
    ## CRF model
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=10, max_iterations=50)
    ## Fit the model
    crf.fit(X_train, y_train)
    ## Get the labels
    labels = list(crf.classes_)
    ## Sort the labels
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    ## Predict on testing data
    y_pred = crf.predict(X_test)
    ## F-Score
    print(f"F-Score = {metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)}")
    ## Classification Report
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    ## Plot Confusion matrix
    get_confusion_matrix(y_test, y_pred,labels=sorted_labels)


# In[19]:


def one_function():
    ## Load the data
    train_path = 'conll/train.txt'
    test_path = 'conll/test.txt'
    conll_train = load__data_conll(train_path)
    conll_test = load__data_conll(test_path)
    ## Extract the features
    feats, labels = get_feats_conll(conll_train)
    testfeats, testlabels = get_feats_conll(conll_test)
    ## Train the model
    train_seq(feats, labels, testfeats, testlabels)


# In[20]:


one_function()

