#!/usr/bin/env python
# coding: utf-8

# # NLP Pipeline
# ## This notebook outlines the main concepts and phases involved in NLP pipeline

# ![NLP Pipeline](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Pipeline.png)

# ![Source formats](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Text-formats.png)

# ### Web scraping
# Scrape the following url and extract the text
# 
# URL: https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
# 
# 
# ![Stack overflow page](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Stackoverflow.png)
# 
# 
# 
# Task
# - look at the url
# - extract question
# - extract answer
# - Display them as shown below
# 
# This is **Text extraction from webpages**

# In[3]:


from bs4 import BeautifulSoup
from urllib.request import urlopen
myurl = "https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python"
html = urlopen(myurl).read()
soupified = BeautifulSoup(html, "html.parser")
question_text = soupified.find("div", {"class": "s-prose js-post-body"})
print(f"Question = \n {question_text.get_text().strip()}")
print("\n\n\n")

answer_text = soupified.find("div", {"class": "answer"})
answer = answer_text.find("div", {"class": "s-prose js-post-body"})
print(f"Answer = \n {answer.get_text().strip()}")


# Task is to extract text from this url: https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_scanned_image.png
# 
# Input:
# 
# ![Scanned Image](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_scanned_image.png)
# 
# 
# Output:
# 
# 
# ’in the nineteenth century the only Kind of linguistics considered\nseriously
# was this comparative and historical study of words in languages\nknown or
# believed to Fe cognate—say the Semitic languages, or the Indo-\nEuropean
# languages. It is significant that the Germans who really made\nthe subject what
# it was, used the term Indo-germanisch. Those who know\nthe popular works of 
# Otto Jespersen will remember how fitmly he\ndeclares that linguistic 
# science is historical. And those who have noticed’

# In[7]:


from PIL import Image
from pytesseract import image_to_string
filename = "NLP_scanned_image.png"
text = image_to_string(Image.open(filename))
print(text)


# In[11]:


text = "I love this!!! 😊  Let's all be happy !😊"


# In[12]:


cleaned_text = text.encode("utf-8")


# In[13]:


cleaned_text


# In[17]:


from textblob import TextBlob
 
a = "cmputr"           
print(f"Original text: {str(a)}")
 
b = TextBlob(a)
 
print(f"corrected text: {str(b.correct())}")

c = str(TextBlob(a).correct())
c


# In[21]:


from spellchecker import SpellChecker
 
spell = SpellChecker()
 
# find those words that may be misspelled
misspelled = spell.unknown(["cmputr", "watr", "study", "wrte"])
 
for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))
 
    # Get a list of `likely` options
    print(spell.candidates(word))


# In[ ]:


import requests
import json

api_key = "<ENTER-KEY-HERE>"
example_text = "Hollo, wrld" 

data = {'text': example_text}
params = {
    'mkt':'en-us',
    'mode':'proof'
    }
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Ocp-Apim-Subscription-Key': api_key,
    }
response = requests.post(endpoint, headers=headers, params=params, data=data)
json_response = response.json()
print(json.dumps(json_response, indent=4))


# In[22]:


mytext = """In the previous chapter, we saw examples of some common NLP applications that we might encounter in everyday life. If we were asked to build such an application, think about how we would approach doing so at our organization. We would normally walk through the requirements and break the problem down into several sub-problems, then try to develop a step-by-step procedure to solve them. Since language processing is involved, we would also list all the forms of text processing needed at each step. This step-by-step processing of text is known as pipeline. """


# In[27]:


from nltk.tokenize import sent_tokenize
my_sentences = sent_tokenize(mytext)
for idx, sent in enumerate(my_sentences):
    print(f"Sentence {idx+1} \n {sent}\n\n")


# In[28]:


sentence = "This step-by-step processing of text is known as pipeline."


# In[29]:


from nltk.tokenize import word_tokenize
print(word_tokenize(sentence))


# In[30]:


from nltk.corpus import stopwords
from string import punctuation
from nltk import sent_tokenize, word_tokenize


# In[48]:


stop_words = set(stopwords.words('english'))
filtered = [word for word in word_tokenize(texts[0]) if word not in stop_words]
print(f"Sentence = {texts[0]}\n\n")
print(f"Cleaned text = {filtered}\n")


# In[33]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
word1, word2 = "cars", "revolution" 
print(stemmer.stem(word1), stemmer.stem(word2))


# In[34]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better", pos="a"))


# In[53]:


word_list = ['well-dressed', 'airliner', 'better', 'was', 'meeting', 'uncomfortable']


# In[54]:


print("Stemmer vs Lemmatizer results\n")
for word in word_list:
    print(f" Stem of {word} = {stemmer.stem(word)}")
    print(f" Lemma of {word} = {lemmatizer.lemmatize(word, pos='a')}\n")


# In[93]:


test_sentence = "Everything we’re doing now is great. However, we don't want to relax now. And this isn't the time to relax at all."


# In[90]:


import re
pattern = r'we[\’\']re'
replacement = 'we are'
expanded_sentence = re.sub(pattern,replacement,test_sentence)
print(expanded_sentence)


# In[94]:


pattern = r'\w[\’\']re'
replacement = ' are'
expanded_sentence = re.sub(pattern,replacement,test_sentence)
print(expanded_sentence)


# In[95]:


pattern = r'\w[\’\']t'
replacement = ' not'
expanded_sentence = re.sub(pattern,replacement,test_sentence)
print(expanded_sentence)


# In[98]:


import spacy


# In[99]:


nlp = spacy.load("en_core_web_sm")


# In[100]:


text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")


# In[101]:


doc = nlp(text)


# In[109]:


[chunk.text for chunk in doc.noun_chunks]


# In[108]:


[token.lemma_ for token in doc if token.pos_ == "VERB"]


# In[107]:


[token.lemma_ for token in doc if token.pos_ == "ADJ"]


# In[104]:


for entity in doc.ents:
    print(entity.text, entity.label_)


# In[110]:


for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text)


# In[111]:


for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])


# In[112]:


from spacy import displacy
displacy.render(doc, style='dep')


# In[113]:


for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


# In[114]:


displacy.render(doc, style='ent')


# In[1]:


import spacy
import neuralcoref


# In[3]:


nlp = spacy.load('en_core_web_sm')


# In[4]:


neuralcoref.add_to_pipe(nlp)


# In[5]:


elon_text = """Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada when he was 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received dual bachelor's degrees in economics and physics. He moved to California in 1995 to begin a Ph.D. in applied physics and material sciences at Stanford University but dropped out after two days to pursue a business career, co-founding web software company Zip2 with his brother Kimbal. The start-up was acquired by Compaq for $307 million in 1999. Musk co-founded online bank X.com that same year, which merged with Confinity in 2000 to form the company PayPal and was subsequently bought by eBay in 2002 for $1.5 billion."""


# In[ ]:


doc = nlp(elon_text)


# In[ ]:


resolved_doc = doc._.coref_resolved
print(resolved_doc)


# #### Classical NLP
# ![Classical NLP](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Classical_FE.png)

# ### Deep Learning NLP
# ![DL NLP](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_DL_FE.png)

# - Feature Extraction happens automatically as part of the model training process
# - **Neurons** extract features

# ![Modeling Principles](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Modeling.png)

# #### Intrinsic Evaluation
# 
# ![Intrinsic Evaluation_1](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Intrinsic_Evaluation1.png)
# 
# 
# ![Intrinsic Evaluation_2](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Intrinsic_Evaluation2.png)
# 
# ![Intrinsic Evaluation_3](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Intrinsic_Evaluation3.png)

# ![Monitoring](https://raw.githubusercontent.com/subashgandyer/datasets/main/images/NLP_Monitoring.png)
