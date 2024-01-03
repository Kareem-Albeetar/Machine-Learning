#!/usr/bin/env python
# coding: utf-8

# # FAQ Chatbot
# 

# In[ ]:


import re, math
from collections import Counter


# In[10]:


CORPUS = [
    {
        "Question": "Am I eligible for financial Aid?",
        "Answer": "Virtually every student qualifies for some type of financial aid. See our HOW TO APPLY information and visit the PAIS (Preliminary Aid Information System) web site for the financial aid eligibility worksheet."
    },
    {
        "Question": "What is an Academic Year?",
        "Answer": "The Academic Year begins with the fall semester. For example: the fall 2014 + spring 2015 semesters = one academic year."
    },
    {
        "Question": "Do I need to wait for a notice of admission to apply for financial aid?",
        "Answer": "No. Submit the FAFSA by the priority processing deadline. Typically, financial aid applications should be submitted during the January preceding your enrollment. You will need to be admitted, however, before financial aid will be awarded."
    },
    {
        "Question": "I applied for financial aid . Why did you give me a loan?",
        "Answer": "Loans are a form of financial aid; in fact, subsidized (interest paid while in college) loans comprise a major portion of the financial aid program."
    },
    {
        "Question": "Are my parent's required to pay the Expected Family Contribution (EFC) to the school?",
        "Answer": "No. The Expected Family Contribution is a federal calculation that determines the amount the family (parent(s) and/or student) should pay toward the educational costs. The calculation examines the contribution by assessing prior year earnings, savings, etc. If the student or parent cannot meet the EFC because of a change in prior year income or other expenses such as medical or elementary school tuition, the student should contact a financial aid representative with this information."
    },
    {
        "Question": "If I am not eligible for a Pell Grant, are there other types of financial aid I can receive?",
        "Answer": "Yes. Most students can apply for low interest loans. If you submit the FAFSA in a timely manner (early in the application cycle), the school may consider you for institutional aid and University grants. Grant funds are not endless -- they are limited to monies in these fund accounts. Grants are awarded until fund accounts are drained. Be prepared and file your FAFSA early!"
    },
    {
        "Question": "Can I be independnet if my parents do not carry me as a tax exemption? If they are unwilling to pay for college? If I am living on my own? If I live with my grandparent (or other relative) who is not my legal guardian?",
        "Answer": "You must be 24-years or older prior to the first day of the calendar year of the award year; or a veteran, or married, or have legal dependents other than a spouse, or be an orphan/ward of the court; or have extenuating circumstances in which a financial aid counselor deems acceptable to warrant a professional judgment override."
    },
    {
        "Question": "Are students who report parental data automatically ineligible?",
        "Answer": "No, it depends on the parents household size, number in college, income, assets and other factors."
    },
    {
        "Question": "Does a Pell Grant cover California State University, Fresno fees?",
        "Answer": "Sometimes. The amount of Pell Grant you receive will depend on your eligibility (as calculated by federal standards) and on your student enrollment status (attending and semester units enrolled)."
    },
    {
        "Question": "Why are students offered so many loans?",
        "Answer": "Stafford/Direct Loans are entitlements (your prerogative) so funding is always available. Other assistance programs, like Grants ( excluding the Pell Grant) have limited funding available per award year."
    },
    {
        "Question": "What special circumstances are considered to revise my income?",
        "Answer": "Common circumstances include loss of employment or non-taxable income, death of one or both parents, and natural disasters. Other cases should be discussed with a financial aid representative."
    },
    {
        "Question": "What is verification and why are students selected for verification?",
        "Answer": "Verification is the process where documents are audited for completeness and accuracy leading to a financial aid award. Students are selected by the U.S. Department of Education. While others are selected by the school based upon quality control measures implemented at the campus level. Documents required are those used to complete the initial application for aid"
    },
    {
        "Question": "Who is eligible for in-state tuition, fee waivers, and scholarships in California?",
        "Answer": "The chart provided can help you determine eligibility."
    }
]


# In[11]:


WORD = re.compile(r'\w+')


# In[12]:


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# In[13]:


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


# In[14]:


def compare_similarity(sent_one, sent_two):
    vector1 = text_to_vector(sent_one.lower())
    vector2 = text_to_vector(sent_two.lower())
    
    return get_cosine(vector1, vector2)


# In[15]:


def find_most_similar(sent):
    max = {"answer": None, "score": 0, "question": None}

    for each in CORPUS:
        score = compare_similarity(sent, each['Question'])
        if score > max['score']:
            max['score'] = score
            max['answer'] = each['Answer']
            max['question'] = each['Question']
    return {"score": max['score'], "answer": max['answer'], "question": max['question']}


# In[16]:


class Bot:

    def __init__(self):
        self.event_stack = []
        self.settings = {
            "min_score": 0.2,
            "help_email": "god@iamgod.com",
            "faq_page": "www.ItsAnAmazingFAQChatbot.com"
        }

        print("Ask a question:")
        while(True):
            self.allow_question()

    def allow_question(self):
        # Check for event stack
        potential_event = None
        if(len(self.event_stack)):
            potential_event = self.event_stack.pop()
        if potential_event:
            text = input("Response: ")
            potential_event.handle_response(text, self)
        else:
            text = input("Question: ")
            answer = self.pre_built_responses_or_none(text)
            if not answer:
                answer = find_most_similar(text)
                self.answer_question(answer, text)

    def answer_question(self, answer, text):
        if answer['score'] > self.settings['min_score']:
            # set off event asking if the response question is what they were looking for
            print("\nBest-fit question: %s (Score: %s)\nAnswer: %s\n" % (answer['question'],
                                                                          answer['score'],
                                                                          answer['answer']))
        else:
            print("Woops! I'm having trouble finding the answer to your question. " \
                  "Would you like to see the list of questions that I am able to answer?\n")
            # set off event for corpus dump
            self.event_stack.append(Event("corpus_dump", text))

    def pre_built_responses_or_none(self, text):
        # only return answer if exact match is found
        pre_built = [
            {
                "Question": "Who made you?",
                "Answer": "I was created by GOD.\n"
            },
            {
                "Question": "When were you born?",
                "Answer": "I first opened my eyes in alpha stage January 9th, 2020.\n"
            },
            {
                "Question": "What is your purpose?",
                "Answer": "I assist user experience by providing an interactive FAQ chat.\n"
            },
            {
                "Question": "Thanks",
                "Answer": "Glad I could help!\n"
            },
            {
                "Question": "Thank you",
                "Answer": "Glad I could help!\n"
            }
        ]
        for each_question in pre_built:
            if each_question['Question'].lower() in text.lower():
                print(each_question['Answer'])
                return each_question


    def dump_corpus(self):
        question_stack = []
        for each_item in CORPUS:
            question_stack.append(each_item['Question'])
        return question_stack


# In[17]:


class Event:

    def __init__(self, kind, text):
        self.kind = kind
        self.CONFIRMATIONS = ["yes", "sure", "okay", "that would be nice", "yep"]
        self.NEGATIONS = ["no", "don't", "dont", "nope"]
        self.original_text = text

    def handle_response(self, text, bot):
        if self.kind == "corpus_dump":
            self.corpus_dump(text, bot)

    def corpus_dump(self, text, bot):
        for each_confirmation in self.CONFIRMATIONS:
            for each_word in text.split(" "):
                if each_confirmation.lower() == each_word.lower():
                    corpus = bot.dump_corpus()
                    corpus = ["-" + s for s in corpus]
                    print("%s%s%s" % ("\n", "\n".join(corpus), "\n"))
                    return 0
        for each_negation in self.NEGATIONS:
            for each_word in text.split(" "):
                if each_negation.lower() == each_word.lower():
                    print("Feel free to ask another question or send an email to %s.\n" % bot.settings['help_email'])
                    bot.allow_question()
                    return 0
        # base case, no confirmation or negation found
        print("I'm having trouble understanding what you are saying. At this time, my ability is quite limited and I am still learning from you. " \
              "Please refer to %s or email %s if I was not able to answer your question correctly. " \
              "For convenience, a google link has been generated below for your perusal: \n%s\n" % (bot.settings['faq_page'],
                                                                                 bot.settings['help_email'],
                                                                                 "https://www.google.com/search?q=%s" %
                                                                                 ("+".join(self.original_text.split(" ")))))
        return 0


# In[ ]:


Bot()

