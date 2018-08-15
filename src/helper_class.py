# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:04:39 2018

@author: Ashutosh Kothiwala
"""
############################
#-------- Imports -----------
############################
from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os

import itertools
import datetime


#############################
#------- Get the Data -------
#############################

def getTrainingData():
    TRAIN_CSV = os.getcwd() + '\\train.csv'
    # Load data
    data = pd.read_csv(TRAIN_CSV)
    return data

current_dir = os.getcwd()
EMBEDDING_FILE = current_dir + '\\glove.42B.300d\\glove.42B.300d.txt'

# Function to read Glove Vectors from stanford dataset
def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding="utf8") as f:
        word_to_vec_map = {}
        for line in f:
            line = line.split()
            word_to_vec_map[line[0]] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map

class preprocessing(object):
    def __init__(self):
        self.count = 0
        self.vocabulary = dict() # the idea is to represent each key (word) by an integer value
        
    # Maximum time is consumed in this function in data preprocessing
    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()
    
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
    
        text = text.split()
    
        return text
    
    def processSentence(self, sentence, stopwords):
        q = []  # Representation of a question as list of mapped indices
        for word in text_to_word_list(sentence):

            # Check for stopwords and autocorrect its spelling if required
            if word not in word_to_vec_map:
                if word in stopwords:
                    continue
                else:
                    word = TextBlob(word).correct()

            
            # If word is not in the vocabulary's keys then add it to it
            # Assign its value to q and append the count
            if word not in vocabulary:
                vocabulary[word] = self.count
                q.append(self.count)
                self.count += 1
            else:
                q.append(vocabulary[word])
        return q
    
    def processDataset(self,data):
        if os.path.isfile("../pickles/processed_data.pkl"):
            return pd.read_pickle("processed_data.pkl")
        else:
            # Function takes a lot of time to run
            # Uncomment the the if code if you wish to log the progress
            print("Total queries in the dataset are " + len(data))
            count1 = 0
            stops = set(stopwords.words('english')) # Stop words in any questions won't be taken in to account
    
            #import time
            #start_time = time.time()
            # Iterate over each question from entire dataset
            for index, row in data.iterrows():
                for question in 'question1', 'question2':
                    # Process question and represent as an array of numbers
                    processedQuestion = self.processSentence(row[question], stops)
                    # Replace questions with its processed representation
                    data.set_value(index, question, processedQuestion)
                count1 += 1
                
                #if(count1%1000 == 0):
                #    print(count1, "Time required: ", time.time() - start_time)
                #    start_time = time.time()
            processedData_path = "../pickles/processed_data.pkl"
            pd.to_pickle(data,processedData_path)
            return data
        

# Replace questions as word to question as number representation
data.set_value(index, question, q)


vocabulary = dict()                     # the idea is to represent each key (word) by an integer value
stops = set(stopwords.words('english')) # Stop words in any questions won't be taken in to account
count = 0
count1 = 0
import time
start_time = time.time()
# Iterate over each question from entire dataset
for index, row in data.iterrows():
    for question in 'question1', 'question2':
        # Process question and represent as an array of numbers
        processedQuestion = self.processSentence(row[question], stops)
        # Replace questions with its processed representation
        data.set_value(index, question, processedQuestion)
    count1 += 1
    if(count1%1000 == 0):
        print(count1, "Time required: ", time.time() - start_time)
        start_time = time.time()
