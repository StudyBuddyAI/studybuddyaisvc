import numpy as np
import pandas as pd
import scipy as sp

import argparse
from allennlp.service.predictors import DemoModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string
import nltk
import re

import common

class StudyBuddyAI:
    # Logger
    logger = common.Logger()
    bidaf_model = None;
    predictor = None;

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = None;
    context_list =  None;

    all_tfidf_vectorizer = TfidfVectorizer()
    all_tfidf_matrix = None;
    all_context_list = None;

    # Trained Models
    trained_models = [
        { 'name': 'Base Model (9/15/2017)','path': '../../allennlp/train_out/bidaf-model-2017.09.15-charpad.tar.gz'}
        , {'name': 'ReTrained Model 1 (12/9/2017)', 'path': '../../allennlp/train_out/model01.tar.gz'}
        , {'name': 'ReTrained Model 2 (12/10/2017)', 'path': '../../allennlp/train_out/model02.tar.gz'}
        , {'name': 'ReTrained Model 3 (12/11/2017)', 'path': '../../allennlp/train_out/model03.tar.gz'}
        , {'name': 'ReTrained Model 4 (12/12/2017)', 'path': '../../allennlp/train_out/model04.tar.gz'}
        , {'name': 'ReTrained Model 5 (12/13/2017)', 'path': '../../allennlp/train_out/model05.tar.gz'}
    ]

    # Class StudyBuddyAI
    def __init__(self):
        self.logger.log("StudyBuddyAI ------------------------------------ Init")
        # Load pretrained model
        self.load_trained_model('../../allennlp/train_out/model04.tar.gz')

    def get_trained_model_list(self):
        return self.trained_models;

    def load_trained_model(self,path):
        self.logger.log("Loading model: " + path)
        self.bidaf_model = DemoModel(path, 'machine-comprehension')
        # predictor
        self.predictor = self.bidaf_model.predictor()


    def load_tfidf_vectorizer(self,context_list,all=False):

        corpus = list()
        if all == True:
            self.all_context_list = context_list
        else:
            self.context_list = context_list;

        for context in context_list:
            # Tokenize
            tokens = self.tokenize_text(context)
            cleaned_context_text = ' '.join(tokens)
            corpus.append(cleaned_context_text)

        # Tf–idf term weighting using TfidfVectorizer
        if all == True:
            self.all_tfidf_matrix = self.all_tfidf_vectorizer.fit_transform(corpus)
        else:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

    def predict_from_passage(self,data):
        prediction = self.predictor.predict_json(data)
        self.logger.log(prediction);
        return prediction

    def predict_for_title(self,question,all=False):
        # Tokenize
        tokens = self.tokenize_text(question)
        cleaned_context_text = ' '.join(tokens)
        if all == False:
            question_vector = self.tfidf_vectorizer.transform([cleaned_context_text])
        else:
            question_vector = self.all_tfidf_vectorizer.transform([cleaned_context_text])

        # Find Cosine Similarity of question with the contexts
        if all == False:
            cs = cosine_similarity(question_vector, self.tfidf_matrix)
        else:
            cs = cosine_similarity(question_vector, self.all_tfidf_matrix)
        #self.logger.log(cs)

        cs_list = cs[0]
        passage = ''
        idx = 0
        threshold = 0.25
        min_value = min(i for i in cs_list if i > 0.0)
        max_value = max(cs_list)
        range = max_value - min_value
        threshold = max_value - range/3
        for cs_val in cs_list:
            if cs_val >= threshold:
                if all == False:
                    passage = passage + self.context_list[idx] + ' '
                else:
                    passage = passage + self.all_context_list[idx] + ' '

            idx = idx + 1;

        data = {}
        data['question'] = question
        data['passage'] = passage

        result = {}
        result['prediction'] = self.predict_from_passage(data)
        result['passage'] = passage

        return result


    # Helper Methods
    # Tokenize text using NLTK
    def tokenize_text(self,text, remove_stop_words=True, stem_words=True, filter_short_token=1):  # split into words
        words = nltk.word_tokenize(text)
        # convert to lower case
        words = [w.lower() for w in words]
        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))  # remove punctuation from each word
        tokens = [re_punc.sub('', w) for w in words]
        # remove not alphabets
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        if remove_stop_words == True:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
        # Perfomring
        if stem_words == True:
            # stemming of words
            porter = nltk.stem.porter.PorterStemmer()
            tokens = [porter.stem(word) for word in tokens]
        # filter out short tokens
        if filter_short_token > 0:
            tokens = [word for word in tokens if len(word) > filter_short_token]
        return tokens