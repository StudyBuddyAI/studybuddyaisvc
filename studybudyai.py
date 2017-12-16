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

    # Context Memory Settings
    context_memory_time = 1 # in minutes
    context_memory_size = 5
    context_memory = []
    context_qa = []

    # Class StudyBuddyAI
    def __init__(self):
        self.logger.log("StudyBuddyAI ------------------------------------ Init")
        # Load pretrained model
        self.load_trained_model('../../allennlp/train_out/model05.tar.gz')

    def get_trained_model_list(self):
        return self.trained_models;

    def load_trained_model(self,path):
        self.logger.log("Loading model: " + path)
        self.bidaf_model = DemoModel(path, 'machine-comprehension')
        # predictor
        self.predictor = self.bidaf_model.predictor()

    def save_in_context_memory(self,context):
        # Save the context
        self.context_memory.insert(0, context)
        if len(self.context_memory) > self.context_memory_size:
            # ensure our context list is limited
            self.context_memory = self.context_memory[:self.context_memory_size]

    def save_qa_in_context_memory(self,qa):
        # Save the context
        self.context_qa.insert(0, qa)

    def clear_context_memory(self):
        self.context_memory = []
        self.context_qa = []

    def get_context_memory(self):
        return {'context_memory':self.context_memory,'context_qa':self.context_qa};

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

    def predict_for_title(self,question,all=False,check_context=False):

        passage = ''
        current_context_list = []
        current_context_start_index = []
        current_context_end_index = []

        # if we need to look at the context only
        if (check_context == True) and (len(self.context_memory) > 0):
            # the top context item
            current_context_list = self.context_memory[:1]
        else:
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
            self.logger.log(cs)

            cs_list = cs[0]
            idx = 0
            threshold = 0.25

            values_greater_than_zero = [i for i in cs_list if i > 0.0]
            if len(values_greater_than_zero) == 0:
                return {'status':0}
            #     for ctx in self.context_memory:
            #         current_context_start_index.append(len(passage))
            #         passage = passage + ctx + ' '
            #         current_context_list.append(ctx)
            #         current_context_end_index.append(len(passage))
            # else:

            min_value = min(values_greater_than_zero)
            max_value = max(cs_list)
            range = max_value - min_value
            threshold = max_value - range/3

            for cs_val in cs_list:
                if cs_val >= threshold:
                    if all == False:
                        current_context_list.append(self.context_list[idx])
                    else:
                        current_context_list.append(self.all_context_list[idx])

                idx = idx + 1;

        # build passage
        for txt in current_context_list:
            current_context_start_index.append(len(passage))
            passage = passage + txt + ' '
            current_context_end_index.append(len(passage))

        data = {}
        data['question'] = question
        data['passage'] = passage

        # Build the return object
        result = {}
        result['status'] = 1
        result['prediction'] = self.predict_from_passage(data)
        result['current_context_list'] = current_context_list

        # print(current_context_start_index)
        # print(current_context_end_index)
        # print(current_context_list)
        # print(passage)

        # Save the context from which answer was predicted from
        # best_span = result['prediction']['best_span']
        # for idx, ctx in enumerate(current_context_end_index):
        #     if (best_span[0] >= current_context_start_index[idx]) and (best_span[1] <= current_context_end_index[idx]):
        #         self.save_in_context_memory(current_context_list[idx],{'question':question,'answer':result['prediction']['best_span_str']})
        #         result['current_context'] = current_context_list[idx]
        #         continue;

        best_span_str = result['prediction']['best_span_str']
        for ctx in current_context_list:
            if best_span_str in ctx:
                self.save_in_context_memory(ctx)
                self.save_qa_in_context_memory({'question': question, 'answer': best_span_str})
                result['current_context'] = ctx
                continue;

        # return the current context memory
        result['context_memory'] = self.context_memory
        result['context_qa'] = self.context_qa

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