#!/usr/bin/env python3

import requests
import time
import argparse
import os
import json
from requests.compat import urljoin
import gensim
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin

class BotHandler(object):
    """
        BotHandler is a class which implements all back-end of the bot.
        It has three main functions:
            'get_updates' — checks for new messages
            'send_message' – posts new message to user
            'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager):
    	# Put the Telegram Access token here
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []

        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question):
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        return self.dialogue_manager.generate_answer(question)


def is_unicode(text):
    return len(text) == len(text.encode())



# We will need this function to prepare text at prediction time
def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()

# need this to convert questions asked by user to vectors
def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    word_tokens = question.split(" ")
    question_len = len(word_tokens)
    question_mat = np.zeros((question_len,dim), dtype = np.float32)
    
    for idx, word in enumerate(word_tokens):
        if word in embeddings:
            question_mat[idx,:] = embeddings[word]
            
    # remove zero-rows which stand for OOV words       
    question_mat = question_mat[~np.all(question_mat == 0, axis = 1)]
    
    # Compute the mean of each word along the sentence
    if question_mat.shape[0] > 0:
        vec = np.array(np.mean(question_mat, axis = 0), dtype = np.float32).reshape((1,dim))
    else:
        vec = np.zeros((1,dim), dtype = np.float32)
        
    return vec

class SimpleDialogueManager(object):
    """
    This is a simple dialogue manager to test the telegram bot.
    The main part of our bot will be written here.
    """

    def __init__(self):

        # Instantiate all the models and TFIDF Objects.
        print("Loading resources...")
        # Instantiate a Chatterbot for Chitchat type questions
        from chatterbot import ChatBot
        from chatterbot.trainers import ChatterBotCorpusTrainer
        chatbot = ChatBot('MLWhizChatterbot')
        trainer = ChatterBotCorpusTrainer(chatbot)
        trainer.train('chatterbot.corpus.english')
        self.chitchat_bot = chatbot
        print("Loading Word2vec model...")
        # Instantiate the Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
        print("Loading Classifier objects...")
        # Load the intent classifier and tag classifier
        self.intent_recognizer =  pickle.load(open('resources/intent_clf.pkl', 'rb'))
        self.tag_classifier =  pickle.load(open('resources/tag_clf.pkl', 'rb'))
        # Load the TFIDF vectorizer object
        self.tfidf_vectorizer = pickle.load(open('resources/tfidf.pkl', 'rb'))
        print("Finished Loading Resources")

    # We created this function just above. We just need to have a function to get most similar question's *post id* in the dataset given we know the programming Language of the question. Here it is:
    def get_similar_question(self,question,tag):
        # get the path where all question embeddings are kept and load the post_ids and post_embeddings
        embeddings_path = 'resources/embeddings_folder/' + tag + ".pkl"
        post_ids, post_embeddings = pickle.load(open(embeddings_path, 'rb'))
        # Get the embeddings for the question
        question_vec = question_to_vec(question, self.model, 300)
        # find index of most similar post
        best_post_index = pairwise_distances_argmin(question_vec,
                                                    post_embeddings)
        # return best post id
        return post_ids[best_post_index]

    def generate_answer(self, question): 
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        # find intent
        intent = self.intent_recognizer.predict(features)[0]
        # Chit-chat part:   
        if intent == 'dialogue':
            response = self.chitchat_bot.get_response(question)
        # Stack Overflow Question
        else:
            # find programming language
            tag = self.tag_classifier.predict(features)[0]
            # find most similar question post id
            post_id = self.get_similar_question(question,tag)[0]
            # respond with 
            response = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s' % (tag, post_id)
        return response

def main():
    token = '839585958:AAEfTDo2X6PgHb9IEdb62ueS4SmdpCkhtmc'
    simple_manager = SimpleDialogueManager()
    bot = BotHandler(token, simple_manager)
    ###############################################################

    print("Ready to talk!")
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print("An update received.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    if is_unicode(text):
                        print("Update content: {}".format(update))
                        bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Hmm, you are sending some weird characters to me...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)

if __name__ == "__main__":
    main()
