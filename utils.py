from datetime import datetime
import os
from pydantic import BaseModel
from pymongo import MongoClient
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from dotenv import load_dotenv


class UnasweredQuestion(BaseModel):
    question:str
    createdAt:datetime

class Answer(BaseModel):
    response:str
    

class Question(BaseModel):
    question:str    
    
    
def clean_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def get_numerical_representation_of_words(sentence,WORDS):
    sentence_words = clean_sentence(sentence)
    result = [0] * len(WORDS)
    for word in sentence_words:
        for index,w in enumerate(WORDS):
            if word == w:
                result[index] = 1
    return np.array(result)



# def get_numerical_representation_of_words_v3(sentence,WORDS):
#     sentence_words = clean_sentence(sentence)
#     result = [0] * len(WORDS)
#     for word in sentence_words:
#         for index,w in enumerate(WORDS):
#             if word == w:
#                 result[index] = 1
#     return np.array(result)


def save_question_to_db(ques: UnasweredQuestion):
    load_dotenv()
    DB_URL = os.environ['DB_URL']
    DB_NAME = os.environ['DB_NAME']
    COLLECTION_NAME = os.environ['COLLECTION_NAME']
    with MongoClient(DB_URL) as client:
        col = client[DB_NAME][COLLECTION_NAME]
        res = col.insert_one(ques.dict())
        return res



def get_response(predicted_class):
    with open('./data/Intent.json','rb') as file:
        intents = json.loads(file.read())['intents']
        response = None
        for intent in intents:
            if intent['intent'] ==  predicted_class:
                response = random.choice(intent['responses'])
                break 
        return response


def get_response_v3(predicted_class):
    with open('./data/Intent_v3.json','rb') as file:
        intents = json.loads(file.read())['intents']
        response = None
        for intent in intents:
            if intent['intent'] ==  predicted_class:
                response = random.choice(intent['responses'])
                break 
        return response
