from pydantic import BaseModel
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json


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
