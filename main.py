import http
from time import sleep, time
from fastapi import FastAPI
import uvicorn
import numpy as np
import os
import pickle
import nltk
import requests


from utils import Answer,Question,get_response,get_response_v3,get_numerical_representation_of_words

from tensorflow.keras.models import load_model

app = FastAPI()

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
MODEL = load_model(os.path.join(os.curdir,'./saved_models/chatbot_model_v2'))
MODEL_V3 = load_model(os.path.join(os.curdir,'./saved_models/chatbot_model_v3'))

WORDS = pickle.load(open('./data/words.pkl','rb'))
WORDS_V3 = pickle.load(open('./data/words_v3.pkl','rb'))
CLASSES = pickle.load(open('./data/classes.pkl','rb'))
CLASSES_V3 = pickle.load(open('./data/classes_v3.pkl','rb'))


def predict(sentence):
    rep = get_numerical_representation_of_words(sentence,WORDS=WORDS)
    prediction = MODEL.predict(np.array([rep]))[0]
    THRESHOLD  = 0.0025
    results = [[i,r] for i,r in enumerate(prediction) if r > THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({ 'intent': CLASSES[r[0]] , 'probability' : str(r[1]) })
    return return_list 

def predict_v3(sentence):
    rep = get_numerical_representation_of_words(sentence,WORDS=WORDS_V3)
    prediction = MODEL_V3.predict(np.array([rep]))[0]
    THRESHOLD  = 0.0025
    results = [[i,r] for i,r in enumerate(prediction) if r > THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({ 'intent': CLASSES_V3[r[0]] , 'probability' : str(r[1]) })
    return return_list 



 
 
@app.get('/')
async def ping():
    return 'ping'



@app.post('/question')
async def get_answer(data:Question):
    sentence = data.question; 
    response = get_response(predict(sentence)[0]['intent'])
    return Answer(response=response)


@app.post('/v3/question')
async def get_answer_v3(data:Question):
    sentence = data.question; 
    response = get_response_v3(predict_v3(sentence)[0]['intent'])
    return Answer(response=response)


def keep_server_alive():
    while True:
        requests.get('https://chat-bot-99.herokuapp.com/pin')
        sleep(120)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=os.environ.get("PORT",8000))
    keep_server_alive()