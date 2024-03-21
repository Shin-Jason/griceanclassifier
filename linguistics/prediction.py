import nltk
from gensim.models import KeyedVectors
from joblib import load
from nltk.tokenize import word_tokenize
import ssl
import numpy as np

#same error as before, something with ssl ceritification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    #internal python handeling environments
    ssl._create_default_https_context = _create_unverified_https_context

#download tokenizer model
nltk.download('punkt')

#load in word2vec model
model_w2v = KeyedVectors.load("word2vec.wordvectors", mmap='r')

#load in the logistic regression models
log_reg_quality = load('log_reg_quality.joblib')
log_reg_quantity = load('log_reg_quantity.joblib')
log_reg_relevance = load('log_reg_relevance.joblib')
log_reg_manner = load('log_reg_manner.joblib')

def sentence_to_vec(sentence, embedding_model):
    #tokenize to lower case
    tokens = word_tokenize(sentence.lower())
    #empty list for vectors
    sentence_vector = []
    for token in tokens:
        if token in embedding_model.key_to_index:
            sentence_vector.append(embedding_model[token])
    if sentence_vector:  #if it's not empty, calculate the mean
        return np.mean(sentence_vector, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)  #if no words, return 0 vector

def predict_maxims(sentence):
    vec = sentence_to_vec(sentence, model_w2v)
    predictions = {
        "Quality": log_reg_quality.predict([vec])[0],
        "Quantity": log_reg_quantity.predict([vec])[0],
        "Relevance": log_reg_relevance.predict([vec])[0],
        "Manner": log_reg_manner.predict([vec])[0],
    }
    return predictions

if __name__ == "__main__":
    while True:
        sentence = input("Enter a sentence to classify (or type 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        predictions = predict_maxims(sentence)
        for maxim, adherence in predictions.items():
            print(f"{maxim}: {'Adheres' if int(adherence) == 1 else 'Does not adhere'}")