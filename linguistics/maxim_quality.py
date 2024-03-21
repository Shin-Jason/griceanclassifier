# comments for these .py file in maxim_relevance.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
import ssl
#from joblib import dump, load
#from sklearn.externals import joblib
import joblib

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#download tokenizer models
nltk.download('punkt')

#load in wordvec model
model_w2v = KeyedVectors.load("word2vec.wordvectors", mmap='r')

def sentence_to_vec(sentence, embedding_model):
    tokens = word_tokenize(sentence.lower())
    sentence_vector = []
    for token in tokens:
        if token in embedding_model.key_to_index:
            sentence_vector.append(embedding_model[token])
    if sentence_vector:
        return np.mean(sentence_vector, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)

df = pd.read_csv('datawords.csv', on_bad_lines='skip')

df['Sentence_Vector'] = df['Sentence'].apply(lambda x: sentence_to_vec(x, model_w2v))

train_df = df.iloc[:-30]
test_df = df.iloc[-30:]
train_df = train_df[train_df['Sentence_Vector'].map(lambda x: x.size) > 0]
test_df = test_df[test_df['Sentence_Vector'].map(lambda x: x.size) > 0]

X_train = np.stack(train_df['Sentence_Vector'].values)
y_train = train_df['Maxim_of_Quality'].values

X_test = np.stack(test_df['Sentence_Vector'].values)
y_test = test_df['Maxim_of_Quality'].values

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

joblib.dump(log_reg, 'log_reg_quality.joblib')

y_pred = log_reg.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))

