import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
import ssl
import joblib

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

#load in the model
model_w2v = KeyedVectors.load("word2vec.wordvectors", mmap='r')

# Define a function to convert sentences to vectors
def sentence_to_vec(sentence, embedding_model):
    #tokenize embeddings
    tokens = word_tokenize(sentence.lower())
    #intialize list for empty sentence vectors
    sentence_vector = []
    for token in tokens:
        if token in embedding_model.key_to_index:
            sentence_vector.append(embedding_model[token])
    if sentence_vector:
        #check the vector is not empty
        return np.mean(sentence_vector, axis=0)
    else:
        #return zero if nothing found
        return np.zeros(embedding_model.vector_size)

df = pd.read_csv('datawords.csv', on_bad_lines='skip')

#convert sentence to vec
df['Sentence_Vector'] = df['Sentence'].apply(lambda x: sentence_to_vec(x, model_w2v))

#split data frames
train_df = df.iloc[:-30]
test_df = df.iloc[-30:]

train_df = train_df[train_df['Sentence_Vector'].map(lambda x: x.size) > 0]
test_df = test_df[test_df['Sentence_Vector'].map(lambda x: x.size) > 0]

X_train = np.stack(train_df['Sentence_Vector'].values)
y_train = train_df['Maxim_of_Quantity'].values

X_test = np.stack(test_df['Sentence_Vector'].values)
y_test = test_df['Maxim_of_Quantity'].values

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

joblib.dump(log_reg, 'log_reg_quantity.joblib')

y_pred = log_reg.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))