import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
import ssl
import joblib


#same above, internal ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

#load word2vec model
model_w2v = KeyedVectors.load("word2vec.wordvectors", mmap='r')

#function to convert sentences to vectors
def sentence_to_vec(sentence, embedding_model):
    #tokenize
    tokens = word_tokenize(sentence.lower())
    #initialize list to hold
    sentence_vector = []
    for token in tokens:
        if token in embedding_model.key_to_index:
            sentence_vector.append(embedding_model[token])
    if sentence_vector:
        #check vector isn't empty
        return np.mean(sentence_vector, axis=0)
    else:
        #if no vectors found, return 0
        return np.zeros(embedding_model.vector_size)

#load dataset
df = pd.read_csv('datawords.csv', on_bad_lines='skip')

#convert each sentence to vector
df['Sentence_Vector'] = df['Sentence'].apply(lambda x: sentence_to_vec(x, model_w2v))

#split for last thirty rows for testing
train_df = df.iloc[:-30]
test_df = df.iloc[-30:]

#make sure to convert sentences to vectors
train_df = train_df[train_df['Sentence_Vector'].map(lambda x: x.size) > 0]
test_df = test_df[test_df['Sentence_Vector'].map(lambda x: x.size) > 0]

#prep feature labels and training
X_train = np.stack(train_df['Sentence_Vector'].values)
y_train = train_df['Maxim_of_Relevance'].values

#prep features and labels to test
X_test = np.stack(test_df['Sentence_Vector'].values)
y_test = test_df['Maxim_of_Relevance'].values

#initialize logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

#make predictions on testing set
y_pred = log_reg.predict(X_test)

#use the log_reg which is the trained model
joblib.dump(log_reg, 'log_reg_relevance.joblib')

#evaluate using testing set
print(classification_report(y_test, y_pred, zero_division=0))