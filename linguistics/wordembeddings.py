import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

#load in dataset
df = pd.read_csv('datawords.csv', on_bad_lines='skip')

#preprocessing: tokenization, lowercasing
df['Processed_Sentences'] = df['Sentence'].apply(lambda x: word_tokenize(x.lower()))

#assum processed sentences are already tokenized
sentences = df['Processed_Sentences'].tolist()
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

#save model
model.save("word2vec.model")

#save word vecotrs
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")

