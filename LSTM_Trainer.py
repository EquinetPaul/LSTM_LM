##################
#   PARAMETERS   #
##################
# Paramètres modifiables Model
vocab_size = 50000
embedding_dim = 128
lstm_units = 256
num_layers = 2
dropout_rate = 0.05
# Paramètres modifiables Data
n_min = 2
n_max = 10
# Paramètres modifiables Training
epochs = 5
batch_size = 258
learning_rate = 0.001
test_size = 0.1
##################
#   PARAMETERS   #
##################

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info("Loading Libraries...")

import warnings
# warnings.filterwarnings("ignore")

# Data
import pandas as pd
# Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
# Preprocessing
from collections import Counter
from typing import List
# Training
import nltk
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Utils
from tqdm import tqdm
import pickle
import re
logging.info("Loaded.")


##################
#     METHODS    #
##################
def preprocess_text(text: str) -> str:
    # Nettoyer le texte
    text = text.replace("\n", " ")
    text = " ".join(text.strip().split())
    return text

def create_vocabulary(corpus: List[str], vocab_size: int) -> dict:
    # Tokenize le corpus
    tokens = [word for sentence in tqdm(corpus) for word in preprocess_text(sentence).split()]

    # Compte les occurrences de chaque token
    token_counts = Counter(tokens)

    # Conserve les vocab_size premiers tokens les plus fréquents
    most_common_tokens = token_counts.most_common(vocab_size - 2)

    # Ajoute les tokens spéciaux <PAD> et <OOV>
    vocabulary = {token: idx + 2 for idx, (token, _) in enumerate(most_common_tokens)}
    vocabulary['<PAD>'] = 0
    vocabulary['<OOV>'] = 1

    return vocabulary

def create_lstm_model(vocab_size, embedding_dim, lstm_units, num_layers, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=None))

    for i in range(num_layers - 1):
        model.add(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))

    model.add(LSTM(lstm_units, dropout=dropout_rate))
    model.add(Dense(vocab_size, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def tokenize_and_pad(corpus, vocabulary, max_seq_length):
    tokenized_corpus = []

    for sentence in corpus:
        tokenized_sentence = [vocabulary.get(preprocess_text(token), vocabulary['<OOV>']) for token in sentence.split()]
        tokenized_corpus.append(tokenized_sentence)

    padded_corpus = pad_sequences(tokenized_corpus, maxlen=max_seq_length, padding='pre', value=vocabulary['<PAD>'])

    return padded_corpus

def sequences_to_sentences(sequence, vocabulary):
    sentences = []
    for s in sequence:
        sentence = []
        for token in s:
            if token == vocabulary['<PAD>']:
                continue
            elif token == vocabulary['<OOV>']:
                sentence.append('<OOV>')
            else:
                sentence.append(list(vocabulary.keys())[list(vocabulary.values()).index(token)])

        sentences.append(' '.join(sentence))
    return sentences

def remove_empty_elements(input_list):
    return [element for element in input_list if element]

def split_text(text: str):
    # Utilise l'expression régulière pour diviser le texte par '.', '!' et '?'
    split_regex = r'(?<=[.!?])\s*'
    result = re.split(split_regex, text)
    return result

def create_training_data(corpus, vocabulary, n_min, n_max):
    X, y = [], []

    corpus_full = []
    for sentence in corpus:
        corpus_full += split_text(sentence)
    corpus = remove_empty_elements(corpus_full)

    for sentence in tqdm(corpus):
        sentence = preprocess_text(sentence)
        tokenized_sentence = [vocabulary.get(token, vocabulary['<OOV>']) for token in sentence.split()]
        for n in range(n_min, n_max + 1):
            ngrams = list(nltk.ngrams(tokenized_sentence, n))
            for ngram in ngrams:
                X.append(list(ngram[:-1]))
                y.append(ngram[-1])

    logging.info("Padding Sequences...")
    X = pad_sequences(X, maxlen=n_max, padding='pre', value=vocabulary['<PAD>'])


    return X, np.array(y)


##################
#      DATA      #
##################
logging.info("Loading Data...")
df = pd.read_csv("../macron.csv", sep=";", encoding="utf-8")
corpus = df["speech"].to_list()
logging.info("Loaded.")

##################
#    TRAINING    #
#      DATA      #
##################
logging.info("Creating Vocabulary...")
vocabulary = create_vocabulary(corpus, vocab_size)

logging.info("Saving Vocabulary...")
file = open("vocab.pickle", "wb")
pickle.dump(vocabulary, file)
file.close()
logging.info("Saved.")
logging.info("Creating Training Data...")
X, y  = create_training_data(corpus, vocabulary, n_min, n_max)
X_train, X_val, y_train, y_val = X, X[:int(test_size*len(X))], y, y[:int(test_size*len(y))]
logging.info("Created.")

##################
#    TRAINING    #
#      MODEL     #
##################
logging.info("Creating LSTM Model...")
model = create_lstm_model(vocab_size, embedding_dim, lstm_units, num_layers, dropout_rate, learning_rate)
logging.info("Created...")
print(model.summary())

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
##################
#      SAVE      #
##################
model_path = "LSTM_LM"
model.save(model_path)
