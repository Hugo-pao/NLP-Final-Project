import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf

from tensorflow import keras

from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import InputLayer
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers import Bidirectional, LSTM, Flatten
from keras.layers import Dropout, Dense, Activation

from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D, Dropout
from keras.layers.core import Dense, Activation

import pickle


data = pd.read_csv("cleaned_data.csv")
data.cleaned.dropna(axis = 0,inplace= True)
no_verb = data[["headline","cleaned","Set","is_sarcastic"]].dropna()

sarcasm = no_verb.loc[no_verb["is_sarcastic"] == 1,["headline","cleaned"]]

################################################### DATA PREPROCESSING ##########################################
corpus = [line.split() for line in sarcasm.headline.tolist()]

X_train = []
y_train = []
for sentence in corpus[:2000]:
    #include start and stop in the sentence
    tokens =  sentence 
    for i in range (len(tokens)-1) :
        X = tokens[:i]
        y = tokens[i+1]
        X_train.append(X)
        y_train.append(y)

y_train = np.array(y_train)
X_train = np.array(X_train)


vocab_size = len(np.unique(y_train))

one_hot = OneHotEncoder()
y_train_one_hot = one_hot.fit_transform(y_train.reshape(-1, 1))


wordset = set()
for words in X_train:
    wordset.update(set(words))

# map words and tags into ints
PAD = '-PAD-'
UNK = '-UNK-'
word2int = {word: i + 2 for i, word in enumerate(sorted(wordset))}
word2int[PAD] = 0  # special token for padding
word2int[UNK] = 1  # special token for unknown words

def convert2ints(instances):
    result = []
    for words in instances:
        # replace words with int, 1 for unknown words
        word_ints = [word2int.get(word, 1) for word in words]
        # replace tags with int
        result.append(word_ints)
    return result 

def convert2int(text):
    result = []
    # replace words with int, 1 for unknown words
    word_ints = [word2int.get(word, 1) for word in text]
    # replace tags with int
    result.append(word_ints)
    return result


train_instances_int = convert2ints(X_train)
# test_instances_int = convert2ints(test_instances)

train_sentences = pad_sequences(train_instances_int, maxlen=9, padding='post')
print(train_sentences[5])


np.random.seed(42)
from keras.optimizers import RMSprop
model_seq = Sequential()
model_seq.add(InputLayer(input_shape=(9, ), name="word_IDs"))
model_seq.add(Embedding(len(word2int), 128, mask_zero=True, name='embeddings'))
model_seq.add(Bidirectional(LSTM(units=256, return_sequences=False), name="Bi-LSTM"))
model_seq.add(Dense(256, name = 'fully_connected1', activation = 'relu'))
model_seq.add(Dropout(0.2, name='dropout'))
model_seq.add(Dense(vocab_size, activation = 'softmax', name='output'))
model_seq.summary()

batch_size = 32
epochs = 45

model_seq.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy']
             )
history = model_seq.fit(train_sentences, y_train_one_hot,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    )

df = pd.DataFrame(history.history)
df[ 'accuracy'].plot.line();
df['loss'].plot.line()

text = ["twenty"]
def generate(text):
    liste = convert2int(text)
    train_sentences = pad_sequences(liste, maxlen=9, padding='post')
    a = np.random.choice(np.array([a for a in range(vocab_size)]), size = 1, p = np.transpose(model_seq.predict(train_sentences)[0]))
    return np.unique(y_train)[a[0]]


def run(text, sentence_length):
    texte = [text]
    for i in range (sentence_length):
        t = generate(texte)
        texte.append(t)
    print(' '.join(texte))

run("", 5)

filename = 'BiLSTM_language_model.sav'
pickle.dump(model_seq, open(filename, 'wb'))
model_seq = pickle.load(open(filename, 'rb'))
