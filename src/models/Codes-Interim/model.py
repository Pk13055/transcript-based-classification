#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import argparse
import math
import os

from IPython.core.debugger import set_trace
from keras.layers import Dense, Input, LSTM, Bidirectional, Embedding, Dropout
from keras.models import Model, Sequential, load_model
from keras.losses import binary_crossentropy, mean_squared_error
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils.np_utils as np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# load the dataset
df = pd.read_csv('modified_davidson.csv', header=None)
print(df.head())
X_train, X_test, y_train, y_test = train_test_split(df[1], df[0], test_size=0.10, random_state=42)
X_train.shape, X_test.shape


# In[3]:


# Load the glove embeddings
X_train = X_train.astype('str')
embedding_ = np.load('300_glove_davidson.npy')
embedding_.shape


# In[4]:


# setup the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train.astype('str'))
vocab_size = len(tokenizer.word_index) + 1  # why the extra +1?
encoded_ = tokenizer.texts_to_matrix(X_train)
encoded_.shape


# In[7]:


# Creating the model
def get_model(embedding_size: tuple, input_length: int, embedding_weights: np.ndarray):
    """Create the model architecture
    :param embedding_size: tuple -> Size of the embedding data
    :param input_length: int -> Length of the input strings
    :param embedding_weights: np array -> the pre trained weights for the embedding layer
    :return mode: Keras sequential model
    """
    model = Sequential()

    model.add(Embedding(*embedding_size, weights=[embedding_weights], input_length=input_length, name='embedding_layer'))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.2)))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax', name='last'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# final model
model = get_model(embedding_.shape, encoded_.shape[-1], embedding_)
model.summary()


# In[ ]:


y_train_onehot = np_utils.to_categorical(y_train, num_classes=3)
y_test_onehot = np_utils.to_categorical(y_test, num_classes=3)
history = model.fit(encoded_, y_train_onehot, batch_size=8, epochs=2, validation_split=0.0, verbose=3)


# In[ ]:


# Test results
model.save('new_model.h5')  # save the current trained model
encoded_test = tokenizer.texts_to_matrix(X_test)
y_hat = model.predict([encoded_test])
results = accuracy_score(y_test_onehot, y_hat)
print(results)

