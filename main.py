#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    df1 = pd.read_csv('data/full_dataset/goemotions_1.csv')
    df2 = pd.read_csv('data/full_dataset/goemotions_2.csv')
    df3 = pd.read_csv('data/full_dataset/goemotions_3.csv')

    # Concatenate all data
    frames = [df1, df2, df3]
    df = pd.concat(frames)

    # Split into train and test
    X = df['text'].values
    X = X.astype(str)
    y = df.iloc[:, 9:].values # 28 emotions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    maxlen = max([len(x) for x in X_train])
    print('Size: {}(train), {}(test)'.format(len(X_train), len(X_test)))

    # Tokenize / Create word dictionary / Getting ad list of word index
    tokenizer = Tokenizer(num_words=10000, oov_token='unk')
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1
    print('vocabulary size: {}'.format(vocab_size))

    # Padding
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # Use pretrained word embedding vector
    embedding_dim = 100
    embedding_index = {}
    with open('glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    print('Number of words: {}'.format(len(embedding_index)))

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    embedding_accuracy = nonzero_elements / vocab_size
    print('Embedding accuracy: {}'.format(embedding_accuracy))

    # Define a model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=True),
        tf.keras.layers.Conv1D(256, 3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(28, activation='sigmoid'),
    ])
    opt = optimizers.Adam(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    model.summary()

    # Do training
    history = model.fit(X_train, y_train, epochs=10, verbose=True, callbacks=callbacks, validation_data=(X_test, y_test), batch_size=100)

    # Evaluate
    y_pred = model.predict(X_test)
    thresholds = np.arange(0.1, 1.0, 0.1)
    for th in thresholds:
        pred = y_pred.copy()
        pred[pred >= th] = 1
        pred[pred < th] = 0

        precision = precision_score(y_test, pred, average='micro')
        recall = recall_score(y_test, pred, average='micro')
        f1 = f1_score(y_test, pred, average='micro')

        print('Threshold: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}'.format(th, precision, recall, f1))

