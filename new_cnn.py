#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: new_cnn.py
#Author: yuxuan
#Created Time: 2016-12-17 11:25:37
############################
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input, Merge, Flatten
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import *
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.utils.visualize_util import plot
from keras.regularizers import l2, activity_l2
import os
from six.moves import cPickle
import jieba

def load_data(fold):
    dataX = []
    dataY = []
    for file_name in os.listdir(fold):
        file_path = os.path.join(fold, file_name)
        split_file_path = os.path.join(fold, file_name)
        with open(split_file_path, "rb") as f:
            num_line = 0
            for line in f:
                if(num_line>7000):
                    break
                num_line += 1
                dataX.append(line)
                dataY.append(int(file_name[:-4]))
    datalen = len(dataX)
    train_len = int(datalen*0.8)
    dataY = to_categorical(dataY, nb_classes=None)
    tempper = np.random.permutation(datalen)
    trainX = np.array([dataX[i] for i in tempper[:train_len]])
    trainY = np.array([dataY[i] for i in tempper[:train_len]])
    textX = np.array([dataX[i] for i in tempper[train_len:]])
    textY = np.array([dataY[i] for i in tempper[train_len:]])
    return ((trainX, trainY),(textX, textY))

def train(maxlen=400,
        max_features = 50000,
        embedding_dims = 50,
        nb_filter = 250,
        batch_size = 32,
        filter_length = [3,4,5],
        hidden_dims = 250,
        nb_epoch = 2):
    (X_train, y_train), (X_test, y_test) = load_data("../split_new")
    token = Tokenizer(nb_words=max_features, filters=base_filter(),
            lower=True, split=" ")
    token.fit_on_texts(X_train)
    cPickle.dump(token, open("tokenfile", "wb"))
    # 不够长的补零，太长的截断
    X_train = token.texts_to_sequences(X_train)
    X_test = token.texts_to_sequences(X_test)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Build model...')
    model = Sequential()
    main_input = Input(shape=(maxlen, ), name='main_input')
    embedding = Embedding(max_features,
        embedding_dims,
        input_length=maxlen,
        dropout=0.2)(main_input)
    # graph_in = Input(shape=(maxlen, embedding_dims))
    convs = []
    for i in filter_length:
        conv = Convolution1D(nb_filter=nb_filter,
                    filter_length=i,
                    border_mode='valid',
                    activation='relu',
                    subsample_length=1)(embedding)
        pool_layer = MaxPooling1D()(conv)
        flatten = Flatten()(pool_layer)
        convs.append(flatten)
    out = Merge(mode='concat', name="merge_name")(convs) 
    # We add a vanilla hidden layer:
    dense1 = Dense(hidden_dims)(out)
    dropout1 = Dropout(0.2)(dense1)
    activation1 = Activation('relu')(dropout1)
    # We project onto a single unit output layer, and squash it with a sigmoid:
    dense2 = Dense(15)(activation1)
    activation2 = Activation('softmax')(dense2)
    model = Model(input=main_input, output=activation2)
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    plot(model, to_file='model.png', show_shapes=True)
    model.fit(X_train, y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        validation_data=(X_test, y_test))
    json_string = model.to_json()
    fs = open("my_model.json", "w")
    fs.write(json_string)
    fs.close()
    model.save_weights('my_model_weights.h5')
    return model

def predict(filepath = 'my_model.json', maxlen=400):
    fs = open("my_model.json", "r")
    json_string = fs.read()
    fs.close()
    model = model_from_json(json_string)
    model.load_weights('my_model_weights.h5')
    token = cPickle.load(open("tokenfile", 'rb'))
    # plot(model, to_file='model.png', show_shapes=True)
    from draw_data import draw_data
    title = draw_data()
    title_item = title.get_title_data('2015-09-25', '2015-12-25', 0)
    id2text = []
    for ti in title_item:
        id2text.append([[i['_id']], list(jieba.cut(ti['title_text'])), [ti['title_content']]])
    arrayid2text = np.array(id2text)
    predictData = sequence.pad_sequences(token.texts_to_sequences(arrayid2text[:, 1]), maxlen=maxlen)
    result = model.predict(predictData)
    result_dict = dict()
    for i in range(id2text):
        print(id2text[i][2] + str(result[i]))
        result_dict[id2text[i][0]] = result[i]
    np.savez("classify.dict", result_dict)

if __name__ == "__main__":
    train()
    # predict()
