import os
# use plaidml lib as the background support for AMD
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

os.environ["RUNFILES_DIR"] = "/usr/local/share/plaidml"
# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path

os.environ["PLAIDML_NATIVE_PATH"] = "/usr/local/lib/libplaidml.dylib"
# libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path
# PLAIDML_NATIVE_PATH=/usr/local/lib/libplaidml.dylib

import numpy as np
import random
from datetime import datetime
from sklearn import preprocessing
from matplotlib import pyplot as plt
from data_loader import data_preprocessor
# from lstm_rnn_init import RNN_MODEL as RM

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam


class rnn_model:
    def __init__(self, input_shape, dim1, dim2, output_size):
        print("init LSTM RNN model ...")
        self.model = Sequential()

        self.model.add(LSTM(units=dim1, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
        print("init the first layer of the RNN")
        self.model.add(LSTM(units=dim2, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        print("init the second layer of the RNN")

        self.model.add(Dense(units=output_size, activation="softmax"))

        # code modified and refered from https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification.git
        # Keras optimizer defaults:
        # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
        # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
        # SGD    : lr=0.01,  momentum=0.,                             decay=0.

        opt = Adam()
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.model.summary()

        print("finsish initialization")

    def train(self, features, target_labels, batch_size, epochs):
        history = self.model.fit(x=features, y=target_labels, batch_size=batch_size, epochs=epochs)
        return history

    def test(self, features, target_labels):
        (score, eval_accuracy) = self.model.evaluate(x=features, y=target_labels, batch_size=35, verbose=1)
        return (score, eval_accuracy)

    def save(self, model_path):
        self.model.save(model_path)


if __name__ == '__main__':
    RNN_DATA = data_preprocessor()
    trainData, trainClass, testData, testClass, num_rows, num_columns= RNN_DATA.load_and_format()
    #output_size = len(np.unique(trainClass, return_index=True))
    output_size = max(trainClass)+1

    trainClassBinLst = []
    trainClassBinTemplate = []
    for i in range(0, output_size):
        trainClassBinTemplate.append(0)
    for t in trainClass:
        currentBinClass = trainClassBinTemplate.copy()
        currentBinClass[t] = 1
        trainClassBinLst.append(currentBinClass)
    trainClassBinLst = np.array(trainClassBinLst)
    testClassBinLst = []
    for t in testClass:
        currentBinClass = trainClassBinTemplate.copy()
        currentBinClass[t] = 1
        testClassBinLst.append(currentBinClass)
    testClassBinLst = np.array(testClassBinLst)
    # init RNN model
    RNN_MODEL =rnn_model((num_rows, num_columns), num_rows, num_columns, output_size)

    # train
    RNN_MODEL.train(trainData, trainClassBinLst, 35, 400)
    # validate 
    print("\n Validating")
    score, accuracy = RNN_MODEL.test(trainData, trainClassBinLst)
    print("Train loss:  ", score)
    print("Train accuracy:  ", accuracy)
    # test
    print("\n Testing")
    score, accuracy = RNN_MODEL.test(testData, testClassBinLst)
    print("Test loss", score)
    print("Test accuracy: ", accuracy)

