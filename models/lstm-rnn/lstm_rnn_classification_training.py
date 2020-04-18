import numpy as np
import random
from datetime import datetime
from sklearn import preprocessing
from matplotlib import pyplot as plt
from data_loader import data_preprocessor
from lstm_rnn_init import RNN_MODEL as RM

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
    RNN_MODEL = RM((num_rows, num_columns), num_rows, num_columns, output_size)

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

