import logging
import os
import ntpath
import sys
import csv  
import numpy as np
import random
from datetime import datetime
from sklearn import preprocessing

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from audio_preprocessor import audio_preprocessor as ap
from feature_extractor import feature_extractor as fe
from feature_extractor import feature_type as ft

# Configure the size of training, testing sets and the number of classes we're matching against, 
# and whether we will randomly picking value or sequentially picking value
class data_preprocess:
    def load_and_format():

        # data init 
        myAP = ap()
        myFE = fe()
        trainTrackID = myFE.get_training_dataset_song_ids()
        classDict = {}
        trainClassStrLst=[]
        trainClassLst=[]
        trainTrackIDLst = []
        trainTrackIdStrLst = []
        nextClassID = 0 

        # feature extraction 
        trainDataLst=myFE.get_all_features_as_nparray(trainTrackID)
        for id in trainTrackID:
            trackIDStr=str(id)
            currentClass = myFE.get_genre(id)
            currentClassID = -1
            if classDict.get(currentClass) == None:
                classDict[currentClass] = nextClassID
                currentClassID = nextClassID
                nextClassID+=1
            else:
                currentClassID = classDict[currentClass]
            trainClassStrLst.append(currentClass)
            trainClassLst.append(currentClassID)
            trainTrackIDLst.append(id)
            trainTrackIdStrLst.append(trackIDStr)
        print('Training data loaded.')

        #Load the testing data...
        print('Now loading testing data...')
        testTrackID = myFE.get_validation_dataset_song_ids()
        testDataLst=myFE.get_all_features_as_nparray(testTrackID)
        testTrackIdLst = []
        testTrackIdStrLst = []
        testClassLst = []
        testClassStrLst = []
        for id in testTrackID:
            trackIDStr=str(id)
            currentClass = myFE.get_genre(id)
            currentClassID = -1
            if classDict.get(currentClass) == None:
                continue
            else:
                currentClassID = classDict[currentClass]
            testClassStrLst.append(currentClass)
            testClassLst.append(currentClassID)
            testTrackIdLst.append(id)
            testTrackIdStrLst.append(trackIDStr)
        print('Testing data loaded.')

        print('Processing data...')
        # reduce the size of 2-dimen audio features into 3-dimen 
        # for the convenience of training data
        reduceFactor = 14
        num_rows = int(len(trainDataLst[0])/reduceFactor)
        num_columns = reduceFactor
        num_channels = 1
        scaler = preprocessing.StandardScaler().fit(trainDataLst)
        trainDataLst = scaler.transform(trainDataLst)
        trainData = trainDataLst.reshape(trainDataLst.shape[0], num_rows, num_columns, num_channels)
        trainClass = np.array(trainClassLst)

        testDataLst = scaler.transform(testDataLst)
        testData = testDataLst.reshape(testDataLst.shape[0], num_rows, num_columns, num_channels)
        testClass = np.array(testClassLst)

        # Reformatting the data for better training
        num_rows = int(len(trainDataLst[0])/reduceFactor)
        num_columns = reduceFactor
        num_channels = 1
        scaler = preprocessing.StandardScaler().fit(trainDataLst)
        trainDataLst = scaler.transform(trainDataLst)
        trainData = trainDataLst.reshape(trainDataLst.shape[0], num_rows, num_columns, num_channels)
        trainClass = np.array(trainClassLst)

        testDataLst = scaler.transform(testDataLst)
        testData = testDataLst.reshape(testDataLst.shape[0], num_rows, num_columns, num_channels)
        testClass = np.array(testClassLst)
        print("data loaded")
        return trainData, trainClass, testData, testClass

