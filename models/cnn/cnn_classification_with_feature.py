import os
import ntpath
import sys
import csv
from cnn_for_feature import cnn
import numpy as np
import random
from datetime import datetime

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from audio_preprocessor import audio_preprocessor as ap
from feature_extractor import feature_extractor as fe
from feature_extractor import feature_type as ft

if __name__ == '__main__':

    #Configure the size of training, testing sets and the number of classes we're matching against, and whether we will randomly picking value or sequentially picking value
    reduceFactor = 14

    myAP = ap()
    myFE = fe()
    trainTrackID = myFE.get_training_dataset_song_ids()
    classDict = {}
    trainClassStrLst=[]
    trainClassLst=[]
    trainTrackIDLst = []
    trainTrackIdStrLst = []
    nextClassID = 0
    print('Now loading training data...')

    #Load the Training data...
    trainDataLst=myFE.get_features_as_nparray(trainTrackID)
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
    testDataLst=myFE.get_features_as_nparray(testTrackID)
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


    #Transform the format of data into something that can be used
    num_rows = int(len(trainDataLst[0])/reduceFactor)
    num_columns = reduceFactor
    num_channels = 1
    trainData = trainDataLst.reshape(trainDataLst.shape[0], num_rows, num_columns, num_channels)
    trainClass = np.array(trainClassLst)

    testData = testDataLst.reshape(testDataLst.shape[0], num_rows, num_columns, num_channels)
    testClass = np.array(testClassLst)

    print('Creating data model...')

    myModel = cnn((num_rows, num_columns, num_channels))
    print('Now training model...')
    myModel.train(trainData, trainClass)

    #Evaluate the model using test set. Code modifed based on https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324fa
    print('Now testing the model we trained...')
    (eval_loss, eval_accuracy) = myModel.test(testData, testClass)
    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    #Save the detailed prediction as a csv file - optional. Code modified based on https://realpython.com/python-csv/
    currentTimeStr = datetime.now().strftime("%Y%m%d%H%M%S")
    reverseclassDict = list(classDict.keys())

    trainPredictedClasses = myModel.predict(trainData)
    trainPredictedClassesStr = []
    for classID in trainPredictedClasses:
        trainPredictedClassesStr.append(reverseclassDict[classID])
    with open('trainset_prediction_result_'+ currentTimeStr +'.csv', mode='w') as csv_file:
        fieldnames = ['trackID', 'predictedClass', 'realClass', 'correct']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for j in range(0, len(trainTrackIdStrLst)):
            #print('Track ' + str(testTrackIdLst[j]) + ' has genre ' + testClassStrLst[j] + ' and our model predict it has genre ' + predictedClassesStr[j])
            writer.writerow({'trackID': trainTrackIdStrLst[j], 'predictedClass': trainPredictedClassesStr[j], 'realClass': trainClassStrLst[j], 'correct': str(trainPredictedClassesStr[j] == trainClassStrLst[j])})

    testPredictedClasses = myModel.predict(testData)
    testPredictedClassesStr = []
    for classID in testPredictedClasses:
        testPredictedClassesStr.append(reverseclassDict[classID])
    with open('testset_prediction_result_'+ currentTimeStr + '.csv', mode='w') as csv_file:
        fieldnames = ['trackID', 'predictedClass', 'realClass', 'correct']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for j in range(0, len(testTrackIdLst)):
            #print('Track ' + str(testTrackIdLst[j]) + ' has genre ' + testClassStrLst[j] + ' and our model predict it has genre ' + predictedClassesStr[j])
            writer.writerow({'trackID': testTrackIdStrLst[j], 'predictedClass': testPredictedClassesStr[j], 'realClass': testClassStrLst[j], 'correct': str(testPredictedClassesStr[j] == testClassStrLst[j])})