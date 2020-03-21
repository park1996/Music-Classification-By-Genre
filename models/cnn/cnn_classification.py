import os
import ntpath
import sys
import csv
from cnn import cnn
import numpy as np
import random
from datetime import datetime

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from audio_preprocessor import audio_preprocessor as ap
from feature_extractor import feature_extractor as fe

if __name__ == '__main__':

    trainingSetSize = 10
    testingSetSize = 10
    classSize = trainingSetSize
    randomize = True

    myAP = ap()
    myFE = fe()
    audioPathList = myAP.list_of_all_audio_files
    if randomize:
        random.shuffle(audioPathList)
    trainTrackID = myFE.get_training_dataset_song_ids()
    trainDataLst = []
    trainClassLst = []
    trainClassStrLst = []
    trainTrackIDLst = []
    trainTrackIdStrLst = []
    classDict = {}
    nextClassID = 0
    print('Now loading training data...')

    spectrogramLowestCol = 2147483647
    spectrogramLowestRow = 2147483647
    #Load the Training data...
    i = 0
    for path in audioPathList:
        trackIDStr = os.path.splitext(ntpath.basename(path))[0]
        trackIDInt = int(trackIDStr)
        if os.path.exists(path) and (trackIDInt in trainTrackID):
            currentClass = myFE.get_genre(trackIDInt)
            currentClassID = -1
            if classDict.get(currentClass) == None and len(classDict) < classSize:
                classDict[currentClass] = nextClassID
                currentClassID = nextClassID
                nextClassID+=1
            elif classDict.get(currentClass) == None and len(classDict) >= classSize:
                continue
            else:
                currentClassID = classDict[currentClass]
            print('Loading track '+trackIDStr + ' with genre '+currentClass + ' ('+str(i+1)+'/'+str(trainingSetSize)+')')
            currentSpectrogram = myAP.get_mel_spectrogram(path)
            if(len(currentSpectrogram[0]) < spectrogramLowestCol):
                spectrogramLowestCol=len(currentSpectrogram[0])
            if(len(currentSpectrogram) < spectrogramLowestRow):
                spectrogramLowestRow=len(currentSpectrogram)
            trainDataLst.append(currentSpectrogram)
            trainClassStrLst.append(currentClass)
            trainClassLst.append(currentClassID)
            trainTrackIDLst.append(trackIDInt)
            trainTrackIdStrLst.append(trackIDStr)
            i+=1
            if i >= trainingSetSize:
                break
    print('Training data loaded.')

    #Load the testing data...
    i = 0
    print('Now loading testing data...')
    testTrackID = myFE.get_test_dataset_song_ids()
    testDataLst = []
    testClassLst = []
    testClassStrLst = []
    testTrackIdLst = []
    testTrackIdStrLst = []
    for path in audioPathList:
        trackIDStr = os.path.splitext(ntpath.basename(path))[0]
        trackIDInt = int(trackIDStr)
        if os.path.exists(path) and (trackIDInt in testTrackID):
            currentClass = myFE.get_genre(trackIDInt)
            currentClassID = -1
            if classDict.get(currentClass) == None:
                continue
            else:
                currentClassID = classDict[currentClass]
            print('Loading track '+trackIDStr + ' with genre '+currentClass + ' ('+str(i+1)+'/'+str(testingSetSize)+')')
            currentSpectrogram = myAP.get_mel_spectrogram(path)
            if(len(currentSpectrogram[0]) < spectrogramLowestCol):
                spectrogramLowestCol=len(currentSpectrogram[0])
            if(len(currentSpectrogram) < spectrogramLowestRow):
                spectrogramLowestRow=len(currentSpectrogram)

            testDataLst.append(currentSpectrogram)
            testClassStrLst.append(currentClass)
            testClassLst.append(currentClassID)
            testTrackIdLst.append(trackIDInt)
            testTrackIdStrLst.append(trackIDStr)
            i+=1
            if i >= testingSetSize:
                break
    print('Testing data loaded.')

    print('Processing data...')
    #Truncate the frame for transformation
    for i in range(0, len(trainDataLst)):
        spectrogram = trainDataLst[i]
        if len(spectrogram[0]) <= spectrogramLowestCol and len(spectrogram) <= spectrogramLowestRow:
            continue;
        while len(spectrogram[0]) > spectrogramLowestCol:
            spectrogram = np.delete(spectrogram, len(spectrogram[0])-1, 1)
        while len(spectrogram) > spectrogramLowestRow:
            spectrogram = np.delete(spectrogram, len(spectrogram)-1, 0)
        trainDataLst[i] = spectrogram

    for i in range(0, len(testDataLst)):
        spectrogram = testDataLst[i]
        if len(spectrogram[0]) <= spectrogramLowestCol and len(spectrogram) <= spectrogramLowestRow:
            continue;
        while len(spectrogram[0]) > spectrogramLowestCol:
            spectrogram = np.delete(spectrogram, len(spectrogram[0])-1, 1)
        while len(spectrogram) > spectrogramLowestRow:
            spectrogram = np.delete(spectrogram, len(spectrogram)-1, 0)
        testDataLst[i] = spectrogram

    #Transform the format of data into something that can be used
    num_rows = spectrogramLowestRow
    num_columns = spectrogramLowestCol
    num_channels = 1
    trainData = np.array(trainDataLst)
    trainData = trainData.reshape(trainData.shape[0], num_rows, num_columns, num_channels)
    trainClass = np.array(trainClassLst)

    testData = np.array(testDataLst)
    testData = testData.reshape(testData.shape[0], num_rows, num_columns, num_channels)
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

    trainPredictedClasses = myModel.predict(testData)
    trainPredictedClassesStr = []
    for classID in trainPredictedClasses:
        trainPredictedClassesStr.append(reverseclassDict[classID])
    with open('trainset_prediction_result_'+ currentTimeStr +'.csv', mode='w') as csv_file:
        fieldnames = ['trackID', 'predictedClass', 'realClass']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for j in range(0, len(trainTrackIDLst)):
            #print('Track ' + str(testTrackIdLst[j]) + ' has genre ' + testClassStrLst[j] + ' and our model predict it has genre ' + predictedClassesStr[j])
            writer.writerow({'trackID': trainTrackIdStrLst[j], 'predictedClass': trainPredictedClassesStr[j], 'realClass': trainClassStrLst[j]})

    testPredictedClasses = myModel.predict(testData)
    testPredictedClassesStr = []
    for classID in testPredictedClasses:
        testPredictedClassesStr.append(reverseclassDict[classID])
    with open('testset_prediction_result_'+ currentTimeStr + '.csv', mode='w') as csv_file:
        fieldnames = ['trackID', 'predictedClass', 'realClass']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for j in range(0, len(testTrackIdLst)):
            #print('Track ' + str(testTrackIdLst[j]) + ' has genre ' + testClassStrLst[j] + ' and our model predict it has genre ' + predictedClassesStr[j])
            writer.writerow({'trackID': testTrackIdStrLst[j], 'predictedClass': testPredictedClassesStr[j], 'realClass': testClassStrLst[j]})