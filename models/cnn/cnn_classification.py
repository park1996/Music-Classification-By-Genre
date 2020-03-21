import os
import ntpath
import sys
from cnn import cnn
import numpy as np

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from audio_preprocessor import audio_preprocessor as ap
from feature_extractor import feature_extractor as fe

if __name__ == '__main__':

    myAP = ap()
    myFE = fe()
    audioPathList = myAP.list_of_all_audio_files
    trainTrackID = myFE.get_training_dataset_song_ids()
    trainDataLst = []
    trainGenreLst = []
    genreDict = {}
    nextGenreID = 0
    print('Now loading training data...')

    spectrogramLowestCol = 2147483647
    spectrogramLowestRow = 2147483647
    #Load the Training data...
    for path in audioPathList:
        trackIDStr = os.path.splitext(ntpath.basename(path))[0]
        trackIDInt = int(trackIDStr)
        if os.path.exists(path) and (trackIDInt in trainTrackID):
            currentGenre = myFE.get_genre(trackIDInt)
            currentGenreID = -1
            currentSpectrogram = myAP.get_mel_spectrogram(path)
            if(len(currentSpectrogram[0]) < spectrogramLowestCol):
                spectrogramLowestCol=len(currentSpectrogram[0])
            if(len(currentSpectrogram) < spectrogramLowestRow):
                spectrogramLowestRow=len(currentSpectrogram)
            print('Loading track '+trackIDStr + ' with genre '+currentGenre)
            if genreDict.get(currentGenre) == None:
                genreDict[currentGenre] = nextGenreID
                currentGenreID = nextGenreID
                nextGenreID+=1
            else:
                currentGenreID = genreDict[currentGenre]
            trainDataLst.append(currentSpectrogram)
            trainGenreLst.append(currentGenreID)
    print('Training data loaded.')

    #Load the testing data...
    print('Now loading testing data...')
    testTrackID = myFE.get_training_dataset_song_ids()
    testDataLst = []
    testGenreLst = []
    for path in audioPathList:
        trackIDStr = os.path.splitext(ntpath.basename(path))[0]
        trackIDInt = int(trackIDStr)
        if os.path.exists(path) and (trackIDInt in testTrackID):
            currentGenre = myFE.get_genre(trackIDInt)
            currentGenreID = -1
            currentSpectrogram = myAP.get_mel_spectrogram(path)
            if(len(currentSpectrogram[0]) < spectrogramLowestCol):
                spectrogramLowestCol=len(currentSpectrogram[0])
            if(len(currentSpectrogram) < spectrogramLowestRow):
                spectrogramLowestRow=len(currentSpectrogram)
            print('Loading track '+trackIDStr + ' with genre '+currentGenre)
            if genreDict.get(currentGenre) == None:
                genreDict[currentGenre] = nextGenreID
                currentGenreID = nextGenreID
                nextGenreID+=1
            else:
                currentGenreID = genreDict[currentGenre]
            testDataLst.append(currentSpectrogram)
            testGenreLst.append(currentGenreID)
    print('Training data loaded.')

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
    trainGenre = np.array(trainGenreLst)

    testData = np.array(testDataLst)
    testData = testData.reshape(testData.shape[0], num_rows, num_columns, num_channels)
    testGenre = np.array(testGenreLst)

    print('Creating data model...')

    myModel = cnn((num_rows, num_columns, num_channels))
    print('Now training model...')
    myModel.train(trainData, trainGenre)

    #Evaluate the model using test set. Code modifed based on https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324fa
    print('Now testing the model we trained...')
    (eval_loss, eval_accuracy) = myModel.test(testData, testGenre)
    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))