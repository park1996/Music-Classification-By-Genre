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
    myModel = cnn((num_rows, num_columns, num_channels))
    audioPathList = myAP.list_of_all_audio_files
    trainTrackID = myFE.get_training_dataset_song_ids()
    trainDataLst = []
    trainGenreLst = []
    trainGenreDict = {}
    print('Now loading training data...')
    i = 0
    nextGenreID = 0
    spectrogramLowestCol = 2147483647
    spectrogramLowestRow = 2147483647
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
                spectrogramLowestRow=len(currentSpectrogram[0])
            print('Loading track '+trackIDStr + ' with genre '+currentGenre)
            if trainGenreDict.get(currentGenre) == None:
                trainGenreDict[currentGenre] = nextGenreID
                currentGenreID = nextGenreID
                nextGenreID+=1
            else:
                currentGenreID = trainGenreDict[currentGenre]
            trainDataLst.append(currentSpectrogram)
            trainGenreLst.append(currentGenreID)
            i+= 1
            if i > 10:
                break;

    for i in range(0, len(trainDataLst)):
        spectrogram = trainDataLst[i]
        if len(spectrogram[0]) <= spectrogramLowestCol and len(spectrogram) <= spectrogramLowestRow:
            continue;
        while len(spectrogram[0]) > spectrogramLowestCol:
            spectrogram = np.delete(spectrogram, len(spectrogram[0])-1, 1)
        while len(spectrogram) > spectrogramLowestRow:
            spectrogram = np.delete(spectrogram, len(spectrogram)-1, 0)
        trainDataLst[i] = spectrogram
    print('Creating data model...')
    num_rows = spectrogramLowestRow
    num_columns = spectrogramLowestCol
    num_channels = 1
    myModel = cnn((num_rows, num_columns, num_channels))
    print('Training data model...')
    trainData = np.array(trainDataLst)
    trainData = trainData.reshape(trainData.shape[0], num_rows, num_columns, num_channels)
    trainGenre = np.array(trainGenreLst)


    print('Now training model...')
    myModel.train(trainData, trainGenre)