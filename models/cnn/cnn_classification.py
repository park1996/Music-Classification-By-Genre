import os
import ntpath
import sys
from cnn import cnn

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from audio_preprocessor import audio_preprocessor as ap
from feature_extractor import feature_extractor as fe

if __name__ == '__main__':
    myAP = ap()
    myFE = fe()
    myModel = cnn()
    audioPathList = myAP.list_of_all_audio_files
    trainTrackID = myFE.get_training_dataset_song_ids()
    trainData = []
    trainGenre = []
    print('Now loading training data...')
    i = 0
    for path in audioPathList:
        trackIDStr = os.path.splitext(ntpath.basename(path))[0]
        trackIDInt = int(trackIDStr)
        if os.path.exists(path) and (trackIDInt in trainTrackID):
            currentGenre = myFE.get_genre(trackIDInt)
            print('Loading track '+trackIDStr + ' it has genre '+currentGenre)
            trainData.append(myAP.audio_read(path))
            trainGenre.append(currentGenre)
            i+= 1
            if i > 10:
                break;
    print('Training data loaded.')
    print('Now training model...')
    myModel.train(trainData, trainGenre)