import audio_preprocessor
from feature_extractor import feature_extractor

from pydub import AudioSegment, effects
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import unittest

extractor = feature_extractor()
PRINT_SIZE = 5

class TestExtractor(unittest.TestCase):
   
    def test_get_training_dataset(self):
        ''' Get training dataset '''

        training_set = extractor.get_training_dataset_song_ids()
        print('Song titles and genre in training dataset (size: ' + str(len(training_set)) + ')')

        for i in training_set[:PRINT_SIZE]:
            print('Title:' + str(extractor.get_title(i)))
            print('Genre:' + str(extractor.get_genre(i)))
            print ('\n')

    def test_get_validation_dataset(self):
        ''' Get validation dataset '''

        validation_set = extractor.get_validation_dataset_song_ids()
        print('Song titles and genre in validation dataset (size: ' + str(len(validation_set)) + ')')

        for i in validation_set[:PRINT_SIZE]:
            print('Title:' + str(extractor.get_title(i)))
            print('Genre:' + str(extractor.get_genre(i)))
            print ('\n')

    def test_get_test_dataset(self):
        ''' Get test dataset '''

        test_set = extractor.get_test_dataset_song_ids()
        print('Song titles and genre in test dataset (size: ' + str(len(test_set)) + ')')

        for i in test_set[:PRINT_SIZE]:
            print('Title:' + str(extractor.get_title(i)))
            print('Genre:' + str(extractor.get_genre(i)))
            print ('\n')

if __name__ == '__main__':
    unittest.main()






