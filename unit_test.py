import audio_preprocessor
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import unittest

from feature_extractor import feature_extractor
from pydub import AudioSegment, effects

extractor = feature_extractor()

class TestExtractor(unittest.TestCase):
   
    def test_get_training_dataset(self):
        ''' Get training dataset '''
        training_set = extractor.get_training_dataset_song_ids()
        SAMPLE_SIZE = 5

        print('Sample data from training dataset (total size: ' + str(len(training_set)) + ')\n')

        for i in training_set[:SAMPLE_SIZE]:
            print('Title: ' + str(extractor.get_title(i)))
            print('Genre: ' + str(extractor.get_genre(i)))
            print ('\n')

    def test_get_validation_dataset(self):
        ''' Get validation dataset '''
        validation_set = extractor.get_validation_dataset_song_ids()
        SAMPLE_SIZE = 5

        print('Sample data from  validation dataset (total size: ' + str(len(validation_set)) + ')\n')

        for i in validation_set[:SAMPLE_SIZE]:
            print('Title: ' + str(extractor.get_title(i)))
            print('Genre: ' + str(extractor.get_genre(i)))
            print ('\n')

    def test_get_test_dataset(self):
        ''' Get test dataset '''
        test_set = extractor.get_test_dataset_song_ids()
        SAMPLE_SIZE = 5

        print('Sample data from test dataset (totalsize: ' + str(len(test_set)) + ')\n')

        for i in test_set[:SAMPLE_SIZE]:
            print('Title: ' + str(extractor.get_title(i)))
            print('Genre: ' + str(extractor.get_genre(i)))
            print ('\n')

if __name__ == '__main__':
    unittest.main()






