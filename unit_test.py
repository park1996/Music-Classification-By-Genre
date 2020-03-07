import audio_preprocessor
import numpy as np
import unittest

from feature_extractor import feature_extractor
from feature_extractor import feature_type
from feature_extractor import statistic_type

extractor = feature_extractor()

class TestExtractor(unittest.TestCase):
   
    def test_get_training_dataset_info(self):
        ''' Get training dataset '''
        training_set_song_ids = extractor.get_training_dataset_song_ids()
        print('---------------------------------------------------------')
        print('Training dataset (total size: ' + str(len(training_set_song_ids)) + ')')
        print('---------------------------------------------------------\n')
        TestExtractor.print_dataset_info(training_set_song_ids)

    def test_get_validation_dataset_info(self):
        ''' Get validation dataset '''
        validation_set_song_ids = extractor.get_validation_dataset_song_ids()
        print('---------------------------------------------------------')
        print('Validation dataset (total size: ' + str(len(validation_set_song_ids)) + ')')
        print('---------------------------------------------------------\n')
        TestExtractor.print_dataset_info(validation_set_song_ids)

    def test_get_test_dataset_info(self):
        ''' Get test dataset '''
        test_set_song_ids = extractor.get_test_dataset_song_ids()
        print('---------------------------------------------------------')
        print('Test dataset (totalsize: ' + str(len(test_set_song_ids)) + ')\n')
        print('---------------------------------------------------------')
        TestExtractor.print_dataset_info(test_set_song_ids)

    def test_get_all_genres(self):
        ''' Get test dataset '''
        genre_list = extractor.get_all_genres()
        print('Genre List (totalsize: ' + str(len(genre_list)) + ')\n')
        print(genre_list)
        print('\n')

    def print_dataset_info(dataset):
        ''' Print dataset info'''
        SAMPLE_SIZE = 2

        print('Printing sample data...\n')

        for i in dataset[:SAMPLE_SIZE]:
            print('Song: #' + str(i))
            print('Title: ' + str(extractor.get_title(i)))
            print('Genre: ' + str(extractor.get_genre(i)))
            print('Median MFCC:\n' + str(extractor.get_feature(i, feature_type.MFCC, statistic_type.MEDIAN)) + '\n')
            print('Median Chroma STFT:\n' + str(extractor.get_feature(i, feature_type.CHROMA_STFT, statistic_type.MEDIAN)) + '\n')
            print('Median Spectral Contrast:\n' + str(extractor.get_feature(i, feature_type.SPEC_CONTRAST, statistic_type.MEDIAN)) + '\n')

if __name__ == '__main__':
    unittest.main()






