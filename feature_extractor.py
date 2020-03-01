from enum import Enum
import os
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
from fma_utils import utils

class feature_type(Enum):
    CHROMA = 1
    TONNETZ = 2
    MFCC = 3
    SPEC_CENTROID = 4
    SPEC_BANDWIDTH = 5
    SPEC_CONTRAST = 6
    SPEC_ROLLOFF = 7
    RMS_ENERGY = 8
    ZERO_CROSSING_RATE = 9
    SPECTROGRAM = 10
    
class statistic_type(Enum):
    KURTOSIS = 1
    MAX = 2
    MEAN = 3
    MEDIAN = 4
    MIN = 5
    SKEW = 6
    STD = 7  

class feature_extractor:
    def __init__(self):
        ''' Constructor for this class '''

        # Metadata csv files
        self.META_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fma_metadata')
        self.TRACKS_FILE = os.path.join(self.META_DATA_DIR, 'tracks.csv')
        self.GENRE_FILE = os.path.join(self.META_DATA_DIR, 'genres.csv')
        self.FEATURES_FILE = os.path.join(self.META_DATA_DIR, 'features.csv')

        print('Loading the following metadata files:')
        print('\t' + self.FEATURES_FILE)        
        print('\t' + self.TRACKS_FILE)
        print('\t' + self.GENRE_FILE)

        # Dataframe keys
        self.TRACK = 'track'
        self.TITLE = 'title'
        self.GENRE = 'genre'
        self.FEATURE = 'feature'
        self.TRAINING = 'training'
        self.VALIDATION = 'validation'
        self.SUBSET = 'subset'
        self.SMALL = 'small'
        self.SPLIT = 'split'
        self.SET = 'set'
        self.TEST = 'test'
        self.GENRES = 'genres'

        self.__load_data()

    def __load_data(self):
        ''' Load metadata and features '''
        self.tracks = utils.load(self.TRACKS_FILE)
        self.genres = utils.load(self.GENRE_FILE)
        print('Genres:' + str(self.genres))

        self.dataset = self.tracks[self.tracks[self.SET, self.SUBSET] == self.SMALL]
        self.training_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TRAINING]
        self.validation_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.VALIDATION]
        self.test_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TEST]

        #self.features = utils.load(self.FEATURES_FILE)
        #np.testing.assert_array_equal(self.features.index, self.tracks.index)

    def get_training_dataset_song_ids(self):
        ''' Get all song ids in training set '''
        return self.training_dataset.index.tolist()

    def get_validation_dataset_song_ids(self):
        ''' Get all song ids in validation dataset '''
        return self.validation_dataset.index.tolist()

    def get_test_dataset_song_ids(self):
        ''' Get all song ids in test dataset '''
        return self.test_dataset.index.tolist()

    def get_title(self, track_id):
        ''' Get title of track '''
        ''' track_id - unique ID of the song in dataset '''
        return self.tracks.loc[track_id, self.TRACK][self.TITLE]

    def get_genre(self, track_id):
        ''' Get genre of track '''
        ''' track_id - unique ID of the song in dataset '''
        index = self.tracks.loc[track_id, self.TRACK][self.GENRES]
        genre_list = []
        for i in index:
            genre_list.append(self.genres.loc[i][self.TITLE])
        
        return genre_list

    def get_track_id(self, filename):
        ''' Get filename of track '''
        ''' track_id - unique ID of the song in dataset ''' 
        pass

    def get_filename(self, track_id):
        ''' Get filename of track '''
        ''' track_id - unique ID of the song in dataset ''' 
        pass

    def get_feature(self, track_id, feat_type, stat_type):
        ''' Return Mel-frequency cepstral coefficients (MFCCs) '''
        ''' track_id - unique ID of the song in dataset '''
        ''' feat_type - feature type: Chroma, Tonnetz, MFCC, etc. '''
        ''' stat_type - statistic type: max, min, median, etc. '''
        pass

    def get_mfcc(self, track_id, stat_type):
        ''' Return Mel-frequency cepstral coefficients (MFCCs) '''
        ''' track_id - unique ID of the song in dataset '''
        ''' stat_type - statistic type: max, min, median, etc. '''
        pass

    def get_chroma(self, track_id, stat_type):
        ''' Return Chroma '''
        ''' track_id - unique ID of the song in dataset ''' 
        ''' stat_type - statistic type: max, min, median, etc. '''
        pass

    def get_spectral_contrast(self, track_id, stat_type):
        ''' Get spectral contrast '''
        ''' stat_type - statistic type: max, min, median, etc. '''
        pass
