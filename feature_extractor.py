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
import ast
import warnings

from enum import Enum
from fma_utils import utils
from pandas.api.types import CategoricalDtype

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
        print('\t' + self.TRACKS_FILE)
        print('\t' + self.GENRE_FILE)
        print('\t' + self.FEATURES_FILE)
        print('\n')

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
        self.FEATURES = 'features'
        self.GENRES = 'genres'
        self.TRACKS = 'tracks'

        self.__load_data()

    def __load_data(self):
        ''' Load datasets and features '''

        # Load tracks and genres dataset
        self.tracks = self.load(self.TRACKS_FILE)
        self.genres = self.load(self.GENRE_FILE)

        # Parse training, validation, and test datasets
        self.dataset = self.tracks[self.tracks[self.SET, self.SUBSET] == self.SMALL]
        self.training_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TRAINING]
        self.validation_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.VALIDATION]
        self.test_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TEST]

        # Load features last
        self.features = self.load(self.FEATURES_FILE)

        # Sanity check: check if number of feature rows is equal to the dataset rows
        np.testing.assert_array_equal(self.features.shape[0], self.dataset.shape[0])


    def load(self, filepath):
        ''' The following method was taken from the FMA repository and heavily modified.'''
        ''' Original source code: ttps://github.com/mdeff/fma/blob/master/utils.py '''

        filename = os.path.basename(filepath)

        if self.FEATURES in filename:
                print ('Loading features...')

                CHUNK_SIZE = 5000
                self.features = pd.DataFrame()
                ids = []
                ids.extend(self.get_training_dataset_song_ids())
                ids.extend(self.get_validation_dataset_song_ids())
                ids.extend(self.get_test_dataset_song_ids())
                ids = list(map(str, ids))

                for file_chunk in pd.read_csv(filepath, low_memory=False, chunksize=CHUNK_SIZE):
                    temp = file_chunk.loc[file_chunk['feature'].isin(ids)]
                    if temp.shape[0] > 0:
                        if self.features.shape[0] == 0:
                            self.features = temp
                        else:
                            self.features = self.features.append(temp)

                print ('Loaded features...')

                return self.features

        if self.GENRES in filename:
                print ('Loading genres...')

                genres = pd.read_csv(filepath, index_col=0)

                print ('Loaded genres...')


        if self.TRACKS in filename:
            print ('Loading tracks...')

            tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

            COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                               ('track', 'genres'), ('track', 'genres_all')]

            for column in COLUMNS:
                    tracks[column] = tracks[column].map(ast.literal_eval)
 
            COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                               ('album', 'date_created'), ('album', 'date_released'),
                               ('artist', 'date_created'), ('artist', 'active_year_begin'),
                               ('artist', 'active_year_end')]

            for column in COLUMNS:
                    tracks[column] = pd.to_datetime(tracks[column])

            SUBSETS = ('small', 'medium', 'large')

            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                       'category', CategoricalDtype(categories=SUBSETS, ordered=True)).cat.as_ordered()

            COLUMNS = [('track', 'license'), ('artist', 'bio'),
                               ('album', 'type'), ('album', 'information')]
            for column in COLUMNS:
                    tracks[column] = tracks[column].astype('category')

            print ('Loaded tracks...')

            return tracks

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
