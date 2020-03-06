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
import time
import itertools

from enum import Enum
from pandas.api.types import CategoricalDtype

class feature_type(Enum):
    CHROMA_STFT = 1
    TONNETZ = 2
    MFCC = 3
    SPEC_CENTROID = 4
    SPEC_BANDWIDTH = 5
    SPEC_CONTRAST = 6
    SPEC_ROLLOFF = 7
    RMS_ENERGY = 8
    ZERO_CROSSING_RATE = 9
    
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

        print('Finding the following metadata files:\n')
        print(self.TRACKS_FILE + '\n')
        print(self.GENRE_FILE + '\n')
        print(self.FEATURES_FILE + '\n')

        # DataFrame keys
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
        self.GENRES_TOP = 'genres_top'
        self.GENRE_ID = 'genre_id'
        self.TRACKS = 'tracks'
        self.TOP_LEVEL = 'top_level'


        self.feature_types_str = {}
        self.feature_types_str[feature_type.CHROMA_STFT] = 'chroma_stft';
        self.feature_types_str[feature_type.MFCC] = 'mfcc';
        self.feature_types_str[feature_type.RMS_ENERGY] = 'rmse';
        self.feature_types_str[feature_type.SPEC_BANDWIDTH] = 'spectral_bandwidth';
        self.feature_types_str[feature_type.SPEC_CENTROID] = 'spectral_centroid';
        self.feature_types_str[feature_type.SPEC_CONTRAST] = 'spectral_contrast';
        self.feature_types_str[feature_type.SPEC_ROLLOFF] = 'spectral_rolloff';
        self.feature_types_str[feature_type.TONNETZ] = 'tonnetz';
        self.feature_types_str[feature_type.ZERO_CROSSING_RATE] = 'zcr';

        self.statistic_types_str = {}
        self.statistic_types_str[statistic_type.KURTOSIS] = 'kurtosis';
        self.statistic_types_str[statistic_type.MEDIAN] = 'median';
        self.statistic_types_str[statistic_type.MEAN] = 'mean';
        self.statistic_types_str[statistic_type.MAX] = 'max';
        self.statistic_types_str[statistic_type.MIN] = 'min';
        self.statistic_types_str[statistic_type.SKEW] = 'skew';
        self.statistic_types_str[statistic_type.STD] = 'std';

        self.__load_data()

    def __load_data(self):
        ''' Load metadata and features'''

        start_time = time.time()

        self.__load_tracks()

        self.__load_genres()

        # Load features last
        self.features = self.load(self.FEATURES_FILE)

        print ('Elapsed time: ' + str(time.time() - start_time) + ' seconds\n')


    def __load_tracks(self):
        '''  Load tracks metadata and dataset '''
        # Load tracks metadata
        self.tracks = self.load(self.TRACKS_FILE)

        # Get training, validation, and test datasets
        self.dataset = self.tracks[self.tracks[self.SET, self.SUBSET] == self.SMALL]
        self.training_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TRAINING]
        self.validation_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.VALIDATION]
        self.test_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TEST]

        self.list_of_all_song_ids = self.get_training_dataset_song_ids()
        self.list_of_all_song_ids.extend(self.get_validation_dataset_song_ids())
        self.list_of_all_song_ids.extend(self.get_validation_dataset_song_ids())

    def __load_genres(self):
        '''  Load genre metadata '''
        # Load genre metadata
        self.genres = self.load(self.GENRE_FILE)

        # Get genres in dataset
        list_of_all_genres = self.training_dataset[self.TRACK, self.GENRES_TOP].to_list()
        list_of_all_genres.extend(self.validation_dataset[self.TRACK, self.GENRES_TOP].to_list())
        list_of_all_genres.extend(self.test_dataset[self.TRACK, self.GENRES_TOP].to_list())

        genre_array = np.array(list_of_all_genres)
        self.list_of_all_genres = np.unique(genre_array).tolist()

    def load(self, filepath):
        ''' The following method was taken from the FMA repository and heavily modified.'''
        ''' Original source code: ttps://github.com/mdeff/fma/blob/master/utils.py '''

        filename = os.path.basename(filepath)
        CHUNK_SIZE = 5000

        print ('Loading ' + filename + '...')

        if self.FEATURES in filename:
            HEADER_SIZE = 1

            features = pd.DataFrame()
            ids = []
            ids.extend(self.get_training_dataset_song_ids())
            ids.extend(self.get_validation_dataset_song_ids())
            ids.extend(self.get_test_dataset_song_ids())
            ids = list(map(str, ids))

            header = pd.read_csv(filepath, nrows=HEADER_SIZE)
            features = header

            for file_chunk in pd.read_csv(filepath, low_memory=False, chunksize=CHUNK_SIZE):
                temp = file_chunk.loc[file_chunk['feature'].isin(ids)]

                if temp.shape[0] > 0:
                    features = features.append(temp)

            features = features.set_index('feature')

            print ('Loaded ' + filename + '\n')

            return features

        if self.GENRES in filename:
            genre_list = []
                
            for chunk in  pd.read_csv(filepath, index_col=0, chunksize=CHUNK_SIZE, low_memory=False):
                genre_list.append(chunk)
  
            print ('Loaded ' + filename  + '\n')

            return pd.concat(genre_list,sort=False)


        if self.TRACKS in filename:
            track_list = []
                
            for chunk in  pd.read_csv(filepath, index_col=0, header=[0, 1], chunksize=CHUNK_SIZE, low_memory=False):
                track_list.append(chunk)

            tracks = pd.concat(track_list,sort=False)
  
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

            print ('Loaded ' + filename  + '\n')

            return tracks

    def get_all_song_ids(self):
        ''' Get all song ids '''
        return self.self.list_of_all_song_ids

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
        return self.tracks.loc[track_id, self.TRACK][self.GENRES_TOP]

    def get_filename(self, track_id):
        ''' Get filename of track '''
        ''' track_id - unique ID of the song in dataset ''' 
        pass

    def get_feature(self, track_id, feat_type, stat_type):
        ''' Return feature '''
        ''' track_id - unique ID of the song in dataset '''
        ''' feat_type - feature type: Chroma, Tonnetz, MFCC, etc. '''
        ''' stat_type - statistic type: max, min, median, etc. '''
        feat_type_str = self.feature_types_str[feat_type]
        stat_type_str = self.statistic_types_str[stat_type]

        ret = self.features.filter(regex=feat_type_str)
        ret = ret.loc[:, ret.loc['statistics'] == stat_type_str]
        ret = ret.loc[str(track_id)]
        ret_list = list(map(np.float32, ret.to_list()))
        return ret_list

    def get_all_genres(self):
        ''' Return all genre types '''
        return self.list_of_all_genres
