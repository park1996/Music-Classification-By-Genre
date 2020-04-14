import os
import numpy as np
import pandas as pd
import ast
import time

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

class echonest_feature_type(Enum):
    ACOUSTICNESS = 1
    DACNEABILITY = 2
    ENERGY = 3
    INSTRUMENTALNESS = 4
    LIVENESS = 5
    SPEECHINESS = 6
    TEMPO = 7
    VALENCE = 8

class statistic_type(Enum):
    KURTOSIS = 1
    MAX = 2
    MEAN = 3
    MEDIAN = 4
    MIN = 5
    SKEW = 6
    STD = 7

class feature_extractor:
    ''' Extracts features from metadata folder '''

    def __init__(self, use_echonest_dataset=False):
        ''' Constructor for this class '''

        # Metadata csv files
        self.META_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fma_metadata')
        self.TRACKS_FILE = os.path.join(self.META_DATA_DIR, 'tracks.csv')
        self.GENRE_FILE = os.path.join(self.META_DATA_DIR, 'genres.csv')
        self.FEATURES_FILE = os.path.join(self.META_DATA_DIR, 'features.csv')
        self.ECHONEST_FILE = os.path.join(self.META_DATA_DIR, 'echonest.csv')
        self.USE_ECHONEST_DATASET = use_echonest_dataset

        print('Finding the following metadata files:\n')
        print(self.TRACKS_FILE + '\n')
        print(self.GENRE_FILE + '\n')
        print(self.FEATURES_FILE + '\n')

        if self.USE_ECHONEST_DATASET == True:
            print(self.ECHONEST_FILE + '\n')

        # Important dataframe keys and strings
        self.TRACK = 'track'
        self.TITLE = 'title'
        self.GENRE = 'genre'
        self.ARTIST = 'artist'
        self.NAME = 'name'
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
        self.GENRES_TOP = 'genre_top'
        self.GENRE_ID = 'genre_id'
        self.TRACKS = 'tracks'
        self.ECHONEST = 'echonest'
        self.TOP_LEVEL = 'top_level'
        self.RAW = 'raw'
        self.STATISTICS = 'statistics'
        self.AUDIO_FEATURES = 'audio_features'

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

        self.echonest_feature_types_str = {}
        self.echonest_feature_types_str[echonest_feature_type.ACOUSTICNESS] = 'acousticness';
        self.echonest_feature_types_str[echonest_feature_type.DACNEABILITY] = 'danceability';
        self.echonest_feature_types_str[echonest_feature_type.ENERGY] = 'energy';
        self.echonest_feature_types_str[echonest_feature_type.INSTRUMENTALNESS] = 'instrumentalness';
        self.echonest_feature_types_str[echonest_feature_type.LIVENESS] = 'liveness';
        self.echonest_feature_types_str[echonest_feature_type.SPEECHINESS] = 'speechiness';
        self.echonest_feature_types_str[echonest_feature_type.TEMPO] = 'tempo';
        self.echonest_feature_types_str[echonest_feature_type.VALENCE] = 'valence';

        self.statistic_types_str = {}
        self.statistic_types_str[statistic_type.KURTOSIS] = 'kurtosis';
        self.statistic_types_str[statistic_type.MEDIAN] = 'median';
        self.statistic_types_str[statistic_type.MEAN] = 'mean';
        self.statistic_types_str[statistic_type.MAX] = 'max';
        self.statistic_types_str[statistic_type.MIN] = 'min';
        self.statistic_types_str[statistic_type.SKEW] = 'skew';
        self.statistic_types_str[statistic_type.STD] = 'std';
        self.list_of_all_song_ids = []

        self.__load_data()

    def __load_data(self):
        ''' Load metadata and features'''

        start_time = time.time()

        # Load echonest metadata first
        if self.USE_ECHONEST_DATASET == True:
            self.__load_echonest_features()

        self.__load_tracks()

        self.__load_genres()

        # Load features last
        self.__load_features()

        print ('Elapsed time: ' + str(time.time() - start_time) + ' seconds\n')

    def __load_tracks(self):
        '''  Load tracks metadata '''
        # Load tracks metadata
        self.tracks = self.__load(self.TRACKS_FILE)

        # Use echonest dataset if required
        if self.USE_ECHONEST_DATASET == True:
            self.dataset = self.tracks.loc[self.tracks.index.intersection([int(i) for i in self.echonest_features.index.tolist()])]
            self.dataset = self.dataset[self.dataset[self.SET, self.SUBSET] == self.SMALL]
        else:
            self.dataset = self.tracks[self.tracks[self.SET, self.SUBSET] == self.SMALL]

        # Get training, validation, and test datasets
        self.training_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TRAINING]
        self.validation_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.VALIDATION]
        self.test_dataset = self.dataset[self.dataset[self.SET, self.SPLIT] == self.TEST]

        self.list_of_all_song_ids.extend(self.get_training_dataset_song_ids())
        self.list_of_all_song_ids.extend(self.get_validation_dataset_song_ids())
        self.list_of_all_song_ids.extend(self.get_test_dataset_song_ids())

    def __load_genres(self):
        '''  Load genre metadata '''
        # Load genre metadata
        self.genres = self.__load(self.GENRE_FILE)

        # Get genres in dataset
        list_of_all_genres = self.training_dataset[self.TRACK, self.GENRES_TOP].to_list()
        list_of_all_genres.extend(self.validation_dataset[self.TRACK, self.GENRES_TOP].to_list())
        list_of_all_genres.extend(self.test_dataset[self.TRACK, self.GENRES_TOP].to_list())

        genre_array = np.array(list_of_all_genres)
        self.list_of_all_genres = np.unique(genre_array).tolist()

    def __load_features(self):
        '''  Load features metadata '''
        self.features = self.__load(self.FEATURES_FILE)

    def __load_echonest_features(self):
        '''  Load echonest metadata '''
        # Load echonest metadata
        self.echonest_features = self.__load(self.ECHONEST_FILE)

    def __load(self, filepath):
        ''' The following method was taken from the FMA repository and heavily modified.'''
        ''' Original source code: https://github.com/mdeff/fma/blob/master/utils.py '''

        filename = os.path.basename(filepath)
        CHUNK_SIZE = 5000

        print ('Loading ' + filename + '...')

        if not os.path.exists(filepath):
            print ("Failed to find metadata file\n")
            return

        if self.FEATURES in filename:
            HEADER_SIZE = 1

            features = pd.DataFrame()
            ids = list(map(str, self.list_of_all_song_ids))

            header = pd.read_csv(filepath, nrows=HEADER_SIZE)
            features = header

            for file_chunk in pd.read_csv(filepath, low_memory=False, chunksize=CHUNK_SIZE):
                temp = file_chunk.loc[file_chunk[self.FEATURE].isin(ids)]

                if temp.shape[0] > 0:
                    features = features.append(temp)

            features = features.set_index('feature')
            features.index = features.index.map(str)

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

        if self.ECHONEST in filename:
            echonest_feature_list = []

            for chunk in  pd.read_csv(filepath, header=[1, 2], index_col=0, chunksize=CHUNK_SIZE, low_memory=False):
                echonest_feature_list.append(chunk)

            print ('Loaded ' + filename  + '\n')
            echonest_features = pd.concat(echonest_feature_list,sort=False)
            echonest_features = echonest_features[self.AUDIO_FEATURES]
            echonest_features.index = echonest_features.index.map(str)

            return echonest_features

    def get_all_song_ids(self):
        ''' Get all song ids '''
        return self.list_of_all_song_ids

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

    def get_artist(self, track_id):
        ''' Get artist of track '''
        ''' track_id - unique ID of the song in dataset '''
        return self.tracks.loc[track_id, self.ARTIST][self.NAME]

    def get_echonest_feature(self, track_id, echonest_feat_type):
        ''' Return feature '''
        ''' track_id - unique ID of the song in dataset '''
        ''' echonest_feat_type - echonest feature type: acousticness, energy, danceability, etc. '''
        if echonest_feat_type not in self.echonest_feature_types_str:
            print("Invalid feature type")
            return None

        if self.USE_ECHONEST_DATASET == False:
            print('Not supported\n')
            return None

        echonest_feat_type_str = self.echonest_feature_types_str[echonest_feat_type]

        ret = self.echonest_features.filter(regex=echonest_feat_type_str)
        ret = ret.loc[str(track_id)]
        ret_list = list(map(np.float32, ret.to_list()))
        return ret_list

    def get_feature(self, track_id, feat_type, stat_type):
        ''' Return feature '''
        ''' track_id - unique ID of the song in dataset '''
        ''' feat_type - feature type: Chroma, Tonnetz, MFCC, etc. '''
        ''' stat_type - statistic type: max, min, median, etc. '''
        if (feat_type not in self.feature_types_str) or (stat_type not in self.statistic_types_str):
            print("Invalid feature or statistic type")
            return None

        feat_type_str = self.feature_types_str[feat_type]
        stat_type_str = self.statistic_types_str[stat_type]

        ret = self.features.filter(regex=feat_type_str)
        ret = ret.loc[str(track_id), ret.loc[self.STATISTICS] == stat_type_str]
        ret_list = list(map(np.float32, ret.to_list()))
        return ret_list
    
    def get_all_features_as_nparray(self, track_ids):
        '''Return feature dataframe as numpy array'''
        statistic_types = self.statistic_types_str.values()
        feature_types = self.feature_types_str.values()
        feature_types_regex = '|'.join(feature_types)
        ret = self.features.filter(regex=feature_types_regex)
        ret = ret.loc[list(map(str, track_ids)), ret.loc[self.STATISTICS].isin(statistic_types)]
        return ret.apply(np.float32).to_numpy()
    
    def get_features_as_nparray(self, track_ids, feature_types, statistic_types):
        feature_vals = []
        statistic_vals = []
        for feature_type in feature_types:
            feature_vals.append(self.feature_types_str[feature_type])
        for statistic_type in statistic_types:
            statistic_vals.append(self.statistic_types_str[statistic_type])
        feature_types_regex = '|'.join(feature_vals)
        ret = self.features.filter(regex=feature_types_regex)
        ret = ret.loc[list(map(str, track_ids)), ret.loc[self.STATISTICS].isin(statistic_vals)]
        return ret.apply(np.float32).to_numpy()

    def get_all_echonest_features_as_nparray(self, track_ids):
        '''Return feature dataframe as numpy array'''
        echonest_feature_types = self.echonest_feature_types_str.values()
        echonest_feature_types_regex = '|'.join(echonest_feature_types)
        ret = self.echonest_features.filter(regex=echonest_feature_types_regex)
        ret = ret.loc[list(map(str, track_ids))]
        return ret.apply(np.float32).to_numpy()
    
    def get_echonest_features_as_nparray(self, track_ids, echonest_feature_types):
        echonest_feature_vals = []
        for echonest_feature_type in echonest_feature_types:
            echonest_feature_vals.append(self.echonest_feature_types_str[echonest_feature_type])
        feature_types_regex = '|'.join(echonest_feature_vals)
        ret = self.echonest_features.filter(regex=feature_types_regex)
        ret = ret.loc[list(map(str, track_ids))]
        return ret.apply(np.float32).to_numpy()

    def get_all_genres(self):
        ''' Return all genre types '''
        return self.list_of_all_genres
