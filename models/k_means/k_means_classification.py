import pandas as pd
import numpy as np
from k_means import k_means
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import os
import ntpath
import sys
import csv
import time

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
ECHO_DIR = os.path.join(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)),'fma_metadata\echonest.csv')

import feature_extractor as feature_extractor


def read_echonest_data():
    df = pd.read_csv(ECHO_DIR, sep=',', header=None, low_memory=False)
    echonest_types = df.iloc[2, 1:9].to_numpy()
    echonest_data = df.iloc[4:, 0:9]
    echonest_data.set_index(0, inplace=True)
    return echonest_types, echonest_data

def get_echonest_features(df, ids):
    return df.loc[(map(str, ids))].apply(np.float32).to_numpy()
   

    
def read_data(fe, song_ids):
    song_genres = get_genres(fe, song_ids)
    features = get_features(fe, song_ids)
    return features, song_genres

def get_genres(fe, ids):
    genres = []
    for i in ids:
        genres.append(fe.get_genre(i))
    return np.array(genres)

def get_number_of_clusters(fe):
    all_genres = fe.get_all_genres()
    return len(all_genres)

def get_features(fe, ids):
    print("Constructing feature matrix...")
    start_time = time.time()
    # features = []
    # for i in ids:
    #     i_features = []
    #     for feat_type in feature_extractor.feature_type:
    #         for stat_type in feature_extractor.statistic_type:
    #             i_features.extend(fe.get_feature(i, feat_type, stat_type))
    #     features.append(i_features)
    features = fe.get_features_as_nparray(ids)
    print('Processing features elapsed time: ' + str(time.time() - start_time) + ' seconds\n')
    return features

feature_types, df = read_echonest_data()
echonest_ids = df.index.to_numpy().astype(int)

fe = feature_extractor.feature_extractor()

scaler = StandardScaler()

train_ids = np.asarray(fe.get_training_dataset_song_ids())

train_ids = np.intersect1d(train_ids, echonest_ids)
train_features = get_echonest_features(df, train_ids)
train_features = scaler.fit_transform(train_features)
train_genres = get_genres(fe, train_ids)

# validation_ids = np.asarray(fe.get_validation_dataset_song_ids())
# validation_ids = np.intersect1d(validation_ids, echonest_ids)
# validation_features = get_echonest_features(df, validation_ids)
# validation_genres = get_genres(fe, validation_ids)

# train_ids = np.asarray(fe.get_training_dataset_song_ids())
# train_features, train_genres = read_data(fe, train_ids)
# train_features = scaler.fit_transform(train_features)

# validation_ids = np.asarray(fe.get_validation_dataset_song_ids())
# validation_features, validation_genres = read_data(fe, validation_ids)

# test_ids = np.asarray(fe.get_test_dataset_song_ids())
# test_features, test_genres = read_data(fe, test_ids)


number_of_clusters = get_number_of_clusters(fe)

all_genre_names = fe.get_all_genres()
km = k_means()
print('All genre names: ')
print(all_genre_names)

start_time = time.time()
train_centers = km.initialize_centers(train_features, train_genres, all_genre_names)

# , n_init=2000
km_model = KMeans(init=train_centers, n_clusters=number_of_clusters, random_state=10).fit(train_features)
# km_model = MiniBatchKMeans(init=train_centers, n_clusters=number_of_clusters, random_state=0, batch_size=10, max_iter=500, reassignment_ratio=0.1, tol=0.000001).fit(train_features)
train_clusters = km_model.labels_
train_its = km_model.n_iter_
print('Number of iteration: ' + str(train_its))
print('k-mean clustering elapsed time: ' + str(time.time() - start_time) + ' seconds\n')

# cluster_genres = km.map_labels(train_features, train_genres, train_clusters, number_of_clusters)
# print('Cluster genres')
# print(cluster_genres)

predicted_train_genres = km.label_clusters(train_clusters, all_genre_names, len(train_genres))

accuracy_rate = km.accuracy_rate(predicted_train_genres, train_genres)
print('Accuracy rate: ' + str(accuracy_rate))

plot_title = 'Plot of Predicted Clusters'
km.display_clusters(train_features, predicted_train_genres, all_genre_names, 0, 1, plot_title)



