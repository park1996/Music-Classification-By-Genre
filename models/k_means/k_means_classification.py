import pandas as pd
import numpy as np
from k_means import k_means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
import ntpath
import sys
import csv
import time
import math

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
ECHO_DIR = os.path.join(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)),'fma_metadata\echonest.csv')

import feature_extractor as feature_extractor

FEATURE_TYPES = [feature_extractor.feature_type.MFCC,
                feature_extractor.feature_type.SPEC_CENTROID,
                feature_extractor.feature_type.SPEC_CONTRAST]
               
STATISTIC_TYPES = [feature_extractor.statistic_type.MEAN,
                feature_extractor.statistic_type.STD]

ECHONEST_FEATURE_TYPES = [feature_extractor.echonest_feature_type.ACOUSTICNESS,
                        feature_extractor.echonest_feature_type.DACNEABILITY,
                        feature_extractor.echonest_feature_type.ENERGY,
                        feature_extractor.echonest_feature_type.INSTRUMENTALNESS,
                        feature_extractor.echonest_feature_type.LIVENESS,
                        feature_extractor.echonest_feature_type.SPEECHINESS,
                        feature_extractor.echonest_feature_type.TEMPO,
                        feature_extractor.echonest_feature_type.VALENCE]

#Remove outlier using IQR
def getValidIndices(features):
    q3 = np.percentile(features, 75, axis=0)
    q1 = np.percentile(features, 25, axis=0)
    iqr = q3 - q1
    return ~((features < (q1 - 1.5 * iqr)) |(features > (q3 + 1.5 * iqr))).any(axis=1)

def read_echonest_data(fe, ids):
    features = get_echonest_features(fe, ids)
    genres = get_genres(fe,ids)
    valid_indices = getValidIndices(features)
    features = features[valid_indices]
    genres = genres[valid_indices]
    return features, genres

def get_echonest_features(fe, ids):
    print("Constructing feature matrix...")
    start_time = time.time()
    features = fe.get_echonest_features_as_nparray(ids, ECHONEST_FEATURE_TYPES)
    print('Processing features elapsed time: ' + str(time.time() - start_time) + ' seconds\n')
    return features
    
def read_data(fe, song_ids):
    genres = get_genres(fe, song_ids)
    features = get_features(fe, song_ids)
    valid_indices = getValidIndices(features)
    features = features[valid_indices]
    genres = genres[valid_indices]
    return features, genres

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
    features = fe.get_features_as_nparray(ids, FEATURE_TYPES, STATISTIC_TYPES) 
    print('Processing features elapsed time: ' + str(time.time() - start_time) + ' seconds\n')
    return features


fe = feature_extractor.feature_extractor(use_echonest_dataset=True)

scaler = StandardScaler()
pca = PCA(n_components=3)

train_ids = np.asarray(fe.get_training_dataset_song_ids())
train_features, train_genres = read_echonest_data(fe, train_ids)
train_features = scaler.fit_transform(train_features)
train_features = pca.fit_transform(train_features)


validation_ids = np.asarray(fe.get_validation_dataset_song_ids())
validation_features, validation_genres = read_echonest_data(fe, validation_ids)
validation_features = scaler.transform(validation_features)
validation_features = pca.transform(validation_features)

test_ids = np.asarray(fe.get_test_dataset_song_ids())
test_features, test_genres = read_echonest_data(fe, test_ids)
test_features = scaler.transform(test_features)
test_features = pca.transform(test_features)

number_of_clusters = get_number_of_clusters(fe)

all_genre_names = fe.get_all_genres()
km = k_means(1987)
print('All genre names: ')
print(all_genre_names)

n_trials = 500
km_models = []
evaluation_values = []
ls_train_genres = []
ls_validatation_genres = []

start_time = time.time()
print('\nK-means clustering model...')
for i in range(n_trials):
    
    train_centers = km.initialize_centers(train_features, train_genres, all_genre_names)

    km_model = KMeans(init=train_centers, n_clusters=number_of_clusters, n_init=1).fit(train_features)
    train_clusters = km_model.labels_
    train_its = km_model.n_iter_

    predicted_train_genres = km.label_clusters(train_clusters, all_genre_names, len(train_genres))

    train_accuracy_rate = km.accuracy_rate(predicted_train_genres, train_genres)

    validation_clusters = km_model.predict(validation_features)
    predicted_validation_genres = km.label_clusters(validation_clusters, all_genre_names, len(validation_genres))
    validation_accuracy_rate = km.accuracy_rate(predicted_validation_genres, validation_genres)
    
    value = validation_accuracy_rate + train_accuracy_rate - math.fabs(validation_accuracy_rate - train_accuracy_rate)
    
    ls_train_genres.append(predicted_train_genres)
    ls_validatation_genres.append(predicted_validation_genres)
    km_models.append(km_model)
    evaluation_values.append(value)
    
print('k-mean clustering elapsed time: ' + str(time.time() - start_time) + ' seconds\n')


max_total_rate = max(evaluation_values)
max_total_rate_idx = evaluation_values.index(max_total_rate)


best_km_model = km_models[max_total_rate_idx]

test_clusters = best_km_model.predict(test_features)
predicted_test_genres = km.label_clusters(test_clusters, all_genre_names, len(test_genres))
test_accuracy_rate = km.accuracy_rate(predicted_test_genres, test_genres)


print('##########################')
print('Best trial: ' + str(max_total_rate_idx))
print('Test Accuracy rate: ' + str(test_accuracy_rate))
print('###########################')

colors = ['b', 'g', 'r', 'c', 'm', 'y','tab:brown','tab:orange']
plot_title = 'Plot of Predicted Clusters for Train Data'
km.display_clusters(train_features, ls_train_genres[max_total_rate_idx], all_genre_names, 0, 1, plot_title, colors, FIG=1)

plot_title = 'Plot of Predicted Clusters for Validation Data'
km.display_clusters(validation_features, ls_validatation_genres[max_total_rate_idx], all_genre_names, 0, 1, plot_title, colors, FIG=2)

plot_title = 'Plot of Predicted Clusters for Test Data'
km.display_clusters(test_features, predicted_test_genres, all_genre_names, 0, 1, plot_title, colors, FIG=3)

all_features = np.vstack((np.vstack((train_features,validation_features)), test_features))
all_predicted_genres = ls_train_genres[max_total_rate_idx] + ls_validatation_genres[max_total_rate_idx] + predicted_test_genres

plot_title = 'Plot of Predicted Clusters for All Data'
km.display_clusters(all_features, all_predicted_genres, all_genre_names, 0, 1, plot_title, colors, FIG=4)



