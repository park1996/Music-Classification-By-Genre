import os
import ntpath
import sys
import csv
import time
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from k_means import k_means

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

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

RANDOM_SEED = 1987

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

def inertia_plot(n_runs, n_init_range, features, number_of_clusters, labels, label_names, random_seed):
    plt.figure()
    plots = []
    legends = []
    n_runs = n_runs
    n_init_range = n_init_range
    train_centers = None
    km_util = k_means(random_seed)
    cases = [
        (KMeans, 'k-means++', {}),
        (KMeans, 'random', {}),
        (KMeans, 'custom', {}),
        (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
        (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500})
    ]

    for factory, init, params in cases:
        print("Evaluation of %s with %s init" % (factory.__name__, init))
        inertia = np.empty((len(n_init_range), n_runs))
        for run_id in range(n_runs):
            for i, n_init in enumerate(n_init_range):
                if init == 'custom':
                    custom_inertias = []
                    for j in range(n_init):
                        centers = km_util.initialize_centers(features, labels, label_names)
                        km = factory(n_clusters=number_of_clusters, init=centers, random_state=run_id,
                                n_init=1, **params).fit(features)
                        custom_inertias.append(km.inertia_)
                    inertia[i, run_id] = min(custom_inertias)
                else:
                    km = factory(n_clusters=number_of_clusters, init=init, random_state=run_id,
                                n_init=n_init, **params).fit(features)
                    inertia[i, run_id] = km.inertia_
        p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
        plots.append(p[0])
        legends.append("%s with %s init" % (factory.__name__, init))

    plt.xlabel('n_init')
    plt.ylabel('inertia')
    plt.legend(plots, legends)
    plt.title("Mean inertia for various k-means init across %d runs" % n_runs)
    plt.show()

'''
Step 1: Read and Preprocess data 
'''
fe = feature_extractor.feature_extractor(use_echonest_dataset=False)
#scaler was used to scale each feature to range [0,1]
scaler = StandardScaler()
#pca was used to reduced feature dimensions
pca = PCA(n_components=2)

#get training data and apply scaler and pca
train_ids = np.asarray(fe.get_training_dataset_song_ids())
train_features, train_genres = read_data(fe, train_ids)#read_echonest_data(fe, train_ids)
train_features = scaler.fit_transform(train_features)
train_features = pca.fit_transform(train_features)

#get validation data and apply scaler and pca
validation_ids = np.asarray(fe.get_validation_dataset_song_ids())
validation_features, validation_genres = read_data(fe, validation_ids)#read_echonest_data(fe, validation_ids)
validation_features = scaler.transform(validation_features)
validation_features = pca.transform(validation_features)

#get test data and apply scaler and pca
test_ids = np.asarray(fe.get_test_dataset_song_ids())
test_features, test_genres = read_data(fe, test_ids) #read_echonest_data(fe, test_ids)
test_features = scaler.transform(test_features)
test_features = pca.transform(test_features)

#number of clusters is number of genres
number_of_clusters = get_number_of_clusters(fe)
#get list of all gernes
all_genre_names = fe.get_all_genres()

'''
Step 2: K-means Clustering
'''
km = k_means(RANDOM_SEED)

#generating inertia plot of different initialization strategies
n_run = 5
n_init_range = np.array([1, 5, 10, 15, 20])
inertia_plot(n_run, n_init_range, train_features, number_of_clusters, train_genres, all_genre_names, RANDOM_SEED)

#initialize 
n_trials = 100
km_models = []
evaluation_values = []
ls_train_genres = []
ls_validatation_genres = []
ls_train_centers = []

start_time = time.time()

print('\nK-means clustering model...')

#iterate through number of trials to select the best kmeans model, using custom nitialization strategy
for i in range(n_trials):
    print('Trial #: ' + str(i))
    train_centers = km.initialize_centers(train_features, train_genres, all_genre_names)

    km_model = KMeans(init=train_centers, n_clusters=number_of_clusters, n_init=1).fit(train_features)
    #km_model = KMeans(init='random', n_clusters=number_of_clusters, n_init=15).fit(train_features)
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

max_eval_value = max(evaluation_values)
max_eval_idx = evaluation_values.index(max_eval_value)

best_km_model = km_models[max_eval_idx]

test_clusters = best_km_model.predict(test_features)
predicted_test_genres = km.label_clusters(test_clusters, all_genre_names, len(test_genres))
test_accuracy_rate = km.accuracy_rate(predicted_test_genres, test_genres)

print('##########################')
print('Best trial: ' + str(max_eval_idx))
print('Test Accuracy rate: ' + str(test_accuracy_rate))
print('###########################')

#Producing plots for analysis
colors = ['b', 'g', 'r', 'c', 'm', 'y','tab:brown','tab:orange']
plot_title = 'Plot of Predicted Clusters for Train Data'
km.display_clusters(train_features, ls_train_genres[max_eval_idx], all_genre_names, 0, 1, plot_title, colors, FIG=1)

plot_title = 'Plot of Actual Labeled Clusters for Train Data'
km.display_clusters(train_features, train_genres, all_genre_names, 0, 1, plot_title, colors, FIG=2)

plot_title = 'Plot of Predicted Clusters for Validation Data'
km.display_clusters(validation_features, ls_validatation_genres[max_eval_idx], all_genre_names, 0, 1, plot_title, colors, FIG=3)

plot_title = 'Plot of Actual Labeled Clusters for Validation Data'
km.display_clusters(validation_features, validation_genres, all_genre_names, 0, 1, plot_title, colors, FIG=4)

plot_title = 'Plot of Predicted Clusters for Test Data'
km.display_clusters(test_features, predicted_test_genres, all_genre_names, 0, 1, plot_title, colors, FIG=5)

plot_title = 'Plot of Actual Labeled Clusters for Test Data'
km.display_clusters(test_features, test_genres, all_genre_names, 0, 1, plot_title, colors, FIG=6)

all_features = np.vstack((np.vstack((train_features,validation_features)), test_features))
all_predicted_genres = ls_train_genres[max_eval_idx] + ls_validatation_genres[max_eval_idx] + predicted_test_genres
all_genres = np.append(np.append(train_genres, validation_genres), test_genres)

plot_title = 'Plot of Predicted Clusters for All Data'
km.display_clusters(all_features, all_predicted_genres, all_genre_names, 0, 1, plot_title, colors, FIG=4)

plot_title = 'Plot of Actual Labeled Clusters for All Data'
km.display_clusters(all_features, all_genres, all_genre_names, 0, 1, plot_title, colors, FIG=4)