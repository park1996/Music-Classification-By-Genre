import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

np.random.seed(0)

class k_means:
    def __init__(self):
        ''' Constructor for this class '''
    
    def trainWithSkLearn(self, features, labels):
        kmeans = KMeans(n_clusters=numberOfClusters)
        kmeans.fit(features)
        predicted_clusters = kmeans.predict(features)

        display_clusters(predicted_clusters, labels)

    
    def train(self, features, labels, trainWithSkLearn=False):
        number_of_clusters = len(labels)
        centers = initialize_centers(features, number_of_clusters)
        predicted_clusters = []
        it = 0 
        while True:
            predicted_clusters.append(assign_clusters(features, centers[-1]))
            new_centers = kmeans_update_centers(features, predicted_clusters[-1], centers)
            if has_converged(centers[-1], new_centers):
                break
            centers.append(new_centers)
            it += 1
        return (centers, predicted_clusters, it)

    def initialize_centers(self, features, number_of_clusters):
        print (type(np.random.choice(features.shape[0], number_of_clusters, replace=False)))
        return features[np.random.choice(features.shape[0], number_of_clusters, replace=False)]

    def update_centers(self, features, labels, centers):
        centers = np.zeros((len(labels), features.shape[0]))
        for iteration in range(len(labels)):
            cluster = features[labels == iteration, :]
            # take average
            centers[iteration,:] = np.mean(cluster, axis = 0)
        return centers

    def assign_clusters(self, features, centers):
        distances = cdist(features, centers)
        return np.argmin(distances, axis=0)

    def is_converged(self, centers, new_centers):  
        return (set([tuple(center) for center in centers]) == 
        set([tuple(center) for center in new_centers]))

    def display_clusters(self, clusters, labels, title, FIG=1, figsize=(8.5, 6), marketsize=4, alpha=0.75):
        plt.figure(FIG, figsize)
        plt.title(title)