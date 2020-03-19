import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

np.random.seed(0)

class k_means:
    def __init__(self):
        ''' Constructor for this class '''
    
    def trainWithSkLearn(self, features, number_of_clusters):
        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(features)
        predicted_clusters = kmeans.predict(features)
    
    def train(self, features, target_labels, number_of_clusters):
        centers, indices = self.initialize_centers(features, number_of_clusters)
        predicted_clusters = []
        it = 0 
        while True:
            predicted_clusters = self.assign_clusters(features, centers)
            new_centers = self.update_centers(features, predicted_clusters, number_of_clusters, centers)
            if self.is_converged(centers, new_centers):
                break
            centers = new_centers
            it += 1
        
        #labelling clusters
        cluster_labels =[]
        print (features.shape)
        print (target_labels.shape)
        input_matrix = np.hstack((features, target_labels))
        print (input_matrix)

        for iteration in range(number_of_clusters):
            cluster = input_matrix[predicted_clusters == iteration, :]
            counts = np.bincount(cluster[:, -1])
            cluster_labels[iteration] = np.argmax(counts)

        return (centers, predicted_clusters, cluster_labels, it)

    def initialize_centers(self, features, number_of_clusters):
        random_indices = np.random.choice(features.shape[0], number_of_clusters, replace=False)
        return features[random_indices], random_indices

    def update_centers(self, features, predicted_clusters, number_of_clusters, centers):
        centers = np.zeros((number_of_clusters, features.shape[1]))
        for iteration in range(number_of_clusters):
            cluster = features[predicted_clusters == iteration, :]
            # take average
            centers[iteration,:] = np.mean(cluster, axis = 0)
        return centers

    def assign_clusters(self, features, centers):
        distances = cdist(features, centers)
        return np.argmin(distances, axis=1)

    def is_converged(self, centers, new_centers):  
        return (set([tuple(center) for center in centers]) == 
        set([tuple(center) for center in new_centers]))
    
    def accuracy_rate(self, predicted_labels, target_labels):
        hit = 0
        for it in range(len(predicted_labels)):
            if target_labels[it] == predicted_labels[it]:
                hit += 1
        return hit/(len(predicted_labels))

    def display_clusters(self, clusters, labels, title, FIG=1, figsize=(8.5, 6), marketsize=4, alpha=0.75):
        plt.figure(FIG, figsize)
        plt.title(title)