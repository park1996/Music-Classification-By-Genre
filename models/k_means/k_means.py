from collections import Counter

import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist

from matching.games import HospitalResident

class k_means:
    def __init__(self,random_seed=0):
        ''' Constructor for this class '''
        np.random.seed(random_seed)
    
    def initialize_centers(self, features, labels, label_names):
        nplabels = np.asarray(labels)
        centers = []
        it = 0
        for label_name in label_names:
            indices = np.array(np.where(nplabels == label_name)[0])
            rand_centers = features[np.random.choice(indices, 10), :]
            center = np.mean(rand_centers, axis=0)
            centers.append(center)
        return np.array(centers)

    #calculate accuracy rate
    def accuracy_rate(self, predicted_labels, target_labels):
        hit = 0
        for it in range(len(predicted_labels)):
            if target_labels[it] == predicted_labels[it]:
                hit += 1
        return hit/(len(predicted_labels))
    
    #label data
    def label_clusters(self, predicted_clusters, cluster_labels, label_size):
        predicted_labels = []
        for i in range(label_size):
            predicted_labels.append(cluster_labels[predicted_clusters[i]])
        return predicted_labels

    #plot the clusters   
    def display_clusters(self, features, labels, label_names, first_feature_idx, second_feature_idx, title, colors, FIG=1, figsize=(8.5, 6), markersize=10, alpha=0.75):
        plt.figure(FIG, figsize)
        plt.title(title)
        nplabels = np.asarray(labels)
        cluster_plots = []
        i = 0
        for label_name in label_names:
            indices = np.array(np.where(nplabels == label_name)[0])
            cluster = features[indices, :]
            color = colors[i]
            i = i+1
            cluster_plot, = plt.plot(cluster[:, first_feature_idx], cluster[:, second_feature_idx], c=color, marker='.', markersize=markersize, alpha=alpha, linestyle='None')
            cluster_plots.append(cluster_plot)
        plt.legend(cluster_plots, label_names)
        plt.axis('equal')
        plt.show()