from collections import Counter

import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist

from matching.games import HospitalResident

class k_means:
    def __init__(self):
        ''' Constructor for this class '''
    
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
    
    def create_cluster_label_by_max_count(self, cluster_label_ratios, number_of_clusters):
        cluster_labels = []
        for cluster_label_ratio in cluster_label_ratios:
                cluster_labels.append(list(cluster_label_ratio.keys())[0])
        return cluster_labels
    
    def create_cluster_label_by_matching(self, cluster_label_ratios, number_of_clusters):
        clusterprefers = {}
        labelprefers = {}
        for i in range(number_of_clusters):
            clusterprefers[i] = list(cluster_label_ratios[i].keys())
       
        for i in range(number_of_clusters):
            cluster_label_ratio = cluster_label_ratios[i]
            for key in cluster_label_ratio:
                value = (i, cluster_label_ratio[key])
                if key in labelprefers:
                    labelprefers[key].append(value)
                else:
                    lst = []
                    lst.append(value)
                    labelprefers[key] = lst
        
        for key in labelprefers:
            labelprefers[key] = sorted(labelprefers[key], reverse = True, key=lambda x: x[1])
            lst = []
            for i in range(len(labelprefers[key])):
                lst.append(labelprefers[key][i][0])
            labelprefers[key] = lst
        
        capacities = {cluster: 1 for cluster in clusterprefers}
        matcher = HospitalResident.create_from_dictionaries(labelprefers, clusterprefers, capacities)
        matched = matcher.solve()
       
        cluster_labels = []
        for key in matched:
            cluster_labels.append(str(matched[key][0]))
        return cluster_labels
    
    #map the cluster indices to the correct labels
    def map_labels(self, features, target_labels, predicted_clusters, number_of_clusters):
        input_matrix = np.hstack((features, target_labels.reshape((len(target_labels), 1))))
        cluster_label_ratios = []
        # get label counters for clusters
        for iteration in range(number_of_clusters):
            cluster = input_matrix[predicted_clusters == iteration, :]
            counter = Counter(cluster[:, -1])
            cluster_label_ratio = {}
            for label in counter:
                cluster_label_ratio[label] = round(counter[label]/len(cluster), 4)
            cluster_label_ratios.append({k: v for k, v in sorted(cluster_label_ratio.items(), reverse = True, key=lambda item: item[1])})
        return self.create_cluster_label_by_matching(cluster_label_ratios, number_of_clusters)

    #label data
    def label_clusters(self, predicted_clusters, cluster_labels, label_size):
        predicted_labels = []
        for i in range(label_size):
            predicted_labels.append(cluster_labels[predicted_clusters[i]])
        return predicted_labels

    #plot the clusters   
    def display_clusters(self, features, labels, label_names, first_feature_idx, second_feature_idx, title, FIG=1, figsize=(8.5, 6), markersize=4, alpha=0.75):
        plt.figure(FIG, figsize)
        plt.title(title)
        nplabels = np.asarray(labels)
        for label_name in label_names:
            indices = np.array(np.where(nplabels == label_name)[0])
            cluster = features[indices, :]
            color = np.random.rand(3,)
            plt.plot(cluster[:, first_feature_idx], cluster[:, second_feature_idx], c=color, marker='.', markersize=markersize, alpha=alpha, linestyle='None')
        plt.axis('equal')
        plt.show()