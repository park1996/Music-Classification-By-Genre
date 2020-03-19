import numpy as np
from k_means import k_means

# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
features = data[:, 0:3]

# Target values, 0 for class 1, 1 for class 2.
target_labels = data[:, 3].astype(int)
km = k_means()
number_of_clusters = 2

centers, predicted_clusters, it = km.train(features, target_labels.T, number_of_clusters)

cluster_labels = km.map_labels(features, target_labels, predicted_clusters, number_of_clusters)
predicted_labels = km.label_clusters(predicted_clusters, cluster_labels, len(target_labels))

accuracy_rate = km.accuracy_rate(predicted_labels, target_labels)

plot_title = 'Plot of Predicted Clusters'
km.display_clusters(features, predicted_labels, number_of_clusters, plot_title)
# new_centers = km.update_centers(features, predicted_clusters[-1], centers)


