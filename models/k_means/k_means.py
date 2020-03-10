import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class k_means:
    def __init__(self, number):
        ''' Constructor for this class '''
    
    def trainWithSkLearn(self, features, numberOfClusters):
        kmeans = KMeans(n_clusters=numberOfClusters)
        kmeans.fit(features)
        predicted_clusters = kmeans.predict(features)