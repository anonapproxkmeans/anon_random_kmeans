# module that holds all the clustering algorithms we used. 
# includes k-means, rse-kmeans, coreset, lightweight coreset
# and minibatch kmeans.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from sklearn.metrics import pairwise_distances

import lightweight_coreset

# all code in the coresets folder taken from: 
# https://github.com/zalanborsos/coresets
from coresets.k_means_coreset import KMeansCoreset

def kmeans_error(data, centroids):
    """
    Calculates value of k-means objective function given dataset
    and centroids.
    """
    distance_matrix = pairwise_distances(data, centroids)

    ans = 0
    for row in distance_matrix:
        ans += min(row)**2
    return ans

def nearest_centroid_extend(data, centroids):
    """
    Given data and centroids, labels all the points.
    """
    return np.argmin(pairwise_distances(data, centroids), axis = 1)

def k_means(data, n_clusters):
    """
    Regular k-means, returns partition labels.
    """
    clustering = KMeans(n_clusters = n_clusters)
    clustering.fit(data)
    return clustering.labels_, clustering.cluster_centers_

def rse_k_means(data, n_clusters, n_samples):
    """
    Does k-means on a random sample then extends labelling to 
    every other point.
    """
    sampled_indices = np.random.choice(data.shape[0], size = n_samples, replace = True)
    sampled_data = data[sampled_indices]
    clustering = KMeans(n_clusters = n_clusters)
    clustering.fit(sampled_data)
    return nearest_centroid_extend(data, clustering.cluster_centers_), clustering.cluster_centers_

def cs_kmeans(data, n_clusters, n_samples):
    """
    Creates a coreset then does k-means on the coreset.
    Reference: Strong Coresets for Hard and Soft Bregman Clustering with Applications 
    to Exponential Family Mixtures, AISTATS2016. 
    """
    coreset_generator = KMeansCoreset(data)
    sampled_data, weights = coreset_generator.generate_coreset(n_samples)
    clustering = KMeans(n_clusters = n_clusters)
    clustering.fit(sampled_data, sample_weight = weights)
    return nearest_centroid_extend(data, clustering.cluster_centers_), clustering.cluster_centers_

def lwcs_kmeans(data, n_clusters, n_samples):
    """
    Creates a lightweight coreset, obtains centroids from the coreset. 
    Then, does extension to other points.

    Reference:Scalable and distributed clustering via lightweight coresets. 
    KDD2018.
    """
    sampled_data, weights = lightweight_coreset.lightweight_coreset(data, n_samples)
    clustering = KMeans(n_clusters = n_clusters)
    clustering.fit(sampled_data, sample_weight = weights)

    return nearest_centroid_extend(data, clustering.cluster_centers_), clustering.cluster_centers_

def minibatch_k_means(data, n_clusters, n_samples):
    """
    Runs MBKM algorithm based on the implementation in sklearn.
    """
    clustering = MiniBatchKMeans(n_clusters = n_clusters, batch_size = n_samples)
    clustering.fit(data)
    return clustering.labels_, clustering.cluster_centers_

