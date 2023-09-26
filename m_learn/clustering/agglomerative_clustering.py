
# import statatments
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

class AgglomerativeClustering(object):

    def __init__(self, k = 2, distance_metric = "euclidean"):
        '''
        Class constructor which initializes parameters of agglomerative clustering
        '''
        self.k = k
        self.distance_metric = distance_metric
        self.features = None
        self.centroids = None

    def convert_to_list_of_arrays(self, data):
        '''
        A method to convert numpy array containing feature matrix to list of arrays
        '''
        # list to store the arrays
        list_of_arrays = []
        # iterate through each row in the numpy array
        # and append it to the designated list
        for row in data:
            list_of_arrays.append(row)
        return list_of_arrays


    def merge_clusters(self, list_of_arrays, i,j):
        '''
        A method to merge 2 clusters within the list of arrays based on index i,j
        '''
        # copy the ith array in the list to a variable
        array_i = list_of_arrays[i].copy()
        # if the ith array is a rank 1 array convert it to (M,1) matrix
        # where M is the number of features
        if len(array_i.shape)==1: array_i = [array_i]
        # copy the jth array in the list to a variable
        array_j = list_of_arrays[j].copy()
        # if the jth array is a rank 1 array convert it to (M,1) matrix
        # where M is the number of features
        if len(array_j.shape)==1: array_j = [array_j]
        # merge the ith and jth arrays representing clusters via numpy concatenate
        # and store the merged cluster in the i index of the list containing the clusters
        list_of_arrays[i] = np.concatenate([array_i,array_j])
        # pop out the jth array in the list
        list_of_arrays.pop(j)
        return list_of_arrays

    def compute_centroids(self, list_of_arrays):
        '''
        A method to compute centroids of clusters stored in the list of arrays
        '''
        # a list to store the centroids
        centroids = []
        # iterate through each array representing a cluster
        for cluster in list_of_arrays:
            # if the array is a rank 1 array convert it to (M,1) matrix
            # where M is the number of features
            if len(cluster.shape)==1: cluster = [cluster.copy()]
            # compute centroid as mean across rows or over column
            centroid = np.mean(cluster, axis=0)
            # append the centroid to the designated list
            centroids.append(centroid)
        return centroids

    def compute_distance(self, centroid_1, centroid_2):
        '''
        A method to compute distance between centroids of 2 clusters
        '''
        distance = pairwise_distances([centroid_1], [centroid_2],metric = self.distance_metric)[0][0]
        return distance


    def fit(self, data, features):
        '''
        A method fits clusters to data via agglomerativeclustering
        '''
        self.features = features
        # extract the feature matrix
        feature_matrix = data[features].values
        # convert the 2D numpy array containing feature matrix to list of arrays
        list_of_arrays = self.convert_to_list_of_arrays(feature_matrix)

        # iterate until there are k number of clusters left after successive merge operations
        while(len(list_of_arrays)>self.k):
            # compute centroids of the list of arrays
            centroids = self.compute_centroids(list_of_arrays)
            # minimum distance between clusters in initialized to large number
            min_distance = 1e100
            # ith and jth index in initialized to a large number
            ith_index = None
            jth_index = None
            # iterate through every combination of clusters using double for loop
            for i in range(len(list_of_arrays)):
                for j in range(len(list_of_arrays)-1):
                    # compute centroid to the ith and jth clusters
                    centroid_1 = centroids[i]
                    centroid_2 = centroids[j]
                    # compute distance between the 2 clusters
                    distance = self.compute_distance(centroid_1, centroid_2)
                    # if the distance between the ith and jth clusters
                    # is less than the minimum registerd so far
                    # then update the indexes of the closest clusters and minimum distance
                    if distance<min_distance and i!=j:
                        ith_index = i
                        jth_index = j
                        min_distance = distance
            # merge the 2 closest clusters
            list_of_arrays = self.merge_clusters(list_of_arrays[:], ith_index, jth_index)

        # compute centroids of the modified clusters
        # and convert it to numpy array
        centroids = self.compute_centroids(list_of_arrays)
        centroids = np.array(centroids)
        # intialize the class field of centroids
        self.centroids = centroids


    def predict(self, data):
        '''
        A method for prediction clusters of data
        '''
        # extracing the fields of features, centroids and distance_metric
        features = self.features
        centroids = self.centroids
        distance_metric = self.distance_metric
        # calculating the distance from each data to each centroid
        distances = pairwise_distances(data[features], centroids, metric=distance_metric)
        # finding the index of the cluster having minimum distance to each data
        cluster_assignments = np.argmin(distances, axis=1)
        # return cluster assignments of the given data
        return cluster_assignments
