
# import statements
import math
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

class KMeans(object):


    def __init__(self, k = 2, initial_centroids = None,\
                distance_metric = "euclidean",track_inertia = False):
         '''
         class constructor which initializes parameters of Kmeans
         '''
         self.k = k
         self.features = None
         self.centroids = initial_centroids
         self.distance_metric = distance_metric
         self.cluster_assignments = None
         self.track_inertia = track_inertia
         self.final_inertia = None
         self.inertia_record = []


    def k_means_plus_plus_init(self, data, seed = None):
        '''
        A method to initialize centroids using kmeans++ algorithm
        '''
        if seed is not None:
            np.random.seed(seed)

        # extracting the field of number of clusters
        k = self.k
        # number of features
        m = data.shape[1]
        # initialize a numpy array which will store the centroids
        centroids = np.zeros((k, m))

        ############################# initializing the first centroid #############################
        # ransomly choosing data index for the first centroid
        index = np.random.randint(data.shape[0])
        # extracting the randomly chosen data
        centroid = data.iloc[index].as_matrix()
        # initializing the first centroid
        centroids[0] = centroid

        ###########################################################################################
        ### initializing the probability disribution based on distances from the first centroid ###
        # calculate the distances of each point from the first centroid
        # flattening the distance to make it a single dimensional vector
        distances = pairwise_distances(data, [centroids[0]], metric=self.distance_metric).flatten()
        # square the distance vector
        distances_squared = distances*distances
        # create probability distribtion for choosing index based on distance squared metric
        probability_distribution = distances_squared/np.sum(distances_squared)
        ###########################################################################################

        # iterating through 2nd to last centroids
        for i in range(1,k):
            # size of the data
            n = data.shape[0]
            # ransomly choosing data index for the ith centroid based on probability distribtion
            index = np.random.choice(n, 1, p = probability_distribution)
            # extracting the randomly chosen data
            centroid = data.iloc[index].as_matrix()
            # initializing the ith centroid
            centroids[i] = centroid
            # calculate the distances of each point from all the initialized centroids
            distances = pairwise_distances(data, centroids[0:i+1], metric=self.distance_metric)
            # square the minimum among the distances for each point
            min_distances_squared = np.min(distances*distances, axis=1)
            # create probability distribtion for choosing index based on distance squared metric
            probability_distribution = min_distances_squared / np.sum(min_distances_squared)

        # initialize the class field of centroids
        self.centroids = centroids


    def assign_clusters(self,data, centroids):
        '''
        A method which assigns cluster to each data
        '''
        # extracing the fields of feature, centroids, distance_metric
        features = self.features
        centroids = self.centroids
        distance_metric = self.distance_metric
        # calculating the distance from each data to each centroid
        distances = pairwise_distances(data, centroids, metric= distance_metric)
        # finding the index of the cluster having minimum distance to each data
        self.cluster_assignments = np.argmin(distances, axis = 1)


    def update_centroids(self,data):
        '''
        A method which calculates the centroids of the clusters
        '''
        # list to store the clusters centroids
        centroids = []
        # extracting the field of number of cluster_assignment
        k = self.k

        # iterating through each cluster
        for i in range(k):
            # extracting the data that belongs to the cluster
            cluster_data = data[self.cluster_assignments == i]
            # calulting the columnwise mean of the coordinates
            centroid = cluster_data.mean(axis = 0)
            # converting the centroid to numpy array and then logistic_regression
            # centroid = centroid.toarray()
            centroid = centroid.tolist()
            # appedning the controid to the designated list
            centroids.append(centroid)

        # converting the centroid data to numpy array
        centroids = np.array(centroids)
        # setting the class field
        self.centroids = centroids


    def inertia(self, data):
        '''
        A method to calculate the inertia of cluster assigments
        '''
		# extract the class field of number of clusters
        k = self.k
        # a variable to store the accumulated inertia of all clusters
        het = 0.0
        # iterate through each cluster
        for i in range(k):
            # extracte the data belonging to ith cluster
            cluster_data = data[self.cluster_assignments == i]
            # if the ith cluster has any data the calculate inertia
            if cluster_data.shape[0] > 0:
                # calculate the distance from each cluster data to its corresponding centroid
                distances = pairwise_distances(cluster_data, [self.centroids[i]],\
                                               metric = self.distance_metric)
                # square the distances
                distances_squared = distances*distances
                # accumulate the inertia
                het += np.sum(distances_squared)

        # return the calculated inertia
        return het


    def fit(self, data, features, max_iter, verbose = False):
        '''
        Method which carries out k means subroutine
        '''
        # setting the field of features
        self.features = features
        # extacting the fields of number of clusters
        k = self.k
        # if centroids are not assigned the use k means++ to initialize the centroids
        if self.centroids is None:
            self.k_means_plus_plus_init(data[features])
        # extract the the initialized centroid
        centroids = self.centroids

        # iterating maximum iterations times
        for i in range(max_iter):
            # assign cluster to each data
            self.assign_clusters(data[features], centroids)
            # update the centroids
            self.update_centroids(data[features])
            # if user wants to track inertia
            # then compute heteroginity and append to the designated list
            if self.track_inertia:
                het = self.inertia(data[features])
                self.inertia_record.append(het)

        # set the field of final inertia
        self.final_inertia = self.inertia(data[features])


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
