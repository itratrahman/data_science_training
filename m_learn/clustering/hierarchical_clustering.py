
# import statemenets
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class Hierarchical(object):

    def __init__(self, k, distance_metric = "euclidean"):

        '''class constructor which initialize the required parameters'''

        self.features = None
        self.k = k
        self.max_depth = None
        self.tree = None
        self.cluster_assignments = []
        self.centroids = []
        self.distance_metric = distance_metric
        self.distance_array = []
        self.nodes_to_be_pruned = None


    @staticmethod
    def bipartition(data, maxiter=400, num_runs=4):

        '''partition method'''

        # partition the data into 2 clusters
        kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs)
        kmeans_model.fit(data)

        # extract the cluster assignments and centroids
        centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

        # extract the data index of left child
        left_data_index= data[cluster_assignment==0].index
        # extract the data index of right child
        right_data_index = data[cluster_assignment==1].index

        # extract the left centroid
        left_centroid = centroids[0]
        # extract the right centroid
        right_centroid = centroids[1]

        return  left_data_index, right_data_index, left_centroid, right_centroid


    def recursive_kmeans(self, data, features, index, current_depth):

        # data index
        data_index = data.index.tolist()

        # calculate centroid
        centroid = data[features].mean(axis = 0)

        # if the current depth is greater than max depth then create leaf
        if current_depth >= self.max_depth:
            return {'left_child' : None,
                    'right_child' : None,
                    'data_index': data_index,
                    'centroid': centroid,
                    'leaf': True,
                    'depth': current_depth,
                    'index': index,
                    'distance_betn_centroids': None}



        # extract the left and right clusters and their centroid
        # using the bipartition function
        left_data_index, right_data_index, left_centroid, right_centroid = Hierarchical.bipartition(data)

        # calculate the distance between the centroids using the pairwise function
        distance = pairwise_distances([left_centroid], [right_centroid], metric= self.distance_metric)

        # add the distance data to the designate array
        self.distance_array.append([current_depth, index, distance[0][0]])

        #left node data
        left_node_data = data.loc[left_data_index]
        # right node data
        right_node_data = data.loc[right_data_index]

        # calculate the left index
        left_index = index*2-1
        # calculate the right index
        right_index = index*2

        # recursively invoke kmeans on the left node data
        left_child = self.recursive_kmeans(left_node_data, features, left_index, current_depth+1)
        # recursively invoke kmeans on the left node data
        right_child = self.recursive_kmeans(right_node_data, features, right_index, current_depth+1)

        # return the dictionary representing the node
        return {'left_child' : left_child,
                'right_child' : right_child,
                'data_index': data_index,
                'centroid': centroid,
                'leaf': False,
                'depth': current_depth,
                'index': index,
                'distance_betn_centroids': distance}


    @staticmethod
    def req_depth_value(k):

        '''a method which returns the required value of depth'''

        # varaible to store the required depth
        depth = 1

        # infinite while loop
        while(True):

            # number of leafs under current depth
            n_leafs = math.pow(2,depth)

            # if number of leafs is >= number of clusters
            # then break from the while loop
            if n_leafs>=k:
                break

            # required depth is incremented
            depth += 1

        # return the required depth
        return depth


    @staticmethod
    def find_num_to_prune(k):

        '''a method which finds the number of nodes to prune'''

        # find the required the depth
        depth = Hierarchical.req_depth_value(k)

        # calculate the number of leafs
        n_leafs = math.pow(2,depth)

        # calculate the number of nodes to prune
        num_to_prune = n_leafs - k

        # return the number of nodes to prune
        return int(num_to_prune)


    def num_of_nodes(self,tree):

        '''a method recursively counts the number of nodes in a tree'''

        if tree['leaf']:
            return 1
        return 1 + self.num_of_nodes(tree['left_child']) + self.num_of_nodes(tree['right_child'])


    def prune(self, tree):

        '''a method for pruning the selected nodes'''

        # if the node under current recursive call is a leaf
        # then do nothing, just return control to the caller
        if tree['leaf']:
            return
        # else investigate further
        else:

            # store the depth and the index of the node in a list
            dept_nd_index = []
            dept_nd_index.append(tree['depth'])
            dept_nd_index.append(tree['index'])

            # if depth and index matches to one of the nodes selected for pruning
            # then just prune the node
            if dept_nd_index in self.nodes_to_be_pruned:
                tree['left_child'] = None
                tree['right_child'] = None
                tree['distance_betn_centroids'] = None
                tree['leaf'] = True

            # else make recursive call on the left and right child
            else:
                self.prune(tree['left_child'])
                self.prune(tree['right_child'])


    def index_cluster(self, tree):

        '''a methods which recursively makes cluster assingments'''

        # if the node under current recursive call is a leaf
        # append the cluster_assignments and centroids to the designated list
        if tree['leaf']:
            self.cluster_assignments.append(tree['data_index'])
            self.centroids.append(tree['centroid'])
            return
        # else recursively search the left and righ childe
        else:
            self.index_cluster(tree['left_child'])
            self.index_cluster(tree['right_child'])


    def fit(self, data, features):

        # extract the k value from the class field
        k = self.k
        # initialize the class field of features
        self.features = features
        # calculate and initialize the class field of max depth of tree
        self.max_depth = Hierarchical.req_depth_value(k)
        # calculate the number of nodes to prune
        num_to_prune = Hierarchical.find_num_to_prune(k)

        # contruct the hierarchical clustering tree using recursive kmeans algorithm
        self.tree = self.recursive_kmeans(data,features,1, 0)

        # storing the array in a pandas dataframe
        self.distance_array = np.array(self.distance_array)
        self.distance_array = pd.DataFrame(self.distance_array,
                                        columns=['depth', 'index', 'dist_betn_centroids'])
        # converting the column of depth & index to int
        self.distance_array['depth'] = self.distance_array['depth'].astype(int)
        self.distance_array['index'] = self.distance_array['index'].astype(int)
        # sorting the dataframe by distance between centroids
        self.distance_array = self.distance_array.sort_values('dist_betn_centroids', ascending=True)

        # creating the list of indexes of nodes to be pruned
        index = list(range(num_to_prune))
        # extracting the nodes to be prunded
        nodes_to_be_pruned = self.distance_array.iloc[index][['depth', 'index']]
        self.nodes_to_be_pruned = nodes_to_be_pruned.as_matrix().tolist()
        # pruning the selected nodes
        self.prune(self.tree)

        # create cluster assignments
        self.index_cluster(self.tree)
        # convert the list to numpy arrays
        self.cluster_assignments = np.array(self.cluster_assignments)
        self.centroids = np.array(self.centroids)


    def predict(self, data):

        '''a method for prediction clusters of data'''

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
