
# import statemenets
import math
import numpy as np
from sklearn.preprocessing import StandardScaler


class pca(object):

    def __init__(self, n_components = 2):

        '''class constructor which initializes the number of components of pca'''
        self.n_components = n_components
        self.features = None


    def mean_normalization(self, X):

        '''a method for computing mean normalization'''
        # create a scalar object
        scaler = StandardScaler()
        # transform the feature matrix
        X_scaled = scaler.fit_transform(X)
        # return the scaled data
        return X_scaled


    def compute_covariance(self, X_scaled):

        '''a method for computing covariance of the scaled data'''
        cov_mat = np.cov(X_scaled.T)
        return cov_mat


    def compute_U(self, cov_mat):

        '''a method for computing singular value decomposition'''
        U,_,_ = np.linalg.svd(cov_mat)
        # _, U = np.linalg.eig(cov_mat)
        return U


    def compute_Z(self,X_scaled, U):

        '''a method for compute Z matrix'''
        # compute U_reduce
        U_reduce = U[:,:self.n_components]

        ##################### compute Z matrix #####################

        # method 1: multiplying individual element through for loop
        # Z = []
        # for i in range(X_scaled.shape[0]):
        #     elem = np.matmul(U_reduce.transpose(), X_scaled[i,:])
        #     elem = elem.tolist()
        #     Z.append(elem)
        # Z = np.array(Z)

        # method 2 through matrix multiplication
        Z = 1*np.matmul(X_scaled, U_reduce)

        ############################################################

        # return Z matrix
        return Z

    def fit_transform(self, data, features):

        '''a method for carrying out pca data decomposition'''
        # set the class of features
        self.features = features
        # extract the feature matrix
        X = data[features].as_matrix()
        # compute mean normalization
        X_scaled = self.mean_normalization(X)
        # compute compute_covariance
        cov_mat = self.compute_covariance(X_scaled)
        # compute singular value decomposition
        U = self.compute_U(cov_mat)
        # compute Z matrix
        Z = self.compute_Z(X_scaled, U)
        # return Z as_matrix
        return Z
