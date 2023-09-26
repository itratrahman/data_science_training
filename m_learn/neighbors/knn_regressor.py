
# import numpy
import numpy as np

class knn_regressor(object):
    '''
    Class module of KNN Regressor
    '''

    def __init__(self, k = 5):
        '''
        Class constructor which initializes the number of neighbors(k),
        and contains class fields to store the feature matrix & output vector
        of the training set.
        '''
        # number of neighbors
        self.k = k
        # class field to store the feature matrix of training set
        self.X = None
        # class field to store the output vector of training set
        self.y = None


    def fit(self, X, y):
        '''
        fit method which initializes the feature matrix and the output vector
        of the training set.
        Arguments:
        X - feature matrix of shape (number of examples, number of features)
        y - output vector of shape (number of examples,)
        '''
        # initializes the class fields of feature matrix and output vector of training set
        self.X = X
        self.y = y


    def find_nearest_neighbors(self, xq):
        '''
        A method which returns the y values of the nearest neighbors based on the training set
        Arguments:
        xq - a single query point of shape (number of features,)
        Returns:
        y_knn - y values of k nearest neighbors
        '''
        # extract the number of nearest neighbors from the class field
        k = self.k
        # compute the difference matrix between the training points and the query point
        difference_matrix = self.X - xq
        # compute the distance vector between the training points and the query point
        distance_vector = np.sqrt(np.sum(difference_matrix**2, axis = 1))
        # compute the indexes of the nearest neighbors using numpy argsort function
        knn_indexes = np.argsort(distance_vector)[:k]
        # extract the y values of k nearest neighbors
        y_knn = self.y[knn_indexes]
        # return the y values of k nearest neighbors
        return y_knn


    def knn_predict(self, xq):
        '''
        A method which computes the predicted value of a single query point
        Arguments:
        xq - a single query point of shape (number of features,)
        Returns:
        prediction - predicted value of the query point
        '''
        # extract the number of nearest neighbors from the class field
        k = self.k
        # compute y values of the nearest neighbors
        y_knn = self.find_nearest_neighbors(xq)
        # calculate predicted value by averaging the y values of the nearest neighbors
        prediction = np.sum(y_knn)/k
        # return the predicted value of the query point
        return prediction


    def predict(self, Xq):
        '''
        A method which computes the predicted values of multiple query points
        Arguments:
        Xq - feature matrix containing multiple query points of shape (number of examples, number of features)
        Returns:
        y_ - predicted values of the multiple query points of shape (number of examples,)
        '''
        # a list to store the predicted values of the query points
        y_ = []

        # iterate through each query point
        for xq in Xq:
            # compute the prediction of the query point
            prediction = self.knn_predict(xq)
            # append the prediction to the designated list
            y_.append(prediction)

        # return the predicted values
        return y_
