
# import numpy and sklearn pairwise_kernels
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class kernel_regressor(object):
    '''
    Class module of kernel regression
    '''
    def __init__(self, kernel = "rbf", gamma = None):
        '''
        Class constructor which initializes the kernel and gamma value of the kernel
        and contains class fields to store the feature matrix & output vector of the training set.
        '''
        # kernel of the kernel regression
        self.kernel = kernel
        # gamma/bandwidth of the kernel
        self.gamma = gamma
        # class field to store the feature matrix
        self.X = None
        # class field to store the output vector
        self.y = None


    def fit(self, X, y):
        '''
        fit method which initializes the feature matrix and the output vector
        of the training set.
        Arguments:
        X - feature matrix of shape (number of examples, number of features)
        y - output vector of shape (number of examples,)
        '''
        # initialize the class fields of feature matrix and output vector
        self.X = X
        self.y = y


    def predict(self, Xq):
        '''
        A method which computes the predicted values of multiple query points
        Arguments:
        Xq - feature matrix containing multiple query points of shape (number of examples, number of features)
        Returns:
        y_ - predicted values of the multiple query points of shape (number of examples,)
        '''
        # compute the kernel distances between train feature matrix and the query feature matrix
        kernel_output = pairwise_kernels(self.X, Xq, metric=self.kernel, gamma=self.gamma)
        # computing the numerator term
        # hint: compute (*) bitwise multiplication between the kernel_output and output_vector y,
        # and then sum across rows or over columns (i.e. axis = 0)
        # hint: use [:, np.newaxis] to convert rank 1 matrix to N X 1 matrix
        numerator = (kernel_output * self.y[:, np.newaxis]).sum(axis=0)
        # compute the denominator term
        # hint: sum across rows or over columns (i.e. axis = 0) of kernel output
        denominator = kernel_output.sum(axis=0)
        # compute the predictions
        y_ = numerator/denominator
        # return the predictions
        return y_
