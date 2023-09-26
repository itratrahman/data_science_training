# import statements
import numpy as np
import math
from ..utility.performance_metrics import residual_sum_of_square

class linear_regression(object):

    '''
    Class module of linear regression
    '''

    def __init__(self, track_rss = False):

        '''
        Class constructor which stores the following fields:
        1) self.coefficients - stores coefficients of the model
        2) self.track_rss - indicator/boolean variable to determine whether user wants to track RSS
        3) self.rss_log - a list to store the log of RSS of the model at every iteration
        '''
        # a class field to store the model coefficients
        self.coefficients = None
        # indicator variable
        self.track_rss = track_rss
        # list to store the log of RSS of the model
        self.rss_log = []


    def gradient_descent_step(self, X, y, weights, step_size):

        '''
        Method to carry out a single step of gradient descent
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        y - output data of shape (number of examples,)
        weights - vector of model weights/coefficients of shape (number of features,)
        step_size - step_size/learning rate of the regression model
        Returns:
        weights - vector of updated weights/coefficients
        gradient_magnitude - magnitude of gradient of cost (a scalar)
        '''

        # calculate the bracketed expression
        # use np.dot function to compute matrix multiplication
        bracketed_expression = y - np.dot(X, weights)

        # take the transpose of the feature matrix (hint: use .transpose() function)
        X_transpose = X.transpose()

        # calculate the gradient of RSS (hint: use np.dot() function)
        gradient_rss = -2*np.dot(X_transpose,bracketed_expression)

        # update the weights using the step size and the gradient vector
        weights = weights - (step_size*gradient_rss)

        # compute the magnitude of the gradient vector
        # (hint: use np.sum to compute summation)
        # (hint: use math.sqrt to compute square root)
        gradient_magnitude = math.sqrt(np.sum(gradient_rss*gradient_rss))

        # return the updated weights, gradient magnitude
        return weights, gradient_magnitude


    def fit(self, X, y, step_size, tolerance, initial_weights = None, verbose = False):

        '''
        Method to carry out gradient descent optimization until convergence
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        y - output data of shape (number of examples,)
        step_size - step_size/learning rate of the regression model
        tolerance - used as a convergence criteria metric
        initial_weights - if given, vector of model weights/coefficients of shape (number of features,)
        verbose - a boolean variable to indicate if verbose is printed at every grad descent iteration
        '''

        # if no initial weight is given then the weights is initialized to a zero vector
        if initial_weights is None:
            # use np.zeros to initialize a vector of zeros
            weights = np.zeros(np.shape(X)[1])
        # else the weight is initialized to initial weight vector passed by the user
        else:
            weights = np.array(initial_weights)

        # counter variable to store the number of iterations of gradient descent
        counter = 0

        # while loop iterates until convergence
        while True:

            # compute gradient descent step to retrieve the updated weights & gradient magnitude
            weights, gradient_magnitude = self.gradient_descent_step(X, y, weights, step_size)
            # increment the counter which stores the number of iterations
            counter += 1

            # track RSS at each iteration if the user wants
            if self.track_rss:
                self.rss_log.append(self.rss(X, y, weights))

            # print out verbose if the user wants
            if verbose and counter%10 == 0:
                print("Iteration: ", counter)
                print("weights: ", weights)
                print("RSS: ", self.rss(X, y, weights))

            # write an if condition statement that breaks from the loop if the convegence criteria is met
            if gradient_magnitude < tolerance:
                break

        # set the model coefficients after gradient descent operation is complete
        self.coefficients = weights


    def predict(self, X, weights = None):

        '''
        Method to predict output vector using a set of feature matrix and weights/coefficients
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        weights - if given, vector of model weights/coefficients of shape (number of features,)
        Returns:
        predictions - vector of predictions of shape (number of examples,)
        '''

        # if no weight vector is given then use the model coefficients as weight vector
        # hint: use np.dot to compute matrix multiplication
        if weights is None:
            predictions = np.dot(X, self.coefficients)
        # else use the given weight vector to get predictions
        else:
            predictions = np.dot(X, weights)

        # return the predictions vector
        return predictions


    def rss(self, X, y, weights = None):

        '''
        Method to compute residual sum of square given X & y data
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        y - output data of shape (number of examples,)
        weights - if given, vector of model weights/coefficients of shape (number of features,)
        Returns:
        _rss - residual sum of square (a scalar)
        '''

        # if no weights is given then use the model coefficients to predict
        if weights is None:
            predictions = self.predict(X)
        # else use the weights given by the caller
        else:
            predictions = self.predict(X, weights)

        # calculate residual sum of square
        _rss = residual_sum_of_square(predictions, y)

        # return rss
        return _rss
