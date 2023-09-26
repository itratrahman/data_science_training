
# import statements
import numpy as np
import math
from ..utility.performance_metrics import residual_sum_of_square



class ridge_regression(object):

    '''
    Class module of ridge regression
    '''

    def __init__(self, l2_penalty = 0.0, track_rss = False):

        '''
        Class constructor which stores the following fields:
        1) self.coefficients - stores coefficients of the model
        2) self.l2_penalty - l2 penalty of ridge regression model
        '''
        # a class field to store the model coefficients
        self.coefficients = None
        # l2 penalty
        self.l2_penalty = l2_penalty


    def gradient_descent_step(self, X, y, weights, step_size):

        '''
        Method to carry out a single step of gradient descent for ridge objective
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
        feature_transpose = X.transpose()

        # extract the intercept
        intercept = weights[0]

        # calculate the regularization term
        regularization_term =  2*self.l2_penalty*weights

        # restore and unpenalize the intercept term
        regularization_term[0] = intercept

        # calculate the gradient of RSS (hint: use np.dot() function)
        gradient_rss = -2*np.dot(feature_transpose,bracketed_expression) + regularization_term

        # update the weights using the step size and the gradient vector
        weights = weights - (step_size*gradient_rss)

        # compute the magnitude of the gradient vector
        # (hint: use np.sum to compute summation)
        # (hint: use math.sqrt to compute square root)
        gradient_magnitude = math.sqrt(np.sum(gradient_rss*gradient_rss))

        # return the updated weights, gradient magnitude
        return weights, gradient_magnitude


    def fit(self, X, y, step_size, tolerance = None, maximum_iterations = 100,\
            initial_weights = None, verbose = False):


        '''
        Method to carry out gradient descent optimization for ridge objective until convergence
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        y - output data of shape (number of examples,)
        step_size - step_size/learning rate of the regression model
        tolerance - used as a convergence criteria metric
        maximum_iterations - maximum number of iterations
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

        # boolean variable to store the while loop condition
        loop_condition = True

        # while loop iterates until convergence
        while loop_condition:

            ################################# Gradient Descent #################################

            # compute gradient descent step to retrieve the updated weights & gradient magnitude
            weights, gradient_magnitude = self.gradient_descent_step(X, y, weights, step_size)
            # increment the counter which stores the number of iterations
            counter += 1

            ##################################### Verbose #####################################

            # print out verbose if the user wants
            if verbose == True and counter%10 == 0:
                print("Iteration ", counter)
                print("weights: ", weights)
                print("RSS: ", self.RSS(X, y, weights))

            ################################# Loop termination #################################

            # break out of the loop if maximum iteration is reached
            if counter>=maximum_iterations:
                break
            # if a tolerance parameter is given then tolerance is used as convergence criteria
            # write an if condition statement that breaks from the loop if the convegence criteria is met
            if (tolerance is not None) and (gradient_magnitude < tolerance):
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
