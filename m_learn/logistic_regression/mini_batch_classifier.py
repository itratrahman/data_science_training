
# import statements
import math
import numpy as np
from ..utility.performance_metrics import accuracy_score

class mini_batch_classifier(object):


    def __init__(self, track_logistic_cost = False):

        '''
        Class constructor stores the following fields:
        1) self.coefficients - stores coefficients of the model
        2) self.track_logistic_cost - indicator/boolean variable to determine whether user wants to track cost
        3) self.log_of_logistic_cost - a list to store the log of logistic reg cost of the model at every iteration
        '''
        # model coefficients
        self.coefficients = None
        # indicator variable to determine whether user wants to track logistic reg cost
        self.track_logistic_cost = track_logistic_cost
        # list to store the cost of the model at every iteration
        self.log_of_logistic_cost = []


    def sigmoid(self, x):

        '''
        A method which computes the sigmoid of a numeric value/numpy array
        Arguments:
        x - a numeric variable or a numpy array
        Returns:
        a numeric value/numpy array that stores the sigmoid of x
        '''
        return 1.0 / (1 + np.exp(-x))


    def sigmoid_function(self, X, weights):

        '''
        A method which computes the probabilities of points, given feature matrix and weights/coefficients
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        weights - vector of model weights/coefficients of shape (number of features,)
        Returns:
        a numpy vector/array of shape (number of examples,) that stores the probability of all the points
        '''
        # compute the dot product/matrix multiplication between the feature matrix and weight vector
        z = np.dot(X, weights)
        # vector of probabilities
        y_ = self.sigmoid(z)
        # return the probability vector
        return y_


    def predict(self, X, weights = None, prob_threshold = 0.5):

        '''
        A method which computes predicted labels using feature matrix and weights/coefficients
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        weights - vector of model weights/coefficients of shape (number of features,)
        prob_threshold - probability threshold in classification
        Returns:
        predictions - a numpy vector of shape (number of examples,)
        that stores the predicted labels of all examples in the feature matrix X
        '''
        # if no weight vector is given then use the model coefficients to compute the probabilities
        if weights is None:
            # compute the probabilities of points
            y_ = self.sigmoid_function(X, self.coefficients)
        # else use the given weight vector to compute the probabilities
        else:
            # compute the probabilities of points
            y_ = self.sigmoid_function(X, weights)

        # compute the list of predicted labels
        # hint: use list comprehension to construct a list,
        # which shows 1 if probability is greater than 0.5 or else 0
        predictions = [+1 if prob>prob_threshold else 0  for prob in y_]
        # convert the list to numpy array
        predictions = np.array(predictions)
        # return the vector of predicted labels
        return predictions


    def compute_logistic_cost(self, X, weights, y):

        '''
        A method which computes logistic regression cost given feature matrix, weight vector, and labels
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        weights - vector of model weights/coefficients of shape (number of features,)
        y - vector of predicted labels of shape (number of examples,)
        Returns:
        cost - logistic regression cost
        '''
        # number of examples in the feature matrix X
        N = len(y)
        # compute probabilities of the points
        y_ = self.sigmoid_function(X, weights)
        # compute cost
        # hint: use * operator to compute bitwise multiplication
        # hint: use numpy sum operator to compute summation across points
        cost = (1/N)*np.sum(-y*np.log(y_)-(1-y)*np.log(1-y_))
        # return cost
        return cost


    def gradient_descent_step(self, X, y, weights, step_size):

        '''
        A method which computes a single gradient descent step
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        y - vector of predicted labels of shape (number of examples,)
        weights - vector of model weights/coefficients of shape (number of features,)
        step_size - step size/learning rate of gradient descent
        Returns:
        weights - updated weights after a gradient descent step
        '''
        # number of training examples
        N = np.shape(X)[0]
        # compute probabilities of the points
        y_ = self.sigmoid_function(X, weights)
        # compute gradient vector
        gradient = (1/N)*np.dot(X.T,y_ - y)
        # gradient descent updated
        weights = weights - step_size*gradient
        # return weights
        return weights


    def fit(self, X, labels, step_size, batch_size, initial_weights = None,\
            maximum_iterations = 1000, verbose = False):

        '''
        A method which carries out mini batch gradient descent of logistic regression
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        labels - vector of predicted labels of shape (number of examples,)
        step_size - step size/learning rate of gradient descent
        batch_size - batch size of the mini batches
        initial_weights - vector of initial weight
        maximum_iterations - maximum of number of iterations
        verbose - boolean variable which indicates if verbose be printed out
        '''

        # if no initial weight is given then the initial is weight is a zero vector
        if initial_weights is None:
            weights = np.zeros(np.shape(X)[1])
        # else the weight is initialized to initial weight passed to the method
        else:
            weights = np.array(initial_weights)

        # loop counter
        counter = 0

        # size of data
        N = np.shape(X)[0]
        # number of complete batches
        num_complete_mini_batches = math.floor(N/batch_size)

        # iterate until maximum number of iterations is reached
        while True:

            # iterate through the mini batches
            for k in range(0, num_complete_mini_batches+1):

                # if the mini batch is a complete batch
                if k < num_complete_mini_batches:
                    X_batch = X[k * batch_size : k * batch_size + batch_size,:]
                    labels_batch = labels[k * batch_size : k * batch_size + batch_size]
                # handling the end case when the last mini batch is less than the batch_size
                elif N % batch_size != 0:
                    X_batch = X[num_complete_mini_batches * batch_size : N,:]
                    labels_batch = labels[num_complete_mini_batches * batch_size : N]

                # compute mini batch gradient descent step
                weights = self.gradient_descent_step(X_batch, labels_batch, weights, step_size)

                num = 50
                step = maximum_iterations//num
                # print verbose
                if  (counter%step == 0 and counter != 0) and verbose:
                    logistic_cost = self.compute_logistic_cost(X_batch, weights, labels_batch)
                    print('iteration %*d: logistic cost = %.8f' % \
                        (int(np.ceil(np.log10(maximum_iterations))), counter, logistic_cost))

                # track cost function at each iteration if the user wants
                if self.track_logistic_cost:
                    self.log_of_logistic_cost.append(self.compute_logistic_cost(X_batch, weights, labels_batch))

                # increment the loop counter
                counter += 1

                # break out of the inner for loop if maximum number of iterations is reached
                if counter>=maximum_iterations:
                    break

            # break out of the while loop if maximum number of iterations is reached
            if counter>=maximum_iterations:
                break

        # print statement
        if verbose:
            print("Completed performing Gradient Descent")

        # initialize the class field containing model coefficients
        self.coefficients = weights


    def accuracy(self, X, y):

        '''
        A method which calculates the accuracy of the model given feature matrix
        and predicted labels
        Arguments:
        X - feature matrix H of shape (number of examples, number of features)
        y - vector of labels of shape (number of examples,)
        Returns:
        accuracy - accuracy of the model
        '''
        # compute predictions of the points
        predictions = self.predict(X)
        # calculate the accuracy
        accuracy = accuracy_score(predictions, y)
        # return the accuracy
        return accuracy
