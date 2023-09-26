# import statements
import numpy as np
from ..utility.performance_metrics import residual_sum_of_square
from ..utility.data_preprocessing import normalize_matrix

class lasso_regression(object):

    '''
    Class module of lasso regression
    '''

    def __init__(self, l1_penalty = 0):

        '''
        Class constructor which stores the following fields:
        1) self.coefficients - stores coefficients of the model
        2) self.l1_penalty - l1 penalty of lasso regression model
        3) self.norm - feature normalizer
        '''

        # a class field to store the model coefficients
        self.coefficients = None
        # l1 penalty
        self.l1_penalty = l1_penalty
        # a class field to store the feature normalizer
        self.norms = None


    def predict(self, X, weights = None):

        '''
        Method to compute predictions vector using a set of feature matrix and weights/coefficients
        Arguments:
        X - feature matrix H of shape (number of examples, number of features), normalized if weights are given
        or unnormalized otherwise
        weights - (if given) vector of model weights/coefficients of shape (number of features,)
        Returns:
        predictions - vector of predictions of shape (number of examples,)
        '''

        # if no weight vector is given then use the model coefficients to compute the predictions vector
        if weights is None:
            # notice we divide the weights by the normalizing vector/feature normalizer
            # becuase we assume user would give unormalized feature matrix
            # when trying to predict using a trained model
            predictions = np.dot(X, self.coefficients/self.norms)
        # else use the given weight vector to compute predictions
        else:
            # else part is used in the coordinate descent step method below
            # when we compute predictions using normalized feature matrix
            # so we dont need to normalize by dividing
            predictions = np.dot(X, weights)

        # return predictions vector
        return predictions


    def coordinate_descent_step(self, j, X, y, weights, l1_penalty = None):

        '''
        Method to carry out a single coordinate descent step to update a particular weight
        Arguments:
        j - index of weight/feature
        X - normalized feature matrix H of shape (number of examples, number of features)
        y - output data of shape (number of examples,)
        weights - vector of model weights/coefficients of shape (number of features,)
        l1_penalty - l1 penalty of lasso regression
        Returns:
        updated_weight - updated jth weight after coordinate descent step (a scalar)
        '''

        # if li_penalty is not given then initialize to that given in the class fields
        if l1_penalty is None:
            l1_penalty = self.l1_penalty

        # compute the predictions using normalized features
        predictions = self.predict(X, weights)

        #################### compute the ro parameter ####################

        # extract the jth feature vector (this is a column vector)
        X_j = X[:,j]
        # compute vector of residuals of points without feature j
        # remember this is a vector so do not use the sum() operator yet
        # we will use the sum() operator in the next step
        # hint: use the jth feature vector above for computation
        rss_rest = predictions - weights[j]*X_j
        # compute ro parameter using the above 2 expressions
        # hint: use the sum() operator to sum across training points
        ro = (X_j*(y - rss_rest)).sum()

        ######################### update weight #########################

        # if the index is 0 then do not regularize the intercept
        if j == 0:
            updated_weight = ro
        # compute updated weight when ro is < -l1_penalty/2.
        elif ro < -l1_penalty/2.:
            updated_weight = ro + l1_penalty/2.0
        # compute updated weight when ro is > -l1_penalty/2.
        elif ro > l1_penalty/2.:
            updated_weight = ro - l1_penalty/2.0
        # compute updated weight when ro is a small value
        else:
            updated_weight = 0

        # return the updated weight
        return updated_weight


    def fit(self, X, y, tolerance, initial_weights = None):

        '''
        Method to carry out lasso coordinate descent
        Arguments:
        X - unnormalized feature matrix H of shape (number of examples, number of features)
        y - output data of shape (number of examples,)
        tolerance - used as a lasso convergence criteria metric
        initial_weights - if given, vector of model weights/coefficients of shape (number of features,)
        '''

        # if no initial weight is given then the initial is weight is a zero vector
        if initial_weights is None:
            # use np.zeros to initialize a vector of zeros
            weights = np.zeros(np.shape(X)[1])
        # else the weight is initialized to initial weight vector passed by the user
        else:
            weights = np.array(initial_weights)

        # normalize the feature matrix
        (X_normalized, norms) = normalize_matrix(X)
        # initialize the class field to store the normalizing vector
        self.norms = norms

        # a variable to store the maximum change,
        # notice maximum change is initialized to a large value
        # so that while loop condition is true for the 1st iteration
        maximum_change = 1e10000000

        # l1 penalty is initialized to the class field
        l1_penalty = self.l1_penalty

        # while loop iterates until maximum change is greater than tolerance
        while(maximum_change>tolerance):

            # a list to store the change
            delta = []

            # iterate through each weight
            for j in range(len(weights)):
                # a variable to store the old weight
                previous_weight = weights[j]
                # carry out coordinate descent step to update the jth weight
                weights[j] = self.coordinate_descent_step(j, X_normalized, y, weights, l1_penalty)
                # calculate the absolute value of change in weight
                change = abs(previous_weight - weights[j])
                # append the change in weight to the designated list
                delta.append(change)

            # extract the maximum change from the list (use np.amax)
            maximum_change = np.amax(delta)

        # set the model coefficients after coordinate descent operation
        self.coefficients = weights


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
