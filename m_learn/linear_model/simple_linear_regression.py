
# import statements
from ..utility.performance_metrics import residual_sum_of_square

class simple_linear_regression(object):

    '''
    Class module of simple linear regression
    '''

    def __init__(self):

        '''
        Class contructor for storing the slope and the intercept of the model
        '''
        self.intercept = None
        self.slope = None


    def fit(self, x, y):

        '''
        Method to compute the closed form solution of linear regression
        Arguments:
        x - feature data of shape (number of examples,)
        y - output data of shape (number of examples,)
        '''
        # compute the sum of input feature and output (Hint: use .sum() operator)
        sum_y = y.sum()
        sum_x = x.sum()

        # compute sum of product of the output and the feature
        xy = x*y
        xy_sum = xy.sum()

        # compute sum of feature squared
        x_squared = x*x
        x_squared_sum = x_squared.sum()

        # size of data
        N = x.size

        # compute the slope of the regression model using above quantities
        self.slope = (xy_sum-((sum_x*sum_y)/N))/(x_squared_sum-((sum_x*sum_x)/N))

        # compute the intercept of the regression model using above quantities
        self.intercept = (sum_y/N) - self.slope*(sum_x/N)


    def predict(self, x):

        '''
        Method to compute predictions given x data
        Arguments:
        x - feature data of shape (number of examples,)
        Returns:
        y_ - vector of predictions of shape (number of examples,)
        '''
        # compute predictions using the equation of a straight line
        y_ = x*self.slope + self.intercept

        # return predictions
        return y_


    def rss(self, x, y):

        '''
        Method to compute residual sum of square given x & y data
        Arguments:
        x - feature data of shape (number of examples,)
        y - output data of shape (number of examples,)
        Return:
        _rss - residual sum of square
        '''
        # compute the vector of predictions using the class method
        y_ = self.predict(x)

        # compute the residual sum of square using the function imported at the top of this file
        _rss = residual_sum_of_square(y_, y)

        # return the rss
        return _rss
