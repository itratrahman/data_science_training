
##import statements
from .knn_regressor import knn_regressor
from scipy.stats import mode

class knn_classifier(knn_regressor):

    '''a class which implements k nearest neighbors classifier
    by importing the class of k nearest neighbors regressor'''

    def knn_predict(self, query_point):

        '''a method which computes the predicted value for one point'''

        ##Extracting the number of nearest neighbors from the class field
        k = self.k

        ##computing indexes of the nearest neighbors
        indexes = self.finding_nearest_neighbors(query_point)

        ##calculating mode of y values of nearest neighbors
        prediction = mode(indexes)[0][0]

        ##returning the y value
        return prediction
