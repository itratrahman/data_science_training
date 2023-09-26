
# import the binary_classifier module
from .binary_classifier import binary_classifier
# import get_xy_data function
from ..utility.data_preprocessing import get_xy_data
# import numpy and pandas
import numpy as np
import pandas as pd


class multiclass_classifier(object):

    def __init__(self):

        '''
        Class constructor which stores the following fields:
        self.classes - stores the classes of the multiclass models
        self.features - stores the features of the model
        self.output - stores the output of the model
        self.models - stores the binary logistic regression classifiers for the n classes
        '''
        self.classes = None
        self.features = None
        self.output = None
        self.models = None


    def fit(self, data, features, output, step_size, maximum_iterations = 1000):

        '''
        A method which carries out one vs rest classification to fit multiclass model
        Arguments:
        data - a pandas dataframe containing the X y data
        features - python list containing the features
        output - python string containing the output
        step_size - step size/learning rate of gradient descent of all n classifiers
        maximum_iterations - maximum of number of iterations of each classifier
        '''
        # store the features and output in designated class fields
        self.features = features
        self.output = output

        # extract the distinct class from the output column
        classes = data[output].tolist()
        classes = set(classes)
        classes = list(classes)
        # set the class field
        self.classes = classes

        # onehot encode the output column i.e. create separate columns for classes
        # and concat the onehot encoded columns to the original data
        # hint: use pandas get_dummies function to create onehot encoded representation
        # hint: use pandas concat function to concat the onehot encoded columns to the original data
        data = pd.concat((data, pd.get_dummies(data[output])), axis=1)
        # a list to store the models of one vs rest classification
        models = []

        # iterate through each class, train and store each model of one vs rest classification
        for class_ in classes:
            # extract the feature matrix and output labels
            X, y = get_xy_data(data, features, class_)
            # create a binary classifier
            model = binary_classifier()
            # fit the binary classifier for the class under consideration
            model.fit(X, y, step_size, maximum_iterations = maximum_iterations, verbose = False)
            # append binary classification model to the disignated list
            models.append(model)

        # store the list of models in the designated class field
        self.models = models


    def predict(self, data, features = None):

        '''
        A method which computes the predicted labels given data and features
        Arguments:
        data - a pandas dataframe containing the X y data
        features - python list containing the features
        Returns:
        predictions - a numpy array containing the predicted labels/predictions
        '''
        # if features are not given then they are taken from the class fields
        if features is None:
            features = self.features

        # extract feature matrix
        X = get_xy_data(data, features)
        # a list to store the scores of all examples for all classes of one vs rest models
        # this list will store a list of 1D numpy array
        scores_all_classes = []

        # iterate through the models of the multiclass classification
        for i, class_ in enumerate(self.classes):
            # calculate the scores of all examples for the class
            # hint: score is the dot product between feature matrix and model weights/coefficients
            score = np.dot(X, self.models[i].coefficients)
            # append the scores of the class to the designated list
            scores_all_classes.append(score)

        # convert the list of 1D arrays to 2D numpy array
        # this will return a n_class by N array,
        # where n_class-number of classes of multiclass classification
        scores_all_classes = np.array(scores_all_classes)
        # create predictions based on maximum score amoung the classes
        # hint: use argmax function to compute maximum index across rows
        predictions = scores_all_classes.argmax(axis = 0)
        # return the predictions
        return predictions


    def accuracy(self, data):

        '''A method which calculates the accuracy of the model
        Arguments:
        data - a pandas dataframe containing the X y data
        Returns:
        accuracy - accraucy of the multiclass model
        '''
        # compute the predictions using the data
        predictions = self.predict(data)
        # extracting the labels from the data
        labels = data[self.output].tolist()
        # count the number of correct predictions by comparing the predicted labels with corresponding true labels
        num_of_correct_preds = np.sum([1 if pred==label else 0 for pred,label in zip(predictions, labels)])
        # calculate the accuracy of the model
        accuracy = num_of_correct_preds/len(predictions)
        # return the accuracy
        return accuracy
