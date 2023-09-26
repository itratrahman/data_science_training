
# import statements
import numpy as np
import pandas as pd
import math

class gaussian_NB_classifier(object):
    '''
    Class module of Gaussian Naive Bayes classifier
    '''
    def __init__(self):
        '''
        Class constructor which initializes the parameters of the gaussian naive bayes classifier
        '''
        self.features = None
        self.output = None
        self.classes = None
        self.cls_frequencies = None
        self.mean_data = None
        self.std_data = None


    def get_classes(self, labels):
        '''
        A method which returns the unique classes in the labels
        Arguments:
        labels - a numpy array containing the true labels
        Returns:
        classes - list containing unique classes in the labels
        '''
        # extract the unique classes/labels
        classes = set(labels.tolist())
        classes = list(classes)
        classes = sorted(classes)
        # return the number of classes
        return classes


    def compute_class_frequencies(self, labels):
        '''
        A method which computes frequencies of classes in the classification and stores it in a list
        Arguments:
        labels - a numpy array containing the true labels
        Returns:
        cls_frequencies - list containing class frequencies of all the classes in classification
        '''
        # list to store the class frequencies
        cls_frequencies = []
        # compute the size of the data
        size = len(labels)
        # extract the classes from the fields
        classes = self.classes

        # iterate through each class
        for cls in classes:
            # calculate the class frequency
            # hint: use pandas filtering to filter the labels belonging to the class
            # hint: use len method to calculate number of datapoints belonging to the class
            freqeuncy = len(labels[labels == cls])/size
            # append the class frequency to the designated list
            cls_frequencies.append(freqeuncy)

        # return the list of class frequencies of all the classes
        return cls_frequencies


    def compute_feature_class_mean(self, data, feature, output,cls):
        '''
        A method which calculates the mean of a single feature for a given class
        Arguments:
        data - pandas dataframe containing the training data
        feature - feature whose mean we want to calculate
        output - output of the model
        cls - the class of interest in the calculation of the mean of feature
        Returns:
        mean - mean of the feature for the given class
        '''
        # filter the data points belonging to the given class
        # hint: use pandas filtering operation
        class_data = data[data[output] == cls]
        # extract the designated feature column from the filtered data
        feature_class_data = class_data[feature]
        # calculate the mean of the feature column for the given class
        mean = feature_class_data.mean()
        # return the mean
        return mean


    def compute_mean_data(self, data, features, output):
        '''
        A method which calculates the means of every feature-class combination
        and tabulates the mean data in a pandas dataframe
        Arguments:
        data - a pandas dataframe containing the training data
        features - a list containing the features of the model
        output - output of the model
        Returns:
        mean_data - a pandas dataframe containing the means of every feature-class combination
        '''
        # extact the classes
        classes = self.classes
        # a dataframe to store the mean data
        mean_data = pd.DataFrame()
        # create a column for the class
        mean_data['class'] = classes

        # iterate through each feature
        for feature in features:
            # a list to store the means of the given feature of all the classes
            means = []
            # iterate through each class
            for cls in classes:
                # calculate the feature mean of the class
                mean = self.compute_feature_class_mean(data, feature, output,cls)
                # append the mean to the designated list
                means.append(mean)
            # create a column of means of the given feature for all the classes
            mean_data[feature] = means

        # set the class column as the index of the dataframe
        mean_data = mean_data.set_index('class')
        # return the mean data
        return mean_data


    def compute_feature_class_std(self, data, feature, output,cls):
        '''
        A method which calculates the std of a single feature for a given class
        Arguments:
        data - pandas dataframe containing the training data
        feature - feature whose std we want to calculate
        output - output of the model
        cls - the class of interest in the calculation of the std of feature
        Returns:
        std - std of the feature for the given class
        '''
        # filter the data points belonging to the given class
        # hint: use pandas filtering operation
        class_data = data[data[output] == cls]
        # extract the designated feature column from the filtered data
        feature_class_data = class_data[feature]
        # calculate the std of the feature for the given class
        std = feature_class_data.std()
        # return the std
        return std


    def compute_std_data(self, data, features, output):
        '''
        A method which calculates the stds of every feature class combination
        and tabulates the std data in pandas dataframe
        Arguments:
        data - a pandas dataframe containing the training data
        features - a list containing the features of the model
        output - output of the model
        Returns:
        std_data - a pandas dataframe containing the stds of every feature-class combination
        '''
        # extract the classes
        classes = self.classes
        # a dataframe to store the std data
        std_data = pd.DataFrame()
        # create a column for the class
        std_data['class'] = classes

        # iterate through each feature
        for feature in features:
            # a list to store the stds of the given feature of all the classes
            stds = []
            # iterate through each class
            for cls in classes:
                # calculate the feature std of the class
                std = self.compute_feature_class_std(data, feature, output,cls)
                # append the std to the designated list
                stds.append(std)
            # create a column of stds of the given feature for all the classes
            std_data[feature] = stds

        # set the class column as the index of the dataframe
        std_data = std_data.set_index('class')
        # return the std data
        return std_data


    def fit(self, data, features, output, verbose = False):
        '''
        A method which fits the model i.e. calculates the means and stds of every feature class combination
        Arguments:
        data - a pandas dataframe containing the training data
        features - a python list containing the features of the model
        output - output of the models
        verbose - boolean variable to indicate whether verbose is wanted
        '''
        # extract the features
        self.features = features
        # extract the output
        self.output = output
        # get the unique classes from the output column/labels
        self.classes = self.get_classes(data[output])
        if verbose: print('Extracted the unique classes from the output column')
        # calculate frequency/proportion of each class (this would act as prior probability)
        self.cls_frequencies = self.compute_class_frequencies(data[output])
        if verbose: print('Calculated the class frequency of each class')
        # calculate and tabulate the mean data of all feature class combinations
        self.mean_data = self.compute_mean_data(data, features, output)
        if verbose: print('Calculated and tabulated the mean data')
        # calculate and tabulate the std data of all feature class combinations
        self.std_data = self.compute_std_data(data, features, output)
        if verbose: print('Calculated and tabulated the std data')
        if verbose: print('Finished fitting the model', "\n")


    def probability(self, x, mean, std):
        '''
        A method to calculate the probability using Gaussian pdf,
        given the feature of a datapoint, mean & std of the feature for a class
        Arguments:
        x - feature of a datapoint/example
        mean - mean of a feature for a class
        std - std of a feature for a class
        Returns:
        prob -  probability using gaussian pdf
        '''
        # compute the exponent of the expression
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
        # calculate probability
        prob = (1 / (math.sqrt(2*math.pi) * std)) * exponent
        # return the probability
        return prob


    def aggregate_probability(self, point, cls):
        '''
        A method which calculates the aggregate probability/likelihood over features,
        of the given datapoint belonging to a particular class
        Arguments:
        point - a single datapoint/example
        cls - class of interest
        Returns:
        aggregate_prob - aggregate probability/likelihood of the datapoint
        belonging to a particular class
        '''
        # extract the mean data, std data, and features from the class fields
        std_data = self.std_data
        mean_data = self.mean_data
        features = self.features
        # initialize the aggregate probability to 1
        aggregate_prob = 1

        # iterate through each feature
        for feature in features:
            # extract the feature value of the data point
            x = point[feature]
            # extract the feature mean of the datapoint for the given class from the mean_data
            # hint: extract the appropriate feature class combination by indexing via
            # column name (feature) and index name (class) of dataframe.
            # hint: the first index points to columns i.e. appropriate feature;
            # the 2nd index points to rows i.e. appropriate class.
            mean = mean_data[feature][cls]
            # extract the feature std of the datapoint for the given class from the std_data
            # hint: extract the appropriate feature class combination by indexing via
            # column name (feature) and index name (class) of dataframe.
            # hint: the first index points to columns i.e. appropriate feature;
            # the 2nd index points to rows i.e. appropriate class.
            std = std_data[feature][cls]
            # hint: calculate the probability using gaussian pdf
            # and aggregate the overall probability through multiplication
            aggregate_prob = aggregate_prob * self.probability(x, mean, std)

        # return the aggregate probability/likelihood
        # of the given datapoint belonging to a particular class
        return aggregate_prob


    def predict_point(self, point):
        '''
        A method which classifies a single datapoint
        Arguments:
        point - a datapoint/example
        Returns:
        prediction - predicted label/class of the datapoint
        '''
        # extract the classes and the class freqeuncies
        classes = self.classes
        cls_frequencies = self.cls_frequencies
        # initialize the max prob to -1
        # and prediction to None
        max_prob = -1
        prediction = None

        # iterate through each class in the classification
        for i, cls in enumerate(classes):
            # extract the class frequency which acts as the prior probability
            frequency = cls_frequencies[i]
            # calculate the probability of the data point belonging to the given class
            prob = frequency*self.aggregate_probability(point, cls)
            # if the probability is greater than the maximum registered
            # then update the max probability and predicted label/class
            if prob > max_prob:
                prediction = cls
                max_prob = prob

        # return the predicted labels/class
        return prediction


    def predict(self, data):
        '''
        A method which classifies the multiple datapoints
        Arguments:
        data - a pandas dataframe containing the columns of features and output
        Returns:
        predictions - predicted labels/class of the datapoints
        '''
        # create predictions by applying predict_point method
        # to all the data points using pandas apply function
        predictions = data.apply(lambda x: self.predict_point(x), axis=1)
        # convert the data to numpy array
        predictions = predictions.values
        # return the predictions
        return predictions
