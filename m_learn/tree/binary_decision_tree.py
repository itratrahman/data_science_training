# import statements
import numpy as np
import pandas as pd

class binary_decision_tree(object):
    """
    Class module of binary decision tree
    """
    def __init__(self, max_depth = 10, min_samples_split = 1, min_impurity_decrease = -10):
        """
        Class constructor which initializes the parameters of the decision tree
        """
        self.tree = None
        self.nodes = None
        self.data_size = None
        self.features = None
        self.output = None
        self.feature_thresholds = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease


    def compute_number_of_cls_examples(self,labels):
        """
        A method to calculate the number of datapoints/examples in each class
        Arguments:
        labels - a pandas series containing the column of labels
        classes - list of unique classes in classification
        Returns:
        classes - list of classes in sorted order
        num_of_examples - list containing the number of examples of each unique class
        """
        # extract the list of classes in sorted order
        # in binary classification this would be [0,1]
        classes = set(labels.tolist())
        classes = list(classes)
        classes = sorted(classes)
        # a list to store the number of examples of each class
        num_of_examples = []

        # iterate through each class
        for class_ in classes:
            # extract the number of examples in the class
            number = len(labels[labels == class_])
            # append the number to the designated list
            num_of_examples.append(number)

        # return the classes and the list of corresonding number of examples
        return classes, num_of_examples


    def compute_gini_index(self, labels):
        """
        A method to calculate the gini index of a labelled data series
        Arguments:
        labels - a pandas series containing the column of labels
        Returns:
        index - gini index of a labelled data series
        """
        # extract the number of examples of each class
        _, num_of_examples = self.compute_number_of_cls_examples(labels)
        # variable to store the summation term in the calculation of gini index
        summation_term = 0
        # total number of data
        size = len(labels)
        # iterate through each number of datapoints/examples in a class
        for num in num_of_examples:
            # calculate the fraction of datapoints/examples belonging to the class
            fraction = num/size
            # square of the fraction
            fraction_square = fraction**2
            # add to the summation term
            summation_term += fraction_square
        # calculate the gini index
        index = 1 - summation_term
        # return the gini index
        return index


    def compute_gini_gain(self, data, feature, output):
        """
        A method to calculate the gini gain of a split with the given splitting feature
        Arguments:
        data - a pandas dataframe containing the data at the node
        splitting_feature - splitting_feature of the node
        output - output of the model
        Returns:
        gini_gain - gini gain of the split with the given splitting feature
        """
        # size of data
        size = len(data)
        # compute gini index of the parent node
        gini_index_before_split = self.compute_gini_index(data[output])
        # extract the data of the left node
        left_node_data = data[data[feature] == 0]
        # extract the data of the right node
        right_node_data = data[data[feature] == 1]
        # compute the gini index of the left node
        gini_index_left_node = self.compute_gini_index(left_node_data[output])
        # compute the gini index of the right node
        gini_index_right_node = self.compute_gini_index(right_node_data[output])
        # calculate the gini gain of the split
        gini_gain = gini_index_before_split - \
                    (len(left_node_data)/size)*gini_index_left_node - \
                    (len(right_node_data)/size)*gini_index_right_node
        # return the gini gain
        return gini_gain


    def find_thresholds(self, data, feature):
        """
        A method which finds all the possible thresholds of a feature
        Arguments
        data - a pandas dataframe containing the training data
        feature - feature whose thresholds we want to find
        Returns
        thresholds - list of thresholds of a feature
        """
        ######## Extract the all the unique values of the feature in ascending order ########
        # extract the feature vector from the data and convert it to a list
        feature_vector = data[feature].tolist()
        # use set operation to extract the unique values
        feature_vector = set(feature_vector)
        # convert the resulting set to list
        feature_vector = list(feature_vector)
        # sort the resulting list
        feature_vector = sorted(feature_vector)
        #####################################################################################

        # a list to store all the thresholds
        thresholds = []
        # if length of data is 1 then return the list of the lone value
        if len(feature_vector) == 1:
            return [feature_vector[0]]

        # iterate through each value
        for i in range(len(feature_vector)-1):
            # index of the value the next item
            j = i+1
            # calculate the threshold using the successive values
            threshold = (feature_vector[i] + feature_vector[j])/2
            # append the threshold to the designated list
            thresholds.append(threshold)

        # return the list of thresholds
        return thresholds


    def binarize_value(self, x, threshold):
        """
        A method which binarize a value based on the given threshold
        Arguments:
        x - a value
        threhold - threshold for binirization
        Returns:
        0 or 1 depending on value of x
        """
        # if x is greater than the threshold then return 1
        if x >= threshold:
            return 1
        # else return 0
        else:
            return 0


    def find_feature_threshold(self, data, feature, output, thresholds):
        """
        A method which selects the best threshold
        for binarization of a feature based on gini gain
        Arguments:
        data - a pandas dataframe containing the training data
        feature - feature whose best threhold we want to find
        output - output of the model
        thresholds - list of thresholds
        Returns:
        selected_threshold - best threshold of the feature
        """
        # create a copy of the pandas dataframe
        df_copy = data.copy()
        # find the threshold of the feature
        thresholds = self.find_thresholds(df_copy, feature)
        # if there is only one threshold then return the lone threshold value
        if len(thresholds) == 1:
            return thresholds[0]
        # highest gini gain is set to an improbable value
        highest_gini_gain = -100
        # variable to store the best threshold
        selected_threshold = None

        # iterate through each threshold
        for threshold in thresholds:
            # binarize the feature column
            df_copy['feature_binary'] = df_copy[feature].apply(lambda x: self.binarize_value(x, threshold))
            # calculate the gini_gain of the instance if a split is made on the feature
            gain = self.compute_gini_gain(df_copy, 'feature_binary', output)
            # if the calculated gini gain is greater than highest gini gain recorded
            # then update the highest gain and update the selected threshold
            if gain > highest_gini_gain:
                highest_gini_gain = gain
                selected_threshold = threshold

        # return the selected threshold
        return selected_threshold


    def find_feat_thresholds(self, data, features, output):
        """
        A method which finds the best thresholds for all the features
        Arguments:
        data - a pandas dataframe containing the training data
        features - list of features of the model
        output - output of the model
        Returns:
        feature_thresholds - list of the best thresholds for all the features
        """
        # a list to store the feature thresholds
        feature_thresholds = []

        # iterate through each feature
        for feature in features:
            # find the thresholds for the feature
            thresholds = self.find_thresholds(data, feature)
            # find the best threshold for the feature given the data
            best_threshold = self.find_feature_threshold(data, feature, output, thresholds)
            # append the threhold to the designated list
            feature_thresholds.append(best_threshold)

        # return a list of the best thresholds for all the features
        return feature_thresholds


    def binarize_data(self, data, features, output, feature_thresholds):
        """
        A method which binarize the whole feature data given the selected feature thresholds
        Arguments:
        data - a pandas dataframe containing the training data
        features - list of features of the model
        output - output of the model
        feature_thresholds - list of the best thresholds for all the features
        Returns:
        binary_data - a pandas dataframe containing columns of binarized features
        """
        # a dataframe to store the binarized data
        binary_data = pd.DataFrame()

        # iterate through each feature and the selected feature threshold
        for feature, feature_threshold in zip(features, feature_thresholds):
            # if the data is single valued then initialize the column based on sign
            if len(data[data[feature]==feature_threshold]) == len(data):
                # if the feature threshold is positive then initialize the column to ones
                if feature_threshold>0:
                    binary_data[feature] = np.ones(len(data))
                # if the feature threshold is negative then initialize the column to zeros
                else:
                    binary_data[feature] = np.zeros(len(data))
                # skip the rest of iteration
                continue
            # binarize the feature column based on the feature threshold
            binary_data[feature] = data[feature].apply(lambda x: self.binarize_value(x, feature_threshold))

        # return the binary data
        return binary_data


    def find_splitting_feature(self, data, features, output):
        """
        A method which finds the best feature to split the decision tree given the dataset at a node
        Arguments:
        data - a pandas dataframe containing the data at node
        features - list of features of the model
        output - output of the model
        Returns:
        splitting_feature - splitting feature of the node
        """
        # variable to store the splitting feature
        splitting_feature = None
        # highest gini gain recorded so far
        highest_gini_gain = -100

        # iterate through each feature
        for feature in features:
            # calculate the gini gain after a split on the feature
            gini_gain = self.compute_gini_gain(data, feature, output)
            # if the gini gain is higher than the best recorded
            # then update the highest gini gain and the splitting feature
            if gini_gain > highest_gini_gain:
                highest_gini_gain = gini_gain
                splitting_feature = feature

        # return the splitting feature
        return splitting_feature


    def create_leaf_node(self,labels):
        """
        A method to create and return a leaf node
        Arguments:
        labels - a pandas data series containing the column of labels of a node
        Returns:
        a dictionary representing the leaf node
        """
        # extract the number of positive examples
        num_positive_examples = len(labels[labels == +1])
        # extract the number of negative examples
        num_negative_examples = len(labels[labels == -1])
        # if number of postive examples is greater than number of negative examples
        # then prediction is set to 1
        if num_positive_examples>num_negative_examples:
            prediction = 1
        # else prediction is set to zero
        else:
            prediction = -1
        # return the dictionary representing the leaf node
        return {'left_node' : None,
                'right_node' : None,
                'splitting_feature' : None,
                'prediction': prediction,
                'leaf': True}


    def create_tree(self, data, features, output, current_depth, verbose = False):
        """
        A function which creates a binary decision tree recursively
        Arguments:
        data - a pandas dataframe containing the data at node
        features - list of features remaining
        output - output of the model
        current_depth - current depth of recursive step
        Returns:
        a dictionary representing the node
        """
        # variable to store the remaining features
        features_remaining = features[:]
        # extract the column of labels from the data
        labels = data[output]
        if verbose:
            print("--------------------------------------------------------------------")
            print("Subtree, depth = %s (contains %s data points)." % (current_depth, len(labels)))

        ##################### EVALUATING STOPPING CONDITIONS #####################
        # if either of the stopping conditions is reached then we create and return decision leaf
        # based on the output vector given in the recursive call

        # evaluate the stopping condition 1 (no mistakes in current node)
        if self.compute_gini_index(labels) == 0:
            if verbose: print("Stopping condition 1 is reached: no mistakes in current node")
            return self.create_leaf_node(labels)
        # evaluate the stopping condition 2 (if there are no remaining features)
        if features_remaining == []:
            if verbose: print("Stopping condition 2 is reached: no remaining features")
            return self.create_leaf_node(labels)
        # evaluate the stopping condition 3 (reached the maximum depth)
        if current_depth >= self.max_depth:
            if verbose: print("Stopping condition 3 is reached: reached the maximum depth")
            return self.create_leaf_node(labels)
        # evaluate the stopping condition 4 (node size reached the minimum acceptable size)
        if len(data)<self.min_samples_split:
            if verbose: print("Stopping condition 4 is reached: node size reached the minimum acceptable size")
            return self.create_leaf_node(labels)
        ##########################################################################

        # find the splitting features
        splitting_feature = self.find_splitting_feature(data, features, output)
        ##################### EVALUATING STOPPING CONDITION 5 #####################

        # calculate the gini gain of the split on the splitting feature
        gini_gain = self.compute_gini_gain(data, splitting_feature, output)
        # size of the node in the current recursive call
        size_of_current_node = len(data)
        # calculate the impurty decrease
        impurity_decrease = (size_of_current_node/self.data_size)*gini_gain
        # evaluate stopping condition 5 (impurity decrease is less then minimum acceptable impurity decrease)
        if impurity_decrease < self.min_impurity_decrease:
            if verbose: print("Stopping condition 5 is reached: impurity decrease is less than\
            minimum acceptable impurity decrease")
            return self.create_leaf_node(labels)
        ###########################################################################

        if verbose:print("splitting feature: ", splitting_feature)

        # extract the data of the left node
        left_node_data = data[data[splitting_feature] == 0]
        # extract the data of the right node
        right_node_data = data[data[splitting_feature] == 1]
        # remove the splitting feature from the remaining features
        features_remaining.remove(splitting_feature)
        # recursively create left node
        left_node = self.create_tree(left_node_data, features_remaining, output,
                                    current_depth + 1,verbose = verbose)
        # recursively create right node
        right_node = self.create_tree(right_node_data, features_remaining, output,
                                    current_depth + 1,verbose = verbose)
        # return the dictionary representing the node
        return {'left_node' : left_node,
                'right_node' : right_node,
                'splitting_feature' : splitting_feature,
                'prediction': None,
                'leaf': False}


    def num_of_nodes(self, tree = None):
        """
        A method which recursively counts the number of nodes in a tree
        Arguments:
        tree - a dictionary representing the decision tree
        Returns:
        number of nodes
        """
        if tree['leaf']:
            return 1
        return 1 + self.num_of_nodes(tree['left_node']) + self.num_of_nodes(tree['right_node'])


    def fit(self, data, features, output, verbose = False):
        """
        A method which creates and stores the decision tree
        Arguments:
        data - a pandas dataframe containing the data at node
        features - list of features of the model
        output - output of the model
        """
        # initialize the class fields of features, output, data size
        self.features = features
        self.output = output
        self.data_size = len(data)
        # initialize current depth is 0
        current_depth = 0
        # find the best thresholds for all the features based on gini gain
        feature_thresholds = self.find_feat_thresholds(data, features, output)
        if verbose: print("Completed finding feature thresholds")
        self.feature_thresholds = feature_thresholds
        # binarize the featuer columns
        binary_data = self.binarize_data(data, features, output, feature_thresholds)
        # add the output column to the binary data
        binary_data[output] = data[output]
        if verbose: print("Completed binarizing data", "\n")
        # create and store the decision tree
        if verbose: print("Fitting decision tree")
        if verbose: print("--------------------------------------------------------------------")
        self.tree = self.create_tree(binary_data, features, output, current_depth, verbose = verbose)
        if verbose: print("--------------------------------------------------------------------")
        if verbose: print("--------------------------------------------------------------------")
        if verbose: print("Compeleted fitting decision tree")
        # count the number of nodes
        self.nodes = self.num_of_nodes(self.tree)


    def classify_point(self, tree, x):
        """
        A method which recursively classifies a single data point
        Arguments:
        tree - a dictionary representing the decision tree/Subtree
        x - a data point
        Returns:
        prediction of the point
        """
        # if the node is a leaf then return the prediction
        if tree['leaf']:
            return tree['prediction']
        # else recursively go a depth down the tree
        else:
            ##### whether to go further down the left or right node will depend
            # on the value of splitting feature of the datapoint #####

            # extract the splitting feature of the current node
            splitting_feature = tree['splitting_feature']
            # extract the value of the splitting feature of the data point
            value = x[splitting_feature]
            # if the value is 0 then recurse down the left node
            if value == 0:
                return self.classify_point(tree['left_node'], x)
            # if value is 1 then recurse down the right node
            if value == 1:
                return self.classify_point(tree['right_node'], x)


    def predict(self, data):
        """
        A method which computes and returns the predictions for multiple data points stored in pandas
        Arguments:
        data - a pandas dataframe containing the data whose predictions we want to find
        Returns:
        predictions - a numpy array containing the predictions of the data
        """
        # binarize the data
        binary_data = self.binarize_data(data, self.features, self.output, self.feature_thresholds)
        # apply the classify_point method to the pandas dataframe to classify all the data points
        predictions = binary_data.apply(lambda x: self.classify_point(self.tree, x), axis=1)
        # convert the predictions to numpy array
        predictions = predictions.values
        # return the predictions
        return predictions
