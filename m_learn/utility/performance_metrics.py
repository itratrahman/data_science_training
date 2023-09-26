
def residual_sum_of_square(y_, y):

    '''
    Function to compute the residual sum of square
    given the predictions and true outputs
    Arguments:
    y - vector of true outputs of shape (number of examples,)
    y_ - vector of predictions of shape (number of examples,)
    Returns:
    rss - residual sum of square
    '''

    # import the numpy package
    import numpy as np

    # compute residual
    residual = y - y_
    # compute square of residual
    residual_squared = np.square(residual)
    # compute sum of the square of residual
    rss = residual_squared.sum()

    # return rss
    return rss


def mean_squared_error(y_, y):

    '''
    Function for computing the mean squared error
    given the predictions and true outputs
    Arguments:
    y - vector of true outputs of shape (number of examples,)
    y_ - vector of predictions of shape (number of examples,)
    Return:
    mse - mean squared error
    '''
    # import the numpy package
    import numpy as np
    # calculate the residual sum of square using the designated function
    rss = residual_sum_of_square(y_, y)
    mse = rss/np.shape(y)[0]
    # returning mse by dividing by number of examples
    return mse


def accuracy_score(predictions, labels):

    '''
    A function for computing accuracy given the predicted labels and true labels
    Arguments:
    predictions - a numpy array containing the predicted labels
    labels - a numpy array containing the corresponding true labels
    Return:
    accuracy - accuracy of the model
    '''
    # import numpy
    import numpy as np

    # count the number of correct predictions by comparing the predicted labels with corresponding true labels
    # hint: use a list comprehension inside the numpy sum operator to create a list
    # which shows 1 when prediction is correct and 0 when prediction is wrong,
    # the sum operator is then used to count the number of correct predictions in the resulting list
    num_of_correct_preds = np.sum([1 if x==y else 0 for x,y in zip(predictions, labels)])
    # calculate the accuracy of the model
    accuracy = num_of_correct_preds/len(predictions)
    # return the accuracy
    return accuracy


def precision_score(prediction_vector, output_vector):

    '''a function for computing the precision
    given the predictions and true outputs'''

    # variables to store the number of true positives and false positives
    tp =  0
    fp = 0

    # iterating through each prediction and output
    for prediction, output in zip(prediction_vector, output_vector):
        # if prediction is positives
        if prediction == 1:
            # if prediction and output are both positives
            # then the true positive is incremented
            if prediction == output:
                tp += 1
            # else if prediction and output does not match
            # false positive is incremented
            else:
                fp += 1

    # calculating the precision
    precision = tp/(tp+fp)
    # returning the precision
    return precision


def recall_score(prediction_vector, output_vector):

    '''a function for computing the recall
    given the predictions and true outputs'''

    # variables to store the number of true positives and false positives
    tp =  0
    fn = 0

    # iterating through each prediction and output
    for prediction, output in zip(prediction_vector, output_vector):
        # if prediction is positives
        if prediction == 1:
            # if prediction and output are both positives
            # then the true positive is incremented
            if prediction == output:
                tp += 1
        # else if prediction is negative
        else:
            # if prediction does not match with output
            # false negavie is incremented
            if prediction != output:
                fn +=1

    # calculating the recall
    recall = tp/(tp+fn)
    # returning the precision
    return recall


def confusion_matrix(fitted_model, test_data, features, output, labels = None, classes = None):

    '''a function for computing the the confusion metrics and return
    it in a pandas dataframe with approriate columns and indexes'''

    # import statements
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    # if no classes are given then the classes are extracted from the fitted model
    if classes is None:
        classes = fitted_model.classes_

    # if no labels are given then labels are initilized to the class labels from the model
    if labels is None:
        labels = fitted_model.classes_

    # Creating a confusion matrix using the sklearn function
    # labels are given to control the order of the class in confusion matrix
    # e.g. +1 is the first column and 0 is the second column in this case
    matrix = confusion_matrix(test_data[output], \
    fitted_model.predict(test_data[features]), labels = labels)

    # Converting the numpy array to pandas dataframe with named indexes and columns
    matrix = pd.DataFrame(matrix, columns=classes, index=classes)

    # returning the matrix
    return matrix
