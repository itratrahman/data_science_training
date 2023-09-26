
def get_xy_data(dataframe, features = None, output = None, intercept = True):

    '''
    Function to parse the feature matrix and output vector from a pandas dataframe
    Arguments:
    dataframe - pandas dataframe for storing the data
    features - a list to store the features
    ouput - a string to store the output
    intercept - a boolean variable to indicate whether the caller wants to create
    an extra column of ones for the intercept
    Returns:
    if variable output is given then the function returns both feature matrix X & output vector y
    if variable output is None then only the feature matrix X is returned
    '''

    # ignore pandas warning
    import warnings
    warnings.filterwarnings('ignore')

    # import numpy
    import numpy as np

    # if no features are given then just return the numpy matrix of the dataframe
    if features == None:
        return dataframe.values

    # if intercept is wanted by the user then create
    # a column of ones in the feature matrix to account for the bias term
    if intercept:
        # create a column of 1s in the dataframe
        dataframe['intercept'] = np.ones(dataframe.shape[0])
        # prepend the term "constant" to the features list
        features = ['intercept'] + features

    # extract the feature matrix and convert it to numpy array
    X = dataframe[features].values

    # if there is no output then just return feature matrix X
    if output == None:
        return X
    # if the output vector is wanted by the user
    # then return both feature matrix X and output vector y
    else:
        # extract the output column and convert it to numpy array
        y = dataframe[output].values
        # return the feature matrix and output vector
        return (X, y)


def polynomial_dataFrame(data, degree):

    '''
    A function to create a pandas dataframe consisting
    of polynomials of a given feature up to a given degree
    Arguments:
    data - a pandas series containing feature data whose polynomials are wanted
    degree - maximum degree of polynomial wanted
    Returns:
    polynomial_dataFrame - a pandas dataframe containing polynomial features upto a given degree
    '''

    # import statements
    import pandas as pd
    import math

    # create a pandas dataframe to store the polynomial features
    polynomial_dataFrame = pd.DataFrame()

    # store the degree 1 polynomial feature, which is the feature itself
    polynomial_dataFrame['power1'] = data

    # if the degree of polynomial wanted is greater than 1
    # then create and store polynomials of the feature
    if degree > 1:
        # iterate through each degree of polynomial
        for power in range(2, degree+1):
            # column name for the polynomial data
            poly = 'power' + str(power)
            # store polynomial feature of the given degree
            polynomial_dataFrame[poly] = data.apply(lambda x: math.pow(x,power))

    # return the polynomial dataframe
    return polynomial_dataFrame


def normalize_matrix(X, features = None, norm_vector = True):

    '''
    A function to carry out columnwise normalization of matrix
    Arguments:
    X - either a numpy array containing the feature matrix
    or a pandas dataframe containing columns of the features
    features - a list of features
    norm_vector - a boolean/indicator variable which indicates whether
    the normalizing vector/feature normalizer is wanted or not
    Returns:
    X_normalized - normalized feature matrix
    norm - (if wanted) the normalizing vector
    '''

    # import numpy
    import numpy as np

    # if no feature list is given then it is assumed a numpy array is given
    if features == None:
        # extract the columnwise norm vector of the matrix
        norm = np.linalg.norm(X,axis = 0)
        # compute the normalized matrix
        X_normalized = X/norm
    # if a feature list is given then it is assumed a pandas dataframe is given
    else:
        # extract the columnwise norm vector of the selected features of dataframe
        norm = np.linalg.norm(X[features],axis = 0)
        # copy the dataframe
        X_normalized = X.copy()
        # compute the normalized matrix only for the selected features
        X_normalized[features] = X[features]/norm

    # if the norm vector is wanted by the user
    # then return the normalizad matrix as well as the norm vector
    if norm_vector == True:
        return (X_normalized, norm)
    # else when the norm vector is not wanted
    # then return just the normalized matrix
    else:
        return X_normalized


def train_test_feature_scaler(scaler, train_data, test_data, features):

    '''
    A function for scaling selected feature columns of train and test data (pandas) using a scalar object
    Arguments:
    scaler - an sklearn scaler object
    train_data - pandas dataframe containing the training set
    test_data - pandas dataframe containing the test set
    features - list of features selected for scaling
    Returns:
    scaled_train_data - pandas dataframe containing the scaled training set
    scaled_test_data - pandas dataframe containing the scaled test set
    '''

    # import pandas
    import pandas as pd

    ################### Find the columns not selected for scaling ####################

    # extract all the columns in the pandas dataframe
    all_columns = train_data.columns.values.tolist()
    # extract the rest of the features that are not selected for scaling
    rest_of_cols = [feature for feature in all_columns if feature not in features]

    ############## Fit and scale selected feature columns of train data ##############

    # extract the unselected columns of the training data and store it in a dataframe
    train_rest = train_data[rest_of_cols]
    # scale the columns of the selected featues of the training data
    train_feature = scaler.fit_transform(train_data[features])
    # above operation returns a numpy array,
    # so we convert this into pandas dataframe with the features as the column names
    scaled_train_data = pd.DataFrame(train_feature, columns = features)
    # add the rest of the unscaled column to the above dataframe to complete the dataframe
    for col in rest_of_cols:
        scaled_train_data[col] = train_rest[col].values

    ################### Scale selected feature columns of test data ###################

    # carry out same operations as above on the test data this time
    # use the same scalar fitted on the train data
    test_rest = test_data[rest_of_cols]
    test_feature = scaler.transform(test_data[features])
    scaled_test_data = pd.DataFrame(test_feature, columns = features)
    for col in rest_of_cols:
        scaled_test_data[col] = test_rest[col].values

    # return the scaled train and test data
    return scaled_train_data, scaled_test_data


def mesh_data(data, features, mesh_step, boudary_extension = .5):

    '''
    A function which creates xx, yy, and flattened mesh points of a 2 feature data
    Arguments:
    data - a pandas dataframe containing the data which gives the 2D span of the mesh plot
    features - the list of 2 features along which mesh plot is going to be plot
    mesh_step - step size in the mesh plot
    boudary_extension - amount of extension beyond the boundary of the mesh plot
    Returns:
    xx - xx points of the mesh plot
    yy - yy points of the mesh plot
    mesh_points - a pandas dataframe containing flattened data of xx and yy matrices
    '''

    # import numpy & pandas
    import numpy as np
    import pandas as pd

    ################################## Compute the 2D span of the mesh plot ##################################
    # calculate the minimum and maximum value along the x axis of the mesh plot
    x_min, x_max = data[features[0]].min() - boudary_extension, data[features[0]].max() + boudary_extension
    # calculate the minimum and maximum value along the y axis of the mesh plot
    y_min, y_max = data[features[1]].min() - boudary_extension, data[features[1]].max() + boudary_extension
    #######################################################################################################

    # create the xx and yy matrices of mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step), np.arange(y_min, y_max, mesh_step))
    # create a two column dataframe which contains flattened columns of xx and yy matrices
    mesh_points = pd.DataFrame({features[0]: xx.ravel(), features[1]: yy.ravel()})
    # return xx, yy, and flattened mesh points
    return xx, yy, mesh_points


def polynomial_dataFrame_multi_features(data, features, degree, output = None):

    '''A function to create polynomial data for multiple features
    Arguments:
    data - pandas dataframe containing feature columns
    degree - maximum degree of polynomial wanted for all the train_test_feature_scaler
    output - python string containing the output
    Returns:
    multi_feature_polynomial_dataframe - a pandas dataframe containing
    polynomial data for multiple features
    '''

    # import the pandas library
    import pandas as pd

    # a list to store the dataframes
    dataframes = []

    # iterating through each feature whose polynomial data is wanted
    for feature in features:

        # contruct the polynomial data of the feature using the specialized function
        dataframe = polynomial_dataFrame(data[feature], degree)

        # rename the columns of the polynomial dataframe
        # by appending the name of the feature
        dataframe.columns = feature + "_" + dataframe.columns

        # appending the polynomial data of the feature to the designated list
        dataframes.append(dataframe)

    # if an output column is not wanted
    # then concatenate just concatenate the polynomial dataframes of each feature side by side
    if output is None:
        # concatenate the polynomial dataframe of all the features
        multi_feature_polynomial_dataframe = pd.concat(dataframes, axis=1)
    # if an output column is wanted
    # then concatenate the output column as well
    else:
        # concatenate the polynomial dataframe of all the features
        dataframes.append(data[output])
        # concatenate the output column
        multi_feature_polynomial_dataframe = pd.concat(dataframes, axis=1)

    # return the multi feature polynomial dataframes
    return multi_feature_polynomial_dataframe
