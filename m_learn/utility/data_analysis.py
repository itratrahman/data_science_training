
def polynomial_plot_regression(degrees, regression_model, train_data,validation_data,\
                               feature, output, subplot_dimension, **kwargs):

    '''a function which plots model of range of complexity complexity'''

    ##Import statements
    import pandas as pd
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mticker
    from .data_preprocessing import polynomial_dataFrame

    ##A list to store the rss of all model complexities
    RSS = []

    ##Extracting the output vector from training data
    output_train = train_data[output]

    ##Iterating through each degree of complexity
    for degree in degrees:

        ##Creating the a polynomial data of the degree complexity under consideration
        polynomial_train = polynomial_dataFrame(train_data[feature], degree)

        ##Fitting the model using the training data
        regression_model.fit(polynomial_train, output_train)

        ##Extracting the power columns from the dataframe
        feature_columns = polynomial_train.columns.values.tolist()

        ##Adding the price columnn to the data
        polynomial_train[output] = train_data[output]

        ##Sorting the dataframe by power 1 (1st degree) so that points in the plots join sequentially
        polynomial_train = polynomial_train.sort_values(['power1', output])

        ##Adding a subplot
        plt.subplot(subplot_dimension[0], subplot_dimension[1], degree)
        plt.plot(polynomial_train['power1'], polynomial_train[output], '.',
                 polynomial_train['power1'], model.predict(polynomial_train[feature_columns]), '-')

        ##Adding the x & y labels
        plt.xlabel(kwargs['xlabel'])
        plt.ylabel(kwargs['ylabel'])

        ##Adding a grid to the plot
        plt.grid()
        ##Adding the title of the plot
        plt.title("Degree of Polynomial: " + str(degree))

        ##Extracting the output column of validation dataset
        validation_output = validation_data[output]

        ##Computing the prediction vector using the validation dataset
        predictions = model.predict(polynomial_dataFrame(validation_data[feature], degree))

        ##Calculating the error vector of the model using validation data
        error = predictions - validation_output

        ##Squaring the error vector
        error_square = error*error

        ##Calculating the residual sum of square
        rss = error_square.sum()

        ##Appending the RSS to the designated list
        RSS.append(rss)

    ##Adjusting height and width space of the subplots
    plt.subplots_adjust(wspace = 0.45, hspace = 0.40)

    ##Adding the grand title to the whole figure
    plt.suptitle(kwargs['fig_title'], fontsize = 16)

    ##Showing the plot
    plt.show()

    ##returning the list of residual sum of square of models of different complexities
    return RSS


def variance_analysis_regression(degree, regression_model, feature, output, datas, subplot_dimension,**kwargs):

    '''a function which plots data for the same regression model fitted on different datasets'''

    ##import statements
    import matplotlib.pyplot as plt
    from .data_preprocessing import polynomial_dataFrame

    ##iterating through each dataset in multiple arguments
    for counter, data in enumerate(datas):

        ##creating a polynomial data to the given degree
        polynomial_data = polynomial_dataFrame(data[feature], degree)

        ##extracting the output data
        output_data = data[output]

        ##fitting the data using the ML model
        regression_model.fit(polynomial_data, output_data)

        ##extracting the power columns from the dataframe
        feature_columns = polynomial_data.columns.values.tolist()

        ##adding the price columnn to the data
        polynomial_data[output] = data[output]

        ##sorting the dataframe by power and price so that points in the plots join sequentially
        polynomial_data = polynomial_data.sort_values(['power1', output])

        ##adding a subplot
        plt.subplot(subplot_dimension[0], subplot_dimension[1], counter+1)
        plt.scatter(polynomial_data['power1'], polynomial_data[output], linewidths  = 0.05, marker = '.')
        plt.plot(polynomial_data['power1'], model.predict(polynomial_data[feature_columns]), '-', linewidth = 1, color = 'green')

        ##adding the x & y labels
        plt.xlabel(kwargs['xlabel'])
        plt.ylabel(kwargs['ylabel'])

        ##adding the title and grid to the plot
        plt.title("Dataset " + str(counter+1), fontsize = 'medium')
        plt.grid()

    ##adjusting height and width space of the subplot
    plt.subplots_adjust(wspace = 0.45, hspace = 0.40)

    ##adding the grand title to the whole figure
    plt.suptitle("Degree: "+ str(degree), fontsize = 'medium')

    ##showing the plot
    plt.show()


def polynomial_plot_classification(data, model, train_test_split_ratio, features, output, degree,**kwargs):


    '''a function for investigating the decision boundary of polynomial model'''

    ##Import statements
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from .data_preprocessing import polynomial_dataFrame_multi_features, mesh_data

    ##Splitting the data into train and test set
    train_data, test_data = train_test_split(data, train_size = train_test_split_ratio, random_state = 0)

    ##Creating polynomial data for train and test set
    polynomial_data_train = polynomial_dataFrame_multi_features(train_data,features,degree,output)
    polynomial_data_test = polynomial_dataFrame_multi_features(test_data,features,degree,output)

    ##Extracting the polynomial features
    polynomial_features = polynomial_data_train.columns.tolist()
    polynomial_features.remove(output)

    ##Constructing and training model using polynomial features
    polynomial_features_model = model(**kwargs)
    polynomial_features_model.fit(polynomial_data_train[polynomial_features], polynomial_data_train[output])

    ##Creating data for mesh plot
    xx, yy, mesh_points = mesh_data(data, features, mesh_step = 0.02)
    polynomial_mesh_data = polynomial_dataFrame_multi_features(mesh_points, features, degree)

    ##Z data for the mesh plot
    polynomial_predictions = polynomial_features_model.predict(polynomial_mesh_data)
    polynomial_predictions = np.array(polynomial_predictions)
    polynomial_predictions = polynomial_predictions.reshape(xx.shape)

    ##Mesh plot
    plt.pcolormesh(xx, yy, polynomial_predictions, cmap=plt.cm.Paired)
    plt.scatter(data[features[0]], data[features[1]], c = data[output], edgecolors='k')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()

    ##Printing the accuracy of the polynomial model
    print("Accuracy of degree "+str(degree)+ " model: ", \
          polynomial_features_model.score(polynomial_data_test[polynomial_features], polynomial_data_test[output]))
