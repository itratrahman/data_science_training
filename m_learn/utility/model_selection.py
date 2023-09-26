

def k_fold_cross_validation(k, model, data, features, output, performance_metric, **kwargs):

    '''a function which carries out k fold cross validation using a ML model'''

    # import function to get the arguments of a function
    from inspect import signature

    # extract the signature of the model fit method
    signature = signature(model.fit)

    # extract the arguments of the fit method
    arguments = list(signature.parameters.keys())

    # a list to store the performance metric values in each fold
    metric_list = []

    # size of data
    n = len(data)

    # iterating through k folds
    for i in range(k):

        # compute the start and end indices of fold i
        start = (n*i)//k
        end = (n*(i+1)//k)-1

        # extract the validation data
        validation_data = data[start:end+1]

        # extract the head & tail of the rest of the data
        head = data[0:start]
        tail = data[end+1:n]

        # append the head & tail to make up the training data
        training_data = head.append(tail)

        # using the fit method that is appropriate for the model
        if 'features' and 'output' in arguments:
            model.fit(training_data, features, output, **kwargs)
        else:
            model.fit(training_data[features], training_data[output], **kwargs)

        # create the prediction vector using the validation set
        prediction_vector = model.predict(validation_data[features])

        # calculate the performance_metric using the function passed by the user
        metric = performance_metric(validation_data[output],prediction_vector)

        # appending the calculated metric to the designated list
        metric_list.append(metric)

    # calculate the average RSS of the various folds
    metric_avg = sum(metric_list)/len(metric_list)

    # return the average RSS
    return metric_avg
