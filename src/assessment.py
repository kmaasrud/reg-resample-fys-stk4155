import numpy as np
from sklearn.utils import resample
from utils import mean_square, mean_value, MSE, split_and_scale
from regression_methods import OLS


def kfolds(X, y, k):
    """Function that splits a dataset into K different folds. Returns a list of k tuples,
    each tuple being an (X, y) pair.

    Arguments:
        X -- Design matrix
        y -- Output array
        k -- Number of splits"""
    # Shuffling the data together, to ensure the relation between input/output persists
    np.random.seed(4155)
    np.random.shuffle(shuffled := np.concatenate((np.array([y]).T, X), axis=1))
    y = shuffled[:, 0]
    X = shuffled[:, 1:]

    # Find number of data points n
    n = X.shape[0]

    # Amount of data per fold:
    folds_n = n // k

    # Determine how many extra observations thats not enough to make
    # a whole fold. Will be scattered around the already made folds
    excessive_n = n % k
    folds = []

    for i in range(k):
        start = folds_n * i
        end = start + folds_n
        fold_X = X[start : end]
        fold_y = y[start : end]
        folds.append((fold_X, fold_y))

    if excessive_n:
        K = folds_n * k
        for i in range(excessive_n):
            np.append(folds[i][0], X[K+i])
            np.append(folds[i][1], y[K+1])

    return folds


def CV(X, y, k, method, lmb=0):
    folds = kfolds(X, y, k)
    
    # Using MSE as the assessment function
    err = mean_square()
    for i, fold in enumerate(folds):
        # Concatenate all folds other than the current one and set them as training data
        train_X = np.concatenate([fold[0] for j, fold in enumerate(folds) if j!=i], axis=0)
        train_y = np.concatenate([fold[1] for j, fold in enumerate(folds) if j!=i], axis=0)

        # Avoid argument errors by just checking which method is used
        if method == OLS:
            reg = method(train_X, train_y)
        else:
            reg = method(train_X, train_y, lmb)

        test_X = fold[0]; test_y = fold[1]

        # Add the predicted and test values to the mean_square class
        for predict, test in zip(reg.predict(test_X), test_y):
            err.add_vals(predict, test)

    return err.result()


def bootstrap(X, y, N_bootstraps, method, lmb=0):
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    vals = []
    bootstrap_predicts = []
    for i in range(N_bootstraps):
        X_, y_ = resample(X_train, y_train)

        if method == OLS:
            model = method(X_, y_)
        else:
            model = method(X_, y_, lmb)

        bootstrap_predicts.append(model.predict(X_test))

    return np.array(bootstrap_predicts), y_test
