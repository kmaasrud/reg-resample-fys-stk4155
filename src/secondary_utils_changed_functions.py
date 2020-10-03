#Just an extra file, with functions that work.
import numpy as np
from utils import design_matrix, R2

def k_folds(n, k=5, seed = None):
    """
    Returns a list with k lists of indexes to be used for test-sets in
    cross-validation. The indexes range from 0 to n, and are randomly distributed
    to the k groups s.t. the groups do not overlap
    """
    indexes = np.arange(n)

    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(indexes)

    min_size = int(n/k)
    extra_points = n % k

    folds = []
    start_index = 0
    for i in range(k):
        if extra_points > 0:
            test_indexes = indexes[start_index: start_index + min_size + 1]
            extra_points -= 1
            start_index += min_size + 1
        else:
            test_indexes = indexes[start_index: start_index + min_size]
            start_index += min_size
        folds.append(test_indexes)
    return folds


def k_Cross_Validation(x, y, z, k=5, d=3, reg_method = 'Linear', lmb = None, seed = None):
    """
    Function that performs k-fold cross-validation with Linear, Lasso, or Ridge regression
    on data x, y, z=f(x, y) for some function f that we are trying to model.
        - x, y and z are 1-dimensional arrays.
        - k specifies the number of folds in the cross validation.
        - d specifies the polynomial degree of the linear model.
        - reg_method specifies the regression method (Linear, Lasso, or Ridge)
        - lmb is the lambda parameter for Lasso and Ridge regression.
    """

    error_test = []
    error_train = []
    r2 = []

    n = len(z)              # Number of "observations"
    i = int(n/k)            # Size of test set
    #print(f"\nPERFORMING {k}-FOLD CROSS VALIDATION (with {reg_method} regression):")
    #print(f"Number of observations (n): {n}")
    #print(f"Minimum size of test set: {i}")
    #print(f"Degree of the polynomial: {d}")

    test_folds = k_folds(n, k=k, seed=seed)

    if reg_method == 'Lasso':
        model = Lasso(alpha=lmb, fit_intercept = False, tol=0.001, max_iter=10e6)

    for indexes in test_folds:
        m = len(indexes)
        x_test = x[indexes]
        y_test = y[indexes]
        z_test = z[indexes]
        x_train = x[np.delete(np.arange(n), indexes)]
        y_train = y[np.delete(np.arange(n), indexes)]
        z_train = z[np.delete(np.arange(n), indexes)]

        X_test = design_matrix(x_test, y_test, d)
        X_train = design_matrix(x_train, y_train, d)

        if reg_method == 'Linear':
            beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
            z_pred_test = X_test @ beta
            z_pred_train = X_train @ beta


        if reg_method == 'Ridge':
            dim = len(X_train.T @ X_train)
            beta = np.linalg.pinv((X_train.T @ X_train) + lmb*np.identity(dim)) @ X_train.T @ z_train
            z_pred_test = X_test @ beta
            z_pred_train = X_train @ beta

        if reg_method == 'Lasso':
            fit = model.fit(X_train, z_train)
            z_pred_train = fit.predict(X_train)
            z_pred_test = fit.predict(X_test)


        error_test.append(sum((z_test - z_pred_test)**2)/len(z_test))
        error_train.append(sum((z_train - z_pred_train)**2)/len(z_train))
        #error_test.append(((z_test - z_pred_test)@(z_test - z_pred_test).T)/len(z_pred))
        #error_train.append(((z_train - z_pred_train)@(z_train - z_pred_train).T)/len(z_pred))
        r2.append(R2(z_test, z_pred_test))

    test_err = np.mean(error_test)
    train_err = np.mean(error_train)
    r2_score = np.mean(r2)

    return test_err, train_err, r2_score
