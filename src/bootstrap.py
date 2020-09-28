import numpy as np 
from sklearn.utils import resample
from resampling import split_and_scale
from regression_methods import OLS 

max_degree = 5 # polynomials in x and y up to fifth order
    
def bootstrap(X, y_data, N_bootstraps, degree):
    X_train, X_test, y_train, y_test = split_and_scale(X, y_data)
    for degree in range(max_degree):
        # (m x N_bootstraps) to hold the column vectors y_predict for each boostrap iteration
        y_predict_train = np.empty((y_train.shape[0], N_bootstraps))
        y_predict_test = np.empty((y_test.shape[0], N_bootstraps))
        for i in range(N_bootstraps):
            X_, y_ = resample(X_train, y_train)

            y_predict_train[:,i] = OLS(X_, y_)
            y_predict_test[:,i] = OLS(X_, y_)
    return y_predict_train
