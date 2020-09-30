from numba import jit
import numpy as np 
from sklearn.utils import resample
from resampling import split_and_scale
from regression_methods import OLS 
from utils import MSE, mean_value

    
@jit
def bootstrap(X, y_data, N_bootstraps, method, lmb=0):
    X_train, X_test, y_train, y_test = split_and_scale(X, y_data)
    MSEs = []
    for i in range(N_bootstraps):
        X_, y_ = resample(X_train, y_train)

        if method == OLS:
            model = method(X_, y_)
        else:
            model = method(X_, y_, lmb)

        y_predict_bootstrap = X_test @ model.beta
        MSEs.append(MSE(y_predict_bootstrap, y_test))

    return mean_value(MSEs)
