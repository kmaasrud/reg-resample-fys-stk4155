import numpy as np
from random import shuffle
import math
from utils import R2    #Correct?
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(X, y, test_size=0.2):
    """
    Function that splits the data in test and training data and scale the data.
    4/5 of the data is training data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


"""
Small test for kfolds
z=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
print(z)
print(kfolds(z,4))
"""

"""
The cross validation, x and y= arrays, z=function,
nlambdas=decide which values of lambda to use, k=amount
of folds, d=degree of polynomial, m=method where 1=OLS,
2=ridge and 3=Lasso. Probably a much more elegant way
to choose the methods, but that can be implemented later.

The function isn't finished yet, but will probably be
easier to finish after all the methods are done
"""
def CV_ridge(x,y,z,nlambdas,k,d,m):
    #Defines the folds
    folds_k = kfolds(z, k)
    #Making the lists ready for the errors
    test_error=[]
    train_error=[]

    R2_score=[]

    for train_inds, test_inds in folds_k:
        xtrain = x[train_inds]
        ytrain = y[train_inds]
        ztrain = z[train_inds]

        xtest = x[test_inds]
        ytest = y[test_inds]
        ztest = z[test_inds]

        #A little bit unsure which parameters to send to "make_data_matrices"
        X_test = make_data_matrices(z, x_test, y_test)
        X_train = make_data_matrices(z, x_train, y_train)

        #Choosing the method:
        if m==1:
            pass
            #OLS()
        elif m==2:
            pass
            #Ridge()
        elif m==3:
            pass
            #Lasso()

        test_error_val=sum((z_test - z_pred_test)**2)/len(z_test)
        train_error_val=sum((z_train - z_pred_train)**2)/len(z_train)
        R2_val=R2(z_test, z_pred_test)

        test_error.append(test_error_val)
        train_error.append(train_error_val)
        R2_score.append(R2_val)
    test_error_mean = np.mean(test_error)
    train_error_mean = np.mean(train_error)
    R2_score_mean = np.mean(R2_score)

    return test_error_mean, train_error_mean, R2_score_mean


