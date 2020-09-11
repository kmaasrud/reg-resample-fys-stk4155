from inspect import getargspec
from numpy import array, linspace, full, ndarray


def MSE(x, y):
    """Evaluates the mean squared error between two lists/arrays."""
    s = 0
    for xval, yval in zip(x, y):
        s += (xval - yval)**2

    s /= len(x)
    return s


def R2(x, y):
    """Evaluates the R2 score of two lists/arrays"""
    deno = MSE(x, full(len(x), meanvalue(y)))
    R2 = 1 - MSE(x, y) / deno
    return R2


def mean_value(y):
    """Evaluates the mean value of a list/array"""
    return sum(y) / len(y)


def make_data_matrices(func, *param_arrays):
    """Takes in a function and a number of parameter vectors and returns
    the design matrix X as well as the true function value array Y.

    Arguments:
        func -- The function to estimate
        *param_arrays -- A number of arrays corresponding to the number of parameters func takes"""

    # Checks if the parameter arrays are all of equal length.
    # If not, raises assertion error.
    is_arrays_equal_len = len({len(param_array)
                               for param_array in param_arrays}) == 1
    assert is_arrays_equal_len, "Parameter arrays are not of equal dimension."

    # Checks that the supplied function actually takes the number of parameters supplied.
    # If not, raises assertion error.
    is_func_args_len_equal_N = len(getargspec(func).args) == len(param_arrays)
    assert is_func_args_len_equal_N, "Function does not take the same number of parameters you have supplied."

    N = len(param_arrays[0])
    # Transposing X to get the correct dimensionality
    X = array([full(N, 1), *param_arrays]).T
    Y = func(*param_arrays)

    return X, Y

# SVD inversion
import numpy as np
def SVDinv(A):
    ''' Morten recommended us to use this code from the lecture slides for inverting
    the matrices while working on the terrain data.

    Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)

    #print(U)
    #print(s)
    #print(VT)

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))

"""
Confidence intervals. A lot of arguments, so might be better to move it
outside a function. Note to self: Remember to test the function, and doublecheck the formulas found at the following site
https://www.investopedia.com/ask/answers/042415/what-difference-between-standard-error-means-and-standard-deviation.asp

X=Design matrix, beta=beta_array, CIpercent=How many percent of data that
contains the mean value(Seems like 95% is quiet common), y=array with
data from franke's function. y_pred=the predicted values, n=observations


# Function is malfunctioning, so I'm just commenting it out now for testing
def CI (X_matrix,beta,CIpercent,y,y_pred,n):
    #The different Zscores representing the CI percentages can be found online
    if CIpercent == 90:
        Zscore = 1.645
    elif CIpercent == 95:
        Zscore = 1.96
    elif CIpercent == 99:
        Zscore = 2.576

    p = len(beta)

    sd = np.sqrt((1/(n−p−1))*np.sum((y-y_pred)∗∗2))  #Standard deviation
    cov_matrix = sd**2*np.linalg.inv(X_matrix.T.dot(X_matrix))
    variance = np.diag(cov_matrix)                 #Variance of betas along diagonal

    CI_min = []
    CI_max = []

    for i in range(p):
        variety=np.sqrt(variance[i]/n)*Zscore
        CI_min_value=beta[i]-variety
        CI_max_value=beta[i]+variety
        CI_min.append(CI_min_value)
        CI_max.append(CI_max_value)

    return CI_min, CI_max    #Returns two lists
"""
