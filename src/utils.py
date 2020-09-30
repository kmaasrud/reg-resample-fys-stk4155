from numba import jit
from inspect import getargspec
import numpy as np
from itertools import product

class mean_square:
    """Evaluates the mean squared error between two lists/arrays."""
    def __init__(self, x=[], y=[]):
        self.xvals = x
        self.yvals = y

    def add_vals(self, x, y):
        self.xvals.append(x)
        self.yvals.append(y)

    def result(self):
        s = 0
        for xval, yval in zip(self.xvals, self.yvals):
            s += (xval - yval)**2
        s /= len(self.xvals)

        return s

def MSE(x, y):
    """Wrapper function for mean_square class"""
    return mean_square(x, y).result()


def R2(x, y):
    """Evaluates the R2 score of two lists/arrays"""
    deno = MSE(x, np.full(len(x), mean_value(y)))
    R2 = 1 - MSE(x, y) / deno
    return R2


def mean_value(y):
    """Evaluates the mean value of a list/array"""
    return sum(y) / len(y)


def design_matrix(x, y, d):
    """Function for setting up a design X-matrix with rows [1, x, y, x², y², xy, ...]
    Input: x and y mesh, keyword argument d is the degree.
    """
    
    if len(x.shape) > 1:
    # reshape input to 1D arrays (easier to work with)
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((d+1)*(d+2)/2)	# number of elements in beta
    X = np.ones((N,p))

    for n in range(1,d+1):
        q = int((n)*(n+1)/2)
        for m in range(n+1):
            X[:,q+m] = x**(n-m)*y**m

    return X


def make_design_matrix(*param_arrays, poly_deg=1):
    """Takes a collection of input arrays and returns the corresponding design matrix.
    The keyword argument poly_deg specifies which degree polynomial you want the design matrix to depict.

    Arguments:
        *param_arrays -- A number of input arrays.

    Keyword arguments:
        poly_deg -- The desired polynomial degree (default = 1)."""

    # Checks if the parameter arrays are all of equal length.
    # If not, raises assertion error.
    is_arrays_equal_len = len({len(param_array)
                               for param_array in param_arrays}) == 1
    assert is_arrays_equal_len, "Parameter arrays are not of equal dimension."

    # Checks that all the input arrays are of type numpy.ndarray.
    # If not, raises assertion error.
    is_params_ndarray = not False in set(type(x) == np.ndarray for x in param_arrays)
    assert is_params_ndarray, "The input arrays must be of type numpy.ndarray"

    n_inputs = len(param_arrays)
    polynomial_permutations = [perm for perm in product(range(poly_deg + 1), repeat=n_inputs) if sum(perm) == poly_deg]
    print(polynomial_permutations)
    print(param_arrays)
    p = len(polynomial_permutations)
    print(p)
    n = len(param_arrays[0])

    X = np.ones((n, p + 1))

    for i, perm in enumerate(polynomial_permutations):
        term = 0
        for j, power in enumerate(perm):
            temp = np.array([x**power for x in param_arrays[j]])
            term *= temp

        X[:, i+1] = term
            #X[:, i+1] = X[:, i+1] * param_arrays[j] ** power


    return X


# SVD inversion
def SVDinv(A):
    ''' Morten recommended us to use this code from the lecture slides for inverting
    the matrices while working on the terrain data.

    Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    import numpy as np

    U, s, VT = np.linalg.svd(A)

    #print(U)
    #print(s)
    #print(VT)

    D = np.zeros((len(U),len(VT)))
    for i in range(len(VT)):
        D[i,i] = s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))


def alternate_design_matrix(x, y, deg):
        # features
        p = int(0.5*( (deg+1)*(deg+2) ))
        X = np.zeros((len(x),p))
        idx=0
        for i in range(deg+1):
            for j in range(deg+1-i):
                X[:,idx] = x**i * y**j
                idx += 1
        return X
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
