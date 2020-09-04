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


def meanvalue(y):
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
