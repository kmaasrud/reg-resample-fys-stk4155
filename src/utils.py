def MSE(x, y):
    """Evaluates the mean squared error between two lists/arrays."""
    s = 0
    for xval, yval in zip(x, y):
        s += (xval - yval)**2

    s /= len(x)
    return s


def R2(x, y):
    """Evaluates the R2 score of two lists/arrays"""
    deno = MSE(x, meanvalue(y))
    R2 = 1 - MSE(x, y) / deno
    return R2


def meanvalue(y):
    """Evaluates the mean value of a list/array"""
    return sum(y) / len(y)
