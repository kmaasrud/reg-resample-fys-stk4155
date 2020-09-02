def MSE(x, y):
    N = len(x)

    s = 0
    for xval, yval in zip(x, y):
        s += (xval - yval)**2

    s /= N
    return s
