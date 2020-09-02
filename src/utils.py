def MSE(x, y):
    N = len(x)

    s = 0
    for xval, yval in zip(x, y):
        s += (xval - yval)**2

    s /= N
    return s

def R2(x, y):
    deno=MSE(x,meanvalue(y))
    R2=1-(MSE(x,y)/deno)
    return R2

def meanvalue(y):
    N = len(y)

    s = sum(y)

    s /= N
    return s
