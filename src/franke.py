from numpy import exp


def franke(x, y):
    nineX = 9 * x
    nineY = 9 * y
    first = 0.75 * exp(-(nineX - 2)**2 * 0.25 - (nineY - 2)**2 * 0.25)
    second = 0.75 * exp(-(nineX + 1)**2 / 49 - (nineY + 1)**2 * 0.1)
    third = 0.5 * exp(-(nineX - 7)**2 * 0.25 - (nineY - 3)**2 * 0.25)
    fourth = - 0.2 * exp(-(nineX - 4)**2 - (nineY - 7)**2)

    return first + second + third + fourth
