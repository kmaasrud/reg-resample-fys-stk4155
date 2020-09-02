from numpy import exp, linspace, meshgrid
import matplotlib.pyplot as plt
from matplotlib import cm


def franke(x, y):
    nineX = 9 * x
    nineY = 9 * y
    first = 0.75 * exp(-(nineX - 2)**2 * 0.25 - (nineY - 2)**2 * 0.25)
    second = 0.75 * exp(-(nineX + 1)**2 / 49 - (nineY + 1)**2 * 0.1)
    third = 0.5 * exp(-(nineX - 7)**2 * 0.25 - (nineY - 3)**2 * 0.25)
    fourth = - 0.2 * exp(-(nineX - 4)**2 - (nineY - 7)**2)

    return first + second + third + fourth


def plot_franke(N):
    x = y = linspace(0, 1, N)
    X, Y = meshgrid(x, y)

    Z = franke(X, Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surface = ax.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()
