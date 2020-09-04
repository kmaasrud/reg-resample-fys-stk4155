import matplotlib.pyplot as plt
from numpy.random import rand

from utils import make_data_matrices
from franke import franke
from ols import OLS
from plotting import plot3D


N = 1000
X, Y = make_data_matrices(franke, rand(N), rand(N))

ols = OLS(X, Y)
beta = ols.fit_beta()
p, Y_hat = ols.predict()

plot3D(X.T[1], X.T[2], Y, Y_hat)
plt.show()
