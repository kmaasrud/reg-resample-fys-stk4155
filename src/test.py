import matplotlib.pyplot as plt
from numpy.random import rand

from utils import make_data_matrices
from franke import franke
from regression_methods import OLS, Ridge
from plotting import plot3D


N = 1000
X, y = make_data_matrices(franke, rand(N), rand(N))

ols = OLS(X, y)
ridge = Ridge(X, y, 4)

plot3D(X.T[1], X.T[2], y, ridge.y_predicted)
plt.show()
