import matplotlib.pyplot as plt
from numpy.random import rand
import numpy as np

from utils import make_design_matrix
from franke import franke
from regression_methods import OLS, Ridge
from plotting import plot3D
from assessment import CV, bootstrap


N = 1000
x = rand(N)
y = rand(N)
Y = franke(x, y)
X = make_design_matrix(x, y, poly_deg=5)
boot = bootstrap(X, Y, 1000, OLS)
print(boot)

#ols = OLS(X, Y)
#ridge = Ridge(X, Y, 10)

#plot3D(x, y, Y, ridge.y_predicted)
#plt.show()
