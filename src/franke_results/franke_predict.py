import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from franke import franke
from utils import MSE, design_matrix, split_and_scale, mean_value
from regression_methods import OLS, Lasso, Ridge
from assessment import bootstrap

out_dir = "../../doc/assets/"

N = 100
deg = 12
lmb = 2
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
X = design_matrix(x, y, deg)
Y = franke(x, y, noise_sigma=0.1, noise=True)
X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=0.3)

ols = OLS(X_train, y_train)
lasso = Lasso(X_train, y_train, lmb)
ridge = Ridge(X_train, y_train, lmb)

ols_predict = ols.predict(X_test)
lasso_predict = lasso.predict(X_test)
ridge_predict = ridge.predict(X_test)

fig = plt.figure()
ax = fig.gca(projection='3d')

def plot(X, y, ax, title):
    ax.plot_trisurf(X[:,1], X[:,2], y, linewidth=0, antialiased=False)
    plt.savefig(os.path.join(out_dir, title))
    plt.cla()
    
plot(X_test, y_test, ax, "actual_franke_plot.png")
plot(X_test, ols_predict, ax, "ols_franke_plot.png")
plot(X_test, lasso_predict, ax, "lasso_franke_plot.png")
plot(X_test, ridge_predict, ax, "ridge_franke_plot.png")