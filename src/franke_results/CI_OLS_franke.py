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
from utils import MSE, design_matrix, split_and_scale, mean_value, CI
from regression_methods import OLS


N = 100
deg = 12
lmb = 2
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
X = design_matrix(x, y, deg)
Y = franke(x, y, noise_sigma=0.1, noise=True)
X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=1)

ols = OLS(X_train, y_train)
ols_fit=ols.fit_beta()
ols_predict = ols.predict(X_test)


fig = plt.figure()
#ax = fig.gca(projection='3d')

"""Calculating and plotting the confidence intervals"""
beta_x = np.arange(len(ols_fit))
beta_lower, beta_upper =CI(X, ols_fit, 95, Y, ols_predict, N)

#Scaling the betas
beta_lower = (np.array(beta_lower))/1000000
beta_upper = (np.array(beta_upper))/1000000

#Plotting the confidence intervals for OLS terrain
plt.plot(beta_x, ols_fit/1000000, 'mo', label='betas')
plt.plot(beta_x, beta_upper, 'k+')
plt.plot(beta_x, beta_lower, 'k+')
plt.xlabel('Betas')
#plt.tight_layout()
#plt.savefig('Terrain_OLS_CI.png',bbox_inches='tight', dpi=300)
plt.show()
