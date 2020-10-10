import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from assessment import bootstrap, CV
from regression_methods import OLS
from utils import MSE, design_matrix, split_and_scale, mean_value
from franke import franke

out_dir = "../../doc/assets/"

N = 18
k = 20
B = 100
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
Y = franke(x, y, noise_sigma=0.1, noise=True)

degs = list(range(1,10))
Xs = [design_matrix(x, y, deg) for deg in degs]
boot_MSE = []; cv_MSE = []
for X in Xs:
    bootstrap_predicts, y_test = bootstrap(X, Y, B, OLS)
    boot_MSE.append(np.array([MSE(predict, y_test) for predict in bootstrap_predicts]).mean())
    cv_MSE.append(CV(X, Y, k, OLS))

plt.plot(degs, boot_MSE, label=f"The Bootstrap ({B} samples)")
plt.plot(degs, cv_MSE, label=f"{k}-fold cross-validation")
plt.xlabel("Complexity (Degree of polynomial)"); plt.ylabel("MSE")
plt.legend()
plt.savefig(os.path.join(out_dir, "cv_boot_mse.png"))