import numpy as np
import matplotlib.pyplot as plt

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from franke import franke
from utils import MSE, design_matrix, split_and_scale, mean_value
from regression_methods import OLS
from assessment import bootstrap

out_dir = "../../doc/assets/"


N = 18
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
Y = franke(x, y, noise_sigma=0.1, noise=True)

degs = list(range(1,10))
Xs = [design_matrix(x, y, deg) for deg in degs]
MSEs_train = []
MSEs_test = []
for i, X in enumerate(Xs):
    print(f"Error/complexity-plot: {i/len(Xs) * 100}%")
    X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=0.3)

    ols = OLS(X_train, y_train)
    y_fit = ols.predict(X_train)
    y_predict= ols.predict(X_test)

    MSEs_train.append(MSE(y_fit, y_train))
    MSEs_test.append(MSE(y_predict, y_test))


plt.plot(degs, MSEs_train, label="Model fit to train data")
plt.plot(degs, MSEs_test, label="Model prediction of test data")
plt.legend()
plt.title("Mean squared error of the predicted data, given model complexity")
plt.xlabel("Complexity (Degree of polynomial)"); plt.ylabel("Error (MSE)")
plt.savefig(os.path.join(out_dir, "complexity.png"))

plt.clf()

N = 40
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
Y = franke(x, y, noise=False)
degs = list(range(1,15))
Xs = [design_matrix(x, y, deg) for deg in degs]
variances = []
biases = []
for i, X in enumerate(Xs):
    print(f"Bias/variance-plot: {i/len(Xs) * 100}%")

    bootstrap_predicts, y_test = bootstrap(X, Y, 100, OLS)
    biases.append(mean_value((y_test - np.mean(bootstrap_predicts))**2))
    variances.append(mean_value(np.var(bootstrap_predicts, axis=1)))

plt.plot(degs, variances, label="Variance")
plt.plot(degs, biases, label="Bias")
plt.legend()
plt.savefig(os.path.join(out_dir, "var.png"))
