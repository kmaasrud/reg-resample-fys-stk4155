import numpy as np
import matplotlib.pyplot as plt

from franke import franke
from utils import MSE, design_matrix, split_and_scale, mean_value
from regression_methods import OLS
from assessment import bootstrap


N = 20
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
Y = franke(x, y, noise_sigma=0.1, noise=True)

degs = list(range(1,11))
MSEs_train = []
MSEs_test = []
variances = []
biases = []
for i, deg in enumerate(degs):
    print(f"{i/len(degs) * 100}%")
    X = design_matrix(x, y, deg)
    X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=0.3)

    beta = OLS(X_train, y_train).beta
    y_fit = X_train @ beta
    y_predict= X_test @ beta

    MSEs_train.append(MSE(y_fit, y_train))
    MSEs_test.append(MSE(y_predict, y_test))

    bootstrap_predicts, _ = bootstrap(X, Y, 100, OLS)
    biases.append(mean_value((y_test - np.mean(bootstrap_predicts, axis=0))**2))
    variances.append(mean_value(np.var(bootstrap_predicts, axis=1)))

plt.plot(degs, MSEs_train, label="Model fit to train data")
plt.plot(degs, MSEs_test, label="Model prediction of test data")
plt.legend()
plt.title("Mean squared error of the predicted data, given model complexity")
plt.xlabel("Complexity"); plt.ylabel("Error")
plt.savefig("../doc/assets/complexity.png")

plt.clf()
plt.plot(degs, variances, label="Variance")
plt.plot(degs, biases, label="Bias")
plt.legend()
plt.savefig("var.png")
