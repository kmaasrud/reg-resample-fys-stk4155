import numpy as np
import matplotlib.pyplot as plt

from franke import franke
from utils import MSE, design_matrix, split_and_scale
from regression_methods import OLS


N = 20
x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
x = np.ravel(x); y = np.ravel(y)
Y = franke(x, y, noise_sigma=0.1, noise=True)

degs = list(range(1,11))
MSEs_train = []
MSEs_test = []
for i, deg in enumerate(degs):
    print(f"{i/len(degs) * 100}%")
    X = design_matrix(x, y, deg)
    X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=0.3)
    beta = OLS(X_train, y_train).beta
    y_fit = X_train @ beta
    y_predict= X_test @ beta
    MSEs_train.append(MSE(y_fit, y_train))
    MSEs_test.append(MSE(y_predict, y_test))

plt.plot(degs, MSEs_train, label="Train")
plt.plot(degs, MSEs_test, label="Test")
plt.legend()
plt.savefig("../doc/assets/complexity.png")
