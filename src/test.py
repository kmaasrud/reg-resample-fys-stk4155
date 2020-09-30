import matplotlib.pyplot as plt
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt

from utils import make_design_matrix, MSE, alternate_design_matrix, design_matrix
from franke import franke
from regression_methods import OLS, Ridge
from plotting import plot3D
from assessment import CV, bootstrap
from resampling import split_and_scale


N = 1000
x = rand(N)
y = rand(N)
Y = franke(x, y, noise_sigma=1, noise=True)


lmbs = np.linspace(0, 10, 100)
degs = [1,2,3,4,5,6,7,8,9,10,11]
MSEs_train = []
MSEs_test = []
for i, deg in enumerate(degs):
    print(f"{i/len(degs) * 100}%")
    X = design_matrix(x, y, deg)
    X_train, X_test, y_train, y_test = split_and_scale(X, Y, test_size=0.3)
    beta = OLS(X_train, y_train).beta
    y_predict_train = X_train @ beta
    y_predict_test = X_test @ beta
    MSEs_train.append(MSE(y_predict_train, y_train))
    MSEs_test.append(MSE(y_predict_test, y_test))

plt.plot(degs, MSEs_train, label="Train")
plt.plot(degs, MSEs_test, label="Test")
plt.legend()
plt.savefig("testplot.png")

#ols = OLS(X, Y)
#ridge = Ridge(X, Y, 10)

#plot3D(x, y, Y, ridge.y_predicted)
#plt.show()
