#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

np.random.seed(2020)        # set a seed to ensure reproducability


class OLS():
    """
    Class for preforming Ordinary Least Square fits without using any statistical package libraries.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit_beta(self):

        # matrix inversion to find beta
        self.beta = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y

        return self.beta

    def predict(self):

        p = len(self.beta)  # number of features p/ complexity of model
        self.y_hat = self.x @ self.beta  # predicted values

        return p, self.y_hat


# simple test for linear case
if __name__ == '__main__':
    x = np.arange(50)[:, np.newaxis]
    y = np.array([i + np.random.rand() for i in range(50)])[:, np.newaxis]
    ols_test = OLS(x, y)
    beta = ols_test.fit_beta()
    p, y_hat = ols_test.predict()
