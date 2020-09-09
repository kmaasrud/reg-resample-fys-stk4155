#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

np.random.seed(2020)        # set a seed to ensure reproducability


class Ridge():
    """
    Class for preforming regression using the Ridge metod
    """

    def __init__(self, x, y, lmb):
        self.x = x
        self.y = y
        self.lmb = lmb

    def fit_beta(self):
	
        I = np.eye(len(x[1]))
        # matrix inversion to find beta
        self.beta = np.linalg.inv(self.x.T @ self.x+self.lmb*I) @ self.x.T @ self.y

        return self.beta

    def predict(self):

        p = len(self.beta)  # number of features p/ complexity of model
        self.y_hat = self.x @ self.beta  # predicted values

        return p, self.y_hat


# simple test for linear case
if __name__ == '__main__':
    x = np.arange(50)[:, np.newaxis]
    y = np.array([i + np.random.rand() for i in range(50)])[:, np.newaxis]
    nlambdas = 20
    lambdas = np.logspace(-4, 1, nlambdas)
    for i in range(nlambdas):
        lmb = lambdas[i] 
    Ridge_test = Ridge(x, y,lmb)
    beta = Ridge_test.fit_beta()
    p, y_hat = Ridge_test.predict()

