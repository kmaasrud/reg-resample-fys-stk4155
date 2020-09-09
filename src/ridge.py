import numpy as np

np.random.seed(2020)        # set a seed to ensure reproducability


class OLS(Ridge):
    """Class for preforming Ordinary Least Square fits without using any statistical package libraries."""

    def __init__(self, X, y):
        super().__init__(X, y, 0)


class Ridge():
    """Class for preforming regression using the Ridge method"""

    def __init__(self, X, y, lmb):
        # Check for correct type and dimensions
        assert x.ndim == 2 and type(x) == np.ndarray, "Parameter X must be of type ndarray and have dimensionality 2."
        assert y.ndim == 1 and type(y) == np.ndarray, "Parameter y must be of type ndarray and have dimensionality 1."

        self.X = X
        self.p = X.shape[0]
        self.n = X.shape[1]
        self.y = y
        self.lmb = lmb

        self.fit_beta()
        self.y_predicted = self.X @ self.beta  # predicted values

    def fit_beta(self):
        """Calculates the Ridge coefficient vector"""
        I = np.eye(self.p)
        # matrix inversion to find beta
        self.beta = np.linalg.inv(self.X.T @ self.X + self.lmb*I) @ self.X.T @ self.y


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

