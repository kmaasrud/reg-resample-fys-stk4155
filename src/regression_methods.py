import numpy as np

np.random.seed(2020)        # set a seed to ensure reproducability


class Ridge():
    """Class for performing regression using the Ridge method"""

    def __init__(self, X, y, lmb):
        # Check for correct type and dimensions
        assert X.ndim == 2 and type(X) == np.ndarray, "Parameter X must be of type ndarray and have dimensionality 2."
        assert y.ndim == 1 and type(y) == np.ndarray, "Parameter y must be of type ndarray and have dimensionality 1."

        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.y = y
        self.lmb = lmb

        self.beta = self.fit_beta()

    def fit_beta(self):
        """Calculates the Ridge coefficient vector"""
        I = np.eye(self.p)
        # Matrix inversion to find beta
        return np.linalg.inv(self.X.T @ self.X + self.lmb*I) @ self.X.T @ self.y

    def predict(self, X):
        return X @ self.beta

class OLS(Ridge):
    """Class for performing Ordinary Least Square fits without using any statistical package libraries."""

    def __init__(self, X, y):
        super().__init__(X, y, 0)

class Lasso(Ridge):
        """Class for performing Lasso regression using sklearn"""

        def __init__(self, X,y):
            super().__init__(X, y, 0)

        def predict(self, X_input, value):
            sklearn_lasso = linear_model.Lasso(alpha= value)
            return sklearn_lasso.fit(X_input)


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
