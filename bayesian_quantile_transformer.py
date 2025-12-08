import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin


class BayesianQuantileTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.quantiles_ = []
        self.references_ = []
        for i in range(X.shape[1]):
            col = X[:, i]
            sorted_col = np.sort(col)
            bayesian_quantiles = np.arange(1, len(col) + 1) / (len(col) + 1)
            self.quantiles_.append(sorted_col)
            self.references_.append(norm.ppf(bayesian_quantiles))
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_transformed = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_transformed[:, i] = np.interp(X[:, i], self.quantiles_[i], self.references_[i])
        return X_transformed

    def inverse_transform(self, X):
        X = np.asarray(X)
        X_inv = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_inv[:, i] = np.interp(X[:, i], self.references_[i], self.quantiles_[i])
        return X_inv


if __name__ == "__main__":
    X_train = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]).astype(float)
    transformer = BayesianQuantileTransformer()
    transformer.fit(X_train)

    X_transformed = transformer.transform(X_train)
    X_reconstructed = transformer.inverse_transform(X_transformed)
    assert np.allclose(X_train, X_reconstructed)

    X_below = np.array([[0.0, 5.0], [-1.0, 0.0]])
    X_below_transformed = transformer.transform(X_below)
    assert np.allclose(X_below_transformed, X_transformed[0])

    X_above = np.array([[6.0, 60.0], [10.0, 100.0]])
    X_above_transformed = transformer.transform(X_above)
    assert np.allclose(X_above_transformed, X_transformed[-1])

    print("All tests passed ✓")
