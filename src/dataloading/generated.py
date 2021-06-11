import numpy as np

def get_classification(p, n):
    X = np.empty(shape=(0, p))
    y = np.empty(shape=(0))
    np.random.seed(1)
    for c in range(2):
        mean = np.random.rand(p)
        covariance = np.eye(p)*0.01
        X = np.concatenate([X, np.random.multivariate_normal(mean, covariance, size=n)])
        y = np.concatenate([y, np.asarray([c for _ in range(n)])])
    return X, y


def get_regression(p, n):
    X = np.empty(shape=(0, p))
    y = np.empty(shape=(0, 1))
    np.random.seed(1)
    mean = np.random.rand(p)
    covariance = np.eye(p)*0.01
    X = np.concatenate([X, np.random.multivariate_normal(mean, covariance, size=n)])
    a = np.random.rand(p)
    eps = np.random.rand(n)
    y = X.dot(a) + eps
    return X, np.squeeze(y)
