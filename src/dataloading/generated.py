import numpy as np

def get_data(p, n):
    X = np.empty(shape=(0, p))
    y = np.empty(shape=(0))
    np.random.seed(1)
    for c in range(2):
        mean = np.random.rand(p)
        covariance = np.eye(p)*0.01
        X = np.concatenate([X, np.random.multivariate_normal(mean, covariance, size=n)])
        y = np.concatenate([y, np.asarray([c for _ in range(n)])])
    return X, y
