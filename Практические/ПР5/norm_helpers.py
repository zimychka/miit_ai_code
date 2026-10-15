import numpy as np


def norm_manual(X):
    n_samples, n_features = X.shape

    mu = [0.0 for _ in range(n_features)]
    sigma = [0.0 for _ in range(n_features)]

    for j in range(n_features):
        summ = 0

        for i in range(n_samples):
            summ += X[i][j]
        mu_val = summ / n_samples
        mu[j] = mu_val

    for j in range(n_features):
        summ = 0

        for i in range(n_samples):
            summ += (X[i][j] - mu[j])**2
        sigma_val = np.sqrt(summ / n_samples)
        sigma[j] = sigma_val

    X_normalized = np.zeros_like(X, dtype='float')
    for i in range(n_samples):
        for j in range(n_features):
            X_normalized[i, j] = (X[i, j] - mu[j]) / sigma[j]

    return X_normalized


def norm_loop(X):
    n_samples, n_features = X.shape
    mu = [0.0 for _ in range(n_features)]
    sigma = [0.0 for _ in range(n_features)]

    for j in range(n_features):
        col = X[:, j].copy()
        mean = np.mean(col)
        std = np.std(col)
        mu[j] = mean
        sigma[j] = std

    X_normalized = np.zeros_like(X, dtype='float')
    for i in range(n_samples):
        for j in range(n_features):
            X_normalized[i, j] = (X[i, j] - mu[j]) / sigma[j]

    return X_normalized


def norm_vector(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized


def norm_minmax(X):
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    return (X - min_values) / (max_values - min_values)


def norm_max(X):
    max_values = np.max(X, axis=0)
    return X / max_values
