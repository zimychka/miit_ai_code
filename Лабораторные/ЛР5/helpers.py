import numpy as np
import matplotlib.pyplot as plt


def add_ones(X):
    n = X.shape[0]
    X_extended = np.hstack([np.ones((n, 1)), X.reshape(n, -1)])
    return X_extended


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def normalize(X, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(X)
        std = np.std(X)
    return (X - mean) / std, mean, std


def extend_X(X):
    X_ones = add_ones(X)
    ones, x, y = X_ones.T
    Xe = np.column_stack([ones, x, y, x * y, x**2, y**2,
                          x**2 * y, x * y**2, x**3, y**3,
                          x * y**3, x**3 * y, x**2 * y**2, x**4, y**4,
                          x * y**4, x**4 * y, x**2 * y**3, x**3 * y**2, x**5, y**5,
                          x * y**5, x**5 * y, x**2 * y**4, x**4 * y**2, x**3 * y**3, x**6, y**6])
    return Xe


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(X, y, theta):
    n = y.shape[0]
    return -1 / n * np.sum(y * np.log(sigmoid(X @ theta)) + (1 - y) * np.log(1 - sigmoid(X @ theta)))


def cost_lasso(X, y, theta, lam):
    cost_1 = cost(X, y, theta)
    mask = np.ones_like(theta)
    mask[0] = 0
    cost_2 = lam * np.sum(np.abs(theta * mask))
    return cost_1 + cost_2


def cost_ridge(X, y, theta, lam):
    cost_1 = cost(X, y, theta)
    mask = np.ones_like(theta)
    mask[0] = 0
    cost_2 = lam * np.sum(theta ** 2 * mask)
    return cost_1 + cost_2


def cost_elastic(X, y, theta, lam):
    cost_1 = cost(X, y, theta)
    mask = np.ones_like(theta)
    mask[0] = 0
    cost_2 = lam * np.sum(np.abs(theta * mask))
    cost_3 = (1 - lam) * np.sum(theta ** 2 * mask)
    return cost_1 + cost_2 + cost_3


class CustomClassifier:
    def __init__(self):
        self.theta = None
        self.eps = 1e-9

    def gd_lasso(self, X, y, theta, alpha, lam):
        n = X.shape[0]
        grad = 1 / n * X.T @ (sigmoid(X @ theta) - y)
        mask = np.ones_like(theta)
        mask[0] = 0
        return theta - alpha * grad - alpha * lam * mask * np.sign(theta)

    def gd_ridge(self, X, y, theta, alpha, lam):
        n = X.shape[0]
        grad = 1 / n * X.T @ (sigmoid(X @ theta) - y)
        mask = np.ones_like(theta)
        mask[0] = 0
        return theta - alpha * grad - alpha * lam * mask * 2 * theta

    def gd_elastic(self, X, y, theta, alpha, lam1, lam2):
        n = X.shape[0]
        grad = 1 / n * X.T @ (sigmoid(X @ theta) - y)
        mask = np.ones_like(theta)
        mask[0] = 0
        return theta - alpha * grad - alpha * lam1 * mask * np.sign(theta) - alpha * lam2 * mask * 2 * theta

    def fit(self, X, y, alpha=0.01, epochs=10_000, lam=1.0, lam2=1.0, method="lasso"):
        Xe = extend_X(X)
        lim = 10
        theta = np.random.uniform(-lim, lim, size=Xe.shape[1])
        for _ in range(epochs):
            if method == "lasso":
                theta = self.gd_lasso(Xe, y, theta, alpha, lam)
            elif method == "ridge":
                theta = self.gd_ridge(Xe, y, theta, alpha, lam)
            elif method == "elastic":
                theta = self.gd_elastic(Xe, y, theta, alpha, lam, lam2)
        self.theta = theta

    def predict_proba(self, X):
        Xe = extend_X(X)
        return np.clip(sigmoid(Xe @ self.theta), self.eps, 1 - self.eps)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


def plotData(X_neg, X_pos):
    plt.scatter(X_neg.T[0], X_neg.T[1], color="red",
                label="Класс 0. Испправные")
    plt.scatter(X_pos.T[0], X_pos.T[1], color="blue",
                label="Класс 1. Не исправные")


def logit(x, y, theta):
    return (
        theta[0] +
        theta[1] * x + theta[2] * y +
        theta[3] * x * y + theta[4] * x**2 + theta[5] * y**2 +
        theta[6] * x**2 * y + theta[7] * x * y**2 + theta[8] * x**3 + theta[9] * y**3 +
        theta[10] * x * y**3 + theta[11] * x**3 * y + theta[12] * x**2 * y**2 + theta[13] * x**4 + theta[14] * y**4 +
        theta[15] * x * y**4 + theta[16] * x**4 * y + theta[17] * x**2 * y**3 + theta[18] * x**3 * y**2 +
        theta[19] * x**5 + theta[20] * y**5 +
        theta[21] * x * y**5 + theta[22] * x**5 * y + theta[23] * x**2 * y**4 + theta[24] * x**4 * y**2 +
        theta[25] * x**3 * y**3 + theta[26] * x**6 + theta[27] * y**6
    )


def plotLines(X_neg, X_pos, classifiers, left=-3, right=3):
    titles = ["Переобучение", "Lasso", "Ridge", "Elastic",
              "Lasso недообучение", "Ridge недообучение", "Elastic недообучение"]

    x1_values = np.linspace(left, right, 1000)
    x2_values = np.linspace(left, right, 1000)

    X1, X2 = np.meshgrid(x1_values, x2_values)
    for clf, title in zip(classifiers, titles):
        plotData(X_neg, X_pos)

        Z = logit(X1, X2, clf.theta)
        H = sigmoid(Z)
        plt.contour(X1, X2, H, levels=[0.5], colors=['black'], linewidths=2)
        plt.contourf(X1, X2, H, levels=[0, 0.5, 1], colors=[
                     "red", "blue"], alpha=0.1)

        plt.grid(alpha=0.3)
        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()

        plt.show()
