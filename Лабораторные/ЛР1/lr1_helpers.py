import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def graph_setup():
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 100


def warmUpExercise_manual(n):
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    return np.array(matrix)


def warmUpExercise_vector(n):
    matrix = np.eye(n)
    return matrix


def train_test(X, test_size=0.2, random_state=42):
    rs = np.random.RandomState(random_state)
    n = X.shape[0]
    indices = rs.permutation(n)
    n_test = int(n * test_size)
    shuffled_X = X[indices, :]

    test_X = shuffled_X[:n_test, :]
    train_X = shuffled_X[n_test:, :]
    return train_X, test_X


def add_ones(X):
    n = X.shape[0]
    X_extended = np.hstack([np.ones((n, 1)), X.reshape(n, -1)])
    return X_extended


def computeCost_vector(X, y, theta):
    error = X @ theta - y
    n = X.shape[0]
    return 1 / (2 * n) * error.T @ error


def gradientDescent_vector(X, y, theta, alpha):
    n = X.shape[0]
    error = X @ theta - y
    grad = 1 / n * error @ X
    theta = theta - alpha * grad
    return theta


def computeCost_loop(X, y, theta):
    n = X.shape[0]
    h = np.sum(X * theta, axis=1)
    errors = h - y
    cost = 1 / (2 * n) * np.sum(errors ** 2)
    return cost


def gradientDescent_loop(X, y, theta, alpha):
    n = X.shape[0]
    h = np.sum(X * theta, axis=1)
    error = h - y
    grad = 1 / n * np.sum(error.reshape(-1, 1) * X, axis=0)
    theta = theta - alpha * grad
    return theta


def computeCost_manual(X, y, theta):
    n = X.shape[0]
    summ = 0
    squared_error = 0
    for i in range(n):
        h = 0
        for j in range(len(theta)):
            h += X[i, j] * theta[j]
        squared_error += (h - y[i]) ** 2
    cost = 1 / (2 * n) * squared_error
    return cost


def gradientDescent_manual(X, y, theta, alpha):
    n = X.shape[0]
    new = np.zeros_like(theta)

    for k in range(len(theta)):
        grad = 0
        for i in range(n):
            h = 0
            for j in range(len(theta)):
                h += X[i, j] * theta[j]

            error = h - y[i]
            grad += error * X[i, k]
        new[k] = theta[k] - alpha / n * grad
    return new


def plotData(X, y, thetas, labels=None, legend=True, title="Linear Regression", figsize=(10, 8)):
    if labels is None:
        labels = [f"LinearRegression_{idx + 1}" for idx in range(len(thetas))]
    colors = ["red", "cyan", "green"]
    graph_setup()
    plt.figure(figsize=figsize)

    plt.scatter(X[:, 1], y)
    for idx, theta in enumerate(thetas):
        plt.plot(X[:, 1], X @ theta, label=labels[idx],
                 color=colors[idx % len(colors)])

    plt.title(title)
    if legend:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plotCost(iterations, costs, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.plot(iterations, costs)
    plt.grid(alpha=0.3)
    plt.title(f"Сходимость", fontsize=18)
    plt.xlabel(f"Итерация")
    plt.ylabel(f"Функция стоимости")
    plt.show()


def r2_score(y_true, y_pred):
    ss_residual = np.sum((y_true - y_pred)**2)
    mean = np.mean(y_true)
    ss_total = np.sum((y_true - mean)**2)
    return 1 - ss_residual / ss_total


class CustomLinearRegression:
    def __init__(self):
        self.theta = None
        self.cost = []

    def fit(self, X, y, init_theta=None, method="vector", alpha=0.001, epochs=50_000, compute_cost=False):
        X = add_ones(X)

        self.cost = []
        n_features = X.shape[1]
        if init_theta is None:
            theta = np.ones(n_features)
        else:
            theta = init_theta.copy()

        if method == "vector":
            for i in range(epochs):
                theta = gradientDescent_vector(X, y, theta, alpha)
                if compute_cost:
                    self.cost.append((i, computeCost_vector(X, y, theta)))
        elif method == "loop":
            for i in range(epochs):
                theta = gradientDescent_loop(X, y, theta, alpha)
                if compute_cost:
                    self.cost.append((i, computeCost_loop(X, y, theta)))
        elif method == "manual":
            for i in range(epochs):
                theta = gradientDescent_manual(X, y, theta, alpha)
                if compute_cost:
                    self.cost.append((i, computeCost_manual(X, y, theta)))

        self.theta = theta

    def save_weights(self, filename="theta.txt"):
        with open(filename, "w") as file:
            file.write(",".join(map(str, self.theta)))
        print(f"Веса успешно сохранены в '{filename}'!")

    def load_weights(self, filename="theta.txt"):
        self.theta = np.loadtxt(filename, delimiter=",")
        print(f"Веса успешно загружены!")

    def predict(self, X):
        X = add_ones(X)
        return X @ self.theta
