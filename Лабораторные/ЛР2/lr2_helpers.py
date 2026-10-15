import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def graph_setup():
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 100


def featureNormalize(X, mean=None, std=None):
    if mean is None or str is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def computeCost(X, y, theta):
    error = X @ theta - y
    n = X.shape[0]
    return 1 / (2 * n) * error.T @ error


def gradientDescent(X, y, theta, alpha):
    n = X.shape[0]
    error = X @ theta - y
    grad = 1 / n * error @ X
    theta = theta - alpha * grad
    return theta


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


def normalEqn(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def r2_score(y_true, y_pred):
    ss_residual = np.sum((y_true - y_pred)**2)
    mean = np.mean(y_true)
    ss_total = np.sum((y_true - mean)**2)
    return 1 - ss_residual / ss_total


def plotNorm(x, y, title="Ненормализованные данные", figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.scatter(x, y)
    plt.xlabel("Скорость оборота трактора")
    plt.ylabel("Число передач")
    plt.title(title, fontsize=18)
    plt.show()


def plotCost(iterations, costs, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.plot(iterations, costs)
    plt.grid(alpha=0.3)
    plt.title(f"Сходимость", fontsize=18)
    plt.xlabel(f"Итерация")
    plt.ylabel(f"Функция стоимости")
    plt.show()


def plot3d(x1, x2, y, theta, figsize=(10, 8)):
    x1_range = np.linspace(x1.min(), x1.max(), 50)
    x2_range = np.linspace(x2.min(), x2.max(), 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    w0, w1, w2 = theta
    Y_grid = w0 + w1 * X1_grid + w2 * X2_grid

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=110)

    ax.scatter(x1, x2, y, c="tab:blue", s=20, alpha=0.7,
               label="данные", depthshade=True)

    ax.plot_surface(X1_grid, X2_grid, Y_grid,
                    color="tab:orange", alpha=0.4, linewidth=0)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title("Linear Regression")
    plt.tight_layout()
    plt.show()


class CustomLinearRegression:
    def __init__(self):
        self.theta = None
        self.cost = []

    def fit(self, X, y, init_theta=None, alpha=0.001, epochs=50_000, compute_cost=False, method="grad"):
        X = add_ones(X)
        self.cost = []

        if method == "analytically":
            self.theta = normalEqn(X, y)
            return

        n_features = X.shape[1]
        if init_theta is None:
            theta = np.ones(n_features)
        else:
            theta = init_theta.copy()

        for i in range(epochs):
            theta = gradientDescent(X, y, theta, alpha)
            if compute_cost:
                self.cost.append((i, computeCost(X, y, theta)))
        self.theta = theta
        return

    def save_weights(self, filename="theta_2.txt"):
        with open(filename, "w") as file:
            file.write(",".join(map(str, self.theta)))
        print(f"Веса успешно сохранены в '{filename}'!")

    def load_weights(self, filename="theta_2.txt"):
        self.theta = np.loadtxt(filename, delimiter=",")
        print(f"Веса успешно загружены!")

    def predict(self, X):
        X = add_ones(X)
        return X @ self.theta
