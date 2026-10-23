import numpy as np
import matplotlib.pyplot as plt
import pickle


def linear(X, theta):
    return X @ theta


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(y_true, y_pred):
    n = y_true.shape[0]
    return -1 / n * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


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


class CustomClassifier:
    def __init__(self, method, random_state=42):
        self.theta = None
        self.rs = np.random.RandomState(random_state)
        self.method = method
        self.Xe = None
        self.eps = 1e-9

        if self.method == "linear":
            self.logit = self.logit_linear
            self.theta = self.rs.uniform(-1, 1, 3)
        elif self.method == "parabolex":
            self.logit = self.logit_parabolex
            self.theta = self.rs.uniform(-1, 1, 4)
        elif self.method == "paraboley":
            self.logit = self.logit_paraboley
            self.theta = self.rs.uniform(-1, 1, 4)
        elif self.method == "hyperbole":
            self.logit = self.logit_hyperbole
            self.theta = self.rs.uniform(-1, 1, 4)
        elif self.method == "ellipse":
            self.logit = self.logit_ellipse
            self.theta = self.rs.uniform(-1, 1, 5)

    def fit(self, X, y, alpha=0.01, epochs=10_000):
        X = add_ones(X)
        for _ in range(epochs):
            self.theta = self.gradientDescent(
                X, y, self.theta, alpha, self.logit)

    def predict_proba(self, X):
        X = add_ones(X)
        z = self.logit(X, self.theta)
        return np.clip(sigmoid(z), self.eps, 1 - self.eps)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def logit_linear(self, X, theta):
        self.Xe = X
        return X @ theta

    def logit_parabolex(self, X, theta):
        x0, x1, x2 = X.T
        Xe = np.column_stack([x0, x1, x2, x1**2])
        self.Xe = Xe
        return Xe @ theta

    def logit_paraboley(self, X, theta):
        x0, x1, x2 = X.T
        Xe = np.column_stack([x0, x1, x2, x2**2])
        self.Xe = Xe
        return Xe @ theta

    def logit_hyperbole(self, X, theta):
        x0, x1, x2 = X.T
        Xe = np.column_stack([x0, x1, x2, x1 * x2])
        self.Xe = Xe
        return Xe @ theta

    def logit_ellipse(self, X, theta):
        x0, x1, x2 = X.T
        Xe = np.column_stack([x0, x1, x2, x1**2, x2**2])
        self.Xe = Xe
        return Xe @ theta

    def gradientDescent(self, X, y, theta, alpha, logit):
        n = X.shape[0]
        z = logit(X, theta)
        error = sigmoid(z) - y
        grad = 1 / n * self.Xe.T @ error
        theta = theta - alpha * grad
        return theta


def logit_linear(x1, x2, theta):
    return theta[0] + x1 * theta[1] + x2 * theta[2]


def logit_parabolex(x1, x2, theta):
    return theta[0] + x1 * theta[1] + x2 * theta[2] + x1 ** 2 * theta[3]


def logit_paraboley(x1, x2, theta):
    return theta[0] + x1 * theta[1] + x2 * theta[2] + x2 ** 2 * theta[3]


def logit_hyperbole(x1, x2, theta):
    return theta[0] + x1 * theta[1] + x2 * theta[2] + x1 * x2 * theta[3]


def logit_ellipse(x1, x2, theta):
    return theta[0] + x1 * theta[1] + x2 * theta[2] + x1 ** 2 * theta[3] + x2 ** 2 * theta[4]


def plotData(X_neg, X_pos):
    plt.scatter(X_neg.T[0], X_neg.T[1], color="red",
                label="Класс 0. Испправные")
    plt.scatter(X_pos.T[0], X_pos.T[1], color="blue",
                label="Класс 1. Не исправные")


def plotLines(X_neg, X_pos, classifiers, names, mean, std, left=0, right=200, plot_test=False, X_test=None):
    left_norm = (left - mean) / std
    right_norm = (right - mean) / std
    ticks = np.linspace(left_norm, right_norm, 10)
    ticks_labels = [f"{tick * std + mean:.1f}" for tick in ticks]

    for method in names:
        clf_linear, clf_parabolex, clf_paraboley, clf_hyperbole, clf_ellipse = classifiers
        plotData(X_neg, X_pos)

        x1_values = np.linspace(left_norm, right_norm, 1000)
        x2_values = np.linspace(left_norm, right_norm, 1000)

        X1, X2 = np.meshgrid(x1_values, x2_values)
        if method == "linear":
            Z = logit_linear(X1, X2, clf_linear.theta)
        elif method == "parabolex":
            Z = logit_parabolex(X1, X2, clf_parabolex.theta)
        elif method == "paraboley":
            Z = logit_paraboley(X1, X2, clf_paraboley.theta)
        elif method == "hyperbole":
            Z = logit_hyperbole(X1, X2, clf_hyperbole.theta)
        elif method == "ellipse":
            Z = logit_ellipse(X1, X2, clf_ellipse.theta)

        H = sigmoid(Z)

        plt.contour(X1, X2, H, levels=[0.5], colors=['black'], linewidths=2)
        plt.contourf(X1, X2, H, levels=[0, 0.5, 1], colors=[
                     "red", "blue"], alpha=0.1)

        if plot_test and X_test is not None:
            plt.scatter(X_test[:, 0], X_test[:, 1], color="green",
                        s=150, marker="D", label="Тестовая точка")
            if method == "linear":
                print(
                    f"Предсказание линейной модели, класс: {clf_linear.predict(X_test)[0]}")
            if method == "parabolex":
                print(
                    f"Предсказание параболлической модели X, класс: {clf_parabolex.predict(X_test)[0]}")
            if method == "paraboley":
                print(
                    f"Предсказание параболической модели Y, класс: {clf_paraboley.predict(X_test)[0]}")
            if method == "hyperbole":
                print(
                    f"Предсказание гиперболической модели, класс: {clf_hyperbole.predict(X_test)[0]}")
            if method == "ellipse":
                print(
                    f"Предсказание эллиптической модели, класс: {clf_ellipse.predict(X_test)[0]}")

        plt.grid(alpha=0.3)
        plt.title(str.capitalize(method))
        plt.xlabel("Вибрация")
        plt.ylabel("Неравномерность вращения")
        plt.legend()

        plt.xticks(ticks, ticks_labels)
        plt.yticks(ticks, ticks_labels)

        plt.show()


def save_model(clf, filename="clf.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(clf, f)


def load_model(filename="clf.pkl"):
    with open(filename, "rb") as f:
        clf = pickle.load(f)
    return clf


def predict_user_data(clf_filename, user_filename, save_filename, mean, std):
    clf = load_model(clf_filename)
    user_data = np.loadtxt(user_filename, delimiter=",")
    user_data_norm, _, _ = normalize(user_data, mean, std)
    y_pred = clf.predict(user_data_norm).reshape(-1, 1)
    pred_data = np.hstack([user_data, y_pred])
    np.savetxt(save_filename, pred_data, fmt='%.1f', delimiter=',')
    print(f"Предсказания успешно записаны в '{save_filename}'")
