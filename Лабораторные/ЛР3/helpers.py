import numpy as np


def to_new_coordinate_sytem(coords, init_coord):
    new_coords = coords - init_coord
    new_coords[:, -1] *= (-1)
    return new_coords


def heaviside(z):
    if z >= 0:
        return 1
    return 0


def add_ones(coords):
    n = len(coords)
    return np.hstack([np.ones((n, 1)), coords.reshape(n, -1)])


def linear(X, weights):
    return X @ weights


def find_solution(y_true, coords, random_state=42):
    X = add_ones(coords)

    rs = np.random.RandomState(random_state)
    weights = rs.uniform(-1, 1, size=X.shape[1])

    y_preds = []
    all_weights = []
    zs = []
    conclusions = []
    epoch = 0

    is_all_correct = False
    while (not is_all_correct):
        is_all_correct = True
        epoch += 1
        for idx, point in enumerate(X):
            old_weights = weights.copy()
            z = linear(point, weights)
            y_pred = heaviside(z)
            conclusion = (y_true[idx] == y_pred)
            if not conclusion and y_pred == 0:
                weights += point
                is_all_correct = False
            elif not conclusion and y_pred == 1:
                weights -= point
                is_all_correct = False
            all_weights.append(old_weights)
            zs.append(z)
            y_preds.append(y_pred)
            conclusions.append(conclusion)
    return y_true, X, all_weights, zs, y_preds, conclusions


def print_results(y_true, X, all_weights, zs, y_preds, conclusions):
    n = len(y_true)

    cols = [
        ("Класс",        6),
        ("x0",           6),
        ("x1",           6),
        ("x2",           6),
        ("w0",          10),
        ("w1",          10),
        ("w2",          10),
        ("z",           10),
        ("Предсказание", 13),
        ("Верно?",       7),
    ]
    header = " | ".join(f"{name:^{w}}" for name, w in cols)
    sep = "-" * len(header)
    double = "=" * len(header)

    print(header)
    print(sep)

    for idx in range(len(conclusions)):
        if idx % n == 0 and idx != 0:
            print(double)

        x0, x1, x2 = X[idx % n]
        w0, w1, w2 = all_weights[idx]
        conclusion = True if conclusions[idx] else False

        row = [
            f"{y_true[idx % n]:>6}",
            f"{x0:>6.2f}",
            f"{x1:>6.2f}",
            f"{x2:>6.0f}",
            f"{w0:>10.4f}",
            f"{w1:>10.4f}",
            f"{w2:>10.4f}",
            f"{zs[idx]:>10.4f}",
            f"{y_preds[idx]:>13}",
            f"{conclusion}",
        ]
        print(" | ".join(row))
