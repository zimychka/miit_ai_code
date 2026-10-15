import numpy as np
import lr2_helpers as h
import sys


def main():
    data_filename = sys.argv[1]
    save_filename = sys.argv[2]

    data = np.loadtxt("ex1data2.txt", delimiter=",")
    train_data, test_data = h.train_test(data, test_size=0.3)
    X_train, y_train = train_data[:, :(-1)], train_data[:, -1]
    X_test, y_test = test_data[:, :(-1)], test_data[:, -1]
    X_train_norm, mean, std = h.featureNormalize(X_train)
    X_test_norm, _, _ = h.featureNormalize(X_test, mean, std)

    model = h.CustomLinearRegression()
    model.fit(X_train_norm, y_train, compute_cost=False,
              method="grad", epochs=50_000)
    model.save_weights(save_filename)


if __name__ == "__main__":
    main()
