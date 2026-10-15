import numpy as np
import lr1_helpers as h
import sys


def main():
    data_filename = sys.argv[1]
    save_filename = sys.argv[2]

    data = np.loadtxt(data_filename, delimiter=",")
    train_data, test_data = h.train_test(data, test_size=0.3)
    X_train, y_train = train_data[:, 0], train_data[:, 1]

    custom_model = h.CustomLinearRegression()
    custom_model.fit(X_train, y_train)
    custom_model.save_weights(save_filename)


if __name__ == "__main__":
    main()
