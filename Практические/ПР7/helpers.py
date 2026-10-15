import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sinh(x):
    return (np.exp(x) - np.exp(-x)) / 2


def cosh(x):
    return (np.exp(x) + np.exp(-x)) / 2


def tanh(x):
    return sinh(x) / cosh(x)


def ctanh(x):
    return cosh(x) / sinh(x)


def tanh_derivative(x):
    return 1 - tanh(x)**2
