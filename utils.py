import numpy as np
import random


def linear(x):
    return x


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_random_pale_color():
    """
    Generates a random pale color in hexadecimal format.

    :return: A random pale color.
    """
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    return '#%02x%02x%02x' % (r, g, b)


def sigmoid_net_gradient(x):
    return x * (1-x)