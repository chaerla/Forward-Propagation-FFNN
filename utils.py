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

def relu_net_gradient(x):
    if x < 0:
        return 0
    else:
        return 1

def linear_net_gradient(x):
    return 1

def softmax_net_gradient(p, targetIdx):
    # target nya uda jadi index
    gradient = p.copy()
    gradient[targetIdx] = gradient[targetIdx] - 1 
    return gradient

def softmax_net_gradient2(p, target):
    # target nya belum ada index
    idx = target.index(max(target))
    gradient = p.copy()
    gradient[idx] = p[idx] - 1
    return gradient