import numpy as np


def f(x):
    return np.sin(4*x) + 2*np.cos(x/8)


def getSampleData():
    x = np.linspace(-2, 2, 100)
    y = f(x)
    return x, y


def relu(x):
    if x < 0.0:
        return np.array([0.0])
    return x


def forwardPropogation(nn, x):
    layers = nn.shape[0]

    op = x * nn[0, 0] + nn[0, 1]
    if (op < 0):
        op = 0

    for i in range(1, layers):
        op = op * nn[i, 0] + nn[i, 1]
        if (op < 0):
            op = 0

    return op


def plotData(x, y):
    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.grid()
    plt.show()
