import numpy as np
import matplotlib.pyplot as plt
from utils import findErrorConstant

FUNCTION = np.sin

def linearInterpolation(x, y, xNew):
    if len(x) != len(y):
        raise ValueError("X and Y must be the same size")

    indices = np.searchsorted(x, xNew)
    indices = np.clip(indices, 1, len(x) - 1)
    yNew = y[indices - 1] + (y[indices] - y[indices - 1]) * \
        (xNew - x[indices - 1]) / (x[indices] - x[indices - 1])
    return yNew



if __name__ == "__main__":
    a = 0
    b = 1
    h = 0.1

    xNew = np.array([0.37454012, 0.59865848, 0.73199394, 0.95071431])

    yNew = FUNCTION(xNew)
    print(xNew, yNew)

    np.set_printoptions(precision=20, suppress=True)
    while h >= 1e-5:
        xBase = np.arange(a, b+h, h)
        yBase = FUNCTION(xBase)
        y = linearInterpolation(xBase, yBase, xNew)
        x = linearInterpolation(yBase, xBase, yNew)
        c = findErrorConstant(yNew, y, h)
        print(c)
        h /= 10

    print(y)
    print(yNew)
    print(x)

    plt.plot(xBase, yBase, label='y = fn(x)')
    plt.scatter(xNew, y, marker='x', color='b')
    plt.title('График функции')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(yBase, xBase, label='x = gn(y)', color='g')
    plt.scatter(yNew, x, marker='x', color='b')
    plt.title('График обратной функции')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.show()
