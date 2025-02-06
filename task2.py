import numpy as np
import matplotlib.pyplot as plt
from utils import findErrorConstant, createInterpolationGrid, selectRandomPoints
#! кубический сплайн не корректно, с ошибками
FUNCTION = np.sin

def cubicSpline(x, y, xNew=None):
    """
    Собственная реализация кубического сплайна.
    Возвращает коэффициенты и, если указаны xNew, вычисляет значения сплайна в этих точках.
    """
    n = len(x)
    h = np.diff(x)

    # Создаем матрицу A и вектор B
    A = np.zeros((n, n))
    B = np.zeros(n)

    # Заполняем матрицу A и вектор B
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Граничные условия (естественный сплайн)
    A[0, 0] = 1
    A[-1, -1] = 1

    # Решаем систему для нахождения коэффициентов c
    c = np.linalg.solve(A, B)

    # Вычисляем коэффициенты a, b, d
    a = y[:-1]
    b = np.diff(y) / h - h * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    # Если xNew не указаны, возвращаем только коэффициенты
    if xNew is None:
        return a, b, c[:-1], d

    # Иначе вычисляем значения сплайна в точках xNew
    idx = np.searchsorted(x, xNew) - 1
    idx = np.clip(idx, 0, len(x) - 2)
    dx = xNew - x[idx]
    yNew = a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3
    return yNew



def tangent(x, y, yNew, max_iter=100, tol=1e-6):
    x_found = np.zeros_like(yNew)

    n = len(x)
    h = np.diff(x)
    a = y.copy()
    b = (a[1:n] - a[0:n-1]) / h
    c = np.zeros(n)

    for i in range(1, n - 1):
        c[i] = (b[i] - b[i - 1]) / h[i - 1]
    for j, y_target in enumerate(yNew):

        idx = np.searchsorted(a, y_target) - 1
        idx = np.clip(idx, 0, n - 2)

        x_current = x[idx] + (y_target - a[idx]) / b[idx]

        for _ in range(max_iter):
            f_value = cubicSpline(x, y, np.array([x_current]))[0] - y_target
            f_derivative = b[idx] + c[idx] * (x_current - x[idx])

            x_new = x_current - f_value / f_derivative

            if abs(x_new - x_current) < tol:
                break

            x_current = x_new

        x_found[j] = x_current

    return x_found


def main():
    a = 0
    b = 1
    h = 0.1

    xNewTrue = np.array([0.37454012, 0.59865848, 0.73199394, 0.95071431])
    yNewTrue = FUNCTION(xNewTrue)

    # ? idk what this shit is about
    np.set_printoptions(precision=20, suppress=True)

    
    errorConstants = []
    errorConstantsTangent = []

    hSet = np.linspace(0.001, 0.01, 11)[::-1]
    for h in hSet:
        xBase = np.arange(a, b + h, h)
        yBase = FUNCTION(xBase)
        yNewCalc = cubicSpline(xBase, yBase, xNewTrue)
        xNewCalc = tangent(xBase, yBase, yNewTrue)
        
        # Для кубического сплайна ошибка ~ O(h^4)
        errorConstantsTangent.append(findErrorConstant(xNewTrue, xNewCalc, h, 4))
        
        errorConstant = findErrorConstant(yNewTrue, yNewCalc, h, 4)
        errorConstants.append(errorConstant)
        
        print("step: {:10f} | error constant: {:20f}".format(h, errorConstant))

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hSet, errorConstants, marker='o')
    plt.title('Зависимость константы ошибки от шага сетки')
    plt.xlabel('Шаг сетки (h)')
    plt.ylabel('Константа ошибки')
    plt.grid()
    
    #plt.subplot(1, 2, 2)
    #plt.plot(hSet, errorConstantsTangent, marker='o')
    #plt.title('Зависимость константы ошибки от шага сетки для обратной интерполяции')
    #plt.xlabel('Шаг сетки (h)')
    #plt.ylabel('Константа ошибки')
    #plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
