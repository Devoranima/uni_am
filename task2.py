import numpy as np
import matplotlib.pyplot as plt
from utils import findErrorConstant, createInterpolationGrid, selectRandomPoints

FUNCTION = np.sin

#TODO: add tangent

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


def main_0():
    a = 0
    b = 1
    h = 0.1

    # Выбираем случайные точки для проверки
    xNew = selectRandomPoints(a, b, num_points=5)
    yNewTrue = FUNCTION(xNew)

    # Подбор шага сетки
    hValues = []
    errorConstants = []
    while h >= 1e-3:
        xBase = createInterpolationGrid(a, b, h)
        yBase = FUNCTION(xBase)
        yNewPred = cubicSpline(xBase, yBase, xNew)

        errorConstant = findErrorConstant(yNewTrue, yNewPred, xBase, 4) # Для кубического сплайна ошибка ~ O(h^4)
        hValues.append(h)
        errorConstants.append(errorConstant)
        h /= 2

    # График зависимости ошибки от шага сетки
    plt.plot(hValues, errorConstants, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Зависимость ошибки от шага сетки')
    plt.xlabel('Шаг сетки (h)')
    plt.ylabel('Константа ошибки')
    plt.grid()
    plt.show()

def main():
    a = 0
    b = 1
    h = 0.1
    #xNew = selectRandomPoints(a, b, num_points=5)
    xNew = np.array([0.37454012, 0.59865848, 0.73199394, 0.95071431])
    yNewTrue = FUNCTION(xNew)
    np.set_printoptions(precision=20, suppress=True)
    while h >= 1e-3:
        xBase = createInterpolationGrid(a, b, h)
        yBase = FUNCTION(xBase)
        yNewPred = cubicSpline(xBase, yBase, xNew)
        errorConstant = findErrorConstant(yNewTrue, yNewPred, h, 4) # Для кубического сплайна ошибка ~ O(h^4)
        print("step: {:10f} | error constant: {:20f}".format(h, errorConstant))
        h /= 2


if __name__ == "__main__":
    main()