import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

FUNCTION = np.sin
FUNCTION_DERIVATIVE_2 = lambda x: -np.sin(x)

def createBasisMatrix(x, z):
    """
    Создает матрицу базисных функций для сегментированного базиса.
    """
    numPoints = len(x)
    numSegments = len(z) - 1
    basisMatrix = np.zeros((numPoints, numSegments + 1))

    for i in range(numPoints):
        for j in range(numSegments + 1):
            if j > 0 and z[j - 1] <= x[i] <= z[j]:
                basisMatrix[i, j] = (x[i] - z[j - 1]) / (z[j] - z[j - 1])
            if j < numSegments and z[j] <= x[i] <= z[j + 1]:
                basisMatrix[i, j] += (z[j + 1] - x[i]) / (z[j + 1] - z[j])
    return basisMatrix

def fitLeastSquares(x, f, numSegments):
    """
    Решает задачу наименьших квадратов для сегментированного базиса.
    """
    z = np.linspace(min(x), max(x), numSegments + 1)
    B = createBasisMatrix(x, z)

    # Минимизация ||Bu - f||^2
    BTB = B.T @ B
    BTf = B.T @ f
    coefficients = solve(BTB, BTf)

    return coefficients, B, z


def calculateMaxInterpolationError(fDerivative2, xPoints):
    """
    Вычисляет максимальную ошибку интерполяции.
    """
    midPoints = (xPoints[:-1] + xPoints[1:]) / 2
    secondDerivativeValues = np.abs(fDerivative2(midPoints))

    # Вычисление ошибки интерполяции
    h = xPoints[1] - xPoints[0]
    return (h**2) / 8 * max(secondDerivativeValues)



if __name__ == "__main__":
    #? Генерация данных
    np.random.seed(13)
    
    x = np.linspace(0, 10, 100)
    yTrue = FUNCTION(x)  # Истинная функция
    noise = np.random.normal(0, 0.1, x.shape)  # Шум
    f = yTrue + noise  # Данные с шумом

    # Параметры аппроксимации
    M = 15
    coefficients, B, z = fitLeastSquares(x, f, M)

    # Аппроксимация на плотной сетке
    xDense = np.linspace(min(x), max(x), 500)
    BDense = createBasisMatrix(xDense, z)
    fApprox = BDense @ coefficients

    # Определение функции и её второй производной
    a, b = 0, np.pi

    # Вычисление ошибки для разных значений N
    NValues = [10, 20, 40, 80, 200]

    for N in NValues:
        xPoints = np.linspace(a, b, N + 1)
        maxError = calculateMaxInterpolationError(FUNCTION_DERIVATIVE_2, xPoints)
        h = (b - a) / N
        c = maxError / (h**2)
        print(f"N = {N}, h = {h:.6f}, Max Error = {maxError:.6f}, C = {c:.6f}")


    plt.figure(figsize=(10, 6))
    plt.scatter(x, f, label="Исходные точки", color="red")
    plt.plot(xDense, fApprox, label="Аппроксимированная функция", color="blue")
    plt.vlines(z, min(f) - 1, max(f) + 1, colors="gray", linestyles="dashed", label="Границы сегментов")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Метод наименьших квадратов с сегментированным базисом")
    plt.grid()
    plt.show()