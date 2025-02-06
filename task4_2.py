import math as m
import pandas as pd
import matplotlib.pyplot as plt


testConvergence=True

def simpsonIntegration(func, a, b, n):
    """
    Численное интегрирование методом Симпсона.
    """
    h = (b - a) / n
    integral = func(a) + func(b)

    # Четные коэффициенты
    for i in range(1, n, 2):
        integral += 4 * func(a + i * h)

    # Нечетные коэффициенты
    for i in range(2, n - 1, 2):
        integral += 2 * func(a + i * h)

    integral *= h / 3

    return integral


def centralRectangleIntegration(func, a, b, n):
    """
    Численное интегрирование методом центральных прямоугольников.
    """
    h = (b - a) / n
    integral = 0

    # Суммируем значения функции в средних точках
    for i in range(n):
        x = a + (i + 0.5) * h
        integral += func(x)

    integral *= h

    return integral


def trapezoidalIntegration(func, a, b, n):
    """
    Численное интегрирование методом трапеций.
    """

    h = (b - a) / n
    integral = 0.5 * (func(a) + func(b))

    # Суммируем значения функции в узлах
    for i in range(1, n):
        x = a + i * h
        integral += func(x)

    integral *= h

    return integral


def gaussIntegration(func, a, b, n):
    """
    Численное интегрирование методом Гаусса.
    """

    h = (b - a) / n
    integral = 0

    # Применяем квадратурную формулу Гаусса
    for i in range(n):
        x0 = a + i * h
        x1 = a + (i + 1) * h

        # Точки и веса для квадратуры Гаусса
        xi1 = (x0 + x1) / 2 - (x1 - x0) / (2 * (3**0.5))
        xi2 = (x0 + x1) / 2 + (x1 - x0) / (2 * (3**0.5))

        w1 = w2 = (x1 - x0) / 2
        integral += w1 * func(xi1) + w2 * func(xi2)

    return integral

# Функции для интегрирования


def f2(phi):
    """
    Функция для f(x) = x^2.
    """
    return m.sin(phi) * m.sqrt(1 + 4 * (m.sin(phi)**4)) / m.sqrt(1 + (m.sin(phi)**2))


def f3(phi):
    """
    Функция для f(x) = x^3.
    """
    return m.sin(phi) * m.sqrt(1 + 9 * (m.sin(phi)**8)) / m.sqrt(1 + (m.sin(phi)**2) + (m.sin(phi)**4))


def f4(phi):
    """
    Функция для f(x) = x^4.
    """
    return m.sin(phi) * m.sqrt(1 + 16 * (m.sin(phi)**18)) / m.sqrt(1 + (m.sin(phi)**2) + (m.sin(phi)**4) + (m.sin(phi)**6))


def calculateIntegration(method, func, a, b):
    """
    Проверка сходимости метода интегрирования.
    """

    tolerance = 10e-5
    n = 2
    previousResult = 0
    iterations = []

    while True:
        currentResult = method[1](func, a, b, n)

        diff = abs(currentResult - previousResult)
        if diff < tolerance:
            break

        previousResult = currentResult
        n *= 2
        iterations.append((n, diff))

    if (testConvergence):
        print(f"Convergence sequence for {method[0]}:")
        for n, diff in iterations:
            print(
                f"\t Interpolation grid length = {n} | values difference = {diff}")

    return currentResult


if __name__ == "__main__":
    # Параметры
    a = 0
    b = m.pi / 2
    g = 9.81
    alpha = m.sqrt(2 * g)

    integrationMethods = [
        ("Central Rectangle", centralRectangleIntegration),
        ("Trapezoidal", trapezoidalIntegration),
        ("Simpson", simpsonIntegration),
        ("Gauss", gaussIntegration)
    ]

    researchFunctions = {
        'x^2': f2,
        'x^3': f3,
        'x^4': f4
    }

    # Вычисление результатов
    results = []
    for name, f in researchFunctions.items():
        if testConvergence:
            print(f"\nRunning integration for {name}")
        row = [name]
        for method in integrationMethods:
            result = calculateIntegration(method, f, a, b)
            row.append(result)
        results.append(row)

    # Создание DataFrame для отображения результатов
    resultsDf = pd.DataFrame(
        results, columns=['Function'] + list(map(lambda x: x[0], integrationMethods)))
    print()
    print(resultsDf)
