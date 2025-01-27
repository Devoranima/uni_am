import math as m
import pandas as pd

def simpsonIntegration(func, a, b, tol=1e-4):
    """
    Численное интегрирование методом Симпсона.
    """
    n = 2
    integralPrev = 0
    integral = 0

    while True:
        h = (b - a) / n
        integral = func(a) + func(b)

        # Четные коэффициенты
        for i in range(1, n, 2):
            integral += 4 * func(a + i * h)

        # Нечетные коэффициенты
        for i in range(2, n - 1, 2):
            integral += 2 * func(a + i * h)

        integral *= h / 3

        # Проверка точности
        if abs(integral - integralPrev) < tol:
            break

        integralPrev = integral
        n *= 2  # Увеличиваем количество подынтервалов

    return integral * 2 / alpha

def centralRectangleIntegration(func, a, b, tolerance=1e-4):
    """
    Численное интегрирование методом центральных прямоугольников.
    """
    n = 4
    previousResult = 0

    while True:
        h = (b - a) / n
        currentResult = 0

        # Суммируем значения функции в средних точках
        for i in range(n):
            x = a + (i + 0.5) * h
            currentResult += func(x)

        currentResult *= h

        # Проверка точности
        if abs(currentResult - previousResult) < tolerance:
            break

        previousResult = currentResult
        n *= 2

    return currentResult * 2 / alpha

def trapezoidalIntegration(func, a, b, tolerance=1e-4):
    """
    Численное интегрирование методом трапеций.
    """
    n = 2
    previousResult = 0

    while True:
        h = (b - a) / n
        currentResult = 0.5 * (func(a) + func(b))

        # Суммируем значения функции в узлах
        for i in range(1, n):
            x = a + i * h
            currentResult += func(x)

        currentResult *= h

        # Проверка точности
        if abs(currentResult - previousResult) < tolerance:
            break

        previousResult = currentResult
        n *= 2

    return currentResult * 2 / alpha

def gaussIntegration(func, a, b, tolerance=1e-4):
    """
    Численное интегрирование методом Гаусса.
    """
    n = 2
    previousResult = 0

    while True:
        h = (b - a) / n
        currentResult = 0

        # Применяем квадратурную формулу Гаусса
        for i in range(n):
            x0 = a + i * h
            x1 = a + (i + 1) * h

            # Точки и веса для квадратуры Гаусса
            xi1 = (x0 + x1) / 2 - (x1 - x0) / (2 * (3**0.5))
            xi2 = (x0 + x1) / 2 + (x1 - x0) / (2 * (3**0.5))

            w1 = w2 = (x1 - x0) / 2
            currentResult += w1 * func(xi1) + w2 * func(xi2)

        # Проверка точности
        if abs(currentResult - previousResult) < tolerance:
            break

        previousResult = currentResult
        n *= 2

    return currentResult * 2 / alpha

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

if __name__ == "__main__":
    # Параметры
    a = 0
    b = m.pi / 2
    g = 9.81
    alpha = m.sqrt(2 * g)

    quadratureMethods = [
        centralRectangleIntegration,
        trapezoidalIntegration,
        simpsonIntegration,
        gaussIntegration
    ]
    
    quadratureNames = ['Central Rectangle', 'Trapezoidal', 'Simpson', 'Gauss']

    researchFunctions = {
        'x^2': f2,
        'x^3': f3,
        'x^4': f4
    }
    
    # Вычисление результатов
    results = []
    for name, f in researchFunctions.items():
        row = [name]
        for method in quadratureMethods:
            result = method(f, a, b)
            row.append(result)
        results.append(row)

    # Создание DataFrame для отображения результатов
    resultsDf = pd.DataFrame(results, columns=['Function'] + quadratureNames)
    print(resultsDf)