import math as m
import pandas as pd
def simpson_integration(func, a, b, tol=1e-4):
    n = 2  
    integral_prev = 0  
    integral = 0  

    while True:
        h = (b - a) / n
        integral = func(a) + func(b)

        for i in range(1, n, 2):
            integral += 4 * func(a + i * h)  # Четные коэффициенты

        for i in range(2, n-1, 2):
            integral += 2 * func(a + i * h)  # Нечетные коэффициенты

        integral *= h / 3 

        if abs(integral - integral_prev) < tol:
            break

        integral_prev = integral
        n *= 2  # Увеличиваем количество подынтервалов

    return integral*2/alpha


# f(x)=x^2
def f2(phi):
    #return m.sqrt(1+4*phi*phi)/(m.sqrt(1-phi)*m.sqrt(1+phi)
    return m.sin(phi) * m.sqrt(1 + 4 * (m.sin(phi)**4)) / m.sqrt(1 + (m.sin(phi)**2))

#f(x)=x^3
def f3(phi):
    return m.sin(phi) * m.sqrt(1 + 9 * (m.sin(phi)**8)) / m.sqrt(1 + (m.sin(phi)**2)+(m.sin(phi)**4))

#f(x)=x^4
def f4(phi):
    return m.sin(phi) * m.sqrt(1 + 16 * (m.sin(phi)**18)) / m.sqrt(1 + (m.sin(phi)**2)+(m.sin(phi)**4)+(m.sin(phi)**6))


def central_rectangle_integration(func, a, b, tolerance=1e-4):
    n = 4  
    previous_result = 0  

    while True:
        h = (b - a) / n
        current_result = 0
        for i in range(n):
            x = a + (i + 0.5) * h  
            current_result += func(x)
        current_result *= h 

        if abs(current_result - previous_result) < tolerance:
            break

        previous_result = current_result
        n *= 2  

    return current_result*2/alpha


def trapezoidal_integration(func, a, b, tolerance=1e-4):
    n = 2  
    previous_result = 0  

    while True:
        h = (b - a) / n
        current_result = 0.5 * (func(a) + func(b))  
        for i in range(1, n):
            x = a + i * h  
            current_result += func(x)

        current_result *= h  

        if abs(current_result - previous_result) < tolerance:
            break

        previous_result = current_result
        n *= 2 

    return current_result*2/alpha

def gauss_integration(func, a, b, tolerance=1e-4):
    n = 2  
    previous_result = 0  

    while True:
        h = (b - a) / n
        current_result = 0

        for i in range(n):
            x0 = a + i * h
            x1 = a + (i + 1) * h
            
            xi1 = (x0 + x1) / 2 - (x1 - x0) / (2 * (3 ** 0.5))
            xi2 = (x0 + x1) / 2 + (x1 - x0) / (2 * (3 ** 0.5))

            w1 = w2 = (x1 - x0) / 2
            current_result += w1 * func(xi1) + w2 * func(xi2)

        if abs(current_result - previous_result) < tolerance:
            break
        previous_result = current_result
        n *= 2 

    return current_result*2/alpha


a = 0
b = m.pi/2
h = 0.01
g = 9.8
alpha = m.sqrt(2 * g)

functions = [f2, f3,f4]
function_names = ['x^2', 'x^3', 'x^4']
quadrature_methods = [
    central_rectangle_integration,
    trapezoidal_integration,
    simpson_integration,
    gauss_integration
]
quadrature_names = ['Central', 'Trapeze', 'Simpson', 'Gauss']


results = []

for f, name in zip(functions, function_names):
    row = [name]
    for method in quadrature_methods:
        result = method(f, a, b)
        row.append(result)
    results.append(row)


results_df = pd.DataFrame(results, columns=['Function'] + quadrature_names)


print(results_df)
