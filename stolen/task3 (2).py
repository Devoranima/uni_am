import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

def create_basis_matrix(x, z):
    num_points = len(x)
    num_segments = len(z) - 1
    basis_matrix = np.zeros((num_points, num_segments + 1))

    for i in range(num_points):
        for j in range(num_segments + 1):
            if j > 0 and z[j - 1] <= x[i] <= z[j]:
                basis_matrix[i, j] = (x[i] - z[j - 1]) / (z[j] - z[j - 1])
            if j < num_segments and z[j] <= x[i] <= z[j + 1]:
                basis_matrix[i, j] += (z[j + 1] - x[i]) / (z[j + 1] - z[j])
    return basis_matrix

def fit_least_squares(x, f, num_segments):
    z = np.linspace(min(x), max(x), num_segments + 1)
    B = create_basis_matrix(x, z)

    # Минимизация ||Bu - f||^2
    BTB = B.T @ B
    BTf = B.T @ f
    coefficients = solve(BTB, BTf)

    return coefficients, B, z

def approximate_function(coefficients, B):
    return B @ coefficients

def display_results(x, f, x_dense, f_approx, z):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, f, label="Исходные точки", color="red")
    plt.plot(x_dense, f_approx, label="Аппроксимированная функция", color="blue")
    plt.vlines(z, min(f) - 1, max(f) + 1, colors="gray", linestyles="dashed", label="Границы сегментов")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Метод наименьших квадратов с сегментированным базисом")
    plt.grid()
    plt.show()

def calculate_max_interpolation_error(f_sym, f_derivative_2, x_points, y_points):
    mid_points = (x_points[:-1] + x_points[1:]) / 2
    second_derivative_values = np.abs(f_derivative_2(mid_points))

    # Вычисление ошибки интерполяции
    h = x_points[1] - x_points[0]
    return (h ** 2) / 8 * max(second_derivative_values)

# Генерация данных
np.random.seed(0)
x = np.linspace(0, 10, 100)
y_true = np.sin(x)  # Истинная функция
noise = np.random.normal(0, 0.1, x.shape)  # Шум
f = y_true + noise  # Данные с шумом

M = 15
coefficients, B, z = fit_least_squares(x, f, M)

x_dense = np.linspace(min(x), max(x), 500)
B_dense = create_basis_matrix(x_dense, z)
f_approx = approximate_function(coefficients, B_dense)

# Определение функции и её производной
a, b = 0, np.pi
f_sym = lambda x: np.cos(x)
f_derivative_2 = lambda x: -np.cos(x)

N_values = [10, 20, 40, 80, 200]
results = []

for N in N_values:
    x_points = np.linspace(a, b, N + 1)
    y_points = f_sym(x_points)

    h = (b - a) / N
    max_error = calculate_max_interpolation_error(f_sym, f_derivative_2, x_points, y_points)
    c = max_error / (h ** 2)

    results.append((N, max_error, c))
    print(f"N = {N}, h = {h:.6f}, Max Error = {max_error:.6f}, C = {c:.6f}")
# Вывод результатов
for N, max_error, c in results:
    print(f"N = {N}, Max Error = {max_error:.6f}, C = {c:.6f}")

display_results(x, f, x_dense, f_approx, z)
