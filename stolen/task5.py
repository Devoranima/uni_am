import numpy as np
import matplotlib.pyplot as plt

def build_adaptive_grid(func, x_min, x_max, num_points=100, sensitivity=1.0):

    x_uniform = np.linspace(x_min, x_max, num_points * 10)
    y_uniform = func(x_uniform)
    
    dydx = np.abs(np.gradient(y_uniform, x_uniform))

    weights = dydx + sensitivity * 1e-3
    weights /= weights.sum()
    
    # Создаем неравномерную сетку на основе накопленной суммы весов
    cumulative_weights = np.cumsum(weights)
    cumulative_weights /= cumulative_weights[-1]  # Нормализация
    x_adaptive = np.interp(np.linspace(0, 1, num_points), cumulative_weights, x_uniform)
    
    return x_adaptive

# Функция f(x) = a * arctan(x / eps)
def target_function(x, a=1, eps=0.1):
    return a * np.arctan(x / eps)

# Параметры функции и сетки
a = 1
eps = 0.001
x_min = -5
x_max = 5
num_points = 100
sensitivity = 1.0

# Строим адаптивную сетку
func = lambda x: target_function(x, a, eps)
x_adaptive = build_adaptive_grid(func, x_min, x_max, num_points, sensitivity)


x_uniform = np.linspace(x_min, x_max, num_points)
y_uniform = target_function(x_uniform, a, eps)
y_adaptive = target_function(x_adaptive, a, eps)

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(x_uniform, y_uniform, '-x', label='Равномерная сетка')
plt.title(f"Равномерная сетка для f(x) =  arctan(x / {eps})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x_adaptive, y_adaptive, label='Адаптивная сетка')
plt.scatter(x_adaptive, y_adaptive, marker='x', color='r', label='Адаптивная сетка')
plt.title(f"Адаптивная сетка для f(x) = arctan(x / {eps})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

plt.tight_layout() 
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def build_euclidean_grid(func, x_min, x_max, num_points=100, sub_points=10):
    x_dense = np.linspace(x_min, x_max, num_points * sub_points)
    y_dense = func(x_dense)
    distances = np.zeros_like(x_dense)
    
    # Вычисляем расстояния для всех промежутков
    for i in range(1, len(x_dense) - 1):
        x1, x2 = x_dense[i - 1], x_dense[i + 1]
        y1, y2 = y_dense[i - 1], y_dense[i + 1]
        xi, yi = x_dense[i], y_dense[i]
        
        # Уравнение прямой через точки (x1, y1) и (x2, y2)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:  
            distances[i] = 0
            continue

        numerator = abs(dy * xi - dx * yi + x2 * y1 - y2 * x1)
        denominator = np.sqrt(dx**2 + dy**2)
        distances[i] = numerator / denominator

    weights = distances + 1e-6 
    weights /= weights.sum()

    cumulative_weights = np.cumsum(weights)
    cumulative_weights /= cumulative_weights[-1] 
    x_adaptive = np.interp(np.linspace(0, 1, num_points), cumulative_weights, x_dense)
    
    return x_adaptive


def target_function(x, a=1, eps=0.1):
    return np.arctan(x / eps)

# Параметры
x_min = -5
x_max = 5
num_points = 100
a = 1
eps = 0.001

func = lambda x: target_function(x, a, eps)
x_adaptive = build_euclidean_grid(func, x_min, x_max, num_points)

x_uniform = np.linspace(x_min, x_max, num_points)
y_uniform = target_function(x_uniform, a, eps)
y_adaptive = target_function(x_adaptive, a, eps)

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(x_uniform, y_uniform, '-x', label='Равномерная сетка')
plt.title(f"Равномерная сетка для f(x) =  arctan(x / {eps})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x_adaptive, y_adaptive, label='Адаптивная сетка')
plt.scatter(x_adaptive, y_adaptive, marker='x', color='r', label='Адаптивная сетка')
plt.title(f"Адаптивная сетка для f(x) = arctan(x / {eps})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

plt.tight_layout() 
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# def build_non_euclidean_grid(func, x_min, x_max, num_points=100, sub_points=10):
#     """
#     Строит неравномерную сетку на основе взвешенного расстояния по оси y (не Евклидово расстояние).
    
#     :param func: Целевая функция.
#     :param x_min: Минимальное значение x.
#     :param x_max: Максимальное значение x.
#     :param num_points: Количество точек в итоговой сетке.
#     :param sub_points: Количество точек для оценки расстояния между сеточными узлами.
#     :return: Массив с координатами сеточных точек.
#     """
#     # Создаем равномерно распределенную густую сетку
#     x_dense = np.linspace(x_min, x_max, num_points * sub_points)
#     y_dense = func(x_dense)
    
#     # Инициализируем массив для хранения расстояний
#     distances = np.zeros_like(x_dense)
    
#     # Вычисляем вертикальное отклонение (по оси y) от аппроксимирующей прямой
#     for i in range(1, len(x_dense) - 1):
#         x1, x2 = x_dense[i - 1], x_dense[i + 1]
#         y1, y2 = y_dense[i - 1], y_dense[i + 1]
#         xi, yi = x_dense[i], y_dense[i]
        
#         # Уравнение прямой: y = mx + b
#         if x2 != x1:
#             m = (y2 - y1) / (x2 - x1)  # Наклон
#             b = y1 - m * x1            # Свободный член
#             y_proj = m * xi + b        # y-координата точки на прямой
#         else:
#             y_proj = y1  # Если x1 == x2, используем фиксированное значение y
        
#         # Расстояние вдоль оси y
#         distances[i] = abs(yi - y_proj)
    
#     # Нормализация весов
#     weights = distances + 1e-6  # Добавляем маленькое число для предотвращения деления на 0
#     weights /= weights.sum()
    
#     # Генерация адаптивной сетки на основе накопленных весов
#     cumulative_weights = np.cumsum(weights)
#     cumulative_weights /= cumulative_weights[-1]  # Нормализация накопленных весов
#     x_adaptive = np.interp(np.linspace(0, 1, num_points), cumulative_weights, x_dense)
    
#     return x_adaptive

# # Целевая функция f(x) = arctan(x / eps)
# def target_function(x, a=1, eps=0.1):
#     return a * np.arctan(x / eps)

# x_min = -5
# x_max = 5
# num_points = 100
# a = 1
# eps = 0.001

# func = lambda x: target_function(x, a, eps)
# x_adaptive = build_euclidean_grid(func, x_min, x_max, num_points)

# x_uniform = np.linspace(x_min, x_max, num_points)
# y_uniform = target_function(x_uniform, a, eps)
# y_adaptive = target_function(x_adaptive, a, eps)

# plt.figure(figsize=(12, 12))

# plt.subplot(2, 1, 1)
# plt.plot(x_uniform, y_uniform, '-x', label='Равномерная сетка')
# plt.title(f"Равномерная сетка для f(x) =  arctan(x / {eps})")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(x_adaptive, y_adaptive, label='Адаптивная сетка')
# plt.scatter(x_adaptive, y_adaptive, marker='x', color='r', label='Адаптивная сетка')
# plt.title(f"Адаптивная сетка для f(x) = arctan(x / {eps})")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.grid(True)
# plt.legend()

# plt.tight_layout() 
# plt.show()