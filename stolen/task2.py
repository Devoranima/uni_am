import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def cubic_spline(x, y, x_new):
    n = len(x)
    a = y.copy()
    h = np.diff(x)
    
    A_row_indices = []
    A_col_indices = []
    A_values = []
    rhs = np.zeros(n)
    
    # Граничные условия
    A_row_indices.append(0)
    A_col_indices.append(0)
    A_values.append(1.0)
    
    for i in range(1, n - 1):
        A_row_indices.append(i)
        A_col_indices.append(i - 1)
        A_values.append(h[i - 1])
        
        A_row_indices.append(i)
        A_col_indices.append(i)
        A_values.append(2 * (h[i - 1] + h[i]))
        
        A_row_indices.append(i)
        A_col_indices.append(i + 1)
        A_values.append(h[i])
    
    A_row_indices.append(n - 1)
    A_col_indices.append(n - 1)
    A_values.append(1.0)
    
    rhs[1:n-1] = 3 * ((a[2:n] - a[1:n-1]) / h[1:n-1] - (a[1:n-1] - a[0:n-2]) / h[0:n-2])
 
    A = csr_matrix((A_values, (A_row_indices, A_col_indices)), shape=(n, n))
    
    c = spsolve(A, rhs)
    b = (a[1:n] - a[0:n-1]) / h - h * (2 * c[0:n-1] + c[1:n]) / 3
    d = (c[1:n] - c[0:n-1]) / (3 * h)
    y_new = np.zeros_like(x_new)
    
    # Находим соответствующий интервал
    for j, x_new_i in enumerate(x_new):
        idx = np.searchsorted(x, x_new_i) - 1
        idx = np.clip(idx, 0, n - 2)  # Убедимся, что индекс в пределах
        dx = x_new_i - x[idx]
        y_new[j] = a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3
    
    return y_new

def find_const(y_new_std, y_new, x):
    norma = np.max((abs(y_new_std - y_new)))
    h = max(np.diff(x))
    return norma / h**4

def tangent_method(y_new, x, y, max_iter=100, tol=1e-6):
    x_found = np.zeros_like(y_new)
    
    n = len(x)
    h = np.diff(x)
    a = y.copy()
    b = (a[1:n] - a[0:n-1]) / h
    c = np.zeros(n)
    
    for i in range(1, n - 1):
        c[i] = (b[i] - b[i - 1]) / h[i - 1]
    for j, y_target in enumerate(y_new):
       
        idx = np.searchsorted(a, y_target) - 1
        idx = np.clip(idx, 0, n - 2)
        
        x_current = x[idx] + (y_target - a[idx]) / b[idx]
        
        for _ in range(max_iter):
            f_value = cubic_spline(x, y, np.array([x_current]))[0] - y_target
            f_derivative = b[idx] + c[idx] * (x_current - x[idx])

            x_new = x_current - f_value / f_derivative
            
            if abs(x_new - x_current) < tol:
                break
            
            x_current = x_new
        
        x_found[j] = x_current
    
    return x_found



a = 0
b = 1
#h = 0.02
x_new=[0.1612345,0.2598765,0.49136682,0.8136682]
y_new_std = np.sin(x_new)
y_reverse = [0.40154015988441366, 0.5097808352615859,  0.9020355868811385 ]
np.set_printoptions(precision=20, suppress=True)
hset=[0.02] #массив расстояший на которых смотрится константа
for h in hset:
    x = np.arange(a, b + h, h)
    y = np.sin(x)   
    y_new = cubic_spline(x, y, x_new)
    c = find_const(y_new_std, y_new, x)
    print(c)

x_reverse = tangent_method(y_reverse, x, y)
print(y_new)
print(y_new_std)
print(x_reverse)
