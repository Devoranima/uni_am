import numpy as np
import matplotlib.pyplot as plt


def linear_interpolation(x, y, x_new):
    if len(x) != len(y):
        raise ValueError("Массивы x и y должны иметь одинаковую длину.")
    x = np.array(x)
    y = np.array(y)
    # Находим индексы, где x_new должен быть вставлен в x
    indices = np.searchsorted(x, x_new)  # Индексы, где x_new попадает в x
    indices = np.clip(indices, 1, len(x) - 1)  # Ограничиваем индексы, чтобы избежать выхода за пределы
    y_new = y[indices - 1] + (y[indices] - y[indices - 1]) * (x_new - x[indices - 1]) / (x[indices] - x[indices - 1])
    return y_new
    
def find_const(y_new_std,y_new,x):
    norma=np.max((abs(y_new_std-y_new)))
    h=max(np.diff(x))
    return norma/h**2


a=0
b=1
h=0.1
x_new = np.array([0.59345678901239997891,0.643711098765432109876,0.71321098765432109876,0.98765432109876543212])
y_new_std=np.sqrt(x_new)
y_reverse = np.array([0.7703614664579328, 0.8023160840726263, 0.8445181985316841, 0.9938079900526832])
np.set_printoptions(precision=20, suppress=True)
while h>=1e-5:
    x = np.arange(a,b+h,h)
    y = np.sqrt(x)   
    y_new=linear_interpolation(x,y,x_new)
    x_reverse=linear_interpolation(y,x,y_reverse)
    c = find_const(y_new_std,y_new,x)
    print(c)
    h/=10
print(y_new)
print(y_new_std)
print(x_reverse)
plt.plot(x, y, label='y = fn(x)')
plt.scatter(x_new, y_new,marker='x',color='g')
plt.title('График функции')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.show()
plt.plot(y, x, label='x = gn(y)',color='g')
plt.scatter(y_reverse,x_reverse,marker='x',color='b')
plt.title('График обратной функции')
plt.xlabel('y')
plt.ylabel('x')
plt.legend()
plt.grid()
plt.show()
