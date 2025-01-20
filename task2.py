import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


FUNCTION = np.sin

def cubicSpline(x, y, xNew):
    n = len(x)
    a = y.copy()
    h = np.diff(x)
    
    ARowIndices = []
    AColIndices = []
    AValues = []
    rhs = np.zeros(n)
    
    # Граничные условия
    ARowIndices.append(0)
    AColIndices.append(0)
    AValues.append(1.0)
    
    for i in range(1, n - 1):
        ARowIndices.append(i)
        AColIndices.append(i - 1)
        AValues.append(h[i - 1])
        
        ARowIndices.append(i)
        AColIndices.append(i)
        AValues.append(2 * (h[i - 1] + h[i]))
        
        ARowIndices.append(i)
        AColIndices.append(i + 1)
        AValues.append(h[i])
    
    ARowIndices.append(n - 1)
    AColIndices.append(n - 1)
    AValues.append(1.0)
    
    rhs[1:n-1] = 3 * ((a[2:n] - a[1:n-1]) / h[1:n-1] - (a[1:n-1] - a[0:n-2]) / h[0:n-2])
 
    A = csr_matrix((AValues, (ARowIndices, AColIndices)), shape=(n, n))
    
    c = spsolve(A, rhs)
    b = (a[1:n] - a[0:n-1]) / h - h * (2 * c[0:n-1] + c[1:n]) / 3
    d = (c[1:n] - c[0:n-1]) / (3 * h)
    yNew = np.zeros_like(xNew)
    

    for j, xNew_i in enumerate(xNew):
        idx = np.searchsorted(x, xNew_i) - 1
        idx = np.clip(idx, 0, n - 2)
        dx = xNew_i - x[idx]
        yNew[j] = a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3
    
    return yNew

def findErrorConstant(yGiven, y, x):
    maxErr = np.max((abs(yGiven-y)))
    h = max(np.diff(x))
    return maxErr/h**2

def tangentMethod(yNew, x, y, maxIterations=100, tolerance=1e-6):
    xFound = np.zeros_like(yNew)
    
    n = len(x)
    h = np.diff(x)
    a = y.copy()
    b = (a[1:n] - a[0:n-1]) / h
    c = np.zeros(n)
    
    for i in range(1, n - 1):
        c[i] = (b[i] - b[i - 1]) / h[i - 1]
        
    for j, yTarget in enumerate(yNew):
        idx = np.searchsorted(a, yTarget) - 1
        idx = np.clip(idx, 0, n - 2)
        
        xCurrent = x[idx] + (yTarget - a[idx]) / b[idx]
        
        for _ in range(maxIterations):
            fValue = cubicSpline(x, y, np.array([xCurrent]))[0] - yTarget
            fDerivative = b[idx] + c[idx] * (xCurrent - x[idx])

            xNew = xCurrent - fValue / fDerivative
            
            if abs(xNew - xCurrent) < tolerance:
                break
            
            xCurrent = xNew
        
        xFound[j] = xCurrent
    
    return xFound


if __name__ == "__main__":
    a = 0
    b = 1

    xNew = [0.1257438, 0.3264987, 0.4679823, 0.8593421]
    yNew = FUNCTION(xNew)

    np.set_printoptions(precision=20, suppress=True)
    hset=[0.02]
    for h in hset:
        xBase = np.arange(a, b+h, h)
        yBase = FUNCTION(xBase)
        y = cubicSpline(xBase, yBase, xNew)
        c = findErrorConstant(yNew, y, xBase)
        print(f"Error constant: {c}")

    x = tangentMethod(yNew, xBase, yBase)
    print(f"Interpolated values: {y}")
    print(f"Exact values: {yNew}")
    print(f"Inverse interpolated x: {x}")
