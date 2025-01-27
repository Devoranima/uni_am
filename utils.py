import numpy as np

def findErrorConstant(yTrue, yPred, h, scale = 2):
    maxErr = np.max(np.abs(yTrue - yPred))
    return maxErr / h**scale


def createInterpolationGrid(a, b, h):
    return np.arange(a, b + h, h)


def selectRandomPoints(a, b, num_points=5):
    return np.sort(np.random.uniform(a, b, num_points))