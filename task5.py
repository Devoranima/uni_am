import numpy as np
import matplotlib.pyplot as plt


def FUNCTION(x, a=1, eps=0.1):
    return a * np.arctan(x / eps)


def buildAdaptiveGrid(func, x_min, x_max, num_points=100, sensitivity=1.0, num_fine=10000):
    """Create adaptive grid with points clustered in regions of high derivative"""
    x_fine = np.linspace(x_min, x_max, num_fine)
    y_fine = func(x_fine)

    # Calculate scaled derivatives
    deriv = np.abs(np.gradient(y_fine, x_fine)) ** sensitivity
    cum_deriv = np.cumsum(deriv)
    cum_deriv = (cum_deriv - cum_deriv[0]) / \
        (cum_deriv[-1] - cum_deriv[0])  # Normalize

    # Generate adaptive grid points
    u = np.linspace(0, 1, num_points)
    return np.interp(u, cum_deriv, x_fine)


def buildEuclideanGrid(func, x_min, x_max, num_points=100, sub_points=10):
    """Create grid with uniform Euclidean distance along the curve"""
    # Create dense grid for accurate arc length calculation
    x_fine = np.linspace(x_min, x_max, num_points * sub_points)
    y_fine = func(x_fine)

    # Calculate arc lengths
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    distances = np.sqrt(dx**2 + dy**2)
    arc_length = np.zeros_like(x_fine)
    arc_length[1:] = np.cumsum(distances)
    arc_length /= arc_length[-1]  # Normalize

    # Generate Euclidean grid points
    target_arc = np.linspace(0, 1, num_points)
    return np.interp(target_arc, arc_length, x_fine)


def main():
    # Parameters
    a = 1
    eps = 0.1
    x_min = -5
    x_max = 5
    num_points = 100
    sensitivity = 1.0

    # Create function with current parameters
    def func(x): return FUNCTION(x, a, eps)

    # Generate grids
    x_default = np.linspace(x_min, x_max, num_points)
    adaptive_grid = buildAdaptiveGrid(
        func, x_min, x_max, num_points, sensitivity)
    euclidean_grid = buildEuclideanGrid(func, x_min, x_max, num_points)

    y_default = func(x_default)
    y_adaptive = func(adaptive_grid)
    y_euclidean = func(euclidean_grid)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.plot(x_default, y_default, label='Target Function', lw=2)
    plt.title(f"Default Grid")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(adaptive_grid, y_adaptive, label='Adaptive Grid Plot')
    plt.scatter(adaptive_grid, y_adaptive, marker='X',
                c='red', label='Adaptive Grid', alpha=0.7)

    plt.title(f"Adaptive Grid")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(adaptive_grid, y_adaptive, label='Euclidean Grid Plot')
    plt.scatter(euclidean_grid, y_euclidean, marker='X',
                c='green', label='Euclidean Grid', alpha=0.7)
    plt.title(f"Euclidean Grid")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
