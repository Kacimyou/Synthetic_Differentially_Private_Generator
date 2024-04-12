# %%
import numpy as np
import itertools


def generate_grid_points(m, d):
    """
    Generate 1/m-net of [0,1]**d.

    Parameters:
        m (int): Number of points in each dimension.
        d (int): Dimensionality of the space.

    Returns:
        numpy.ndarray: Array of grid points with shape (m^d, d).
    """
    # Generate grid points for each dimension
    grid_points = [np.linspace(1 / (2 * m), 1 - 1 / (2 * m), m) for _ in range(d)]
    # Take the Cartesian product of grid points
    points = np.array(list(itertools.product(*grid_points)))
    return points


def sample_from_grid(m, d, k):
    """
    Generate k independent samples from the grid points.

    Parameters:
        m (int): Number of points in each dimension.
        d (int): Dimensionality of the space.
        k (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Array of sampled points with shape (k, d).
    """
    grid_points = generate_grid_points(m, d)
    indices = np.random.choice(len(grid_points), k, replace=True)
    return grid_points[indices]


def get_histogram_indices(m, d):
    """
    Generate all indices of a d multidimensional histogram.

    Parameters:
        m (int): Number of bins in each dimension.
        d (int): Dimensionality of the space.

    Returns:
        list: List of all histogram indices.
    """
    index_ranges = [range(m) for _ in range(d)]
    return list(itertools.product(*index_ranges))


def scale(X):
    # Calculate min and max along each axis
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    rescaling_factors = np.vstack((min_values, max_values))

    X_scaled = (X - rescaling_factors[0]) / (
        rescaling_factors[1] - rescaling_factors[0]
    )

    return X_scaled, rescaling_factors
