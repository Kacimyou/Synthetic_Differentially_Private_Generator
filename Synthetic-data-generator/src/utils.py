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
    grid_points = [np.linspace(0, 1, m) for _ in range(d)]
    # Take the Cartesian product of grid points
    points = np.array(list(itertools.product(*grid_points)))
    return points


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
